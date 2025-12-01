use std::{
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
    sync::{
        Arc, RwLock,
        atomic::{AtomicU16, Ordering},
    },
    time::{Duration, Instant},
};

use bitflags::bitflags;
use petgraph::{Direction, visit::EdgeRef};
use tokio::{task, time::timeout};
use tracing::warn;
use uuid::Uuid;

use crate::{
    analysis,
    graph::{
        InvariantCode,
        model::*,
        specs::{AnchorsSpec, AssessesSpec, EdgeSpec, PrecedesSpec, RequiresSpec, SupportsSpec},
        traversal,
    },
    schema::types::{AssessmentScope, KnowledgeType, SourceRef},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValidationSeverity {
    Warning,
    Error,
}

#[derive(Clone, Debug)]
struct ValidationIssue {
    code:              crate::graph::InvariantCode,
    severity:          ValidationSeverity,
    message:           String,
    promote_in_strict: bool,
}

type GuardCallback<'a> = Box<dyn FnOnce(&mut GraphService) + 'a>;

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct InvariantFamilies: u16 {
        const STATEMENTS    = 1 << 0;
        const PROVENANCE    = 1 << 1;
        const REQUIRES_DAG  = 1 << 2;
        const FADEABILITY   = 1 << 3;
        const COVERAGE      = 1 << 4;
        const SUPPORTS      = 1 << 5;
        const PURITY        = 1 << 6;
        const DISCOURSE     = 1 << 7;
        const INTRODUCTIONS = 1 << 8;
        const ALL           = Self::STATEMENTS.bits()
            | Self::PROVENANCE.bits()
            | Self::REQUIRES_DAG.bits()
            | Self::FADEABILITY.bits()
            | Self::COVERAGE.bits()
            | Self::SUPPORTS.bits()
            | Self::PURITY.bits()
            | Self::DISCOURSE.bits()
            | Self::INTRODUCTIONS.bits();
    }
}

struct ValidationState {
    dirty: AtomicU16,
}

#[allow(dead_code)]
pub mod test_support {
    use std::sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    };

    static TEST_VALIDATION_DELAY_MS: OnceLock<AtomicU64> = OnceLock::new();

    pub fn set_test_validation_delay_ms(delay: u64) -> u64 {
        TEST_VALIDATION_DELAY_MS
            .get_or_init(|| AtomicU64::new(0))
            .swap(delay, Ordering::Relaxed)
    }

    pub fn test_validation_delay_ms() -> u64 {
        TEST_VALIDATION_DELAY_MS
            .get_or_init(|| AtomicU64::new(0))
            .load(Ordering::Relaxed)
    }
}

impl ValidationState {
    fn new_empty() -> Self {
        Self {
            dirty: AtomicU16::new(0),
        }
    }
}

#[derive(Clone, Debug)]
enum ValidationScope {
    Full,
    Targeted {
        coverage_los:      Vec<NodeId>,
        skip_requires_dag: bool,
        skip_fadeability:  bool,
    },
}

#[derive(Clone, Debug)]
struct ValidationContext {
    strict:                bool,
    expected_revision:     Option<String>,
    rubric_prev:           HashMap<String, u64>,
    include_rubric_update: bool,
}

/// Lightweight helper for provenance checks to keep error text consistent.
struct Provenance<'a> {
    expected: Option<&'a str>,
}

impl<'a> Provenance<'a> {
    fn new(expected: Option<&'a str>) -> Self {
        Self { expected }
    }

    fn check(&self, spans: &[SourceRef], label: &str) -> Result<(), GraphError> {
        let Some(expected) = self.expected else {
            return Ok(());
        };
        for span in spans {
            if span.revision != expected {
                return Err(GraphError::Schema(format!(
                    "{label} source_ref revision `{}` must equal course_commit `{}`",
                    span.revision, expected
                )));
            }
        }
        Ok(())
    }
}

/// RAII helper to mark invariant families as dirty, run validations, and roll
/// back on error. Intended to reduce repetition across mutation handlers.
struct ValidationGuard<'a> {
    svc:        &'a mut GraphService,
    scope:      ValidationScope,
    rollback:   Option<GuardCallback<'a>>,
    on_success: Option<GuardCallback<'a>>,
    committed:  bool,
}

impl<'a> ValidationGuard<'a> {
    fn new(
        svc: &'a mut GraphService,
        families: InvariantFamilies,
        scope: ValidationScope,
        rollback: impl FnOnce(&mut GraphService) + 'a,
    ) -> Self {
        svc.mark_dirty(families);
        Self {
            svc,
            scope,
            rollback: Some(Box::new(rollback)),
            on_success: None,
            committed: false,
        }
    }

    fn on_success(mut self, f: impl FnOnce(&mut GraphService) + 'a) -> Self {
        self.on_success = Some(Box::new(f));
        self
    }

    fn commit(mut self) -> Result<(), GraphError> {
        let result = match &self.scope {
            ValidationScope::Full => {
                let families = self.svc.planned_families(InvariantFamilies::ALL);
                self.svc
                    .validate_invariants(ValidationScope::Full, families)
            }
            ValidationScope::Targeted {
                coverage_los,
                skip_requires_dag,
                skip_fadeability,
            } => self.svc.validate_targeted_invariants(
                coverage_los.clone(),
                *skip_requires_dag,
                *skip_fadeability,
            ),
        };

        match result {
            Ok(_) => {
                if let Some(cb) = self.on_success.take() {
                    cb(self.svc);
                }
                self.svc.bump_version();
                self.committed = true;
                Ok(())
            }
            Err(err) => {
                if let Some(rb) = self.rollback.take() {
                    rb(self.svc);
                }
                self.committed = true;
                Err(err)
            }
        }
    }
}

impl<'a> Drop for ValidationGuard<'a> {
    fn drop(&mut self) {
        if !self.committed
            && let Some(rb) = self.rollback.take()
        {
            rb(self.svc);
        }
    }
}

/// Core graph owner with slug lookup and optional strict quality mode.
pub struct GraphService {
    graph:             Arc<CurriculumGraph>,
    slug_to_node:      HashMap<String, NodeId>,
    strict_quality:    bool,
    graph_version:     u64,
    expected_revision: Option<String>,
    rubric_hashes:     RwLock<HashMap<String, u64>>,
    validation_state:  ValidationState,
    fade_cache:        RwLock<Option<(u64, analysis::FadeabilityContext)>>,
}

impl GraphService {
    pub fn new() -> Self {
        Self {
            graph:             Arc::new(CurriculumGraph::default()),
            slug_to_node:      HashMap::new(),
            strict_quality:    false,
            graph_version:     0,
            expected_revision: None,
            rubric_hashes:     RwLock::new(HashMap::new()),
            validation_state:  ValidationState::new_empty(),
            fade_cache:        RwLock::new(None),
        }
    }

    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict_quality = strict;
        self
    }

    pub fn from_graph(graph: CurriculumGraph) -> Self {
        Self::from_parts(graph, false, 0, None)
            .expect("graph provided to from_graph must have unique slugs and valid invariants")
    }

    pub fn from_parts(
        mut graph: CurriculumGraph,
        strict_quality: bool,
        graph_version: u64,
        expected_revision: Option<String>,
    ) -> Result<Self, GraphError> {
        normalize_assesses_claims_graph(&mut graph);
        let rubric_hashes = RwLock::new(compute_rubric_hashes(&graph));
        let graph = Arc::new(graph);
        let mut svc = Self {
            graph,
            slug_to_node: HashMap::new(),
            strict_quality,
            graph_version,
            expected_revision: expected_revision.filter(|s| !s.is_empty()),
            rubric_hashes,
            validation_state: ValidationState::new_empty(),
            fade_cache: RwLock::new(None),
        };
        svc.rebuild_slug_index()?;
        svc.validate_global_invariants()?;
        Ok(svc)
    }

    /// Cheap shared pointer for read-heavy callers.
    pub fn shared_graph(&self) -> Arc<CurriculumGraph> {
        Arc::clone(&self.graph)
    }

    pub fn graph(&self) -> &CurriculumGraph {
        &self.graph
    }

    pub fn graph_version(&self) -> u64 {
        self.graph_version
    }

    pub(crate) fn fade_ctx(&self) -> analysis::FadeabilityContext {
        if let Ok(cache) = self.fade_cache.read()
            && let Some((ver, ctx)) = cache.as_ref()
            && *ver == self.graph_version
        {
            return ctx.clone();
        }
        let fresh = analysis::FadeabilityContext::compute(self.graph());
        if let Ok(mut cache) = self.fade_cache.write() {
            *cache = Some((self.graph_version, fresh.clone()));
        }
        fresh
    }

    fn mark_dirty(&self, families: InvariantFamilies) {
        if families.is_empty() {
            return;
        }
        self.validation_state
            .dirty
            .fetch_or(families.bits(), Ordering::Relaxed);
        if let Ok(mut cache) = self.fade_cache.write() {
            cache.take();
        }
    }

    fn planned_families(&self, required: InvariantFamilies) -> InvariantFamilies {
        let dirty = InvariantFamilies::from_bits_truncate(
            self.validation_state.dirty.load(Ordering::Relaxed),
        );
        dirty | required
    }

    fn clear_validated_families(&self, families: InvariantFamilies) {
        if families.is_empty() {
            return;
        }
        let mask = !families.bits();
        self.validation_state
            .dirty
            .fetch_and(mask, Ordering::Relaxed);
    }

    fn families_for_scope(
        mut families: InvariantFamilies,
        scope: &ValidationScope,
    ) -> InvariantFamilies {
        match scope {
            ValidationScope::Full => {}
            ValidationScope::Targeted {
                coverage_los,
                skip_requires_dag,
                skip_fadeability,
            } => {
                if *skip_requires_dag {
                    families.remove(InvariantFamilies::REQUIRES_DAG);
                }
                if *skip_fadeability {
                    families.remove(InvariantFamilies::FADEABILITY);
                }
                if coverage_los.is_empty() {
                    families.remove(InvariantFamilies::COVERAGE);
                }
            }
        }
        families
    }

    pub fn strict_quality(&self) -> bool {
        self.strict_quality
    }

    pub fn expected_revision(&self) -> Option<&str> {
        self.expected_revision.as_deref()
    }

    pub fn set_expected_revision(&mut self, revision: Option<String>) {
        self.expected_revision = revision.filter(|s| !s.is_empty());
    }

    pub fn set_strict_quality(&mut self, strict: bool) -> Result<(), GraphError> {
        let prev = self.strict_quality;
        self.strict_quality = strict;
        if strict && let Err(err) = self.validate_global_invariants() {
            self.strict_quality = prev;
            return Err(err);
        }
        Ok(())
    }

    pub(crate) fn graph_mut(&mut self) -> &mut CurriculumGraph {
        Arc::make_mut(&mut self.graph)
    }

    pub(crate) fn bump_version(&mut self) {
        self.graph_version = self.graph_version.saturating_add(1);
    }

    pub fn upsert_slug(&mut self, slug: String, id: NodeId) {
        self.slug_to_node.insert(slug, id);
    }

    pub fn node_by_slug(&self, slug: &str) -> Result<NodeId, GraphError> {
        self.slug_to_node
            .get(slug)
            .copied()
            .ok_or_else(|| GraphError::MissingSlug(slug.to_string()))
    }

    pub fn add_knowledge_node(
        &mut self,
        slug: String,
        payload: KnowledgeNode,
        tags: Vec<String>,
    ) -> Result<NodeId, GraphError> {
        if self.slug_to_node.contains_key(&slug) {
            return Err(GraphError::Schema(format!(
                "slug `{}` already exists; use update_knowledge_node",
                slug
            )));
        }
        if payload.statement.trim().is_empty() {
            return Err(GraphError::Schema(
                "knowledge node statement must not be empty".to_string(),
            ));
        }
        if payload.source_refs.is_empty() {
            return Err(GraphError::Schema(
                "knowledge nodes must include at least one source_ref".to_string(),
            ));
        }
        validate_source_refs(&payload.source_refs)?;
        self.validate_revision(&payload.source_refs, "knowledge")?;
        let mut dirty = InvariantFamilies::STATEMENTS
            | InvariantFamilies::PROVENANCE
            | InvariantFamilies::SUPPORTS;
        if payload.knowledge_type == KnowledgeType::LearningOutcome
            || payload.knowledge_type.is_assessment_item()
        {
            dirty.insert(InvariantFamilies::COVERAGE);
        }
        if payload.knowledge_type.is_assessment_item() {
            dirty.insert(InvariantFamilies::PURITY);
        }
        let logical_id = Uuid::new_v4();
        let node = NodePayload {
            logical_id,
            slug: slug.clone(),
            kind: NodeKind::Knowledge(payload),
            tags,
        };
        let id = self.graph_mut().add_node(node);
        self.upsert_slug(slug.clone(), id);
        let coverage_los = match &self.graph()[id].kind {
            NodeKind::Knowledge(k) if k.knowledge_type == KnowledgeType::LearningOutcome => {
                vec![id]
            }
            _ => Vec::new(),
        };
        let guard = ValidationGuard::new(
            self,
            dirty,
            ValidationScope::Targeted {
                coverage_los,
                skip_requires_dag: true,
                skip_fadeability: true,
            },
            move |svc| {
                svc.graph_mut().remove_node(id);
                svc.slug_to_node.remove(&slug);
            },
        )
        .on_success(|svc| svc.refresh_rubric_hashes());
        guard.commit()?;
        Ok(id)
    }

    /// Update an existing knowledge node and revalidate all incident edges.
    pub fn update_knowledge_node(
        &mut self,
        slug: &str,
        payload: KnowledgeNode,
        tags: Vec<String>,
    ) -> Result<NodeId, GraphError> {
        let id = self.node_by_slug(slug)?;
        let old_kind = self.graph()[id].kind.clone();
        let old_tags = self.graph()[id].tags.clone();
        if !matches!(&old_kind, NodeKind::Knowledge(_)) {
            return Err(GraphError::Schema(format!(
                "slug `{}` exists as teaching_step; cannot update knowledge",
                slug
            )));
        }

        if payload.statement.trim().is_empty() {
            return Err(GraphError::Schema(
                "knowledge node statement must not be empty".to_string(),
            ));
        }
        if payload.source_refs.is_empty() {
            return Err(GraphError::Schema(
                "knowledge nodes must include at least one source_ref".to_string(),
            ));
        }
        validate_source_refs(&payload.source_refs)?;
        self.validate_revision(&payload.source_refs, "knowledge")?;
        let mut dirty = InvariantFamilies::STATEMENTS
            | InvariantFamilies::PROVENANCE
            | InvariantFamilies::SUPPORTS;
        let mut needs_coverage = false;
        let mut needs_purity = false;
        if let NodeKind::Knowledge(k) = &old_kind {
            if k.knowledge_type == KnowledgeType::LearningOutcome
                || k.knowledge_type.is_assessment_item()
            {
                needs_coverage = true;
            }
            if k.knowledge_type.is_assessment_item() {
                needs_purity = true;
            }
        }
        if payload.knowledge_type == KnowledgeType::LearningOutcome
            || payload.knowledge_type.is_assessment_item()
        {
            needs_coverage = true;
        }
        if payload.knowledge_type.is_assessment_item() {
            needs_purity = true;
        }
        if needs_coverage {
            dirty.insert(InvariantFamilies::COVERAGE);
        }
        if needs_purity {
            dirty.insert(InvariantFamilies::PURITY);
        }

        // apply tentative change
        self.graph_mut()[id].kind = NodeKind::Knowledge(payload);

        // validate incident edges against the new kind
        if let Err(err) = self.validate_incident_edges(id) {
            // rollback
            self.graph_mut()[id].kind = old_kind;
            return Err(err);
        }

        self.graph_mut()[id].tags = tags;
        let mut coverage_los = Vec::new();
        if let NodeKind::Knowledge(k) = &old_kind
            && k.knowledge_type == KnowledgeType::LearningOutcome
        {
            coverage_los.push(id);
        }
        if let NodeKind::Knowledge(k) = &old_kind
            && k.knowledge_type.is_assessment_item()
        {
            coverage_los.extend(self.los_assessed_by(id));
        }
        if let NodeKind::Knowledge(k) = &self.graph()[id].kind
            && k.knowledge_type == KnowledgeType::LearningOutcome
        {
            coverage_los.push(id);
        }
        if let NodeKind::Knowledge(k) = &self.graph()[id].kind
            && k.knowledge_type.is_assessment_item()
        {
            coverage_los.extend(self.los_assessed_by(id));
        }
        let coverage_los: Vec<NodeId> = coverage_los
            .into_iter()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let guard = ValidationGuard::new(
            self,
            dirty,
            ValidationScope::Targeted {
                coverage_los,
                skip_requires_dag: true,
                skip_fadeability: true,
            },
            move |svc| {
                svc.graph_mut()[id].kind = old_kind;
                svc.graph_mut()[id].tags = old_tags;
            },
        )
        .on_success(|svc| svc.refresh_rubric_hashes());
        guard.commit()?;
        Ok(id)
    }

    fn validate_incident_edges(&self, node: NodeId) -> Result<(), GraphError> {
        for edge in self.graph().edges_directed(node, Direction::Incoming) {
            self.validate_edge_kind(
                edge.source(),
                node,
                &edge.weight().kind,
                edge.weight().confidence,
            )?;
        }
        for edge in self.graph().edges_directed(node, Direction::Outgoing) {
            self.validate_edge_kind(
                node,
                edge.target(),
                &edge.weight().kind,
                edge.weight().confidence,
            )?;
        }
        Ok(())
    }

    fn validate_edge_kind(
        &self,
        from: NodeId,
        to: NodeId,
        kind: &EdgeKind,
        confidence: f32,
    ) -> Result<(), GraphError> {
        match kind {
            EdgeKind::Requires(a) => {
                RequiresSpec::validate(self, from, to, a, confidence)?;
                self.validate_revision(&a.evidence_refs, "requires")?;
                Ok(())
            }
            EdgeKind::Supports(a) => {
                SupportsSpec::validate(self, from, to, a, confidence)?;
                self.validate_revision(&a.evidence_refs, "supports")?;
                Ok(())
            }
            EdgeKind::Assesses(a) => AssessesSpec::validate(self, from, to, a, confidence),
            EdgeKind::Precedes(a) => PrecedesSpec::validate(self, from, to, a, confidence),
            EdgeKind::Anchors(a) => AnchorsSpec::validate(self, from, to, a, confidence),
        }
    }

    pub fn rename_node(&mut self, old_slug: &str, new_slug: String) -> Result<(), GraphError> {
        if self.slug_to_node.contains_key(&new_slug) {
            return Err(GraphError::Schema(format!("slug `{}` already exists", new_slug)));
        }

        let id = self.node_by_slug(old_slug)?;
        let old_index = self.slug_to_node.clone();
        let old_rubric = self
            .rubric_hashes
            .read()
            .expect("rubric_hashes lock")
            .clone();
        let incoming: Vec<_> = self
            .graph()
            .edges_directed(id, Direction::Incoming)
            .map(|e| e.id())
            .collect();
        let incoming_for_rollback = incoming.clone();

        self.slug_to_node.remove(old_slug);
        self.slug_to_node.insert(new_slug.clone(), id);
        self.graph_mut()[id].slug = new_slug.clone();

        // keep denormalized claims in sync for assesses edges targeting this node
        for edge_id in incoming {
            if let EdgeKind::Assesses(attrs) = &mut self.graph_mut()[edge_id].kind {
                attrs.evidence_link.claim = new_slug.clone();
            }
        }

        let guard = ValidationGuard::new(self, InvariantFamilies::ALL, ValidationScope::Full, {
            let old_slug = old_slug.to_string();
            let old_index = old_index.clone();
            let old_rubric = old_rubric.clone();
            let incoming = incoming_for_rollback.clone();
            move |svc| {
                svc.slug_to_node = old_index;
                svc.graph_mut()[id].slug = old_slug.clone();
                for edge_id in incoming.iter().copied() {
                    if let EdgeKind::Assesses(attrs) = &mut svc.graph_mut()[edge_id].kind {
                        attrs.evidence_link.claim = old_slug.clone();
                    }
                }
                *svc.rubric_hashes.write().expect("rubric_hashes lock") = old_rubric;
            }
        })
        .on_success(|svc| svc.refresh_rubric_hashes());
        guard.commit()
    }

    pub fn remove_node(&mut self, slug: &str) -> Result<(), GraphError> {
        let id = self.node_by_slug(slug)?;

        // Stash current state for rollback on invariant failure.
        let old_graph = self.graph.clone();
        let old_index = self.slug_to_node.clone();
        let old_version = self.graph_version;
        let old_rubric = self
            .rubric_hashes
            .read()
            .expect("rubric_hashes lock")
            .clone();

        self.slug_to_node.remove(slug);
        self.graph_mut().remove_node(id);

        let guard =
            ValidationGuard::new(self, InvariantFamilies::ALL, ValidationScope::Full, move |svc| {
                svc.graph = old_graph;
                svc.slug_to_node = old_index;
                svc.graph_version = old_version;
                *svc.rubric_hashes.write().expect("rubric_hashes lock") = old_rubric;
            })
            .on_success(|svc| svc.refresh_rubric_hashes());
        guard.commit()?;
        Ok(())
    }

    pub fn snapshot_graph(&self) -> CurriculumGraph {
        (*self.graph).clone()
    }

    pub fn install_graph(
        &mut self,
        mut graph: CurriculumGraph,
        graph_version: u64,
    ) -> Result<(), GraphError> {
        normalize_assesses_claims_graph(&mut graph);

        // Build slug index first; only commit if it succeeds.
        let candidate = Arc::new(graph);
        let index = self.build_slug_index_map(candidate.as_ref())?;

        let old_graph = self.graph.clone();
        let old_index = self.slug_to_node.clone();
        let old_version = self.graph_version;

        self.graph = candidate;
        self.slug_to_node = index;
        self.graph_version = graph_version;
        self.rubric_hashes = RwLock::new(compute_rubric_hashes(self.graph.as_ref()));
        self.mark_dirty(InvariantFamilies::ALL);
        if let Err(err) = self.validate_global_invariants() {
            // rollback on failure
            self.graph = old_graph;
            self.slug_to_node = old_index;
            self.graph_version = old_version;
            self.rubric_hashes = RwLock::new(compute_rubric_hashes(self.graph.as_ref()));
            return Err(err);
        }
        Ok(())
    }

    pub fn replace_graph(&mut self, graph: CurriculumGraph) -> Result<(), GraphError> {
        let next = self.graph_version.saturating_add(1);
        self.install_graph(graph, next)
    }

    fn rebuild_slug_index(&mut self) -> Result<(), GraphError> {
        let map = self.build_slug_index_map(self.graph.as_ref())?;
        self.slug_to_node = map;
        Ok(())
    }

    fn build_slug_index_map(
        &self,
        graph: &CurriculumGraph,
    ) -> Result<HashMap<String, NodeId>, GraphError> {
        let mut map = HashMap::new();
        for n in graph.node_indices() {
            let slug = graph[n].slug.clone();
            if map.insert(slug.clone(), n).is_some() {
                return Err(GraphError::Schema(format!(
                    "duplicate slug `{}` found while rebuilding index",
                    slug
                )));
            }
        }
        Ok(map)
    }

    pub fn add_teaching_step(
        &mut self,
        slug: String,
        payload: TeachingStepNode,
        tags: Vec<String>,
    ) -> Result<NodeId, GraphError> {
        if self.slug_to_node.contains_key(&slug) {
            return Err(GraphError::Schema(format!(
                "slug `{}` already exists; use update_teaching_step",
                slug
            )));
        }
        if payload.statement.trim().is_empty() {
            return Err(GraphError::Schema(
                "teaching_step statement must not be empty".to_string(),
            ));
        }
        if payload.source_refs.is_empty() {
            return Err(GraphError::Schema(
                "teaching steps must include at least one source_ref".to_string(),
            ));
        }
        validate_source_refs(&payload.source_refs)?;
        self.validate_revision(&payload.source_refs, "teaching_step")?;
        let logical_id = Uuid::new_v4();
        let node = NodePayload {
            logical_id,
            slug: slug.clone(),
            kind: NodeKind::TeachingStep(payload),
            tags,
        };
        let id = self.graph_mut().add_node(node);
        self.upsert_slug(slug.clone(), id);
        let guard = ValidationGuard::new(
            self,
            InvariantFamilies::STATEMENTS
                | InvariantFamilies::PROVENANCE
                | InvariantFamilies::DISCOURSE,
            ValidationScope::Targeted {
                coverage_los:      Vec::new(),
                skip_requires_dag: true,
                skip_fadeability:  true,
            },
            move |svc| {
                svc.graph_mut().remove_node(id);
                svc.slug_to_node.remove(&slug);
            },
        )
        .on_success(|svc| svc.refresh_rubric_hashes());
        guard.commit()?;
        Ok(id)
    }

    pub fn update_teaching_step(
        &mut self,
        slug: &str,
        payload: TeachingStepNode,
        tags: Vec<String>,
    ) -> Result<NodeId, GraphError> {
        let id = self.node_by_slug(slug)?;
        let old_kind = self.graph()[id].kind.clone();
        let old_tags = self.graph()[id].tags.clone();
        if !matches!(&old_kind, NodeKind::TeachingStep(_)) {
            return Err(GraphError::Schema(format!(
                "slug `{}` exists as knowledge; cannot update teaching_step",
                slug
            )));
        }
        if payload.statement.trim().is_empty() {
            return Err(GraphError::Schema(
                "teaching_step statement must not be empty".to_string(),
            ));
        }
        if payload.source_refs.is_empty() {
            return Err(GraphError::Schema(
                "teaching steps must include at least one source_ref".to_string(),
            ));
        }
        validate_source_refs(&payload.source_refs)?;
        self.validate_revision(&payload.source_refs, "teaching_step")?;
        self.graph_mut()[id].kind = NodeKind::TeachingStep(payload);

        if let Err(err) = self.validate_incident_edges(id) {
            self.graph_mut()[id].kind = old_kind;
            return Err(err);
        }
        self.graph_mut()[id].tags = tags;
        let guard = ValidationGuard::new(
            self,
            InvariantFamilies::STATEMENTS
                | InvariantFamilies::PROVENANCE
                | InvariantFamilies::DISCOURSE,
            ValidationScope::Targeted {
                coverage_los:      Vec::new(),
                skip_requires_dag: true,
                skip_fadeability:  true,
            },
            move |svc| {
                svc.graph_mut()[id].kind = old_kind;
                svc.graph_mut()[id].tags = old_tags;
            },
        )
        .on_success(|svc| svc.refresh_rubric_hashes());
        guard.commit()?;
        Ok(id)
    }

    /// Generic edge insertion using the EdgeSpec trait.
    pub fn add_edge<S: EdgeSpec>(
        &mut self,
        from: NodeId,
        to: NodeId,
        attrs: S::Attrs,
        confidence: f32,
    ) -> Result<EdgeId, GraphError> {
        // Duplicate edge guard runs only for new insertions; validation passes
        // during node updates should not trigger it.
        let duplicate = match S::NAME {
            "requires" => self.has_edge_of_kind(from, to, |k| matches!(k, EdgeKind::Requires(_))),
            "supports" => self.has_edge_of_kind(from, to, |k| matches!(k, EdgeKind::Supports(_))),
            "assesses" => self.has_edge_of_kind(from, to, |k| matches!(k, EdgeKind::Assesses(_))),
            "precedes" => self.has_edge_of_kind(from, to, |k| matches!(k, EdgeKind::Precedes(_))),
            "anchors" => self.has_edge_of_kind(from, to, |k| matches!(k, EdgeKind::Anchors(_))),
            _ => false,
        };
        if duplicate {
            return Err(GraphError::Schema(format!(
                "duplicate {} edge between these nodes",
                S::NAME
            )));
        }

        S::validate(self, from, to, &attrs, confidence)?;
        let payload = S::make_payload(attrs, confidence);
        let edge_id = self.graph_mut().add_edge(from, to, payload);

        let coverage_los = match S::NAME {
            "assesses" => vec![to],
            "requires" => self.impacted_los_from_requires(from),
            _ => Vec::new(),
        };

        let (skip_requires_dag, skip_fadeability) = match S::NAME {
            // Requires edges can introduce cycles and affect fadeability via prerequisite
            // structure.
            "requires" => (false, false),
            // Supports edges can affect fadeability but not the requires DAG.
            "supports" => (true, false),
            // Other edges do not impact requires/fadeability invariants.
            _ => (true, true),
        };

        let mut dirty = match S::NAME {
            "requires" => {
                InvariantFamilies::REQUIRES_DAG
                    | InvariantFamilies::FADEABILITY
                    | InvariantFamilies::PURITY
            }
            "supports" => InvariantFamilies::FADEABILITY | InvariantFamilies::SUPPORTS,
            "assesses" => InvariantFamilies::PURITY,
            "precedes" => InvariantFamilies::DISCOURSE,
            "anchors" => InvariantFamilies::DISCOURSE | InvariantFamilies::INTRODUCTIONS,
            _ => InvariantFamilies::empty(),
        };
        if !coverage_los.is_empty() {
            dirty.insert(InvariantFamilies::COVERAGE);
        }
        let guard = ValidationGuard::new(
            self,
            dirty,
            ValidationScope::Targeted {
                coverage_los,
                skip_requires_dag,
                skip_fadeability,
            },
            move |svc| {
                svc.graph_mut().remove_edge(edge_id);
            },
        );
        guard.commit()?;
        Ok(edge_id)
    }

    fn has_edge_of_kind(&self, from: NodeId, to: NodeId, pred: impl Fn(&EdgeKind) -> bool) -> bool {
        self.graph
            .edges_directed(from, Direction::Outgoing)
            .any(|e| e.target() == to && pred(&e.weight().kind))
    }

    fn validate_revision(&self, spans: &[SourceRef], label: &str) -> Result<(), GraphError> {
        Provenance::new(self.expected_revision.as_deref()).check(spans, label)
    }

    fn refresh_rubric_hashes(&self) {
        *self.rubric_hashes.write().expect("rubric_hashes lock") =
            compute_rubric_hashes(self.graph());
    }

    /// Ensure every assesses edge claim matches its target LO slug. Intended
    /// for snapshot loads and node renames performed outside the
    /// GraphService API.
    pub fn normalize_assesses_claims(&mut self) {
        let graph = Arc::make_mut(&mut self.graph);
        normalize_assesses_claims_graph(graph);
    }

    pub fn node(&self, id: NodeId) -> &NodePayload {
        &self.graph[id]
    }

    pub(crate) fn node_kinds(
        &self,
        edge_name: &'static str,
        from: NodeId,
        to: NodeId,
    ) -> Result<(&NodeKind, &NodeKind), GraphError> {
        let from_kind = &self
            .graph
            .node_weight(from)
            .ok_or_else(|| {
                GraphError::Schema(format!(
                    "{} edge references missing from-node index {:?}",
                    edge_name,
                    from.index()
                ))
            })?
            .kind;
        let to_kind = &self
            .graph
            .node_weight(to)
            .ok_or_else(|| {
                GraphError::Schema(format!(
                    "{} edge references missing to-node index {:?}",
                    edge_name,
                    to.index()
                ))
            })?
            .kind;
        Ok((from_kind, to_kind))
    }

    /// Path existence along requires edges (petgraph backed).
    pub fn has_requires_path(&self, start: NodeId, goal: NodeId) -> bool {
        traversal::requires_path_exists(self.graph(), start, goal)
    }

    /// Return one requires-path from `start` to `goal`, if it exists (petgraph
    /// backed).
    pub fn requires_path(&self, start: NodeId, goal: NodeId) -> Option<Vec<NodeId>> {
        traversal::requires_one_path(self.graph(), start, goal)
    }

    pub(crate) fn has_precedes_path(&self, start: NodeId, goal: NodeId, episode: &str) -> bool {
        traversal::precedes_path_exists(self.graph(), episode, start, goal)
    }

    /// Identify redundant requires edges (edges removable without changing
    /// reachability).
    pub fn redundant_requires(&self) -> Vec<(NodeId, NodeId)> {
        traversal::requires_transitive_reduction(self.graph())
            .map(|set| set.into_iter().collect())
            .unwrap_or_default()
    }

    /// Remove redundant requires edges and return the number pruned.
    pub fn prune_redundant_requires(&mut self) -> usize {
        let redundant = self.redundant_requires();
        let mut removed = 0;
        for (u, v) in redundant {
            if let Some(edge) = self
                .graph
                .find_edge(u, v)
                .filter(|e| matches!(self.graph[*e].kind, EdgeKind::Requires(_)))
            {
                self.graph_mut().remove_edge(edge);
                removed += 1;
            }
        }
        if removed > 0 {
            self.mark_dirty(
                InvariantFamilies::REQUIRES_DAG
                    | InvariantFamilies::FADEABILITY
                    | InvariantFamilies::COVERAGE
                    | InvariantFamilies::PURITY,
            );
            self.bump_version();
        }
        removed
    }

    /// Run global audits and return violations as errors.
    pub fn validate_global_invariants(&self) -> Result<(), GraphError> {
        let families = self.planned_families(InvariantFamilies::ALL);
        self.validate_invariants(ValidationScope::Full, families)
    }

    /// Run global audits on a snapshot in a blocking task with a timeout. This
    /// keeps the actor mailbox responsive during heavy analyses.
    pub async fn validate_global_invariants_off_thread(
        &self,
        timeout_ms: Duration,
    ) -> Result<(), GraphError> {
        let scope = ValidationScope::Full;
        let families = self.planned_families(InvariantFamilies::ALL);
        let effective = GraphService::families_for_scope(families, &scope);
        if effective.is_empty() {
            return Ok(());
        }
        let graph = self.graph.clone();
        let ctx = ValidationContext {
            strict:                self.strict_quality,
            expected_revision:     self.expected_revision.clone(),
            rubric_prev:           self
                .rubric_hashes
                .read()
                .expect("rubric_hashes lock")
                .clone(),
            include_rubric_update: true,
        };
        let handle = task::spawn_blocking(move || {
            let delay_ms = crate::graph::service::test_support::test_validation_delay_ms();
            if delay_ms > 0 {
                std::thread::sleep(Duration::from_millis(delay_ms));
            }

            run_invariants_for_graph(&graph, scope, &ctx, effective)
        });
        let join_result = match timeout(timeout_ms, handle).await {
            Ok(res) => res,
            Err(_) => {
                let timeout_ms = timeout_ms.as_millis() as u64;
                tracing::warn!(
                    target: "weaver.graph.validation",
                    code = "validation_timeout",
                    timeout_ms
                );
                return Err(GraphError::InvariantTimeout { timeout_ms });
            }
        };
        let rubric_current = match join_result {
            Ok(inner) => inner?,
            Err(join_err) => {
                tracing::warn!(
                    target: "weaver.graph.validation",
                    code = "validation_task_failed",
                    error = %join_err
                );
                return Err(GraphError::InvariantTaskFailed {
                    message: format!("graph invariant validation task failed: {join_err}"),
                });
            }
        };
        if let Some(rubric) = rubric_current {
            *self.rubric_hashes.write().expect("rubric_hashes lock") = rubric;
        }
        self.clear_validated_families(effective);
        Ok(())
    }

    fn validate_invariants(
        &self,
        scope: ValidationScope,
        families: InvariantFamilies,
    ) -> Result<(), GraphError> {
        let ctx = ValidationContext {
            strict:                self.strict_quality,
            expected_revision:     self.expected_revision.clone(),
            rubric_prev:           self
                .rubric_hashes
                .read()
                .expect("rubric_hashes lock")
                .clone(),
            include_rubric_update: true,
        };
        let effective = GraphService::families_for_scope(families, &scope);
        if effective.is_empty() {
            return Ok(());
        }
        let result = run_invariants_for_graph(self.graph(), scope, &ctx, effective)?;
        if let Some(rubric) = result {
            *self.rubric_hashes.write().expect("rubric_hashes lock") = rubric;
        }
        self.clear_validated_families(effective);
        Ok(())
    }

    fn validate_targeted_invariants(
        &self,
        coverage_los: Vec<NodeId>,
        skip_requires_dag: bool,
        skip_fadeability: bool,
    ) -> Result<(), GraphError> {
        if coverage_los.is_empty()
            && skip_requires_dag
            && skip_fadeability
            && self.validation_state.dirty.load(Ordering::Relaxed) == 0
        {
            return Ok(());
        }

        let required = {
            let mut mask = InvariantFamilies::empty();
            if !coverage_los.is_empty() {
                mask.insert(InvariantFamilies::COVERAGE);
            }
            if !skip_requires_dag {
                mask.insert(InvariantFamilies::REQUIRES_DAG);
            }
            if !skip_fadeability {
                mask.insert(InvariantFamilies::FADEABILITY);
            }
            mask
        };
        let families = self.planned_families(required);
        self.validate_invariants(
            ValidationScope::Targeted {
                coverage_los,
                skip_requires_dag,
                skip_fadeability,
            },
            families,
        )
    }

    fn los_assessed_by(&self, assessment: NodeId) -> Vec<NodeId> {
        let mut los = Vec::new();
        for edge in self
            .graph()
            .edges_directed(assessment, Direction::Outgoing)
            .filter(|e| matches!(&e.weight().kind, EdgeKind::Assesses(_)))
        {
            if let EdgeKind::Assesses(attrs) = &edge.weight().kind
                && attrs.evidence_link.scope == AssessmentScope::Target
            {
                los.push(edge.target());
            }
        }
        los
    }

    fn impacted_los_from_requires(&self, start: NodeId) -> Vec<NodeId> {
        let mut stack = vec![start];
        let mut seen = HashSet::new();
        let mut assessments = HashSet::new();

        while let Some(node) = stack.pop() {
            for edge in self
                .graph()
                .edges_directed(node, Direction::Outgoing)
                .filter(|e| matches!(&e.weight().kind, EdgeKind::Requires(_)))
            {
                let next = edge.target();
                if seen.insert(next) {
                    stack.push(next);
                }
            }
            if let NodeKind::Knowledge(k) = &self.graph()[node].kind
                && k.knowledge_type.is_assessment_item()
            {
                assessments.insert(node);
            }
        }

        let mut los: HashSet<NodeId> = HashSet::new();
        for assessment in assessments {
            for lo in self.los_assessed_by(assessment) {
                los.insert(lo);
            }
        }
        los.into_iter().collect()
    }
}

fn run_invariants_for_graph(
    g: &CurriculumGraph,
    scope: ValidationScope,
    ctx: &ValidationContext,
    families: InvariantFamilies,
) -> Result<Option<HashMap<String, u64>>, GraphError> {
    let start = Instant::now();
    let slug_index: HashMap<String, NodeId> =
        g.node_indices().map(|n| (g[n].slug.clone(), n)).collect();
    let has_teaching_steps = g
        .node_indices()
        .any(|n| matches!(&g[n].kind, NodeKind::TeachingStep(_)));

    let (coverage_los, include_requires_dag, include_fadeability) = match &scope {
        ValidationScope::Full => (
            g.node_indices()
                .filter(|&n| {
                    matches!(
                        &g[n].kind,
                        NodeKind::Knowledge(k) if k.knowledge_type == KnowledgeType::LearningOutcome
                    )
                })
                .collect(),
            true,
            true,
        ),
        ValidationScope::Targeted {
            coverage_los,
            skip_requires_dag,
            skip_fadeability,
        } => (coverage_los.clone(), !skip_requires_dag, !skip_fadeability),
    };

    let include_requires_dag =
        include_requires_dag && families.contains(InvariantFamilies::REQUIRES_DAG);
    let include_fadeability =
        include_fadeability && families.contains(InvariantFamilies::FADEABILITY);
    let needs_coverage = !coverage_los.is_empty() && families.contains(InvariantFamilies::COVERAGE);

    let rubric_current = compute_rubric_hashes(g);
    let (first_principles, rubric_prev) =
        if needs_coverage || matches!(scope, ValidationScope::Full) || include_fadeability {
            (analysis::first_principles(g), ctx.rubric_prev.clone())
        } else {
            (Vec::new(), HashMap::new())
        };
    let fade_ctx = if include_fadeability {
        Some(analysis::FadeabilityContext::from_first_principles(g, &first_principles))
    } else {
        None
    };

    let mut issues = Vec::new();
    if families.contains(InvariantFamilies::STATEMENTS) {
        issues.extend(check_statements(g));
    }
    if families.contains(InvariantFamilies::PROVENANCE) {
        issues.extend(check_provenance(g, ctx.expected_revision.as_deref()));
    }
    if include_requires_dag || include_fadeability {
        issues.extend(check_requires_and_fadeability(
            g,
            include_requires_dag,
            include_fadeability,
            fade_ctx.as_ref(),
        ));
    }
    if needs_coverage {
        issues.extend(check_reachability_and_coverage(
            g,
            &first_principles,
            &coverage_los,
            &rubric_prev,
            &rubric_current,
        ));
    }
    if families.contains(InvariantFamilies::SUPPORTS) {
        issues.extend(check_supports_and_practice(g));
    }
    if families.contains(InvariantFamilies::PURITY) {
        issues.extend(check_purity(g, &slug_index));
    }
    if has_teaching_steps && families.contains(InvariantFamilies::DISCOURSE) {
        issues.extend(check_discourse(g));
    }
    if has_teaching_steps && families.contains(InvariantFamilies::INTRODUCTIONS) {
        issues.extend(check_introductions(g));
    }

    let mut errors: Vec<crate::graph::InvariantViolation> = Vec::new();
    let mut warnings: Vec<crate::graph::InvariantViolation> = Vec::new();
    for mut issue in issues {
        if ctx.strict && issue.promote_in_strict {
            issue.severity = ValidationSeverity::Error;
        }
        match issue.severity {
            ValidationSeverity::Error => errors.push(crate::graph::InvariantViolation {
                code:    issue.code,
                message: issue.message,
            }),
            ValidationSeverity::Warning => warnings.push(crate::graph::InvariantViolation {
                code:    issue.code,
                message: issue.message,
            }),
        }
    }

    if ctx.strict {
        errors.extend(warnings);
    } else {
        for w in warnings {
            warn!(
                target: "weaver.graph.invariants",
                code = %w.code.as_str(),
                "{message}",
                message = w.message
            );
        }
    }

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    tracing::debug!(
        target: "weaver.graph.validation",
        scope = %match scope {
            ValidationScope::Full => "full",
            ValidationScope::Targeted { .. } => "targeted",
        },
        coverage_los = coverage_los.len(),
        include_requires_dag,
        include_fadeability,
        elapsed_ms
    );

    if errors.is_empty() {
        if ctx.include_rubric_update {
            Ok(Some(rubric_current))
        } else {
            Ok(None)
        }
    } else {
        Err(GraphError::InvariantViolation { violations: errors })
    }
}

fn make_issue(
    code: crate::graph::InvariantCode,
    severity: ValidationSeverity,
    promote_in_strict: bool,
    message: String,
) -> ValidationIssue {
    ValidationIssue {
        code,
        severity,
        message,
        promote_in_strict,
    }
}

fn check_statements(g: &CurriculumGraph) -> Vec<ValidationIssue> {
    g.node_indices()
        .filter_map(|n| match &g[n].kind {
            NodeKind::Knowledge(k) if k.statement.trim().is_empty() => Some(make_issue(
                InvariantCode::StatementEmpty,
                ValidationSeverity::Error,
                true,
                format!("knowledge `{}` has empty statement", g[n].slug),
            )),
            NodeKind::TeachingStep(ts) if ts.statement.trim().is_empty() => Some(make_issue(
                InvariantCode::StatementEmpty,
                ValidationSeverity::Error,
                true,
                format!("teaching_step `{}` has empty statement", g[n].slug),
            )),
            _ => None,
        })
        .collect()
}

fn check_provenance(g: &CurriculumGraph, expected_revision: Option<&str>) -> Vec<ValidationIssue> {
    let Some(expected) = expected_revision else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for n in g.node_indices() {
        match &g[n].kind {
            NodeKind::Knowledge(k) => {
                for span in &k.source_refs {
                    if span.revision != expected {
                        out.push(make_issue(
                            InvariantCode::ProvenanceRevision,
                            ValidationSeverity::Error,
                            true,
                            format!(
                                "knowledge `{}` source_ref revision `{}` must equal course_commit \
                                 `{}`",
                                g[n].slug, span.revision, expected
                            ),
                        ));
                    }
                }
            }
            NodeKind::TeachingStep(ts) => {
                for span in &ts.source_refs {
                    if span.revision != expected {
                        out.push(make_issue(
                            InvariantCode::ProvenanceRevision,
                            ValidationSeverity::Error,
                            true,
                            format!(
                                "teaching_step `{}` source_ref revision `{}` must equal \
                                 course_commit `{}`",
                                g[n].slug, span.revision, expected
                            ),
                        ));
                    }
                }
            }
        }
    }

    for edge in g.edge_indices() {
        if let Some((u, v)) = g.edge_endpoints(edge) {
            match &g[edge].kind {
                EdgeKind::Requires(attrs) => {
                    for span in &attrs.evidence_refs {
                        if span.revision != expected {
                            out.push(make_issue(
                                InvariantCode::ProvenanceRevision,
                                ValidationSeverity::Error,
                                true,
                                format!(
                                    "{} -> {} evidence_ref revision `{}` must equal course_commit \
                                     `{}`",
                                    g[u].slug, g[v].slug, span.revision, expected
                                ),
                            ));
                        }
                    }
                }
                EdgeKind::Supports(attrs) => {
                    for span in &attrs.evidence_refs {
                        if span.revision != expected {
                            out.push(make_issue(
                                InvariantCode::ProvenanceRevision,
                                ValidationSeverity::Error,
                                true,
                                format!(
                                    "{} -> {} evidence_ref revision `{}` must equal course_commit \
                                     `{}`",
                                    g[u].slug, g[v].slug, span.revision, expected
                                ),
                            ));
                        }
                    }
                }
                _ => {}
            }
        }
    }
    out
}

fn check_requires_and_fadeability(
    g: &CurriculumGraph,
    include_dag: bool,
    include_fadeability: bool,
    fade_ctx: Option<&analysis::FadeabilityContext>,
) -> Vec<ValidationIssue> {
    let mut out = Vec::new();
    if include_dag && !analysis::requires_is_dag(g) {
        out.push(make_issue(
            InvariantCode::RequiresDag,
            ValidationSeverity::Error,
            true,
            "requires layer must remain acyclic".to_string(),
        ));
    }
    if include_fadeability {
        let ctx = fade_ctx
            .cloned()
            .unwrap_or_else(|| analysis::FadeabilityContext::compute(g));
        for issue in analysis::fadeability_issues_with_context(g, &ctx) {
            let assessment_slug = g[issue.assessment].slug.clone();
            let edges: Vec<String> = issue
                .support_edges
                .iter()
                .filter_map(|e| g.edge_endpoints(*e))
                .map(|(u, v)| format!("{} -> {}", g[u].slug, g[v].slug))
                .collect();
            out.push(make_issue(
                InvariantCode::Fadeability,
                ValidationSeverity::Error,
                true,
                format!(
                    "assessment `{}` reachable only via supports that carry prerequisite load: \
                     [{}]",
                    assessment_slug,
                    edges.join("; ")
                ),
            ));
        }
    }
    out
}

fn check_reachability_and_coverage(
    g: &CurriculumGraph,
    first_principles: &[NodeId],
    lo_nodes: &[NodeId],
    rubric_prev: &HashMap<String, u64>,
    rubric_current: &HashMap<String, u64>,
) -> Vec<ValidationIssue> {
    let mut out = Vec::new();
    for &lo_id in lo_nodes {
        let report = analysis::lo_reachability(g, lo_id, first_principles);
        if report.assessments.is_empty() {
            out.push(make_issue(
                InvariantCode::LoTargetAssessment,
                ValidationSeverity::Warning,
                true,
                format!(
                    "learning_outcome `{}` has no assesses(scope=target) assessments",
                    g[lo_id].slug
                ),
            ));
        } else if !report
            .assessments
            .iter()
            .any(|a| a.reachable_from_first_principle)
        {
            let assessments = report
                .assessments
                .iter()
                .map(|a| g[a.assessment].slug.clone())
                .collect::<Vec<_>>()
                .join(", ");
            out.push(make_issue(
                InvariantCode::LoReachability,
                ValidationSeverity::Warning,
                true,
                format!(
                    "learning_outcome `{}` not reachable from first principles via assessments \
                     [{}]",
                    g[lo_id].slug, assessments
                ),
            ));
        }

        let report = analysis::coverage_report(g, lo_id);
        if !report.missing_criteria.is_empty() {
            out.push(make_issue(
                InvariantCode::RubricCoverage,
                ValidationSeverity::Warning,
                true,
                format!(
                    "learning_outcome `{}` missing coverage for rubric criteria: {}",
                    g[lo_id].slug,
                    report.missing_criteria.join(", ")
                ),
            ));
        }
        if !report.unused_observation_features.is_empty() {
            out.push(make_issue(
                InvariantCode::RubricUnusedObservationFeatures,
                ValidationSeverity::Warning,
                true,
                format!(
                    "learning_outcome `{}` has observation_features not present in rubric: {}",
                    g[lo_id].slug,
                    report.unused_observation_features.join(", ")
                ),
            ));
        }
        if let (Some(cur), Some(prev)) =
            (rubric_current.get(&g[lo_id].slug), rubric_prev.get(&g[lo_id].slug))
            && cur != prev
        {
            out.push(make_issue(
                InvariantCode::RubricDrift,
                ValidationSeverity::Warning,
                true,
                format!(
                    "learning_outcome `{}` rubric_criteria changed; coverage rechecked",
                    g[lo_id].slug
                ),
            ));
        }
    }
    out
}

fn check_supports_and_practice(g: &CurriculumGraph) -> Vec<ValidationIssue> {
    let mut out = Vec::new();
    for gap in analysis::example_gaps(g) {
        out.push(make_issue(
            InvariantCode::ExampleMinimums,
            ValidationSeverity::Warning,
            true,
            format!("{}: {}", g[gap.node].slug, gap.description),
        ));
    }
    for gap in analysis::procedural_practice_gaps(g) {
        out.push(make_issue(
            InvariantCode::ProceduralPractice,
            ValidationSeverity::Warning,
            true,
            format!(
                "procedural `{}` lacks reachable assessment with assesses(scope=target)",
                g[gap.node].slug
            ),
        ));
    }
    out
}

fn check_purity(g: &CurriculumGraph, slug_index: &HashMap<String, NodeId>) -> Vec<ValidationIssue> {
    let mut out = Vec::new();
    for edge in g.edge_indices() {
        if let EdgeKind::Assesses(attrs) = &g[edge].kind
            && attrs.evidence_link.scope == AssessmentScope::Target
            && let Some((assessment, lo)) = g.edge_endpoints(edge)
        {
            let intended = analysis::intended_knowledge_from_anchors(g, lo);
            if intended.is_empty() {
                out.push(make_issue(
                    InvariantCode::PurityIntendedMissing,
                    ValidationSeverity::Warning,
                    true,
                    format!(
                        "purity check skipped for `{}` -> `{}` (no target anchors with intended \
                         knowledge)",
                        g[assessment].slug, g[lo].slug
                    ),
                ));
                continue;
            }

            let mut allowed = HashSet::new();
            if let NodeKind::Knowledge(k) = &g[assessment].kind {
                for cid in &k.construct_irrelevant_demands {
                    if let Some(id) = slug_index.get(cid) {
                        allowed.insert(*id);
                    }
                }
            }
            let mut extraneous = analysis::extraneous_knowledge(g, assessment, &intended);
            extraneous.retain(|n| !allowed.contains(n));
            if !extraneous.is_empty() {
                let slugs: Vec<String> = extraneous.iter().map(|n| g[*n].slug.clone()).collect();
                out.push(make_issue(
                    InvariantCode::PurityExtraneous,
                    ValidationSeverity::Error,
                    true,
                    format!(
                        "purity violation: assessment `{}` -> LO `{}` requires extraneous \
                         knowledge [{}]",
                        g[assessment].slug,
                        g[lo].slug,
                        slugs.join(", ")
                    ),
                ));
            }
        }
    }
    out
}

fn check_discourse(g: &CurriculumGraph) -> Vec<ValidationIssue> {
    let mut out = Vec::new();
    let episodes: HashSet<String> = g
        .node_indices()
        .filter_map(|n| {
            if let NodeKind::TeachingStep(ts) = &g[n].kind {
                Some(ts.episode.clone())
            } else {
                None
            }
        })
        .collect();
    for ep in episodes {
        for result in analysis::borrow_ahead(g, &ep) {
            if matches!(
                &g[result.target].kind,
                NodeKind::Knowledge(k) if k.knowledge_type == KnowledgeType::LearningOutcome
            ) {
                continue;
            }
            let step_slug = g[result.step].slug.clone();
            let target_slug = g[result.target].slug.clone();
            match result.severity {
                analysis::BorrowSeverity::CrossEpisode | analysis::BorrowSeverity::NoIntro => {
                    out.push(make_issue(
                        InvariantCode::BorrowAhead,
                        ValidationSeverity::Error,
                        true,
                        format!(
                            "borrow-ahead error in episode `{}`: step `{}` uses `{}` before \
                             introduction (severity {:?})",
                            ep, step_slug, target_slug, result.severity
                        ),
                    ));
                }
                analysis::BorrowSeverity::InEpisode => out.push(make_issue(
                    InvariantCode::BorrowAhead,
                    ValidationSeverity::Warning,
                    true,
                    format!(
                        "borrow-ahead warning in episode `{}`: step `{}` uses `{}` before \
                         introduction",
                        ep, step_slug, target_slug
                    ),
                )),
                analysis::BorrowSeverity::Suppressed => {}
            }
        }
    }

    for orphan in analysis::discourse_orphans(g, None) {
        out.push(make_issue(
            InvariantCode::DiscourseOrphan,
            ValidationSeverity::Warning,
            true,
            format!(
                "teaching_step `{}` is orphaned within its episode (no precedes links)",
                g[orphan].slug
            ),
        ));
    }

    for n in g.node_indices() {
        if let NodeKind::TeachingStep(ts) = &g[n].kind {
            let has_anchor = g
                .edges_directed(n, Direction::Outgoing)
                .any(|e| matches!(&e.weight().kind, EdgeKind::Anchors(_)));
            let has_rationale = ts
                .rationale
                .as_ref()
                .map(|r| !r.trim().is_empty())
                .unwrap_or(false);
            let missing_anchor = !has_anchor;
            let missing_rationale = !has_rationale;
            if matches!(ts.purpose, TeachingPurpose::Use) && missing_anchor && !missing_rationale {
                out.push(make_issue(
                    InvariantCode::BorrowAhead,
                    ValidationSeverity::Warning,
                    true,
                    format!(
                        "teaching_step `{}` has purpose=use but no anchors; cannot verify \
                         introduction order despite provided rationale",
                        g[n].slug
                    ),
                ));
            }
            if missing_anchor && missing_rationale {
                out.push(make_issue(
                    InvariantCode::TeachingStepAnchorOrRationale,
                    ValidationSeverity::Warning,
                    true,
                    format!(
                        "teaching_step `{}` must have at least one anchor or a rationale",
                        g[n].slug
                    ),
                ));
            }
        }
    }

    out
}

fn check_introductions(g: &CurriculumGraph) -> Vec<ValidationIssue> {
    let mut out = Vec::new();
    for n in g.node_indices() {
        if let NodeKind::Knowledge(k) = &g[n].kind
            && k.knowledge_type.is_instructional_knowledge()
            && matches!(k.introduction_scope, crate::graph::IntroductionScope::InCourse)
        {
            let anchored = g
                .edges_directed(n, Direction::Incoming)
                .any(|e| matches!(&e.weight().kind, EdgeKind::Anchors(_)));
            if !anchored {
                continue;
            }
            let has_intro = g.edges_directed(n, Direction::Incoming).any(|e| {
                matches!(
                    &e.weight().kind,
                    EdgeKind::Anchors(attrs) if matches!(attrs.impact, AnchorImpact::Introduce)
                )
            });
            if !has_intro {
                out.push(make_issue(
                    InvariantCode::IntroduceAnchor,
                    ValidationSeverity::Warning,
                    true,
                    format!(
                        "knowledge `{}` (in_course) missing anchors(impact=introduce)",
                        g[n].slug
                    ),
                ));
            }
        }
    }
    out
}
impl Default for GraphService {
    fn default() -> Self {
        Self::new()
    }
}

fn compute_rubric_hashes(g: &CurriculumGraph) -> HashMap<String, u64> {
    g.node_indices()
        .filter_map(|n| {
            if let NodeKind::Knowledge(k) = &g[n].kind
                && k.knowledge_type == KnowledgeType::LearningOutcome
            {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                k.rubric_criteria.hash(&mut hasher);
                Some((g[n].slug.clone(), hasher.finish()))
            } else {
                None
            }
        })
        .collect()
}

/// Ensure assesses.claim mirrors its target LO slug to keep denormalized data
/// consistent.
fn normalize_assesses_claims_graph(graph: &mut CurriculumGraph) {
    let edges: Vec<_> = graph
        .edge_indices()
        .filter(|e| matches!(graph[*e].kind, EdgeKind::Assesses(_)))
        .collect();
    for e in edges {
        if let Some((_, target)) = graph.edge_endpoints(e) {
            let target_slug = graph[target].slug.clone();
            if let EdgeKind::Assesses(ref mut attrs) = graph[e].kind
                && attrs.evidence_link.claim != target_slug
            {
                attrs.evidence_link.claim = target_slug;
            }
        }
    }
}
