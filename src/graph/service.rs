//! Core graph owner responsible for routing mutations through validation and
//! keeping derived caches in sync.
use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc, RwLock,
        atomic::{AtomicU16, Ordering},
    },
    time::Duration,
};

use petgraph::{Direction, visit::EdgeRef};
use serde_json::json;
use tokio::{task, time::timeout};
use uuid::Uuid;

use crate::{
    analysis,
    graph::{
        audit::{self, MutationKind},
        model::*,
        specs::{AnchorsSpec, AssessesSpec, EdgeSpec, PrecedesSpec, RequiresSpec, SupportsSpec},
        traversal,
        validation::{
            self, InvariantFamilies, Provenance, ValidationContext, ValidationScope,
            compute_rubric_hashes,
        },
    },
    schema::types::{AssessmentScope, KnowledgeType, SourceRef},
};

type GuardCallback<'a> = Box<dyn FnOnce(&mut GraphService) + 'a>;

struct ValidationState {
    dirty: AtomicU16,
}

struct GraphValidator<'a> {
    strict_quality:    bool,
    expected_revision: Option<String>,
    rubric_hashes:     &'a RwLock<HashMap<String, u64>>,
}

impl<'a> GraphValidator<'a> {
    fn new(
        strict_quality: bool,
        expected_revision: Option<String>,
        rubric_hashes: &'a RwLock<HashMap<String, u64>>,
    ) -> Self {
        Self {
            strict_quality,
            expected_revision,
            rubric_hashes,
        }
    }

    fn context(&self, include_rubric_update: bool) -> ValidationContext {
        ValidationContext {
            strict: self.strict_quality,
            expected_revision: self.expected_revision.clone(),
            rubric_prev: self
                .rubric_hashes
                .read()
                .expect("rubric_hashes lock")
                .clone(),
            include_rubric_update,
        }
    }

    fn run(
        &self,
        graph: &CurriculumGraph,
        scope: ValidationScope,
        families: InvariantFamilies,
    ) -> Result<Option<HashMap<String, u64>>, GraphError> {
        let ctx = self.context(true);
        validation::run_invariants_for_graph(graph, scope, &ctx, families)
    }

    async fn run_blocking(
        &self,
        graph: Arc<CurriculumGraph>,
        scope: ValidationScope,
        families: InvariantFamilies,
        timeout_ms: Duration,
    ) -> Result<Option<HashMap<String, u64>>, GraphError> {
        let ctx = self.context(true);
        let handle = task::spawn_blocking(move || {
            let delay_ms = crate::graph::service::test_support::test_validation_delay_ms();
            if delay_ms > 0 {
                std::thread::sleep(Duration::from_millis(delay_ms));
            }

            validation::run_invariants_for_graph(&graph, scope, &ctx, families)
        });
        match timeout(timeout_ms, handle).await {
            Ok(res) => res.map_err(|join_err| {
                GraphError::Operational(GraphOperationalError::InvariantTaskFailed {
                    message: format!("graph invariant validation task failed: {join_err}"),
                })
            })?,
            Err(_) => {
                let timeout_ms = timeout_ms.as_millis() as u64;
                tracing::warn!(
                    target: "weaver.graph.validation",
                    code = "validation_timeout",
                    timeout_ms
                );
                Err(GraphOperationalError::InvariantTimeout { timeout_ms }.into())
            }
        }
    }

    fn update_rubric(&self, rubric_current: Option<HashMap<String, u64>>) {
        if let Some(rubric) = rubric_current {
            *self.rubric_hashes.write().expect("rubric_hashes lock") = rubric;
        }
    }
}

impl ValidationState {
    fn new_empty() -> Self {
        Self {
            dirty: AtomicU16::new(0),
        }
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
    audit_sink:        audit::SharedMutationSink,
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
            audit_sink:        Arc::new(audit::NoopMutationSink),
        }
    }

    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict_quality = strict;
        self
    }

    fn validator(&self) -> GraphValidator<'_> {
        GraphValidator::new(
            self.strict_quality,
            self.expected_revision.clone(),
            &self.rubric_hashes,
        )
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
            audit_sink: Arc::new(audit::NoopMutationSink),
        };
        svc.rebuild_slug_index()?;
        svc.validate_global_invariants()?;
        Ok(svc)
    }

    pub fn with_audit_sink(mut self, sink: audit::SharedMutationSink) -> Self {
        self.audit_sink = sink;
        self
    }

    pub fn set_audit_sink(&mut self, sink: audit::SharedMutationSink) {
        self.audit_sink = sink;
    }

    fn record_mutation(&self, kind: MutationKind, payload: serde_json::Value) {
        let event = audit::MutationEvent::new(kind, self.graph_version, payload);
        self.audit_sink.record(&event);
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
            if let Some((ver, ctx)) = cache.as_ref()
                && *ver == self.graph_version
            {
                return ctx.clone();
            }
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

    fn ensure_node_capacity(&self) -> Result<(), GraphError> {
        if self.graph.node_count() >= crate::constants::MAX_GRAPH_NODES {
            return Err(GraphError::Schema(format!(
                "graph node cap {} reached",
                crate::constants::MAX_GRAPH_NODES
            )));
        }
        Ok(())
    }

    fn ensure_edge_capacity(&self) -> Result<(), GraphError> {
        if self.graph.edge_count() >= crate::constants::MAX_GRAPH_EDGES {
            return Err(GraphError::Schema(format!(
                "graph edge cap {} reached",
                crate::constants::MAX_GRAPH_EDGES
            )));
        }
        Ok(())
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
        if prev != strict {
            self.record_mutation(
                MutationKind::SetStrictQuality { strict },
                json!({
                    "previous": prev,
                }),
            );
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
        self.ensure_node_capacity()?;
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
        dirty.insert(InvariantFamilies::STRUCTURE);
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
        if let NodeKind::Knowledge(k) = &self.graph()[id].kind {
            self.record_mutation(
                MutationKind::InsertKnowledge,
                json!({
                    "slug": self.graph()[id].slug.clone(),
                    "knowledge_type": k.knowledge_type,
                    "tags": self.graph()[id].tags.clone(),
                }),
            );
        }
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
        dirty.insert(InvariantFamilies::STRUCTURE);
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
        if let NodeKind::Knowledge(k) = &self.graph()[id].kind {
            self.record_mutation(
                MutationKind::UpdateKnowledge,
                json!({
                    "slug": self.graph()[id].slug.clone(),
                    "knowledge_type": k.knowledge_type,
                    "tags": self.graph()[id].tags.clone(),
                }),
            );
        }
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
        guard.commit()?;
        self.record_mutation(
            MutationKind::RenameNode,
            json!({
                "from": old_slug,
                "to": new_slug,
            }),
        );
        Ok(())
    }

    pub fn remove_node(&mut self, slug: &str) -> Result<(), GraphError> {
        let id = self.node_by_slug(slug)?;

        // Stash current state for rollback on invariant failure.
        let removed_slug = self.graph()[id].slug.clone();
        let removed_kind = match &self.graph()[id].kind {
            NodeKind::Knowledge(k) => json!({
                "kind": "knowledge",
                "knowledge_type": k.knowledge_type
            }),
            NodeKind::TeachingStep(_) => json!({
                "kind": "teaching_step"
            }),
        };
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
        self.record_mutation(
            MutationKind::RemoveNode,
            json!({
                "slug": removed_slug,
                "kind": removed_kind,
            }),
        );
        Ok(())
    }

    /// Cheap shared snapshot of the current graph for read-heavy callers.
    pub fn snapshot_graph(&self) -> Arc<CurriculumGraph> {
        Arc::clone(&self.graph)
    }

    /// Owned snapshot of the current graph for persistence/serialization.
    pub fn snapshot_graph_owned(&self) -> CurriculumGraph {
        (*self.graph).clone()
    }

    pub fn install_graph(
        &mut self,
        mut graph: CurriculumGraph,
        graph_version: u64,
    ) -> Result<(), GraphError> {
        normalize_assesses_claims_graph(&mut graph);
        if graph.node_count() > crate::constants::MAX_GRAPH_NODES {
            return Err(GraphError::Schema(format!(
                "graph node cap {} exceeded in snapshot",
                crate::constants::MAX_GRAPH_NODES
            )));
        }
        if graph.edge_count() > crate::constants::MAX_GRAPH_EDGES {
            return Err(GraphError::Schema(format!(
                "graph edge cap {} exceeded in snapshot",
                crate::constants::MAX_GRAPH_EDGES
            )));
        }

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
        self.record_mutation(
            MutationKind::InstallGraph,
            json!({
                "graph_version": self.graph_version,
            }),
        );
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
        self.ensure_node_capacity()?;
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
        self.record_mutation(
            MutationKind::InsertTeachingStep,
            json!({
                "slug": self.graph()[id].slug.clone(),
                "tags": self.graph()[id].tags.clone(),
            }),
        );
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
        self.record_mutation(
            MutationKind::UpdateTeachingStep,
            json!({
                "slug": self.graph()[id].slug.clone(),
                "tags": self.graph()[id].tags.clone(),
            }),
        );
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
        self.ensure_edge_capacity()?;
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
                    | InvariantFamilies::STRUCTURE
            }
            "supports" => {
                InvariantFamilies::FADEABILITY
                    | InvariantFamilies::SUPPORTS
                    | InvariantFamilies::STRUCTURE
            }
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
        self.record_mutation(
            MutationKind::AddEdge { edge: S::NAME },
            json!({
                "edge": S::NAME,
                "from": self.graph()[from].slug.clone(),
                "to": self.graph()[to].slug.clone(),
                "confidence": confidence,
            }),
        );
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
                    | InvariantFamilies::PURITY
                    | InvariantFamilies::STRUCTURE,
            );
            self.bump_version();
            self.record_mutation(
                MutationKind::PruneRequires,
                json!({
                    "removed": removed,
                }),
            );
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
        let effective = validation::families_for_scope(families, &scope);
        if effective.is_empty() {
            return Ok(());
        }
        let result = self
            .validator()
            .run_blocking(self.graph.clone(), scope, effective, timeout_ms)
            .await?;
        self.validator().update_rubric(result);
        self.clear_validated_families(effective);
        Ok(())
    }

    fn validate_invariants(
        &self,
        scope: ValidationScope,
        families: InvariantFamilies,
    ) -> Result<(), GraphError> {
        let effective = validation::families_for_scope(families, &scope);
        if effective.is_empty() {
            return Ok(());
        }
        let result = self.validator().run(self.graph(), scope, effective)?;
        self.validator().update_rubric(result);
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

impl Default for GraphService {
    fn default() -> Self {
        Self::new()
    }
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
