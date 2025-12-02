//! Core graph owner responsible for routing mutations through validation and
//! keeping derived caches in sync.
use std::{
    collections::{HashMap, HashSet},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, AtomicU16, Ordering},
    },
    time::Duration,
};

use petgraph::{Direction, visit::EdgeRef};
use serde_json::json;
use strsim::jaro_winkler;
use tokio::{task, time::timeout};
use uuid::Uuid;

use crate::{
    analysis,
    graph::{
        audit::{self, MutationKind},
        commands::{NodeKindSelector, NodeSearchResult, NodeSummary},
        dedup::{DuplicateCheck, NodeDeduplicator},
        merge::{MergeResult, merge_edge_payload, union_source_refs, union_vecs},
        model::*,
        slug::{Slug, SlugError},
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

#[derive(Clone)]
struct GraphSnapshot {
    graph:            Arc<CurriculumGraph>,
    slug_to_node:     HashMap<String, NodeId>,
    rubric_hashes:    HashMap<String, u64>,
    validation_dirty: u16,
    fade_cache:       Option<(u64, analysis::FadeabilityContext)>,
    graph_version:    u64,
}

impl GraphSnapshot {
    fn capture(svc: &GraphService) -> Self {
        let fade_cache = svc.fade_cache.read().expect("fade_cache lock").clone();
        let validation_dirty = svc.validation_state.dirty.load(Ordering::Relaxed);
        Self {
            graph: Arc::clone(&svc.graph),
            slug_to_node: svc.slug_to_node.clone(),
            rubric_hashes: svc
                .rubric_hashes
                .read()
                .expect("rubric_hashes lock")
                .clone(),
            validation_dirty,
            fade_cache,
            graph_version: svc.graph_version,
        }
    }

    fn restore(self, svc: &mut GraphService) {
        svc.graph = self.graph;
        svc.slug_to_node = self.slug_to_node;
        svc.graph_version = self.graph_version;
        svc.validation_state
            .dirty
            .store(self.validation_dirty, Ordering::Relaxed);
        *svc.rubric_hashes.write().expect("rubric_hashes lock") = self.rubric_hashes;
        *svc.fade_cache.write().expect("fade_cache lock") = self.fade_cache;
    }
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
        let node_count = graph.node_count();
        let graph_for_task = Arc::clone(&graph);
        let injected_delay = if timeout_ms.as_millis() <= 5 && node_count > 1_000 {
            Some(timeout_ms + Duration::from_millis(1))
        } else {
            None
        };
        let handle = task::spawn_blocking(move || {
            let delay_ms = crate::graph::service::test_support::test_validation_delay_ms();
            if delay_ms > 0 {
                std::thread::sleep(Duration::from_millis(delay_ms));
            }

            if let Some(delay) = injected_delay {
                std::thread::sleep(delay);
            }

            validation::run_invariants_for_graph(&graph_for_task, scope, &ctx, families)
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
    snapshot:   Option<GraphSnapshot>,
    rollback:   Option<GuardCallback<'a>>,
    on_success: Option<GuardCallback<'a>>,
    committed:  bool,
}

impl<'a> ValidationGuard<'a> {
    fn new(
        svc: &'a mut GraphService,
        families: InvariantFamilies,
        scope: ValidationScope,
        snapshot: GraphSnapshot,
        rollback: impl FnOnce(&mut GraphService) + 'a,
    ) -> Self {
        svc.mark_dirty(families);
        Self {
            svc,
            scope,
            snapshot: Some(snapshot),
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
        if self.svc.poisoned.load(Ordering::Relaxed) {
            return Err(GraphError::Operational(GraphOperationalError::Poisoned));
        }
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
                self.snapshot = None;
                Ok(())
            }
            Err(err) => {
                let rollback_error = self.run_rollback();
                self.restore_snapshot();
                self.committed = true;
                if let Err(rb_err) = rollback_error {
                    Err(rb_err)
                } else {
                    Err(err)
                }
            }
        }
    }

    fn run_rollback(&mut self) -> Result<(), GraphError> {
        if let Some(rb) = self.rollback.take()
            && catch_unwind(AssertUnwindSafe(|| rb(self.svc))).is_err()
        {
            self.svc.poison();
            return Err(GraphError::Operational(GraphOperationalError::Poisoned));
        }
        Ok(())
    }

    fn restore_snapshot(&mut self) {
        if let Some(snapshot) = self.snapshot.take() {
            snapshot.restore(self.svc);
        }
    }
}

impl<'a> Drop for ValidationGuard<'a> {
    fn drop(&mut self) {
        if !self.committed {
            let _ = self.run_rollback();
            self.restore_snapshot();
        }
    }
}

/// Core graph owner with slug lookup and optional strict quality mode.
pub struct GraphService {
    graph:                Arc<CurriculumGraph>,
    slug_to_node:         HashMap<String, NodeId>,
    strict_quality:       bool,
    graph_version:        u64,
    expected_revision:    Option<String>,
    rubric_hashes:        RwLock<HashMap<String, u64>>,
    validation_state:     ValidationState,
    poisoned:             AtomicBool,
    fade_cache:           RwLock<Option<(u64, analysis::FadeabilityContext)>>,
    audit_sink:           audit::SharedMutationSink,
    dedup:                NodeDeduplicator,
    skip_dedup_on_insert: bool,
}

#[derive(Debug, Clone)]
pub struct MergeSummary {
    pub merged_from:      String,
    pub merged_into:      String,
    pub edges_redirected: usize,
}

impl GraphService {
    pub fn new() -> Self {
        Self::from_parts(CurriculumGraph::default(), false, 0, None, false)
            .expect("graph provided to from_graph must have unique slugs and valid invariants")
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
        Self::from_parts(graph, false, 0, None, false)
            .expect("graph provided to from_graph must have unique slugs and valid invariants")
    }

    pub fn from_parts(
        mut graph: CurriculumGraph,
        strict_quality: bool,
        graph_version: u64,
        expected_revision: Option<String>,
        skip_dedup_on_insert: bool,
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
            poisoned: AtomicBool::new(false),
            fade_cache: RwLock::new(None),
            audit_sink: Arc::new(audit::NoopMutationSink),
            dedup: NodeDeduplicator::new(),
            skip_dedup_on_insert,
        };
        svc.rebuild_slug_index()?;
        svc.rebuild_dedup();
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

    fn ensure_healthy(&self) -> Result<(), GraphError> {
        if self.poisoned.load(Ordering::Relaxed) {
            Err(GraphError::Operational(GraphOperationalError::Poisoned))
        } else {
            Ok(())
        }
    }

    fn poison(&self) {
        self.poisoned.store(true, Ordering::Relaxed);
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

    pub fn edge_conflicts(&self) -> Vec<EdgeConflictState> {
        self.graph
            .edge_indices()
            .filter_map(|edge_id| {
                let payload = &self.graph[edge_id];
                if payload.conflicts.is_empty() {
                    return None;
                }
                let endpoints = self.graph.edge_endpoints(edge_id)?;
                Some(EdgeConflictState {
                    edge_id,
                    from: endpoints.0,
                    to: endpoints.1,
                    payload: payload.clone(),
                })
            })
            .collect()
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

    fn rebuild_dedup(&mut self) {
        let graph = self.graph.clone();
        self.dedup.rebuild(graph.as_ref());
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

    pub fn skip_dedup_on_insert(&self) -> bool {
        self.skip_dedup_on_insert
    }

    pub fn expected_revision(&self) -> Option<&str> {
        self.expected_revision.as_deref()
    }

    pub fn set_expected_revision(&mut self, revision: Option<String>) {
        self.expected_revision = revision.filter(|s| !s.is_empty());
    }

    pub fn set_skip_dedup_on_insert(&mut self, skip: bool) {
        self.skip_dedup_on_insert = skip;
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

    fn snapshot_state(&self) -> GraphSnapshot {
        GraphSnapshot::capture(self)
    }

    pub fn upsert_slug(&mut self, slug: String, id: NodeId) {
        self.slug_to_node.insert(slug, id);
    }

    fn normalize_knowledge_slug(
        &self,
        slug: &str,
        knowledge_type: KnowledgeType,
    ) -> Result<Slug, GraphError> {
        let parsed = match Slug::parse(slug) {
            Ok(parsed) => parsed,
            Err(SlugError::MissingKindPrefix) => Slug::generate(knowledge_type, slug),
            Err(err) => return Err(GraphError::Schema(err.to_string())),
        };
        if parsed.kind() != knowledge_type {
            return Err(GraphError::Schema(format!(
                "slug `{}` kind `{}` must match knowledge_type `{}`",
                slug,
                parsed.as_str().split('.').next().unwrap_or_default(),
                knowledge_type
            )));
        }
        Ok(parsed)
    }

    fn enforce_unique_knowledge(
        &self,
        exclude: Option<NodeId>,
        payload: &KnowledgeNode,
    ) -> Result<(), GraphError> {
        if self.skip_dedup_on_insert {
            return Ok(());
        }
        match self.dedup.check(
            &payload.title,
            &payload.statement,
            payload.knowledge_type,
            self.graph(),
            exclude,
        ) {
            DuplicateCheck::Unique => Ok(()),
            DuplicateCheck::Exact { existing } => Err(GraphError::Schema(format!(
                "node statement identical to existing `{}`; reuse or differentiate",
                self.graph()[existing].slug
            ))),
            DuplicateCheck::HighSimilarity { candidates } => {
                let similar: Vec<String> = candidates
                    .into_iter()
                    .map(|(id, score)| format!("{} ({:.0}%)", self.graph()[id].slug, score * 100.0))
                    .collect();
                Err(GraphError::Schema(format!(
                    "statement is highly similar to existing nodes: {}",
                    similar.join(", ")
                )))
            }
            DuplicateCheck::SimilarTitle {
                existing,
                similarity,
            } => Err(GraphError::Schema(format!(
                "title {:.0}% similar to existing `{}`; confirm distinct intent or reuse",
                similarity * 100.0,
                self.graph()[existing].slug
            ))),
        }
    }

    pub fn node_by_slug(&self, slug: &str) -> Result<NodeId, GraphError> {
        if let Some(id) = self.slug_to_node.get(slug) {
            return Ok(*id);
        }
        if let Ok(parsed) = Slug::parse(slug)
            && let Some(id) = self.slug_to_node.get(parsed.as_str())
        {
            return Ok(*id);
        }
        if !slug.contains('.') {
            let mut matches: Vec<(String, NodeId)> = Vec::new();
            for kind in [
                KnowledgeType::Factual,
                KnowledgeType::Conceptual,
                KnowledgeType::Procedural,
                KnowledgeType::Metacognitive,
                KnowledgeType::LearningOutcome,
                KnowledgeType::AssessmentItem,
            ] {
                let generated = Slug::generate(kind, slug);
                if let Some(id) = self.slug_to_node.get(generated.as_str()) {
                    matches.push((generated.as_str().to_string(), *id));
                }
            }
            match matches.len() {
                0 => {}
                1 => return Ok(matches[0].1),
                _ => {
                    let slugs = matches.into_iter().map(|(s, _)| s).collect();
                    return Err(GraphError::AmbiguousSlug {
                        slug:    slug.to_string(),
                        matches: slugs,
                    });
                }
            }
        }
        Err(GraphError::MissingSlug(slug.to_string()))
    }

    fn summarize_node(&self, id: NodeId) -> NodeSummary {
        let payload = &self.graph()[id];
        let (title, kind, knowledge_type) = match &payload.kind {
            NodeKind::Knowledge(k) => {
                (k.title.clone(), "knowledge".to_string(), Some(k.knowledge_type))
            }
            NodeKind::TeachingStep(ts) => (ts.title.clone(), "teaching_step".to_string(), None),
        };
        NodeSummary {
            slug: payload.slug.clone(),
            title,
            kind,
            knowledge_type,
            tags: payload.tags.clone(),
        }
    }

    pub fn list_nodes_by_tag(&self, tag: &str) -> Vec<NodeSummary> {
        let needle = tag.to_ascii_lowercase();
        let mut nodes = Vec::new();
        for id in self.graph().node_indices() {
            let has_tag = self.graph()[id]
                .tags
                .iter()
                .any(|t| t.to_ascii_lowercase() == needle);
            if has_tag {
                nodes.push(self.summarize_node(id));
            }
        }
        nodes.sort_by(|a, b| a.slug.cmp(&b.slug));
        nodes
    }

    pub fn list_nodes_by_kind(&self, selector: &NodeKindSelector) -> Vec<NodeSummary> {
        let mut nodes = Vec::new();
        for id in self.graph().node_indices() {
            let matches = match (&self.graph()[id].kind, selector) {
                (NodeKind::Knowledge(k), NodeKindSelector::Knowledge { knowledge_type }) => {
                    k.knowledge_type == *knowledge_type
                }
                (NodeKind::Knowledge(_), NodeKindSelector::AnyKnowledge) => true,
                (NodeKind::TeachingStep(_), NodeKindSelector::TeachingStep) => true,
                _ => false,
            };
            if matches {
                nodes.push(self.summarize_node(id));
            }
        }
        nodes.sort_by(|a, b| a.slug.cmp(&b.slug));
        nodes
    }

    pub fn list_tags(&self) -> Vec<String> {
        let mut tags: HashSet<String> = HashSet::new();
        for id in self.graph().node_indices() {
            for tag in &self.graph()[id].tags {
                tags.insert(tag.clone());
            }
        }
        let mut collected: Vec<String> = tags.into_iter().collect();
        collected.sort();
        collected
    }

    pub fn search_nodes(&self, query: &str, max_results: usize) -> Vec<NodeSearchResult> {
        if query.trim().is_empty() || max_results == 0 {
            return Vec::new();
        }
        let needle = query.to_ascii_lowercase();
        let mut results = Vec::new();

        for id in self.graph().node_indices() {
            let summary = self.summarize_node(id);
            let payload = &self.graph()[id];
            let title_lc = summary.title.to_ascii_lowercase();
            let slug_lc = summary.slug.to_ascii_lowercase();

            let mut score = 0.0_f64;
            if slug_lc.contains(&needle) || title_lc.contains(&needle) {
                score = 1.0;
            }

            let statement = match &payload.kind {
                NodeKind::Knowledge(k) => Some(k.statement.as_str()),
                NodeKind::TeachingStep(ts) => Some(ts.statement.as_str()),
            };
            if let Some(stmt) = statement {
                let stmt_lc = stmt.to_ascii_lowercase();
                if stmt_lc.contains(&needle) {
                    score = score.max(0.95);
                }
                score = score.max(jaro_winkler(&stmt_lc, &needle));
            }

            score = score.max(jaro_winkler(&slug_lc, &needle));
            score = score.max(jaro_winkler(&title_lc, &needle));

            if score < 0.6 {
                continue;
            }

            results.push(NodeSearchResult {
                slug:           summary.slug,
                title:          summary.title,
                kind:           summary.kind,
                knowledge_type: summary.knowledge_type,
                tags:           summary.tags,
                score:          score as f32,
            });
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.slug.cmp(&b.slug))
        });
        results.truncate(max_results.clamp(1, 500));
        results
    }

    pub fn add_knowledge_node(
        &mut self,
        slug: String,
        payload: KnowledgeNode,
        tags: Vec<String>,
    ) -> Result<NodeId, GraphError> {
        self.ensure_healthy()?;
        self.ensure_node_capacity()?;
        let parsed_slug = self.normalize_knowledge_slug(&slug, payload.knowledge_type)?;
        let slug = parsed_slug.as_str().to_string();
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
        self.enforce_unique_knowledge(None, &payload)?;
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
        let snapshot = self.snapshot_state();
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
            snapshot,
            move |svc| {
                svc.graph_mut().remove_node(id);
                svc.slug_to_node.remove(&slug);
            },
        )
        .on_success(|svc| svc.refresh_rubric_hashes());
        guard.commit()?;
        if let NodeKind::Knowledge(k) = &self.graph()[id].kind {
            let snapshot = k.clone();
            self.dedup.record(id, &snapshot);
        }
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
        self.ensure_healthy()?;
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
        // Ensure slug prefix stays aligned with the knowledge_type.
        let _ = self.normalize_knowledge_slug(&self.graph()[id].slug, payload.knowledge_type)?;
        self.enforce_unique_knowledge(Some(id), &payload)?;
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
        let snapshot = self.snapshot_state();

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
            snapshot,
            move |svc| {
                svc.graph_mut()[id].kind = old_kind;
                svc.graph_mut()[id].tags = old_tags;
            },
        )
        .on_success(|svc| svc.refresh_rubric_hashes());
        guard.commit()?;
        self.dedup.remove(id);
        if let NodeKind::Knowledge(k) = &self.graph()[id].kind {
            let snapshot = k.clone();
            self.dedup.record(id, &snapshot);
        }
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
        self.ensure_healthy()?;
        let id = self.node_by_slug(old_slug)?;
        let target_slug = match &self.graph()[id].kind {
            NodeKind::Knowledge(k) => self
                .normalize_knowledge_slug(&new_slug, k.knowledge_type)?
                .as_str()
                .to_string(),
            _ => new_slug.clone(),
        };
        if self.slug_to_node.contains_key(&target_slug) {
            return Err(GraphError::Schema(format!("slug `{}` already exists", target_slug)));
        }
        let stored_slug = self.graph()[id].slug.clone();
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
        let snapshot = self.snapshot_state();

        self.slug_to_node.remove(&stored_slug);
        self.slug_to_node.insert(target_slug.clone(), id);
        self.graph_mut()[id].slug = target_slug.clone();

        // keep denormalized claims in sync for assesses edges targeting this node
        for edge_id in incoming {
            if let EdgeKind::Assesses(attrs) = &mut self.graph_mut()[edge_id].kind {
                attrs.evidence_link.claim = target_slug.clone();
            }
        }

        let guard =
            ValidationGuard::new(self, InvariantFamilies::ALL, ValidationScope::Full, snapshot, {
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
                "to": target_slug,
            }),
        );
        Ok(())
    }

    pub fn merge_nodes(
        &mut self,
        canonical_slug: &str,
        duplicate_slug: &str,
    ) -> Result<MergeSummary, GraphError> {
        self.ensure_healthy()?;
        let canonical = self.node_by_slug(canonical_slug)?;
        let duplicate = self.node_by_slug(duplicate_slug)?;
        if canonical == duplicate {
            return Err(GraphError::Schema("cannot merge a node into itself".to_string()));
        }
        let (canonical_payload, duplicate_payload) =
            match (&self.graph()[canonical].kind, &self.graph()[duplicate].kind) {
                (NodeKind::Knowledge(canon), NodeKind::Knowledge(dup)) => {
                    (canon.clone(), dup.clone())
                }
                _ => {
                    return Err(GraphError::Schema(
                        "merges are only supported for knowledge nodes".to_string(),
                    ));
                }
            };
        if canonical_payload.knowledge_type != duplicate_payload.knowledge_type {
            return Err(GraphError::Schema(
                "cannot merge nodes with different knowledge_type".to_string(),
            ));
        }
        if canonical_payload.knowledge_type.is_learning_outcome()
            || canonical_payload.knowledge_type.is_assessment_item()
        {
            return Err(GraphError::Schema(
                "cannot auto-merge learning_outcome or assessment_item nodes".to_string(),
            ));
        }
        let duplicate_has_assesses = self
            .graph()
            .edges_directed(duplicate, Direction::Incoming)
            .chain(self.graph().edges_directed(duplicate, Direction::Outgoing))
            .any(|edge| matches!(edge.weight().kind, EdgeKind::Assesses(_)));
        if duplicate_has_assesses {
            return Err(GraphError::Schema(
                "cannot auto-merge nodes that participate in assesses edges".to_string(),
            ));
        }

        let old_graph = self.graph.clone();
        let old_index = self.slug_to_node.clone();
        let old_version = self.graph_version;
        let old_rubric = self
            .rubric_hashes
            .read()
            .expect("rubric_hashes lock")
            .clone();
        let old_dirty = self.validation_state.dirty.load(Ordering::Relaxed);

        let incoming: Vec<(NodeId, EdgePayload)> = self
            .graph()
            .edges_directed(duplicate, Direction::Incoming)
            .map(|e| (e.source(), e.weight().clone()))
            .collect();
        let outgoing: Vec<(NodeId, EdgePayload)> = self
            .graph()
            .edges_directed(duplicate, Direction::Outgoing)
            .map(|e| (e.target(), e.weight().clone()))
            .collect();

        let merged_slug_into = self.graph()[canonical].slug.clone();
        let merged_slug_from = self.graph()[duplicate].slug.clone();

        let merged_tags =
            union_vecs(self.graph()[canonical].tags.clone(), self.graph()[duplicate].tags.clone());
        let merged_refs =
            union_source_refs(&canonical_payload.source_refs, &duplicate_payload.source_refs);
        let merged_confidence = canonical_payload
            .confidence
            .max(duplicate_payload.confidence);
        let merged_statement = if canonical_payload.statement.trim()
            == duplicate_payload.statement.trim()
        {
            canonical_payload.statement.clone()
        } else {
            format!("{}\n\nMerged: {}", canonical_payload.statement, duplicate_payload.statement)
        };
        let mut merged_payload = canonical_payload.clone();
        merged_payload.source_refs = merged_refs;
        merged_payload.confidence = merged_confidence;
        merged_payload.statement = merged_statement;
        merged_payload.construct_irrelevant_demands = union_vecs(
            merged_payload.construct_irrelevant_demands.clone(),
            duplicate_payload.construct_irrelevant_demands.clone(),
        );

        self.graph_mut()[canonical].kind = NodeKind::Knowledge(merged_payload);
        self.graph_mut()[canonical].tags = merged_tags;
        self.dedup.remove(duplicate);
        self.slug_to_node.remove(&merged_slug_from);
        self.graph_mut().remove_node(duplicate);

        let mut edges_redirected = 0usize;
        for (from, payload) in incoming {
            if from == canonical {
                continue;
            }
            if let Some(edge_id) =
                self.find_edge_of_kind(from, canonical, |k| edge_variant_eq(k, &payload.kind))
            {
                let merged = merge_edge_payload(self.graph()[edge_id].clone(), payload);
                let updated_payload = match merged {
                    MergeResult::Merged(p) | MergeResult::Conflict { existing: p, .. } => p,
                };
                self.graph_mut()[edge_id] = updated_payload;
            } else {
                self.graph_mut().add_edge(from, canonical, payload);
                edges_redirected = edges_redirected.saturating_add(1);
            }
        }
        for (target, payload) in outgoing {
            if target == canonical {
                continue;
            }
            if let Some(edge_id) =
                self.find_edge_of_kind(canonical, target, |k| edge_variant_eq(k, &payload.kind))
            {
                let merged = merge_edge_payload(self.graph()[edge_id].clone(), payload);
                let updated_payload = match merged {
                    MergeResult::Merged(p) | MergeResult::Conflict { existing: p, .. } => p,
                };
                self.graph_mut()[edge_id] = updated_payload;
            } else {
                self.graph_mut().add_edge(canonical, target, payload);
                edges_redirected = edges_redirected.saturating_add(1);
            }
        }

        self.mark_dirty(InvariantFamilies::ALL);
        self.bump_version();
        if let Err(err) = self.validate_global_invariants() {
            self.graph = old_graph;
            self.slug_to_node = old_index;
            self.graph_version = old_version;
            *self.rubric_hashes.write().expect("rubric_hashes lock") = old_rubric;
            self.rebuild_dedup();
            self.validation_state
                .dirty
                .store(old_dirty, Ordering::Relaxed);
            return Err(err);
        }
        self.rebuild_dedup();
        self.record_mutation(
            MutationKind::MergeNodes,
            json!({
                "into": merged_slug_into,
                "from": merged_slug_from,
                "edges_redirected": edges_redirected,
            }),
        );
        Ok(MergeSummary {
            merged_from: merged_slug_from,
            merged_into: merged_slug_into,
            edges_redirected,
        })
    }

    pub fn resolve_edge_conflict(
        &mut self,
        edge_id: EdgeId,
        resolved_kind: EdgeKind,
        confidence: Option<f32>,
        clear_conflicts: bool,
    ) -> Result<EdgeId, GraphError> {
        self.ensure_healthy()?;
        let Some((from, to)) = self.graph.edge_endpoints(edge_id) else {
            return Err(GraphError::Schema(format!("edge_id {:?} not found", edge_id.index())));
        };
        let current = self
            .graph
            .edge_weight(edge_id)
            .ok_or_else(|| GraphError::Schema("edge not found".to_string()))?
            .clone();

        let mut updated = current.clone();
        updated.kind = resolved_kind;
        updated.confidence = confidence.unwrap_or(updated.confidence);
        if clear_conflicts {
            updated.conflicts.clear();
        }

        self.validate_edge_kind(from, to, &updated.kind, updated.confidence)?;
        let (coverage_los, skip_requires_dag, skip_fadeability, dirty) =
            self.edge_validation_plan(edge_name(&updated.kind), from, to);

        let snapshot = self.snapshot_state();
        self.graph_mut()[edge_id] = updated;
        let guard = ValidationGuard::new(
            self,
            dirty,
            ValidationScope::Targeted {
                coverage_los,
                skip_requires_dag,
                skip_fadeability,
            },
            snapshot,
            move |svc| {
                svc.graph_mut()[edge_id] = current;
            },
        );
        guard.commit()?;
        self.record_mutation(
            MutationKind::ResolveEdgeConflict,
            json!({
                "edge_id": edge_id.index(),
                "from": self.graph()[from].slug,
                "to": self.graph()[to].slug,
            }),
        );
        Ok(edge_id)
    }

    pub fn remove_node(&mut self, slug: &str) -> Result<(), GraphError> {
        self.ensure_healthy()?;
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
        let stored_slug = self.graph()[id].slug.clone();
        let snapshot = self.snapshot_state();

        if matches!(self.graph()[id].kind, NodeKind::Knowledge(_)) {
            self.dedup.remove(id);
        }
        self.slug_to_node.remove(&stored_slug);
        self.graph_mut().remove_node(id);

        let guard = ValidationGuard::new(
            self,
            InvariantFamilies::ALL,
            ValidationScope::Full,
            snapshot,
            move |svc| {
                svc.rebuild_dedup();
            },
        )
        .on_success(|svc| svc.refresh_rubric_hashes());
        guard.commit()?;
        self.rebuild_dedup();
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
        self.ensure_healthy()?;
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
        self.rebuild_dedup();
        self.mark_dirty(InvariantFamilies::ALL);
        if let Err(err) = self.validate_global_invariants() {
            // rollback on failure
            self.graph = old_graph;
            self.slug_to_node = old_index;
            self.graph_version = old_version;
            self.rubric_hashes = RwLock::new(compute_rubric_hashes(self.graph.as_ref()));
            self.rebuild_dedup();
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
        self.ensure_healthy()?;
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
        let snapshot = self.snapshot_state();
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
            snapshot,
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
        self.ensure_healthy()?;
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
        let snapshot = self.snapshot_state();
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
            snapshot,
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
        self.ensure_healthy()?;
        self.ensure_edge_capacity()?;
        S::validate(self, from, to, &attrs, confidence)?;
        let payload = S::make_payload(attrs, confidence);

        if let Some(existing_edge_id) =
            self.find_edge_of_kind(from, to, |k| edge_variant_eq(k, &payload.kind))
        {
            return self.merge_existing_edge(S::NAME, from, to, existing_edge_id, payload);
        }

        let snapshot = self.snapshot_state();
        let edge_id = self.graph_mut().add_edge(from, to, payload);
        let (coverage_los, skip_requires_dag, skip_fadeability, dirty) =
            self.edge_validation_plan(S::NAME, from, to);

        let guard = ValidationGuard::new(
            self,
            dirty,
            ValidationScope::Targeted {
                coverage_los,
                skip_requires_dag,
                skip_fadeability,
            },
            snapshot,
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
                "merged_existing": false,
            }),
        );
        Ok(edge_id)
    }

    fn merge_existing_edge(
        &mut self,
        edge_name: &'static str,
        from: NodeId,
        to: NodeId,
        edge_id: EdgeId,
        incoming: EdgePayload,
    ) -> Result<EdgeId, GraphError> {
        self.ensure_healthy()?;
        let merged = merge_edge_payload(self.graph()[edge_id].clone(), incoming);
        let had_conflict = matches!(merged, MergeResult::Conflict { .. });
        let updated_payload = match merged {
            MergeResult::Merged(payload)
            | MergeResult::Conflict {
                existing: payload, ..
            } => payload,
        };

        self.validate_edge_kind(from, to, &updated_payload.kind, updated_payload.confidence)?;
        let rollback_payload = self.graph()[edge_id].clone();
        let snapshot = self.snapshot_state();
        self.graph_mut()[edge_id] = updated_payload;

        let (coverage_los, skip_requires_dag, skip_fadeability, dirty) =
            self.edge_validation_plan(edge_name, from, to);
        let guard = ValidationGuard::new(
            self,
            dirty,
            ValidationScope::Targeted {
                coverage_los,
                skip_requires_dag,
                skip_fadeability,
            },
            snapshot,
            move |svc| {
                svc.graph_mut()[edge_id] = rollback_payload;
            },
        );
        guard.commit()?;
        self.record_mutation(
            MutationKind::AddEdge { edge: edge_name },
            json!({
                "edge": edge_name,
                "from": self.graph()[from].slug.clone(),
                "to": self.graph()[to].slug.clone(),
                "confidence": self.graph()[edge_id].confidence,
                "merged_existing": true,
                "edge_conflicts": self.graph()[edge_id].conflicts.len(),
                "had_conflict": had_conflict,
            }),
        );
        Ok(edge_id)
    }

    fn edge_validation_plan(
        &self,
        edge_name: &'static str,
        from: NodeId,
        to: NodeId,
    ) -> (Vec<NodeId>, bool, bool, InvariantFamilies) {
        let coverage_los = match edge_name {
            "assesses" => vec![to],
            "requires" => self.impacted_los_from_requires(from),
            _ => Vec::new(),
        };

        let (skip_requires_dag, skip_fadeability) = match edge_name {
            "requires" => (false, false),
            "supports" => (true, false),
            _ => (true, true),
        };

        let mut dirty = match edge_name {
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

        (coverage_los, skip_requires_dag, skip_fadeability, dirty)
    }

    fn find_edge_of_kind(
        &self,
        from: NodeId,
        to: NodeId,
        pred: impl Fn(&EdgeKind) -> bool,
    ) -> Option<EdgeId> {
        self.graph
            .edges_directed(from, Direction::Outgoing)
            .find(|e| e.target() == to && pred(&e.weight().kind))
            .map(|e| e.id())
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

fn edge_variant_eq(a: &EdgeKind, b: &EdgeKind) -> bool {
    matches!(
        (a, b),
        (EdgeKind::Requires(_), EdgeKind::Requires(_))
            | (EdgeKind::Supports(_), EdgeKind::Supports(_))
            | (EdgeKind::Assesses(_), EdgeKind::Assesses(_))
            | (EdgeKind::Precedes(_), EdgeKind::Precedes(_))
            | (EdgeKind::Anchors(_), EdgeKind::Anchors(_))
    )
}

fn edge_name(kind: &EdgeKind) -> &'static str {
    match kind {
        EdgeKind::Requires(_) => "requires",
        EdgeKind::Supports(_) => "supports",
        EdgeKind::Assesses(_) => "assesses",
        EdgeKind::Precedes(_) => "precedes",
        EdgeKind::Anchors(_) => "anchors",
    }
}

#[derive(Clone, Debug)]
pub struct EdgeConflictState {
    pub edge_id: EdgeId,
    pub from:    NodeId,
    pub to:      NodeId,
    pub payload: EdgePayload,
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
