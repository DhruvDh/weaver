use std::{
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
    sync::{Arc, RwLock},
};

use petgraph::{Direction, visit::EdgeRef};
use tracing::warn;
use uuid::Uuid;

use crate::{
    analysis,
    graph::{
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
    _rule:             &'static str,
    severity:          ValidationSeverity,
    message:           String,
    promote_in_strict: bool,
}

/// Core graph owner with slug lookup and optional strict quality mode.
pub struct GraphService {
    graph:             Arc<CurriculumGraph>,
    slug_to_node:      HashMap<String, NodeId>,
    strict_quality:    bool,
    graph_version:     u64,
    expected_revision: Option<String>,
    rubric_hashes:     RwLock<HashMap<String, u64>>,
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
        let logical_id = Uuid::new_v4();
        let node = NodePayload {
            logical_id,
            slug: slug.clone(),
            kind: NodeKind::Knowledge(payload),
            tags,
        };
        let id = self.graph_mut().add_node(node);
        self.upsert_slug(slug.clone(), id);
        self.validate_global_invariants_or_rollback_node(id, &slug)?;
        self.refresh_rubric_hashes();
        self.bump_version();
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

        // apply tentative change
        self.graph_mut()[id].kind = NodeKind::Knowledge(payload);

        // validate incident edges against the new kind
        if let Err(err) = self.validate_incident_edges(id) {
            // rollback
            self.graph_mut()[id].kind = old_kind;
            return Err(err);
        }

        self.graph_mut()[id].tags = tags;
        if let Err(err) = self.validate_global_invariants() {
            // rollback
            self.graph_mut()[id].kind = old_kind;
            self.graph_mut()[id].tags = old_tags;
            return Err(err);
        }
        self.refresh_rubric_hashes();
        self.bump_version();
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
        self.slug_to_node.remove(old_slug);
        self.slug_to_node.insert(new_slug.clone(), id);
        self.graph_mut()[id].slug = new_slug.clone();

        // keep denormalized claims in sync for assesses edges targeting this node
        let incoming: Vec<_> = self
            .graph()
            .edges_directed(id, Direction::Incoming)
            .map(|e| e.id())
            .collect();
        for edge_id in incoming {
            if let EdgeKind::Assesses(attrs) = &mut self.graph_mut()[edge_id].kind {
                attrs.evidence_link.claim = new_slug.clone();
            }
        }
        self.refresh_rubric_hashes();
        self.bump_version();
        Ok(())
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

        if let Err(err) = self.validate_global_invariants() {
            // rollback
            self.graph = old_graph;
            self.slug_to_node = old_index;
            self.graph_version = old_version;
            *self.rubric_hashes.write().expect("rubric_hashes lock") = old_rubric;
            return Err(err);
        }

        self.refresh_rubric_hashes();
        self.bump_version();
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
        self.validate_global_invariants_or_rollback_node(id, &slug)?;
        self.refresh_rubric_hashes();
        self.bump_version();
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
        if let Err(err) = self.validate_global_invariants() {
            self.graph_mut()[id].kind = old_kind;
            self.graph_mut()[id].tags = old_tags;
            return Err(err);
        }
        self.refresh_rubric_hashes();
        self.bump_version();
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

        if let Err(err) = self.validate_global_invariants() {
            // rollback
            self.graph_mut().remove_edge(edge_id);
            return Err(err);
        }

        self.bump_version();
        Ok(edge_id)
    }

    fn has_edge_of_kind(&self, from: NodeId, to: NodeId, pred: impl Fn(&EdgeKind) -> bool) -> bool {
        self.graph
            .edges_directed(from, Direction::Outgoing)
            .any(|e| e.target() == to && pred(&e.weight().kind))
    }

    fn validate_revision(&self, spans: &[SourceRef], label: &str) -> Result<(), GraphError> {
        if let Some(expected) = &self.expected_revision {
            for span in spans {
                if span.revision != *expected {
                    return Err(GraphError::Schema(format!(
                        "{label} source_ref revision `{}` must equal course_commit `{}`",
                        span.revision, expected
                    )));
                }
            }
        }
        Ok(())
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
            self.bump_version();
        }
        removed
    }

    /// Run global audits and return violations as errors.
    pub fn validate_global_invariants(&self) -> Result<(), GraphError> {
        let g = self.graph();
        let slug_index: HashMap<String, NodeId> =
            g.node_indices().map(|n| (g[n].slug.clone(), n)).collect();
        let has_teaching_steps = g
            .node_indices()
            .any(|n| matches!(&g[n].kind, NodeKind::TeachingStep(_)));
        let first_principles = analysis::first_principles(g);
        let lo_nodes: Vec<NodeId> = g
            .node_indices()
            .filter(|&n| matches!(&g[n].kind, NodeKind::Knowledge(k) if k.knowledge_type == KnowledgeType::LearningOutcome))
            .collect();
        let rubric_prev = self
            .rubric_hashes
            .read()
            .expect("rubric_hashes lock")
            .clone();
        let rubric_current = compute_rubric_hashes(g);

        let mut issues = Vec::new();
        issues.extend(check_statements(g));
        issues.extend(check_provenance(g, self.expected_revision()));
        issues.extend(check_requires_and_fadeability(g));
        issues.extend(check_reachability_and_coverage(
            g,
            &first_principles,
            &lo_nodes,
            &rubric_prev,
            &rubric_current,
        ));
        issues.extend(check_supports_and_practice(g));
        issues.extend(check_purity(g, &slug_index));
        if has_teaching_steps {
            issues.extend(check_discourse(g));
            issues.extend(check_introductions(g));
        }

        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        for mut issue in issues {
            if self.strict_quality && issue.promote_in_strict {
                issue.severity = ValidationSeverity::Error;
            }
            match issue.severity {
                ValidationSeverity::Error => errors.push(issue.message),
                ValidationSeverity::Warning => warnings.push(issue.message),
            }
        }

        if self.strict_quality {
            errors.extend(warnings);
        } else {
            for w in warnings {
                warn!(target: "weaver.graph.invariants", "{w}");
            }
        }

        if errors.is_empty() {
            *self.rubric_hashes.write().expect("rubric_hashes lock") = rubric_current;
            Ok(())
        } else {
            Err(GraphError::InvariantViolation { violations: errors })
        }
    }

    fn validate_global_invariants_or_rollback_node(
        &mut self,
        id: NodeId,
        slug: &str,
    ) -> Result<(), GraphError> {
        if let Err(err) = self.validate_global_invariants() {
            self.graph_mut().remove_node(id);
            self.slug_to_node.remove(slug);
            return Err(err);
        }
        Ok(())
    }
}

fn make_issue(
    rule: &'static str,
    severity: ValidationSeverity,
    promote_in_strict: bool,
    message: String,
) -> ValidationIssue {
    ValidationIssue {
        _rule: rule,
        severity,
        message,
        promote_in_strict,
    }
}

fn check_statements(g: &CurriculumGraph) -> Vec<ValidationIssue> {
    g.node_indices()
        .filter_map(|n| match &g[n].kind {
            NodeKind::Knowledge(k) if k.statement.trim().is_empty() => Some(make_issue(
                "statement_nonempty",
                ValidationSeverity::Error,
                true,
                format!("knowledge `{}` has empty statement", g[n].slug),
            )),
            NodeKind::TeachingStep(ts) if ts.statement.trim().is_empty() => Some(make_issue(
                "statement_nonempty",
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
                            "provenance_revision",
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
                            "provenance_revision",
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
                                "provenance_revision",
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
                                "provenance_revision",
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

fn check_requires_and_fadeability(g: &CurriculumGraph) -> Vec<ValidationIssue> {
    let mut out = Vec::new();
    if !analysis::requires_is_dag(g) {
        out.push(make_issue(
            "requires_dag",
            ValidationSeverity::Error,
            true,
            "requires layer must remain acyclic".to_string(),
        ));
    }
    for issue in analysis::fadeability_issues(g) {
        let assessment_slug = g[issue.assessment].slug.clone();
        let edges: Vec<String> = issue
            .support_edges
            .iter()
            .filter_map(|e| g.edge_endpoints(*e))
            .map(|(u, v)| format!("{} -> {}", g[u].slug, g[v].slug))
            .collect();
        out.push(make_issue(
            "fadeability",
            ValidationSeverity::Error,
            true,
            format!(
                "assessment `{}` reachable only via supports that carry prerequisite load: [{}]",
                assessment_slug,
                edges.join("; ")
            ),
        ));
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
                "lo_target_assessment",
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
                "lo_reachability",
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
                "rubric_coverage",
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
                "rubric_unused_observation_features",
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
                "rubric_drift",
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
            "example_minimums",
            ValidationSeverity::Warning,
            true,
            format!("{}: {}", g[gap.node].slug, gap.description),
        ));
    }
    for gap in analysis::procedural_practice_gaps(g) {
        out.push(make_issue(
            "procedural_practice",
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
                    "purity_intended_missing",
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
                    "purity_extraneous",
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
                        "borrow_ahead",
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
                    "borrow_ahead",
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
            "discourse_orphan",
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
            if !has_anchor && !has_rationale {
                out.push(make_issue(
                    "teaching_step_anchor_or_rationale",
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
                    "introduce_anchor",
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
