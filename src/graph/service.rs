use std::{collections::HashMap, sync::Arc};

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
    schema::types::KnowledgeType,
};

/// Core graph owner with slug lookup and optional strict quality mode.
pub struct GraphService {
    graph:          Arc<CurriculumGraph>,
    slug_to_node:   HashMap<String, NodeId>,
    strict_quality: bool,
    graph_version:  u64,
}

impl GraphService {
    pub fn new() -> Self {
        Self {
            graph:          Arc::new(CurriculumGraph::default()),
            slug_to_node:   HashMap::new(),
            strict_quality: false,
            graph_version:  0,
        }
    }

    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict_quality = strict;
        self
    }

    pub fn from_graph(graph: CurriculumGraph) -> Self {
        Self::from_parts(graph, false, 0)
            .expect("graph provided to from_graph must have unique slugs and valid invariants")
    }

    pub fn from_parts(
        graph: CurriculumGraph,
        strict_quality: bool,
        graph_version: u64,
    ) -> Result<Self, GraphError> {
        let mut svc = Self {
            graph: Arc::new(graph),
            slug_to_node: HashMap::new(),
            strict_quality,
            graph_version,
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
        if payload.source_refs.is_empty() {
            return Err(GraphError::Schema(
                "knowledge nodes must include at least one source_ref".to_string(),
            ));
        }
        validate_source_refs(&payload.source_refs)?;
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

        if payload.source_refs.is_empty() {
            return Err(GraphError::Schema(
                "knowledge nodes must include at least one source_ref".to_string(),
            ));
        }
        validate_source_refs(&payload.source_refs)?;

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
            EdgeKind::Requires(a) => RequiresSpec::validate(self, from, to, a, confidence),
            EdgeKind::Supports(a) => SupportsSpec::validate(self, from, to, a, confidence),
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
        self.bump_version();
        Ok(())
    }

    pub fn remove_node(&mut self, slug: &str) -> Result<(), GraphError> {
        let id = self.node_by_slug(slug)?;
        self.slug_to_node.remove(slug);
        self.graph_mut().remove_node(id);
        self.bump_version();
        Ok(())
    }

    pub fn snapshot_graph(&self) -> CurriculumGraph {
        (*self.graph).clone()
    }

    pub fn install_graph(
        &mut self,
        graph: CurriculumGraph,
        graph_version: u64,
    ) -> Result<(), GraphError> {
        // Build slug index first; only commit if it succeeds.
        let candidate = Arc::new(graph);
        let index = self.build_slug_index_map(candidate.as_ref())?;

        let old_graph = self.graph.clone();
        let old_index = self.slug_to_node.clone();
        let old_version = self.graph_version;

        self.graph = candidate;
        self.slug_to_node = index;
        self.graph_version = graph_version;

        if let Err(err) = self.validate_global_invariants() {
            // rollback on failure
            self.graph = old_graph;
            self.slug_to_node = old_index;
            self.graph_version = old_version;
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
        if payload.source_refs.is_empty() {
            return Err(GraphError::Schema(
                "teaching steps must include at least one source_ref".to_string(),
            ));
        }
        validate_source_refs(&payload.source_refs)?;
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
        if payload.source_refs.is_empty() {
            return Err(GraphError::Schema(
                "teaching steps must include at least one source_ref".to_string(),
            ));
        }
        validate_source_refs(&payload.source_refs)?;
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

    /// Ensure every assesses edge claim matches its target LO slug. Intended
    /// for snapshot loads and node renames performed outside the
    /// GraphService API.
    pub fn normalize_assesses_claims(&mut self) {
        let graph = Arc::make_mut(&mut self.graph);
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
        let mut warnings = Vec::new();
        let mut errors = Vec::new();

        if !analysis::requires_is_dag(g) {
            errors.push("requires layer must remain acyclic".to_string());
        }

        for lo in analysis::lo_missing_target_assessments(g) {
            warnings
                .push(format!("learning_outcome `{}` missing assesses(scope=target)", g[lo].slug));
        }

        for lo_id in g.node_indices().filter(|&n| matches!(&g[n].kind, NodeKind::Knowledge(k) if k.knowledge_type == KnowledgeType::LearningOutcome)) {
            let report = analysis::coverage_report(g, lo_id);
            if !report.missing_criteria.is_empty() {
                warnings.push(format!(
                    "learning_outcome `{}` missing coverage for rubric criteria: {}",
                    g[lo_id].slug,
                    report.missing_criteria.join(", ")
                ));
            }
        }

        for gap in analysis::example_gaps(g) {
            warnings.push(format!("{}: {}", g[gap.node].slug, gap.description));
        }

        for gap in analysis::procedural_practice_gaps(g) {
            warnings.push(format!(
                "procedural `{}` lacks reachable assessment with assesses(scope=target)",
                g[gap.node].slug
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
            errors.push(format!(
                "assessment `{}` reachable only via supports that carry prerequisite load: [{}]",
                assessment_slug,
                edges.join("; ")
            ));
        }

        if self.strict_quality {
            errors.extend(warnings);
        } else {
            for w in warnings {
                warn!(target: "weaver.graph.invariants", "{w}");
            }
        }

        if errors.is_empty() {
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

impl Default for GraphService {
    fn default() -> Self {
        Self::new()
    }
}
