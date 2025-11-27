use std::{collections::HashMap, path::PathBuf, sync::Arc};

use petgraph::{
    Directed, Direction,
    stable_graph::{EdgeIndex, NodeIndex, StableGraph},
    visit::EdgeRef,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::warn;
use uuid::Uuid;

use crate::schema::types::{
    EvidenceLink, IntendedEffect, KnowledgeType, SourceRef, Strength, SupportKind,
};

pub mod manager;
pub mod persist;
pub mod traversal;

fn validate_source_refs(spans: &[SourceRef]) -> Result<(), GraphError> {
    for span in spans {
        crate::schema::validate::validate_source_ref(span)
            .map_err(|e| GraphError::Schema(e.to_string()))?;
    }
    Ok(())
}

/// Runtime configuration for graph persistence/metadata.
#[derive(Clone, Debug)]
pub struct GraphConfig {
    pub course_commit: String,
    pub autosave_path: PathBuf,
    pub autosave_secs: u64,
}

/// Primary graph type alias (stable indices survive deletions).
pub type GraphIx = u32;
pub type NodeId = NodeIndex<GraphIx>;
pub type EdgeId = EdgeIndex<GraphIx>;
pub type CurriculumGraph = StableGraph<NodePayload, EdgePayload, Directed, GraphIx>;

/// Payload attached to every node.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodePayload {
    pub logical_id: Uuid,
    pub slug:       String,
    pub kind:       NodeKind,
    pub tags:       Vec<String>,
}

/// Distinguishes knowledge/assessment/LO nodes from discourse TeachingSteps.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum NodeKind {
    Knowledge(KnowledgeNode),
    TeachingStep(TeachingStepNode),
}

/// Knowledge node payload (covers LOs and assessments via `knowledge_type`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KnowledgeNode {
    pub title: String,
    pub statement: String,
    pub knowledge_type: KnowledgeType,
    pub source_refs: Vec<SourceRef>,
    pub confidence: f32,
    pub rubric_criteria: Vec<String>, // non-empty for LOs
    pub construct_irrelevant_demands: Vec<String>, // for assessment items
    pub grain_level: Option<GrainLevel>,
    pub intrinsic_load: Option<IntrinsicLoad>,
    pub introduction_scope: IntroductionScope,
}

/// Teaching step payload for discourse layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TeachingStepNode {
    pub title:       String,
    pub statement:   String,
    pub purpose:     TeachingPurpose,
    pub method_tags: Vec<String>,
    pub episode:     String,
    pub source_refs: Vec<SourceRef>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TeachingPurpose {
    Setup,
    Idea,
    Use,
    Consolidate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GrainLevel {
    Macro,
    Mid,
    Micro,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum IntrinsicLoad {
    Low,
    Medium,
    High,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum IntroductionScope {
    InCourse,
    Prior,
    External,
}

/// Edge payload plus layer discriminator.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EdgePayload {
    pub kind:       EdgeKind,
    pub confidence: f32,
}

/// Multiplex edge types.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EdgeKind {
    Requires(RequiresAttrs),
    Supports(SupportsAttrs),
    Assesses(AssessesAttrs),
    Precedes(PrecedesAttrs),
    Anchors(AnchorsAttrs),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequiresAttrs {
    pub strength:      Strength,
    pub rationale:     String,
    pub evidence_refs: Vec<SourceRef>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SupportsAttrs {
    pub support_kind:    SupportKind,
    pub intended_effect: IntendedEffect,
    pub case_tag:        Option<CaseTag>,
    pub coverage_tags:   Vec<String>,
    pub evidence_refs:   Vec<SourceRef>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CaseTag {
    Typical,
    Edge,
    ErrorCase,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AssessesAttrs {
    pub evidence_link: EvidenceLink,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrecedesAttrs {
    pub episode: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnchorsAttrs {
    pub impact: AnchorImpact,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AnchorImpact {
    Introduce,
    Use,
    Refine,
    Motivate,
    Target,
}

/// Errors returned by GraphService.
#[derive(thiserror::Error, Debug)]
pub enum GraphError {
    #[error("slug `{0}` not found")]
    MissingSlug(String),
    #[error("invalid edge endpoints for {edge}: from={from:?}, to={to:?}")]
    InvalidEndpoints {
        edge: &'static str,
        from: Option<NodeKindPreview>,
        to:   Option<NodeKindPreview>,
    },
    #[error("edge would create a cycle in requires layer: {cycle_slugs:?}")]
    RequiresCycle { cycle_slugs: Vec<String> },
    #[error("validator error: {0}")]
    Schema(String),
    #[error("graph invariant violation(s): {violations:?}")]
    InvariantViolation { violations: Vec<String> },
}

/// Lightweight snapshot of a node's kind used for error messages.
#[derive(Clone, Debug)]
pub enum NodeKindPreview {
    Knowledge(KnowledgeType),
    TeachingStep,
}

impl NodeKindPreview {
    fn from_node(node: &NodeKind) -> Self {
        match node {
            NodeKind::Knowledge(k) => NodeKindPreview::Knowledge(k.knowledge_type),
            NodeKind::TeachingStep(_) => NodeKindPreview::TeachingStep,
        }
    }
}

/// Core graph owner with slug lookup.
pub struct GraphService {
    graph:        Arc<CurriculumGraph>,
    slug_to_node: HashMap<String, NodeId>,
}

/// Trait implemented per edge type to centralize validation and payload
/// construction. Implementations may use GraphService to enforce global
/// constraints (e.g., DAG guards).
pub trait EdgeSpec {
    type Attrs;
    const NAME: &'static str;

    fn validate(
        svc: &GraphService,
        from: NodeId,
        to: NodeId,
        attrs: &Self::Attrs,
        confidence: f32,
    ) -> Result<(), GraphError>;

    fn make_payload(attrs: Self::Attrs, confidence: f32) -> EdgePayload;
}

impl GraphService {
    pub fn new() -> Self {
        Self {
            graph:        Arc::new(StableGraph::default()),
            slug_to_node: HashMap::new(),
        }
    }

    pub fn from_graph(graph: CurriculumGraph) -> Self {
        let mut svc = Self {
            graph:        Arc::new(graph),
            slug_to_node: HashMap::new(),
        };
        svc.rebuild_slug_index()
            .expect("graph provided to from_graph must have unique slugs");
        svc
    }

    /// Cheap shared pointer for read-heavy callers.
    pub fn shared_graph(&self) -> Arc<CurriculumGraph> {
        self.graph.clone()
    }

    pub fn graph(&self) -> &CurriculumGraph {
        &self.graph
    }

    pub(crate) fn graph_mut(&mut self) -> &mut CurriculumGraph {
        Arc::make_mut(&mut self.graph)
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
        Ok(())
    }

    pub fn remove_node(&mut self, slug: &str) -> Result<(), GraphError> {
        let id = self.node_by_slug(slug)?;
        self.slug_to_node.remove(slug);
        self.graph_mut().remove_node(id);
        Ok(())
    }

    pub fn snapshot_graph(&self) -> CurriculumGraph {
        (*self.graph).clone()
    }

    pub fn replace_graph(&mut self, graph: CurriculumGraph) -> Result<(), GraphError> {
        // Build slug index first; only commit if it succeeds.
        let candidate = Arc::new(graph);
        let index = self.build_slug_index_map(candidate.as_ref())?;
        self.graph = candidate;
        self.slug_to_node = index;
        Ok(())
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

        Ok(edge_id)
    }

    fn has_edge_of_kind(&self, from: NodeId, to: NodeId, pred: impl Fn(&EdgeKind) -> bool) -> bool {
        use petgraph::Direction;
        self.graph
            .edges_directed(from, Direction::Outgoing)
            .any(|e| e.target() == to && pred(&e.weight().kind))
    }

    pub fn node(&self, id: NodeId) -> &NodePayload {
        &self.graph[id]
    }

    fn node_kinds(
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

    fn has_precedes_path(&self, start: NodeId, goal: NodeId, episode: &str) -> bool {
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
        removed
    }

    /// Run global audits and return violations as errors.
    fn validate_global_invariants(&self) -> Result<(), GraphError> {
        use crate::analysis;

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

        for w in warnings {
            warn!(target: "weaver.graph.invariants", "{w}");
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

// ---------- EdgeSpec implementations ----------

pub struct RequiresSpec;
impl EdgeSpec for RequiresSpec {
    type Attrs = RequiresAttrs;
    const NAME: &'static str = "requires";

    fn validate(
        svc: &GraphService,
        from: NodeId,
        to: NodeId,
        attrs: &Self::Attrs,
        _confidence: f32,
    ) -> Result<(), GraphError> {
        let (from_kind, to_kind) = svc.node_kinds(Self::NAME, from, to)?;
        let (from_kt, to_kt) = match (from_kind, to_kind) {
            (NodeKind::Knowledge(f), NodeKind::Knowledge(t)) => {
                (f.knowledge_type, t.knowledge_type)
            }
            _ => {
                return Err(GraphError::InvalidEndpoints {
                    edge: Self::NAME,
                    from: Some(NodeKindPreview::from_node(from_kind)),
                    to:   Some(NodeKindPreview::from_node(to_kind)),
                });
            }
        };

        crate::schema::validate::validate_requires(
            from_kt,
            to_kt,
            attrs.strength,
            &attrs.rationale,
            &attrs.evidence_refs,
        )
        .map_err(|e| GraphError::Schema(e.to_string()))?;

        if let Some(path) = svc.requires_path(to, from) {
            let cycle_slugs = path
                .into_iter()
                .map(|id| svc.graph()[id].slug.clone())
                .collect();
            return Err(GraphError::RequiresCycle { cycle_slugs });
        }
        Ok(())
    }

    fn make_payload(attrs: Self::Attrs, confidence: f32) -> EdgePayload {
        EdgePayload {
            kind: EdgeKind::Requires(attrs),
            confidence,
        }
    }
}

pub struct SupportsSpec;
impl EdgeSpec for SupportsSpec {
    type Attrs = SupportsAttrs;
    const NAME: &'static str = "supports";

    fn validate(
        svc: &GraphService,
        from: NodeId,
        to: NodeId,
        attrs: &Self::Attrs,
        _confidence: f32,
    ) -> Result<(), GraphError> {
        let (from_kind, to_kind) = svc.node_kinds(Self::NAME, from, to)?;
        if from == to {
            return Err(GraphError::Schema("supports self-loops are not allowed".to_string()));
        }
        let (from_kt, to_kt) = match (from_kind, to_kind) {
            (NodeKind::Knowledge(f), NodeKind::Knowledge(t)) => {
                (f.knowledge_type, t.knowledge_type)
            }
            (NodeKind::Knowledge(f), NodeKind::TeachingStep(_)) => {
                return Err(GraphError::InvalidEndpoints {
                    edge: Self::NAME,
                    from: Some(NodeKindPreview::from_node(&NodeKind::Knowledge(f.clone()))),
                    to:   Some(NodeKindPreview::TeachingStep),
                });
            }
            (NodeKind::TeachingStep(_), _) => {
                return Err(GraphError::InvalidEndpoints {
                    edge: Self::NAME,
                    from: Some(NodeKindPreview::TeachingStep),
                    to:   Some(NodeKindPreview::from_node(to_kind)),
                });
            }
        };

        crate::schema::validate::validate_supports(
            from_kt,
            to_kt,
            attrs.support_kind,
            attrs.intended_effect,
            &attrs.evidence_refs,
        )
        .map_err(|e| GraphError::Schema(e.to_string()))
    }

    fn make_payload(attrs: Self::Attrs, confidence: f32) -> EdgePayload {
        EdgePayload {
            kind: EdgeKind::Supports(attrs),
            confidence,
        }
    }
}

pub struct AssessesSpec;
impl EdgeSpec for AssessesSpec {
    type Attrs = AssessesAttrs;
    const NAME: &'static str = "assesses";

    fn validate(
        svc: &GraphService,
        from: NodeId,
        to: NodeId,
        attrs: &Self::Attrs,
        _confidence: f32,
    ) -> Result<(), GraphError> {
        let (from_kind, to_kind) = svc.node_kinds(Self::NAME, from, to)?;
        let (from_kt, to_kt) = match (from_kind, to_kind) {
            (NodeKind::Knowledge(f), NodeKind::Knowledge(t)) => {
                (f.knowledge_type, t.knowledge_type)
            }
            _ => {
                return Err(GraphError::InvalidEndpoints {
                    edge: Self::NAME,
                    from: Some(NodeKindPreview::from_node(from_kind)),
                    to:   Some(NodeKindPreview::from_node(to_kind)),
                });
            }
        };

        crate::schema::validate::validate_assesses(from_kt, to_kt, &attrs.evidence_link)
            .map_err(|e| GraphError::Schema(e.to_string()))?;

        // evidence_link.claim must match target LO slug
        let target_slug = &svc.graph[to].slug;
        if &attrs.evidence_link.claim != target_slug {
            return Err(GraphError::Schema(format!(
                "assesses.claim `{}` must equal target LO slug `{}`",
                attrs.evidence_link.claim, target_slug
            )));
        }
        Ok(())
    }

    fn make_payload(attrs: Self::Attrs, confidence: f32) -> EdgePayload {
        EdgePayload {
            kind: EdgeKind::Assesses(attrs),
            confidence,
        }
    }
}

pub struct PrecedesSpec;
impl EdgeSpec for PrecedesSpec {
    type Attrs = PrecedesAttrs;
    const NAME: &'static str = "precedes";

    fn validate(
        svc: &GraphService,
        from: NodeId,
        to: NodeId,
        attrs: &Self::Attrs,
        _confidence: f32,
    ) -> Result<(), GraphError> {
        let (from_kind, to_kind) = svc.node_kinds(Self::NAME, from, to)?;
        match (from_kind, to_kind) {
            (NodeKind::TeachingStep(_), NodeKind::TeachingStep(_)) => {}
            _ => {
                return Err(GraphError::InvalidEndpoints {
                    edge: Self::NAME,
                    from: Some(NodeKindPreview::from_node(from_kind)),
                    to:   Some(NodeKindPreview::from_node(to_kind)),
                });
            }
        }

        if let (NodeKind::TeachingStep(ts_from), NodeKind::TeachingStep(ts_to)) =
            (from_kind, to_kind)
            && (ts_from.episode != attrs.episode || ts_to.episode != attrs.episode)
        {
            return Err(GraphError::Schema(format!(
                "precedes episode `{}` must match both steps (`{}`, `{}`)",
                attrs.episode, ts_from.episode, ts_to.episode
            )));
        }

        if svc.has_precedes_path(to, from, &attrs.episode) {
            return Err(GraphError::Schema(
                "precedes edge would create a cycle in this episode".to_string(),
            ));
        }
        Ok(())
    }

    fn make_payload(attrs: Self::Attrs, confidence: f32) -> EdgePayload {
        EdgePayload {
            kind: EdgeKind::Precedes(attrs),
            confidence,
        }
    }
}

pub struct AnchorsSpec;
impl EdgeSpec for AnchorsSpec {
    type Attrs = AnchorsAttrs;
    const NAME: &'static str = "anchors";

    /// Discourse anchors semantics:
    /// - Source must be TeachingStep.
    /// - impact = introduce/refine: target must be instructional knowledge (not
    ///   LO or assessment).
    /// - impact = target: target must be a LearningOutcome.
    /// - impact = use/motivate: general knowledge allowed; if target is an
    ///   assessment item, only `use` is permitted.
    fn validate(
        svc: &GraphService,
        from: NodeId,
        to: NodeId,
        attrs: &Self::Attrs,
        _confidence: f32,
    ) -> Result<(), GraphError> {
        let (from_kind, to_kind) = svc.node_kinds(Self::NAME, from, to)?;
        match from_kind {
            NodeKind::TeachingStep(_) => {}
            _ => {
                return Err(GraphError::InvalidEndpoints {
                    edge: Self::NAME,
                    from: Some(NodeKindPreview::from_node(from_kind)),
                    to:   Some(NodeKindPreview::from_node(to_kind)),
                });
            }
        }

        match (&to_kind, attrs.impact) {
            (NodeKind::Knowledge(k), AnchorImpact::Introduce | AnchorImpact::Refine) => {
                if k.knowledge_type.is_learning_outcome() || k.knowledge_type.is_assessment_item() {
                    return Err(GraphError::Schema(
                        "introduce/refine anchors must target instructional knowledge".to_string(),
                    ));
                }
            }
            (NodeKind::Knowledge(k), AnchorImpact::Target) => {
                if k.knowledge_type != KnowledgeType::LearningOutcome {
                    return Err(GraphError::Schema(
                        "target anchors must point to learning_outcome nodes".to_string(),
                    ));
                }
            }
            (NodeKind::Knowledge(k), AnchorImpact::Use | AnchorImpact::Motivate) => {
                if k.knowledge_type.is_assessment_item() && attrs.impact != AnchorImpact::Use {
                    return Err(GraphError::Schema(
                        "anchors to assessment items must use impact=use".to_string(),
                    ));
                }
            }
            (NodeKind::TeachingStep(_), _) => {
                return Err(GraphError::InvalidEndpoints {
                    edge: Self::NAME,
                    from: Some(NodeKindPreview::TeachingStep),
                    to:   Some(NodeKindPreview::TeachingStep),
                });
            }
        }
        Ok(())
    }

    fn make_payload(attrs: Self::Attrs, confidence: f32) -> EdgePayload {
        EdgePayload {
            kind: EdgeKind::Anchors(attrs),
            confidence,
        }
    }
}
