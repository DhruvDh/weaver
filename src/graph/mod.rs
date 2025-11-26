use std::collections::HashMap;

use petgraph::{
    Directed,
    stable_graph::{EdgeIndex, NodeIndex, StableGraph},
    visit::EdgeRef,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::schema::types::{
    EvidenceLink, IntendedEffect, KnowledgeType, SourceRef, Strength, SupportKind,
};

pub mod persist;

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
    #[error("edge would create a cycle in requires layer")]
    RequiresCycle,
    #[error("validator error: {0}")]
    Schema(String),
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
    graph:        CurriculumGraph,
    slug_to_node: HashMap<String, NodeId>,
}

impl GraphService {
    pub fn new() -> Self {
        Self {
            graph:        StableGraph::default(),
            slug_to_node: HashMap::new(),
        }
    }

    pub fn from_graph(graph: CurriculumGraph) -> Self {
        let mut svc = Self {
            graph,
            slug_to_node: HashMap::new(),
        };
        svc.rebuild_slug_index();
        svc
    }

    pub fn graph(&self) -> &CurriculumGraph {
        &self.graph
    }

    pub fn graph_mut(&mut self) -> &mut CurriculumGraph {
        &mut self.graph
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
        if let Some(existing) = self.slug_to_node.get(&slug).copied() {
            match &mut self.graph[existing].kind {
                NodeKind::Knowledge(_) => {
                    // update in place, preserve logical_id
                    self.graph[existing].kind = NodeKind::Knowledge(payload);
                    self.graph[existing].tags = tags;
                    Ok(existing)
                }
                NodeKind::TeachingStep(_) => Err(GraphError::Schema(format!(
                    "slug `{}` already exists as teaching_step; cannot upsert knowledge node",
                    slug
                ))),
            }
        } else {
            let logical_id = Uuid::new_v4();
            let node = NodePayload {
                logical_id,
                slug: slug.clone(),
                kind: NodeKind::Knowledge(payload),
                tags,
            };
            let id = self.graph.add_node(node);
            self.upsert_slug(slug, id);
            Ok(id)
        }
    }

    pub fn snapshot_graph(&self) -> CurriculumGraph {
        self.graph.clone()
    }

    pub fn replace_graph(&mut self, graph: CurriculumGraph) {
        self.graph = graph;
        self.rebuild_slug_index();
    }

    fn rebuild_slug_index(&mut self) {
        self.slug_to_node.clear();
        for n in self.graph.node_indices() {
            let slug = self.graph[n].slug.clone();
            self.slug_to_node.insert(slug, n);
        }
    }

    pub fn add_teaching_step(
        &mut self,
        slug: String,
        payload: TeachingStepNode,
        tags: Vec<String>,
    ) -> Result<NodeId, GraphError> {
        if let Some(existing) = self.slug_to_node.get(&slug).copied() {
            match &mut self.graph[existing].kind {
                NodeKind::TeachingStep(_) => {
                    self.graph[existing].kind = NodeKind::TeachingStep(payload);
                    self.graph[existing].tags = tags;
                    Ok(existing)
                }
                NodeKind::Knowledge(_) => Err(GraphError::Schema(format!(
                    "slug `{}` already exists as knowledge; cannot upsert teaching_step",
                    slug
                ))),
            }
        } else {
            let logical_id = Uuid::new_v4();
            let node = NodePayload {
                logical_id,
                slug: slug.clone(),
                kind: NodeKind::TeachingStep(payload),
                tags,
            };
            let id = self.graph.add_node(node);
            self.upsert_slug(slug, id);
            Ok(id)
        }
    }

    /// Add a requires edge after schema validation and cycle guard.
    pub fn add_requires_edge(
        &mut self,
        from: NodeId,
        to: NodeId,
        attrs: RequiresAttrs,
        confidence: f32,
    ) -> Result<EdgeId, GraphError> {
        let (from_kind, to_kind) = self.node_kinds(from, to)?;
        let (from_kt, to_kt) = match (from_kind, to_kind) {
            (NodeKind::Knowledge(f), NodeKind::Knowledge(t)) => {
                (f.knowledge_type, t.knowledge_type)
            }
            _ => {
                return Err(GraphError::InvalidEndpoints {
                    edge: "requires",
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

        if self.has_requires_path(to, from) {
            return Err(GraphError::RequiresCycle);
        }

        let id = self.graph.add_edge(
            from,
            to,
            EdgePayload {
                kind: EdgeKind::Requires(attrs),
                confidence,
            },
        );
        Ok(id)
    }

    /// Add a supports edge after schema validation.
    pub fn add_supports_edge(
        &mut self,
        from: NodeId,
        to: NodeId,
        attrs: SupportsAttrs,
        confidence: f32,
    ) -> Result<EdgeId, GraphError> {
        let (from_kind, to_kind) = self.node_kinds(from, to)?;
        if from == to {
            return Err(GraphError::Schema("supports self-loops are not allowed".to_string()));
        }
        let (from_kt, to_kt) = match (from_kind, to_kind) {
            (NodeKind::Knowledge(f), NodeKind::Knowledge(t)) => {
                (f.knowledge_type, t.knowledge_type)
            }
            (NodeKind::Knowledge(f), NodeKind::TeachingStep(_)) => {
                return Err(GraphError::InvalidEndpoints {
                    edge: "supports",
                    from: Some(NodeKindPreview::from_node(&NodeKind::Knowledge(f.clone()))),
                    to:   Some(NodeKindPreview::TeachingStep),
                });
            }
            (NodeKind::TeachingStep(_), _) => {
                return Err(GraphError::InvalidEndpoints {
                    edge: "supports",
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
        .map_err(|e| GraphError::Schema(e.to_string()))?;

        let id = self.graph.add_edge(
            from,
            to,
            EdgePayload {
                kind: EdgeKind::Supports(attrs),
                confidence,
            },
        );
        Ok(id)
    }

    /// Add an assesses edge after schema validation.
    pub fn add_assesses_edge(
        &mut self,
        from: NodeId,
        to: NodeId,
        attrs: AssessesAttrs,
        confidence: f32,
    ) -> Result<EdgeId, GraphError> {
        let (from_kind, to_kind) = self.node_kinds(from, to)?;
        let (from_kt, to_kt) = match (from_kind, to_kind) {
            (NodeKind::Knowledge(f), NodeKind::Knowledge(t)) => {
                (f.knowledge_type, t.knowledge_type)
            }
            _ => {
                return Err(GraphError::InvalidEndpoints {
                    edge: "assesses",
                    from: Some(NodeKindPreview::from_node(from_kind)),
                    to:   Some(NodeKindPreview::from_node(to_kind)),
                });
            }
        };

        crate::schema::validate::validate_assesses(from_kt, to_kt, &attrs.evidence_link)
            .map_err(|e| GraphError::Schema(e.to_string()))?;

        // evidence_link.claim must match target LO slug
        let target_slug = &self.graph[to].slug;
        if &attrs.evidence_link.claim != target_slug {
            return Err(GraphError::Schema(format!(
                "assesses.claim `{}` must equal target LO slug `{}`",
                attrs.evidence_link.claim, target_slug
            )));
        }

        let id = self.graph.add_edge(
            from,
            to,
            EdgePayload {
                kind: EdgeKind::Assesses(attrs),
                confidence,
            },
        );
        Ok(id)
    }

    /// Add a precedes edge (discourse). Enforces that both ends are
    /// TeachingSteps.
    pub fn add_precedes_edge(
        &mut self,
        from: NodeId,
        to: NodeId,
        episode: String,
        confidence: f32,
    ) -> Result<EdgeId, GraphError> {
        let (from_kind, to_kind) = self.node_kinds(from, to)?;
        match (from_kind, to_kind) {
            (NodeKind::TeachingStep(_), NodeKind::TeachingStep(_)) => {}
            _ => {
                return Err(GraphError::InvalidEndpoints {
                    edge: "precedes",
                    from: Some(NodeKindPreview::from_node(from_kind)),
                    to:   Some(NodeKindPreview::from_node(to_kind)),
                });
            }
        }

        if let (NodeKind::TeachingStep(ts_from), NodeKind::TeachingStep(ts_to)) =
            (from_kind, to_kind)
            && (ts_from.episode != episode || ts_to.episode != episode)
        {
            return Err(GraphError::Schema(format!(
                "precedes episode `{}` must match both steps (`{}`, `{}`)",
                episode, ts_from.episode, ts_to.episode
            )));
        }

        if self.has_precedes_path(to, from, &episode) {
            return Err(GraphError::Schema(
                "precedes edge would create a cycle in this episode".to_string(),
            ));
        }

        let id = self.graph.add_edge(
            from,
            to,
            EdgePayload {
                kind: EdgeKind::Precedes(PrecedesAttrs { episode }),
                confidence,
            },
        );
        Ok(id)
    }

    /// Add an anchors edge (discourse).
    pub fn add_anchors_edge(
        &mut self,
        from: NodeId,
        to: NodeId,
        impact: AnchorImpact,
        confidence: f32,
    ) -> Result<EdgeId, GraphError> {
        let (from_kind, to_kind) = self.node_kinds(from, to)?;
        match from_kind {
            NodeKind::TeachingStep(_) => {}
            _ => {
                return Err(GraphError::InvalidEndpoints {
                    edge: "anchors",
                    from: Some(NodeKindPreview::from_node(from_kind)),
                    to:   Some(NodeKindPreview::from_node(to_kind)),
                });
            }
        }

        match (&to_kind, impact) {
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
                if k.knowledge_type.is_assessment_item() && impact != AnchorImpact::Use {
                    return Err(GraphError::Schema(
                        "anchors to assessment items must use impact=use".to_string(),
                    ));
                }
            }
            (NodeKind::TeachingStep(_), _) => {
                return Err(GraphError::InvalidEndpoints {
                    edge: "anchors",
                    from: Some(NodeKindPreview::TeachingStep),
                    to:   Some(NodeKindPreview::TeachingStep),
                });
            }
        }

        let id = self.graph.add_edge(
            from,
            to,
            EdgePayload {
                kind: EdgeKind::Anchors(AnchorsAttrs { impact }),
                confidence,
            },
        );
        Ok(id)
    }

    pub fn node(&self, id: NodeId) -> &NodePayload {
        &self.graph[id]
    }

    fn node_kinds(&self, from: NodeId, to: NodeId) -> Result<(&NodeKind, &NodeKind), GraphError> {
        let from_kind = &self
            .graph
            .node_weight(from)
            .ok_or(GraphError::InvalidEndpoints {
                edge: "generic",
                from: None,
                to:   None,
            })?
            .kind;
        let to_kind = &self
            .graph
            .node_weight(to)
            .ok_or(GraphError::InvalidEndpoints {
                edge: "generic",
                from: None,
                to:   None,
            })?
            .kind;
        Ok((from_kind, to_kind))
    }

    /// BFS along requires edges only.
    pub fn has_requires_path(&self, start: NodeId, goal: NodeId) -> bool {
        use petgraph::Direction;
        let mut stack = vec![start];
        let mut seen = std::collections::HashSet::new();
        while let Some(node) = stack.pop() {
            if node == goal {
                return true;
            }
            if !seen.insert(node) {
                continue;
            }
            for edge in self
                .graph
                .edges_directed(node, Direction::Outgoing)
                .filter(|e| matches!(e.weight().kind, EdgeKind::Requires(_)))
            {
                stack.push(edge.target());
            }
        }
        false
    }

    fn has_precedes_path(&self, start: NodeId, goal: NodeId, episode: &str) -> bool {
        use petgraph::Direction;
        let mut stack = vec![start];
        let mut seen = std::collections::HashSet::new();
        while let Some(node) = stack.pop() {
            if node == goal {
                return true;
            }
            if !seen.insert(node) {
                continue;
            }
            for edge in self.graph.edges_directed(node, Direction::Outgoing).filter(
                |e| matches!(e.weight().kind, EdgeKind::Precedes(ref p) if p.episode == episode),
            ) {
                stack.push(edge.target());
            }
        }
        false
    }
}

impl Default for GraphService {
    fn default() -> Self {
        Self::new()
    }
}
