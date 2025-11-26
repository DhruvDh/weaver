use std::{convert::Infallible, path::PathBuf, sync::Arc};

use anyhow::Result;
use kameo::prelude::*;
use petgraph::{Direction, visit::EdgeRef};
use schemars::JsonSchema;

use crate::graph::{
    AnchorImpact, AnchorsAttrs, AssessesAttrs, CurriculumGraph, EdgeKind, GraphConfig, GraphError,
    GraphService, KnowledgeNode, NodeId, NodePayload, PrecedesAttrs, RequiresAttrs, SupportsAttrs,
    TeachingStepNode, persist,
};

#[derive(Clone, Debug, serde::Serialize)]
pub struct Neighbor {
    pub neighbor_slug: String,
    pub edge_kind:     String,
    pub direction:     String,
}

#[derive(Clone, Copy, Debug, serde::Deserialize, serde::Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKindFilter {
    Requires,
    Supports,
    Assesses,
    Precedes,
    Anchors,
}

#[derive(Clone, Debug, serde::Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NeighborDirection {
    Incoming,
    Outgoing,
    Both,
}

#[derive(Actor)]
pub struct GraphManager {
    service: GraphService,
    config:  GraphConfig,
}

impl GraphManager {
    pub fn new(service: GraphService, config: GraphConfig) -> Self {
        Self { service, config }
    }
}

fn edge_kind_name(k: &EdgeKind) -> &'static str {
    match k {
        EdgeKind::Requires(_) => "requires",
        EdgeKind::Supports(_) => "supports",
        EdgeKind::Assesses(_) => "assesses",
        EdgeKind::Precedes(_) => "precedes",
        EdgeKind::Anchors(_) => "anchors",
    }
}

fn edge_kind_matches(kind: EdgeKindFilter, edge: &EdgeKind) -> bool {
    match kind {
        EdgeKindFilter::Requires => matches!(edge, EdgeKind::Requires(_)),
        EdgeKindFilter::Supports => matches!(edge, EdgeKind::Supports(_)),
        EdgeKindFilter::Assesses => matches!(edge, EdgeKind::Assesses(_)),
        EdgeKindFilter::Precedes => matches!(edge, EdgeKind::Precedes(_)),
        EdgeKindFilter::Anchors => matches!(edge, EdgeKind::Anchors(_)),
    }
}

// ---- Messages ----

pub struct InsertKnowledge {
    pub slug:    String,
    pub payload: KnowledgeNode,
    pub tags:    Vec<String>,
}

pub struct UpdateKnowledge {
    pub slug:    String,
    pub payload: KnowledgeNode,
    pub tags:    Vec<String>,
}

impl Message<InsertKnowledge> for GraphManager {
    type Reply = Result<NodeId, GraphError>;

    async fn handle(
        &mut self,
        InsertKnowledge {
            slug,
            payload,
            tags,
        }: InsertKnowledge,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        self.service.add_knowledge_node(slug, payload, tags)
    }
}

impl Message<UpdateKnowledge> for GraphManager {
    type Reply = Result<NodeId, GraphError>;

    async fn handle(
        &mut self,
        UpdateKnowledge {
            slug,
            payload,
            tags,
        }: UpdateKnowledge,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        self.service.update_knowledge_node(&slug, payload, tags)
    }
}

pub struct InsertTeachingStep {
    pub slug:    String,
    pub payload: TeachingStepNode,
    pub tags:    Vec<String>,
}

pub struct UpdateTeachingStep {
    pub slug:    String,
    pub payload: TeachingStepNode,
    pub tags:    Vec<String>,
}

impl Message<InsertTeachingStep> for GraphManager {
    type Reply = Result<NodeId, GraphError>;

    async fn handle(
        &mut self,
        InsertTeachingStep {
            slug,
            payload,
            tags,
        }: InsertTeachingStep,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        self.service.add_teaching_step(slug, payload, tags)
    }
}

impl Message<UpdateTeachingStep> for GraphManager {
    type Reply = Result<NodeId, GraphError>;

    async fn handle(
        &mut self,
        UpdateTeachingStep {
            slug,
            payload,
            tags,
        }: UpdateTeachingStep,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        self.service.update_teaching_step(&slug, payload, tags)
    }
}

pub struct AddRequires {
    pub from:       String,
    pub to:         String,
    pub attrs:      RequiresAttrs,
    pub confidence: f32,
}

impl Message<AddRequires> for GraphManager {
    type Reply = Result<(), GraphError>;

    async fn handle(
        &mut self,
        AddRequires {
            from,
            to,
            attrs,
            confidence,
        }: AddRequires,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let from_id = self.service.node_by_slug(&from)?;
        let to_id = self.service.node_by_slug(&to)?;
        self.service
            .add_edge::<crate::graph::RequiresSpec>(from_id, to_id, attrs, confidence)?;
        Ok(())
    }
}

pub struct AddSupports {
    pub from:       String,
    pub to:         String,
    pub attrs:      SupportsAttrs,
    pub confidence: f32,
}

impl Message<AddSupports> for GraphManager {
    type Reply = Result<(), GraphError>;

    async fn handle(
        &mut self,
        AddSupports {
            from,
            to,
            attrs,
            confidence,
        }: AddSupports,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let from_id = self.service.node_by_slug(&from)?;
        let to_id = self.service.node_by_slug(&to)?;
        self.service
            .add_edge::<crate::graph::SupportsSpec>(from_id, to_id, attrs, confidence)?;
        Ok(())
    }
}

pub struct AddAssesses {
    pub from:       String,
    pub to:         String,
    pub attrs:      AssessesAttrs,
    pub confidence: f32,
}

impl Message<AddAssesses> for GraphManager {
    type Reply = Result<(), GraphError>;

    async fn handle(
        &mut self,
        AddAssesses {
            from,
            to,
            attrs,
            confidence,
        }: AddAssesses,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let from_id = self.service.node_by_slug(&from)?;
        let to_id = self.service.node_by_slug(&to)?;
        self.service
            .add_edge::<crate::graph::AssessesSpec>(from_id, to_id, attrs, confidence)?;
        Ok(())
    }
}

pub struct AddPrecedes {
    pub from:       String,
    pub to:         String,
    pub episode:    String,
    pub confidence: f32,
}

impl Message<AddPrecedes> for GraphManager {
    type Reply = Result<(), GraphError>;

    async fn handle(
        &mut self,
        AddPrecedes {
            from,
            to,
            episode,
            confidence,
        }: AddPrecedes,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let from_id = self.service.node_by_slug(&from)?;
        let to_id = self.service.node_by_slug(&to)?;
        self.service.add_edge::<crate::graph::PrecedesSpec>(
            from_id,
            to_id,
            PrecedesAttrs { episode },
            confidence,
        )?;
        Ok(())
    }
}

pub struct AddAnchors {
    pub from:       String,
    pub to:         String,
    pub impact:     AnchorImpact,
    pub confidence: f32,
}

impl Message<AddAnchors> for GraphManager {
    type Reply = Result<(), GraphError>;

    async fn handle(
        &mut self,
        AddAnchors {
            from,
            to,
            impact,
            confidence,
        }: AddAnchors,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let from_id = self.service.node_by_slug(&from)?;
        let to_id = self.service.node_by_slug(&to)?;
        self.service.add_edge::<crate::graph::AnchorsSpec>(
            from_id,
            to_id,
            AnchorsAttrs { impact },
            confidence,
        )?;
        Ok(())
    }
}

pub struct GetNode {
    pub slug: String,
}

impl Message<GetNode> for GraphManager {
    type Reply = Result<NodePayload, GraphError>;

    async fn handle(
        &mut self,
        GetNode { slug }: GetNode,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let id = self.service.node_by_slug(&slug)?;
        Ok(self.service.graph()[id].clone())
    }
}

pub struct GetGraph;

impl Message<GetGraph> for GraphManager {
    type Reply = Result<Arc<CurriculumGraph>, Infallible>;

    async fn handle(
        &mut self,
        _msg: GetGraph,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        Ok(self.service.shared_graph())
    }
}

impl Message<Neighbors> for GraphManager {
    type Reply = Result<Vec<Neighbor>, GraphError>;

    async fn handle(
        &mut self,
        Neighbors {
            slug,
            edge_kind,
            direction,
        }: Neighbors,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let node = self.service.node_by_slug(&slug)?;

        let kind_filter = edge_kind;

        let dirs: Vec<Direction> = match direction.unwrap_or(NeighborDirection::Both) {
            NeighborDirection::Incoming => vec![Direction::Incoming],
            NeighborDirection::Outgoing => vec![Direction::Outgoing],
            NeighborDirection::Both => vec![Direction::Incoming, Direction::Outgoing],
        };

        let mut out = Vec::new();
        let g = self.service.graph();
        for dir in dirs {
            for edge in g.edges_directed(node, dir) {
                if let Some(k) = kind_filter
                    && !edge_kind_matches(k, &edge.weight().kind)
                {
                    continue;
                }
                let other = if dir == Direction::Outgoing {
                    edge.target()
                } else {
                    edge.source()
                };
                out.push(Neighbor {
                    neighbor_slug: g[other].slug.clone(),
                    edge_kind:     edge_kind_name(&edge.weight().kind).to_string(),
                    direction:     if dir == Direction::Outgoing {
                        "outgoing".to_string()
                    } else {
                        "incoming".to_string()
                    },
                });
            }
        }

        Ok(out)
    }
}

impl Message<ResolveSlug> for GraphManager {
    type Reply = Result<NodeId, GraphError>;

    async fn handle(
        &mut self,
        ResolveSlug { slug }: ResolveSlug,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        self.service.node_by_slug(&slug)
    }
}

impl Message<RenameNode> for GraphManager {
    type Reply = Result<(), GraphError>;

    async fn handle(
        &mut self,
        RenameNode { old_slug, new_slug }: RenameNode,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        self.service.rename_node(&old_slug, new_slug)
    }
}

impl Message<RemoveNode> for GraphManager {
    type Reply = Result<(), GraphError>;

    async fn handle(
        &mut self,
        RemoveNode { slug }: RemoveNode,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        self.service.remove_node(&slug)
    }
}

pub struct SaveSnapshot {
    pub path: PathBuf,
}

pub struct Neighbors {
    pub slug:      String,
    pub edge_kind: Option<EdgeKindFilter>, // requires, supports, assesses, precedes, anchors
    pub direction: Option<NeighborDirection>, // incoming, outgoing, both
}

pub struct RenameNode {
    pub old_slug: String,
    pub new_slug: String,
}

pub struct RemoveNode {
    pub slug: String,
}

pub struct ResolveSlug {
    pub slug: String,
}

impl Message<SaveSnapshot> for GraphManager {
    type Reply = Result<()>;

    async fn handle(
        &mut self,
        SaveSnapshot { path }: SaveSnapshot,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let graph = self.service.shared_graph();
        persist::save_graph(graph.as_ref(), path, &self.config.course_commit).await
    }
}

pub struct LoadSnapshot {
    pub path: PathBuf,
}

impl Message<LoadSnapshot> for GraphManager {
    type Reply = Result<()>;

    async fn handle(
        &mut self,
        LoadSnapshot { path }: LoadSnapshot,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let snapshot = persist::load_graph(&path).await?;
        self.service.replace_graph(snapshot.graph)?;
        self.config.course_commit = snapshot.course_commit;
        Ok(())
    }
}
