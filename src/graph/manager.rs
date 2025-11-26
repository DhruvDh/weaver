use std::{convert::Infallible, path::PathBuf, sync::Arc};

use anyhow::Result;
use kameo::prelude::*;
use petgraph::{Direction, visit::EdgeRef};

use crate::graph::{
    AnchorImpact, AssessesAttrs, CurriculumGraph, EdgeKind, GraphError, GraphService,
    KnowledgeNode, NodeId, NodePayload, RequiresAttrs, SupportsAttrs, TeachingStepNode, persist,
};

#[derive(Clone, Debug, serde::Serialize)]
pub struct Neighbor {
    pub neighbor_slug: String,
    pub edge_kind:     String,
    pub direction:     String,
}

#[derive(Actor)]
pub struct GraphManager {
    service:       GraphService,
    course_commit: String,
}

impl GraphManager {
    pub fn new(service: GraphService, course_commit: String) -> Self {
        Self {
            service,
            course_commit,
        }
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

fn edge_kind_matches(kind: &str, edge: &EdgeKind) -> bool {
    match kind {
        "requires" => matches!(edge, EdgeKind::Requires(_)),
        "supports" => matches!(edge, EdgeKind::Supports(_)),
        "assesses" => matches!(edge, EdgeKind::Assesses(_)),
        "precedes" => matches!(edge, EdgeKind::Precedes(_)),
        "anchors" => matches!(edge, EdgeKind::Anchors(_)),
        _ => true,
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
        if self.service.node_by_slug(&slug).is_ok() {
            self.service.update_knowledge_node(&slug, payload, tags)
        } else {
            self.service.add_knowledge_node(slug, payload, tags)
        }
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
        if self.service.node_by_slug(&slug).is_ok() {
            self.service.update_teaching_step(&slug, payload, tags)
        } else {
            self.service.add_teaching_step(slug, payload, tags)
        }
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
            .add_requires_edge(from_id, to_id, attrs, confidence)?;
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
            .add_supports_edge(from_id, to_id, attrs, confidence)?;
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
            .add_assesses_edge(from_id, to_id, attrs, confidence)?;
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
        self.service
            .add_precedes_edge(from_id, to_id, episode, confidence)?;
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
        self.service
            .add_anchors_edge(from_id, to_id, impact, confidence)?;
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

        let kind_filter = edge_kind.map(|k| k.to_ascii_lowercase());
        if let Some(ref k) = kind_filter
            && !matches!(k.as_str(), "requires" | "supports" | "assesses" | "precedes" | "anchors")
        {
            return Err(GraphError::Schema(format!("unknown edge_kind `{}`", k)));
        }

        let dirs: Vec<Direction> = match direction.as_deref() {
            Some("incoming") => vec![Direction::Incoming],
            Some("outgoing") => vec![Direction::Outgoing],
            Some("both") | None => vec![Direction::Incoming, Direction::Outgoing],
            Some(other) => {
                return Err(GraphError::Schema(format!("unknown direction `{}`", other)));
            }
        };

        let mut out = Vec::new();
        let g = self.service.graph();
        for dir in dirs {
            for edge in g.edges_directed(node, dir) {
                if let Some(ref k) = kind_filter
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
    pub edge_kind: Option<String>, // requires, supports, assesses, precedes, anchors
    pub direction: Option<String>, // incoming, outgoing, both
}

pub struct RenameNode {
    pub old_slug: String,
    pub new_slug: String,
}

pub struct RemoveNode {
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
        persist::save_graph(graph.as_ref(), path, &self.course_commit).await
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
        let graph = persist::load_graph(&path).await?;
        self.service.replace_graph(graph);
        Ok(())
    }
}
