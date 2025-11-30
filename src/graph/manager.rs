use std::{
    convert::Infallible,
    path::PathBuf,
    sync::{Arc, LazyLock, RwLock},
    time::Instant,
};

use anyhow::Result;
use kameo::prelude::*;
use kameo_persistence::{BiHashMap, PersistentActor};
use petgraph::{Direction, visit::EdgeRef};
use schemars::JsonSchema;
use tracing::warn;
use url::Url;

use crate::graph::{
    AnchorImpact, AnchorsAttrs, AssessesAttrs, CurriculumGraph, EdgeKind, GraphConfig, GraphError,
    GraphService, KnowledgeNode, NodeId, NodePayload, PrecedesAttrs, RequiresAttrs, SupportsAttrs,
    TeachingStepNode, persist,
};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GraphManagerState {
    pub graph:          CurriculumGraph,
    pub course_commit:  String,
    pub strict_quality: bool,
    #[serde(default)]
    pub graph_version:  u64,
}

impl GraphManagerState {
    pub fn new(
        graph: CurriculumGraph,
        course_commit: String,
        strict_quality: bool,
        graph_version: u64,
    ) -> Self {
        Self {
            graph,
            course_commit,
            strict_quality,
            graph_version,
        }
    }
}

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

pub struct GraphManager {
    service:       GraphService,
    course_commit: String,
}

impl GraphManager {
    pub fn new(service: GraphService, config: GraphConfig) -> Self {
        Self {
            service,
            course_commit: config.course_commit,
        }
    }

    fn log_write_latency(op: &str, start: Instant) {
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        tracing::debug!(target: "weaver.graph.write_latency", op, elapsed_ms);
    }
}

impl From<&GraphManager> for GraphManagerState {
    fn from(manager: &GraphManager) -> Self {
        GraphManagerState {
            graph:          manager.service.snapshot_graph(),
            course_commit:  manager.course_commit.clone(),
            strict_quality: manager.service.strict_quality(),
            graph_version:  manager.service.graph_version(),
        }
    }
}

impl Actor for GraphManager {
    type Args = GraphManagerState;
    type Error = Infallible;

    async fn on_start(state: Self::Args, _actor_ref: ActorRef<Self>) -> Result<Self, Self::Error> {
        let strict = state.strict_quality;
        let service = match GraphService::from_parts(state.graph, strict, state.graph_version) {
            Ok(svc) => svc,
            Err(err) => {
                warn!(error = %err, "invalid persisted graph; starting with empty graph");
                GraphService::new().with_strict(strict)
            }
        };

        Ok(Self {
            service,
            course_commit: state.course_commit,
        })
    }
}

static GRAPH_MANAGER_REGISTRY: LazyLock<RwLock<BiHashMap<Url, WeakActorRef<GraphManager>>>> =
    LazyLock::new(|| RwLock::new(BiHashMap::new()));

impl PersistentActor for GraphManager {
    type Snapshot = GraphManagerState;

    fn register_persistent(persistence_key: Url, actor_ref: &ActorRef<Self>) -> anyhow::Result<()> {
        let mut registry = GRAPH_MANAGER_REGISTRY
            .write()
            .map_err(|_| anyhow::anyhow!("graph manager registry poisoned"))?;
        let _ = registry.insert(persistence_key, actor_ref.downgrade());
        Ok(())
    }

    fn persistence_key(actor_ref: &ActorRef<Self>) -> Option<Url> {
        let registry = GRAPH_MANAGER_REGISTRY.read().ok()?;
        registry.get_left(&actor_ref.downgrade()).cloned()
    }

    fn lookup_persistent(persistence_key: &Url) -> Option<ActorRef<Self>> {
        let registry = GRAPH_MANAGER_REGISTRY.read().ok()?;
        registry
            .get_right(persistence_key)
            .and_then(|weak| weak.upgrade())
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
        let start = Instant::now();
        let res = self.service.add_knowledge_node(slug, payload, tags);
        if res.is_ok() {
            GraphManager::log_write_latency("insert_knowledge", start);
        }
        res
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
        let start = Instant::now();
        let res = self.service.update_knowledge_node(&slug, payload, tags);
        if res.is_ok() {
            GraphManager::log_write_latency("update_knowledge", start);
        }
        res
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
        let start = Instant::now();
        let res = self.service.add_teaching_step(slug, payload, tags);
        if res.is_ok() {
            GraphManager::log_write_latency("insert_teaching_step", start);
        }
        res
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
        let start = Instant::now();
        let res = self.service.update_teaching_step(&slug, payload, tags);
        if res.is_ok() {
            GraphManager::log_write_latency("update_teaching_step", start);
        }
        res
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
        let start = Instant::now();
        let from_id = self.service.node_by_slug(&from)?;
        let to_id = self.service.node_by_slug(&to)?;
        let res = self
            .service
            .add_edge::<crate::graph::RequiresSpec>(from_id, to_id, attrs, confidence);
        if res.is_ok() {
            GraphManager::log_write_latency("add_requires", start);
        }
        res.map(|_| ())
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
        let start = Instant::now();
        let from_id = self.service.node_by_slug(&from)?;
        let to_id = self.service.node_by_slug(&to)?;
        let res = self
            .service
            .add_edge::<crate::graph::SupportsSpec>(from_id, to_id, attrs, confidence);
        if res.is_ok() {
            GraphManager::log_write_latency("add_supports", start);
        }
        res.map(|_| ())
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
        let start = Instant::now();
        let from_id = self.service.node_by_slug(&from)?;
        let to_id = self.service.node_by_slug(&to)?;
        let res = self
            .service
            .add_edge::<crate::graph::AssessesSpec>(from_id, to_id, attrs, confidence);
        if res.is_ok() {
            GraphManager::log_write_latency("add_assesses", start);
        }
        res.map(|_| ())
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
        let start = Instant::now();
        let from_id = self.service.node_by_slug(&from)?;
        let to_id = self.service.node_by_slug(&to)?;
        let res = self.service.add_edge::<crate::graph::PrecedesSpec>(
            from_id,
            to_id,
            PrecedesAttrs { episode },
            confidence,
        );
        if res.is_ok() {
            GraphManager::log_write_latency("add_precedes", start);
        }
        res.map(|_| ())
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
        let start = Instant::now();
        let from_id = self.service.node_by_slug(&from)?;
        let to_id = self.service.node_by_slug(&to)?;
        let res = self.service.add_edge::<crate::graph::AnchorsSpec>(
            from_id,
            to_id,
            AnchorsAttrs { impact },
            confidence,
        );
        if res.is_ok() {
            GraphManager::log_write_latency("add_anchors", start);
        }
        res.map(|_| ())
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

impl Message<ResolveSlugs> for GraphManager {
    type Reply = Result<Vec<NodeId>, GraphError>;

    async fn handle(
        &mut self,
        ResolveSlugs { slugs }: ResolveSlugs,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let mut ids = Vec::with_capacity(slugs.len());
        for slug in slugs {
            ids.push(self.service.node_by_slug(&slug)?);
        }
        Ok(ids)
    }
}

impl Message<RenameNode> for GraphManager {
    type Reply = Result<(), GraphError>;

    async fn handle(
        &mut self,
        RenameNode { old_slug, new_slug }: RenameNode,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let start = Instant::now();
        let res = self.service.rename_node(&old_slug, new_slug);
        if res.is_ok() {
            GraphManager::log_write_latency("rename_node", start);
        }
        res
    }
}

impl Message<RemoveNode> for GraphManager {
    type Reply = Result<(), GraphError>;

    async fn handle(
        &mut self,
        RemoveNode { slug }: RemoveNode,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let start = Instant::now();
        let res = self.service.remove_node(&slug);
        if res.is_ok() {
            GraphManager::log_write_latency("remove_node", start);
        }
        res
    }
}

pub struct SaveSnapshot {
    pub path: PathBuf,
}

pub struct PersistSnapshot;

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

pub struct ResolveSlugs {
    pub slugs: Vec<String>,
}

pub struct RedundantRequires {
    pub prune: bool,
}

pub struct ApplyRuntimeConfig {
    pub course_commit:  String,
    pub strict_quality: bool,
}

impl Message<SaveSnapshot> for GraphManager {
    type Reply = Result<()>;

    async fn handle(
        &mut self,
        SaveSnapshot { path }: SaveSnapshot,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let graph = self.service.shared_graph();
        persist::save_graph(graph.as_ref(), path, &self.course_commit, self.service.graph_version())
            .await
    }
}

impl Message<PersistSnapshot> for GraphManager {
    type Reply = Result<()>;

    async fn handle(
        &mut self,
        _msg: PersistSnapshot,
        ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        self.save_snapshot(ctx.actor_ref()).await
    }
}

impl Message<RedundantRequires> for GraphManager {
    type Reply = Result<Vec<(String, String)>, GraphError>;

    async fn handle(
        &mut self,
        RedundantRequires { prune }: RedundantRequires,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let start = Instant::now();
        let edges: Vec<(String, String)> = self
            .service
            .redundant_requires()
            .into_iter()
            .map(|(u, v)| {
                (self.service.graph()[u].slug.clone(), self.service.graph()[v].slug.clone())
            })
            .collect();
        if prune {
            let removed = self.service.prune_redundant_requires();
            if removed > 0 {
                GraphManager::log_write_latency("prune_redundant_requires", start);
            }
        }
        Ok(edges)
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
        let start = Instant::now();
        let snapshot = persist::load_graph(&path).await?;
        self.service
            .install_graph(snapshot.graph, snapshot.graph_version)?;
        self.service.bump_version();
        self.course_commit = snapshot.course_commit;
        GraphManager::log_write_latency("load_snapshot", start);
        Ok(())
    }
}

impl Message<ApplyRuntimeConfig> for GraphManager {
    type Reply = Result<()>;

    async fn handle(
        &mut self,
        ApplyRuntimeConfig {
            course_commit,
            strict_quality,
        }: ApplyRuntimeConfig,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let prev_strict = self.service.strict_quality();

        if strict_quality != prev_strict
            && let Err(err) = self.service.set_strict_quality(strict_quality)
        {
            return Err(err.into());
        }

        self.course_commit = course_commit;

        Ok(())
    }
}
