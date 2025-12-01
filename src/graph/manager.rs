use std::{
    path::PathBuf,
    sync::{Arc, LazyLock, RwLock},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use kameo::{error::Infallible, message::Context as MsgContext, prelude::*};
use kameo_persistence::{BiHashMap, PersistentActor};
use petgraph::{Direction, visit::EdgeRef};
use schemars::JsonSchema;
use serde::Serialize;
use tracing::error;
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

#[derive(Clone, Serialize)]
struct QuarantinedSnapshot {
    error:          String,
    course_commit:  String,
    strict_quality: bool,
    graph_version:  u64,
    saved_at_sec:   u64,
    graph:          CurriculumGraph,
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

#[derive(Clone, Debug, serde::Serialize)]
pub struct GraphMeta {
    pub graph_version:  u64,
    pub course_commit:  String,
    pub strict_quality: bool,
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

    async fn quarantine_rejected_snapshot(
        state: &GraphManagerState,
        actor_ref: &ActorRef<Self>,
        err: &GraphError,
    ) -> anyhow::Result<Option<PathBuf>> {
        let Some(key) = GraphManager::persistence_key(actor_ref) else {
            return Ok(None);
        };

        let mut dir = key
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("invalid persistence key path"))?;

        let saved_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let filename = format!("quarantine-{}-{}.json", state.graph_version, saved_at);
        dir.push(filename);

        let envelope = QuarantinedSnapshot {
            error:          err.to_string(),
            course_commit:  state.course_commit.clone(),
            strict_quality: state.strict_quality,
            graph_version:  state.graph_version,
            saved_at_sec:   saved_at,
            graph:          state.graph.clone(),
        };

        let payload = serde_json::to_vec_pretty(&envelope)?;
        tokio::fs::write(&dir, payload)
            .await
            .with_context(|| format!("write quarantine snapshot {}", dir.display()))?;
        Ok(Some(dir))
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
    type Error = Arc<anyhow::Error>;

    async fn on_start(state: Self::Args, actor_ref: ActorRef<Self>) -> Result<Self, Self::Error> {
        let strict = state.strict_quality;
        let expected_revision = if state.course_commit.is_empty() {
            None
        } else {
            Some(state.course_commit.clone())
        };
        let service = match GraphService::from_parts(
            state.graph.clone(),
            strict,
            state.graph_version,
            expected_revision,
        ) {
            Ok(svc) => svc,
            Err(err) => {
                let quarantine_path =
                    GraphManager::quarantine_rejected_snapshot(&state, &actor_ref, &err).await?;
                error!(
                    target: "weaver.graph.restore_failed",
                    error = %err,
                    graph_version = state.graph_version,
                    course_commit = %state.course_commit,
                    quarantine_path = %quarantine_path.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "<none>".into()),
                    "refusing to start with invalid persisted graph"
                );
                return Err(Arc::new(anyhow::anyhow!(err)));
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
        _ctx: &mut MsgContext<Self, Self::Reply>,
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
        _ctx: &mut MsgContext<Self, Self::Reply>,
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
        _ctx: &mut MsgContext<Self, Self::Reply>,
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
        _ctx: &mut MsgContext<Self, Self::Reply>,
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
        _ctx: &mut MsgContext<Self, Self::Reply>,
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
        _ctx: &mut MsgContext<Self, Self::Reply>,
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
        _ctx: &mut MsgContext<Self, Self::Reply>,
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
        _ctx: &mut MsgContext<Self, Self::Reply>,
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
        _ctx: &mut MsgContext<Self, Self::Reply>,
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
        _ctx: &mut MsgContext<Self, Self::Reply>,
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
        _ctx: &mut MsgContext<Self, Self::Reply>,
    ) -> Self::Reply {
        Ok(self.service.shared_graph())
    }
}

pub struct GetGraphWithVersion;

impl Message<GetGraphWithVersion> for GraphManager {
    type Reply = Result<(Arc<CurriculumGraph>, u64), Infallible>;

    async fn handle(
        &mut self,
        _msg: GetGraphWithVersion,
        _ctx: &mut MsgContext<Self, Self::Reply>,
    ) -> Self::Reply {
        Ok((self.service.shared_graph(), self.service.graph_version()))
    }
}

pub struct GetGraphVersion;

impl Message<GetGraphVersion> for GraphManager {
    type Reply = Result<u64, Infallible>;

    async fn handle(
        &mut self,
        _msg: GetGraphVersion,
        _ctx: &mut MsgContext<Self, Self::Reply>,
    ) -> Self::Reply {
        Ok(self.service.graph_version())
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
        _ctx: &mut MsgContext<Self, Self::Reply>,
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
        _ctx: &mut MsgContext<Self, Self::Reply>,
    ) -> Self::Reply {
        self.service.node_by_slug(&slug)
    }
}

impl Message<ResolveSlugs> for GraphManager {
    type Reply = Result<Vec<NodeId>, GraphError>;

    async fn handle(
        &mut self,
        ResolveSlugs { slugs }: ResolveSlugs,
        _ctx: &mut MsgContext<Self, Self::Reply>,
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
        _ctx: &mut MsgContext<Self, Self::Reply>,
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
        _ctx: &mut MsgContext<Self, Self::Reply>,
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

pub struct AuditInvariants;

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

pub struct GetCourseCommit;

impl Message<GetCourseCommit> for GraphManager {
    type Reply = String;

    async fn handle(
        &mut self,
        _msg: GetCourseCommit,
        _ctx: &mut MsgContext<Self, Self::Reply>,
    ) -> Self::Reply {
        self.course_commit.clone()
    }
}

pub struct GetGraphMeta;

impl Message<GetGraphMeta> for GraphManager {
    type Reply = Result<GraphMeta, Infallible>;

    async fn handle(
        &mut self,
        _msg: GetGraphMeta,
        _ctx: &mut MsgContext<Self, Self::Reply>,
    ) -> Self::Reply {
        Ok(GraphMeta {
            graph_version:  self.service.graph_version(),
            course_commit:  self.course_commit.clone(),
            strict_quality: self.service.strict_quality(),
        })
    }
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
        _ctx: &mut MsgContext<Self, Self::Reply>,
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
        ctx: &mut MsgContext<Self, Self::Reply>,
    ) -> Self::Reply {
        self.save_snapshot(ctx.actor_ref()).await
    }
}

impl Message<AuditInvariants> for GraphManager {
    type Reply = Result<()>;

    async fn handle(
        &mut self,
        _msg: AuditInvariants,
        _ctx: &mut MsgContext<Self, Self::Reply>,
    ) -> Self::Reply {
        let start = Instant::now();
        let res = self
            .service
            .validate_global_invariants_off_thread(Duration::from_secs(2))
            .await;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        tracing::info!(
            target: "weaver.graph.validation.audit",
            elapsed_ms,
            success = res.is_ok()
        );
        res.map_err(Into::into)
    }
}

impl Message<RedundantRequires> for GraphManager {
    type Reply = Result<Vec<(String, String)>, GraphError>;

    async fn handle(
        &mut self,
        RedundantRequires { prune }: RedundantRequires,
        _ctx: &mut MsgContext<Self, Self::Reply>,
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
        _ctx: &mut MsgContext<Self, Self::Reply>,
    ) -> Self::Reply {
        let start = Instant::now();
        let snapshot = persist::load_graph(&path).await?;
        if !self.course_commit.is_empty() && snapshot.course_commit != self.course_commit {
            anyhow::bail!(
                "snapshot course_commit `{}` does not match runtime course_commit `{}`",
                snapshot.course_commit,
                self.course_commit
            );
        }
        let prev_commit = self.course_commit.clone();
        let prev_expected = self.service.expected_revision().map(|s| s.to_string());

        let new_commit = if self.course_commit.is_empty() {
            snapshot.course_commit.clone()
        } else {
            self.course_commit.clone()
        };
        let expected = if new_commit.is_empty() {
            None
        } else {
            Some(new_commit.clone())
        };

        self.service.set_expected_revision(expected);
        if let Err(err) = self
            .service
            .install_graph(snapshot.graph, snapshot.graph_version)
        {
            // rollback commit/expected on failure
            self.course_commit = prev_commit;
            self.service.set_expected_revision(prev_expected);
            return Err(err.into());
        }

        self.course_commit = new_commit;
        self.service.bump_version();
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
        _ctx: &mut MsgContext<Self, Self::Reply>,
    ) -> Self::Reply {
        let prev_strict = self.service.strict_quality();

        if strict_quality != prev_strict
            && let Err(err) = self.service.set_strict_quality(strict_quality)
        {
            return Err(err.into());
        }

        let prev_commit = self.course_commit.clone();
        let prev_strict_mode = self.service.strict_quality();
        let expected = if course_commit.is_empty() {
            None
        } else {
            Some(course_commit.clone())
        };
        self.service.set_expected_revision(expected);
        self.course_commit = course_commit;

        if let Err(err) = self.service.validate_global_invariants() {
            if strict_quality != prev_strict_mode {
                let _ = self.service.set_strict_quality(prev_strict_mode);
            }
            self.course_commit = prev_commit.clone();
            let prev_expected = if prev_commit.is_empty() {
                None
            } else {
                Some(prev_commit)
            };
            self.service.set_expected_revision(prev_expected);
            return Err(err.into());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use uuid::Uuid;

    use super::*;
    use crate::schema::types::{KnowledgeType, SourceRef};

    fn mk_kn(title: &str, kt: KnowledgeType) -> KnowledgeNode {
        KnowledgeNode {
            title: title.to_string(),
            statement: title.to_string(),
            knowledge_type: kt,
            source_refs: vec![SourceRef {
                path:       "dummy".into(),
                start_line: 1,
                end_line:   2,
                revision:   "deadbeef".into(),
            }],
            confidence: 1.0,
            rubric_criteria: vec![],
            construct_irrelevant_demands: vec![],
            grain_level: None,
            intrinsic_load: None,
            introduction_scope: crate::graph::IntroductionScope::InCourse,
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn graph_manager_persists_and_restores_state() -> anyhow::Result<()> {
        let state_dir: PathBuf =
            std::env::temp_dir().join(format!("weaver-graph-state-{}", Uuid::new_v4()));
        fs::create_dir_all(&state_dir)?;
        let state_url = Url::from_directory_path(&state_dir)
            .map_err(|_| anyhow::anyhow!("invalid state url"))?;

        let mut svc = GraphService::new();
        svc.add_knowledge_node("k1".into(), mk_kn("k1", KnowledgeType::Conceptual), vec![])?;
        let base_version = svc.graph_version();

        let state =
            GraphManagerState::new(svc.snapshot_graph(), String::new(), false, base_version);

        let actor = GraphManager::spawn_persistent(state_url.clone(), state).await?;

        actor
            .ask(InsertKnowledge {
                slug:    "k2".into(),
                payload: mk_kn("k2", KnowledgeType::Procedural),
                tags:    vec![],
            })
            .await?;

        actor.ask(PersistSnapshot).await?;
        actor.stop_gracefully().await.expect("stop graph manager");
        actor.wait_for_shutdown().await;
        drop(actor);

        let restored = GraphManager::respawn_persistent(state_url.clone()).await?;
        restored
            .ask(ResolveSlug { slug: "k1".into() })
            .await
            .expect("k1 restored");
        restored
            .ask(ResolveSlug { slug: "k2".into() })
            .await
            .expect("k2 restored");

        let snapshot_bytes = fs::read(state_dir.join("index.bin"))?;
        let snapshot: GraphManagerState = postcard::from_bytes(&snapshot_bytes)?;
        assert_eq!(snapshot.graph_version, base_version + 1);
        assert!(snapshot.course_commit.is_empty());

        restored
            .stop_gracefully()
            .await
            .expect("stop restored graph manager");
        restored.wait_for_shutdown().await;

        Ok(())
    }
}
