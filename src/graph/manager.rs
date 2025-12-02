use std::{
    path::PathBuf,
    sync::{Arc, LazyLock, RwLock},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use kameo::{error::Infallible, message::Context as MsgContext, prelude::*};
use kameo_persistence::{BiHashMap, PersistentActor};
use petgraph::{Direction, visit::EdgeRef};
use serde::Serialize;
use thiserror::Error;
use tracing::error;
use url::Url;

use crate::graph::{
    CurriculumGraph, EdgeKind, GraphConfig, GraphError, GraphOperationalError, GraphService,
    NodeId, NodePayload, commands::*, persist,
};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GraphManagerState {
    pub graph:                 CurriculumGraph,
    pub course_commit:         String,
    pub strict_quality:        bool,
    #[serde(default)]
    pub graph_version:         u64,
    #[serde(default = "default_validation_timeout_ms")]
    pub validation_timeout_ms: u64,
    #[serde(default)]
    pub skip_dedup_on_insert:  bool,
}

const fn default_validation_timeout_ms() -> u64 {
    crate::constants::GRAPH_VALIDATION_TIMEOUT_MS
}

#[derive(Debug, Error)]
pub enum GraphManagerError {
    #[error(transparent)]
    Domain(#[from] GraphError),
    #[error("graph manager operational error: {0}")]
    Operational(#[from] anyhow::Error),
}

impl GraphManagerState {
    pub fn new(
        graph: CurriculumGraph,
        course_commit: String,
        strict_quality: bool,
        graph_version: u64,
        validation_timeout_ms: u64,
        skip_dedup_on_insert: bool,
    ) -> Self {
        Self {
            graph,
            course_commit,
            strict_quality,
            graph_version,
            validation_timeout_ms,
            skip_dedup_on_insert,
        }
    }
}

#[derive(Clone, Serialize)]
struct QuarantinedSnapshot {
    error:                 String,
    course_commit:         String,
    strict_quality:        bool,
    graph_version:         u64,
    validation_timeout_ms: u64,
    saved_at_sec:          u64,
    graph:                 CurriculumGraph,
    skip_dedup_on_insert:  bool,
}

pub struct GraphManager {
    service:               GraphService,
    course_commit:         String,
    validation_timeout_ms: u64,
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
            validation_timeout_ms: config.validation_timeout_ms,
        }
    }

    fn next_graph_version_after_snapshot(current_version: u64, snapshot_version: u64) -> u64 {
        current_version.max(snapshot_version).saturating_add(1)
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
            error:                 err.to_string(),
            course_commit:         state.course_commit.clone(),
            strict_quality:        state.strict_quality,
            graph_version:         state.graph_version,
            validation_timeout_ms: state.validation_timeout_ms,
            saved_at_sec:          saved_at,
            graph:                 state.graph.clone(),
            skip_dedup_on_insert:  state.skip_dedup_on_insert,
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
            graph:                 manager.service.snapshot_graph_owned(),
            course_commit:         manager.course_commit.clone(),
            strict_quality:        manager.service.strict_quality(),
            graph_version:         manager.service.graph_version(),
            validation_timeout_ms: manager.validation_timeout_ms,
            skip_dedup_on_insert:  manager.service.skip_dedup_on_insert(),
        }
    }
}

impl Actor for GraphManager {
    type Args = GraphManagerState;
    type Error = Arc<GraphManagerError>;

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
            state.skip_dedup_on_insert,
        ) {
            Ok(svc) => svc,
            Err(err) => {
                let quarantine_path = match GraphManager::quarantine_rejected_snapshot(
                    &state, &actor_ref, &err,
                )
                .await
                {
                    Ok(path) => path,
                    Err(qerr) => {
                        return Err(Arc::new(GraphManagerError::Operational(qerr)));
                    }
                };
                error!(
                    target: "weaver.graph.restore_failed",
                    error = %err,
                    graph_version = state.graph_version,
                    course_commit = %state.course_commit,
                    quarantine_path = %quarantine_path.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "<none>".into()),
                    "refusing to start with invalid persisted graph"
                );
                return Err(Arc::new(GraphManagerError::Domain(err)));
            }
        };

        Ok(Self {
            service,
            course_commit: state.course_commit,
            validation_timeout_ms: state.validation_timeout_ms,
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

impl Message<AddPrecedes> for GraphManager {
    type Reply = Result<(), GraphError>;

    async fn handle(
        &mut self,
        AddPrecedes {
            from,
            to,
            attrs,
            confidence,
        }: AddPrecedes,
        _ctx: &mut MsgContext<Self, Self::Reply>,
    ) -> Self::Reply {
        let start = Instant::now();
        let from_id = self.service.node_by_slug(&from)?;
        let to_id = self.service.node_by_slug(&to)?;
        let res = self
            .service
            .add_edge::<crate::graph::PrecedesSpec>(from_id, to_id, attrs, confidence);
        if res.is_ok() {
            GraphManager::log_write_latency("add_precedes", start);
        }
        res.map(|_| ())
    }
}

impl Message<AddAnchors> for GraphManager {
    type Reply = Result<(), GraphError>;

    async fn handle(
        &mut self,
        AddAnchors {
            from,
            to,
            attrs,
            confidence,
        }: AddAnchors,
        _ctx: &mut MsgContext<Self, Self::Reply>,
    ) -> Self::Reply {
        let start = Instant::now();
        let from_id = self.service.node_by_slug(&from)?;
        let to_id = self.service.node_by_slug(&to)?;
        let res = self
            .service
            .add_edge::<crate::graph::AnchorsSpec>(from_id, to_id, attrs, confidence);
        if res.is_ok() {
            GraphManager::log_write_latency("add_anchors", start);
        }
        res.map(|_| ())
    }
}

impl Message<GetEdgeConflicts> for GraphManager {
    type Reply = Result<Vec<EdgeConflictView>, GraphError>;

    async fn handle(
        &mut self,
        _msg: GetEdgeConflicts,
        _ctx: &mut MsgContext<Self, Self::Reply>,
    ) -> Self::Reply {
        let conflicts = self.service.edge_conflicts();
        let graph = self.service.graph();
        let views = conflicts
            .into_iter()
            .map(|entry| EdgeConflictView {
                edge_id:    entry.edge_id.index() as u32,
                from_slug:  graph[entry.from].slug.clone(),
                to_slug:    graph[entry.to].slug.clone(),
                kind:       entry.payload.kind.clone(),
                confidence: entry.payload.confidence,
                conflicts:  entry.payload.conflicts.clone(),
            })
            .collect();
        Ok(views)
    }
}

impl Message<ResolveEdgeConflict> for GraphManager {
    type Reply = Result<(), GraphError>;

    async fn handle(
        &mut self,
        ResolveEdgeConflict {
            edge_id,
            resolved_kind,
            confidence,
            clear_conflicts,
        }: ResolveEdgeConflict,
        _ctx: &mut MsgContext<Self, Self::Reply>,
    ) -> Self::Reply {
        let start = Instant::now();
        let idx = petgraph::stable_graph::EdgeIndex::new(edge_id as usize);
        let res =
            self.service
                .resolve_edge_conflict(idx, resolved_kind, confidence, clear_conflicts);
        if res.is_ok() {
            GraphManager::log_write_latency("resolve_edge_conflict", start);
        }
        res.map(|_| ())
    }
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

impl Message<MergeNodes> for GraphManager {
    type Reply = Result<crate::graph::MergeSummary, GraphError>;

    async fn handle(
        &mut self,
        MergeNodes {
            canonical,
            duplicate,
        }: MergeNodes,
        _ctx: &mut MsgContext<Self, Self::Reply>,
    ) -> Self::Reply {
        let start = Instant::now();
        let res = self.service.merge_nodes(&canonical, &duplicate);
        if res.is_ok() {
            GraphManager::log_write_latency("merge_nodes", start);
        }
        res
    }
}

pub struct SaveSnapshot {
    pub path: PathBuf,
}

pub struct PersistSnapshot;

pub struct AuditInvariants;

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
    pub course_commit:         String,
    pub strict_quality:        bool,
    pub validation_timeout_ms: u64,
}

pub struct SetAuditSink {
    pub sink: crate::graph::audit::SharedMutationSink,
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
        let timeout = Duration::from_millis(self.validation_timeout_ms.max(1));
        let res = self
            .service
            .validate_global_invariants_off_thread(timeout)
            .await;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        if let Err(GraphError::Operational(GraphOperationalError::InvariantTimeout {
            timeout_ms,
        })) = res.as_ref()
        {
            tracing::warn!(
                target: "weaver.graph.validation.audit",
                code = "validation_timeout",
                timeout_ms
            );
        }
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
        let prev_version = self.service.graph_version();
        let target_version =
            GraphManager::next_graph_version_after_snapshot(prev_version, snapshot.graph_version);

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
        if let Err(err) = self.service.install_graph(snapshot.graph, target_version) {
            // rollback commit/expected on failure
            self.course_commit = prev_commit;
            self.service.set_expected_revision(prev_expected);
            return Err(err.into());
        }

        self.course_commit = new_commit;
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
            validation_timeout_ms,
        }: ApplyRuntimeConfig,
        _ctx: &mut MsgContext<Self, Self::Reply>,
    ) -> Self::Reply {
        let prev_strict = self.service.strict_quality();
        let prev_timeout = self.validation_timeout_ms;
        let new_timeout = validation_timeout_ms.max(1);

        self.validation_timeout_ms = new_timeout;

        if strict_quality != prev_strict
            && let Err(err) = self.service.set_strict_quality(strict_quality)
        {
            self.validation_timeout_ms = prev_timeout;
            return Err(err.into());
        }

        let prev_commit = self.course_commit.clone();
        let expected = if course_commit.is_empty() {
            None
        } else {
            Some(course_commit.clone())
        };
        self.service.set_expected_revision(expected);
        self.course_commit = course_commit;

        if let Err(err) = self
            .service
            .validate_global_invariants_off_thread(Duration::from_millis(new_timeout))
            .await
        {
            if strict_quality != prev_strict {
                let _ = self.service.set_strict_quality(prev_strict);
            }
            self.course_commit = prev_commit.clone();
            let prev_expected = if prev_commit.is_empty() {
                None
            } else {
                Some(prev_commit)
            };
            self.service.set_expected_revision(prev_expected);
            self.validation_timeout_ms = prev_timeout;
            return Err(err.into());
        }

        Ok(())
    }
}

impl Message<SetAuditSink> for GraphManager {
    type Reply = std::result::Result<(), Infallible>;

    async fn handle(
        &mut self,
        SetAuditSink { sink }: SetAuditSink,
        _ctx: &mut MsgContext<Self, Self::Reply>,
    ) -> Self::Reply {
        self.service.set_audit_sink(sink);
        Ok(())
    }
}
