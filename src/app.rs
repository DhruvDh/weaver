use std::{
    convert::Infallible,
    fs,
    path::{Component, Path, PathBuf},
    process::Command,
    sync::Arc,
    time::Duration,
};

use anyhow::{Context as _, Result, anyhow, bail};
use bpaf::{OptionParser, Parser, construct, long, positional};
use futures::future::{BoxFuture, join_all};
use kameo::{error::SendError, prelude::*};
use kameo_actors::scheduler::{Scheduler, SetInterval};
use kameo_persistence::PersistentActor;
use tracing::{debug, error, info, warn};
use tracing_subscriber::EnvFilter;
use url::Url;

use crate::{
    agents::deduplication::{DeduplicationAgent, RunDeduplication},
    constants::PRETEXT_SUBDIR,
    file_reader::{
        FileReader, FileReaderQuery, HarvesterFocus, HarvesterReader, WeaverFocus, WeaverReader,
    },
    graph::{
        CurriculumGraph, GraphConfig,
        audit::{FanoutMutationSink, RerunMutationSink},
        commands::ListNodesByTag,
        manager::{
            ApplyRuntimeConfig, AuditInvariants, GetCourseCommit, GraphManager, GraphManagerState,
            PersistSnapshot, RedundantRequires, SaveSnapshot, SetAuditSink,
        },
        persist,
        viz::{GraphVisualizer, GraphVizConfig, GraphVizSink, PrimeRender},
    },
    llm_gateway::{
        GatewayMetrics, GetGatewayMetrics, LLMGateway, LLMGatewayState, PersistGatewaySnapshot,
    },
    rerun_sink::{RerunSink, RerunTarget},
};

fn log_scalar(rerun: &Option<ActorRef<RerunSink>>, path: impl Into<String>, value: f64) {
    if let Some(sink) = rerun {
        let msg = crate::rerun_sink::LogScalar {
            path: path.into(),
            value,
            time_ns: None,
        };
        let sink = sink.clone();
        tokio::spawn(async move {
            let _ = sink.tell(msg).await;
        });
    }
}

#[derive(Debug)]
struct PersistenceReport {
    audit_error:   Option<anyhow::Error>,
    persist_error: Option<anyhow::Error>,
    save_error:    Option<anyhow::Error>,
    gateway_error: Option<anyhow::Error>,
    audit_ms:      f64,
    persist_ms:    f64,
    total_ms:      f64,
}

impl PersistenceReport {
    fn success(&self) -> bool {
        self.audit_error.is_none()
            && self.persist_error.is_none()
            && self.save_error.is_none()
            && self.gateway_error.is_none()
    }

    fn first_error(&self) -> Option<PersistStage<'_>> {
        if let Some(err) = self.audit_error.as_ref() {
            return Some(PersistStage::Audit(err));
        }
        if let Some(err) = self.persist_error.as_ref() {
            return Some(PersistStage::Persist(err));
        }
        if let Some(err) = self.save_error.as_ref() {
            return Some(PersistStage::Save(err));
        }
        if let Some(err) = self.gateway_error.as_ref() {
            return Some(PersistStage::Gateway(err));
        }
        None
    }

    fn log_autosave_metrics(&self, rerun: &Option<ActorRef<RerunSink>>) {
        log_scalar(rerun, "metrics/autosave/audit_duration_ms", self.audit_ms);
        log_scalar(
            rerun,
            "metrics/autosave/audit_success",
            if self.audit_error.is_none() { 1.0 } else { 0.0 },
        );
        log_scalar(rerun, "metrics/autosave/duration_ms", self.total_ms);
        log_scalar(rerun, "metrics/autosave/success", if self.success() { 1.0 } else { 0.0 });
    }

    fn log_labeled_metrics(&self, rerun: &Option<ActorRef<RerunSink>>, label: &str) {
        log_scalar(rerun, format!("metrics/{label}/audit_ms"), self.audit_ms);
        log_scalar(rerun, format!("metrics/{label}/persist_ms"), self.persist_ms);
    }
}

enum PersistStage<'a> {
    Audit(&'a anyhow::Error),
    Persist(&'a anyhow::Error),
    Save(&'a anyhow::Error),
    Gateway(&'a anyhow::Error),
}

trait IntoAnyhow {
    fn into_anyhow(self) -> anyhow::Error;
}

impl IntoAnyhow for Arc<anyhow::Error> {
    fn into_anyhow(self) -> anyhow::Error {
        Arc::try_unwrap(self).unwrap_or_else(|arc| anyhow::Error::msg(arc.to_string()))
    }
}

impl IntoAnyhow for anyhow::Error {
    fn into_anyhow(self) -> anyhow::Error {
        self
    }
}

impl IntoAnyhow for Arc<crate::graph::manager::GraphManagerError> {
    fn into_anyhow(self) -> anyhow::Error {
        match Arc::try_unwrap(self) {
            Ok(err) => err.into_anyhow(),
            Err(shared) => anyhow::Error::msg(shared.to_string()),
        }
    }
}

impl IntoAnyhow for crate::graph::manager::GraphManagerError {
    fn into_anyhow(self) -> anyhow::Error {
        anyhow::Error::new(self)
    }
}

fn flatten_send_error<A, E>(err: SendError<A, E>) -> anyhow::Error
where
    E: IntoAnyhow + std::fmt::Debug,
{
    match err {
        SendError::HandlerError(e) => e.into_anyhow(),
        other => anyhow!(format!("{other:?}")),
    }
}

async fn run_persistence_flow(
    graph: &ActorRef<GraphManager>,
    gateway: Option<&ActorRef<LLMGateway>>,
    autosave_path: &Path,
) -> PersistenceReport {
    let started = std::time::Instant::now();

    let audit_started = std::time::Instant::now();
    let audit_res = graph.ask(AuditInvariants).await.map_err(flatten_send_error);
    let audit_ms = audit_started.elapsed().as_secs_f64() * 1000.0;
    if let Err(err) = audit_res {
        return PersistenceReport {
            audit_error: Some(err),
            persist_error: None,
            save_error: None,
            gateway_error: None,
            audit_ms,
            persist_ms: 0.0,
            total_ms: started.elapsed().as_secs_f64() * 1000.0,
        };
    }

    let persist_started = std::time::Instant::now();
    let persist_res = graph.ask(PersistSnapshot).await.map_err(flatten_send_error);
    if let Err(err) = persist_res {
        let elapsed_ms = persist_started.elapsed().as_secs_f64() * 1000.0;
        return PersistenceReport {
            audit_error: None,
            persist_error: Some(err),
            save_error: None,
            gateway_error: None,
            audit_ms,
            persist_ms: elapsed_ms,
            total_ms: started.elapsed().as_secs_f64() * 1000.0,
        };
    }

    let save_res = graph
        .ask(SaveSnapshot {
            path: autosave_path.to_path_buf(),
        })
        .await
        .map_err(flatten_send_error);
    if let Err(err) = save_res {
        let elapsed_ms = persist_started.elapsed().as_secs_f64() * 1000.0;
        return PersistenceReport {
            audit_error: None,
            persist_error: None,
            save_error: Some(err),
            gateway_error: None,
            audit_ms,
            persist_ms: elapsed_ms,
            total_ms: started.elapsed().as_secs_f64() * 1000.0,
        };
    }

    let gateway_res = if let Some(gateway) = gateway {
        gateway
            .ask(PersistGatewaySnapshot)
            .await
            .map_err(flatten_send_error)
    } else {
        Ok(())
    };

    let persist_ms = persist_started.elapsed().as_secs_f64() * 1000.0;
    match gateway_res {
        Ok(()) => PersistenceReport {
            audit_error: None,
            persist_error: None,
            save_error: None,
            gateway_error: None,
            audit_ms,
            persist_ms,
            total_ms: started.elapsed().as_secs_f64() * 1000.0,
        },
        Err(err) => PersistenceReport {
            audit_error: None,
            persist_error: None,
            save_error: None,
            gateway_error: Some(err),
            audit_ms,
            persist_ms,
            total_ms: started.elapsed().as_secs_f64() * 1000.0,
        },
    }
}

#[derive(Clone)]
struct AutosaveTick;

#[derive(Clone)]
struct PruneTick;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RerunMode {
    Grpc,
    File,
    Both,
    None,
}

#[derive(Clone)]
struct AutosaveWorker {
    graph:         ActorRef<GraphManager>,
    gateway:       Option<ActorRef<LLMGateway>>,
    rerun:         Option<ActorRef<RerunSink>>,
    autosave_path: PathBuf,
}

impl Actor for AutosaveWorker {
    type Args = AutosaveWorker;
    type Error = Infallible;

    async fn on_start(state: Self::Args, _actor_ref: ActorRef<Self>) -> Result<Self, Self::Error> {
        Ok(state)
    }
}

#[derive(Clone)]
struct PruneWorker {
    graph: ActorRef<GraphManager>,
    rerun: Option<ActorRef<RerunSink>>,
}

impl Actor for PruneWorker {
    type Args = PruneWorker;
    type Error = Infallible;

    async fn on_start(state: Self::Args, _actor_ref: ActorRef<Self>) -> Result<Self, Self::Error> {
        Ok(state)
    }
}

impl Message<AutosaveTick> for AutosaveWorker {
    type Reply = ();

    async fn handle(
        &mut self,
        _msg: AutosaveTick,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let report =
            run_persistence_flow(&self.graph, self.gateway.as_ref(), &self.autosave_path).await;

        report.log_autosave_metrics(&self.rerun);

        match report.first_error() {
            Some(PersistStage::Audit(err)) => {
                error!(error = %err, "autosave aborted: graph audit failed");
            }
            Some(PersistStage::Persist(err)) => {
                error!(error = %err, "autosave failed: graph persistence");
            }
            Some(PersistStage::Save(err)) => {
                error!(error = %err, "autosave failed: legacy snapshot write");
            }
            Some(PersistStage::Gateway(err)) => {
                error!(error = %err, "autosave failed: gateway persistence");
            }
            None => {
                debug!("graph autosave completed");
            }
        }
    }
}

impl Message<PruneTick> for PruneWorker {
    type Reply = ();

    async fn handle(
        &mut self,
        _msg: PruneTick,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let start = std::time::Instant::now();
        match self.graph.ask(RedundantRequires { prune: true }).await {
            Ok(edges) => {
                debug!(pruned = edges.len(), "graph prune redundant requires");
                log_scalar(&self.rerun, "metrics/prune_redundant/edges", edges.len() as f64);
                log_scalar(
                    &self.rerun,
                    "metrics/prune_redundant/duration_ms",
                    start.elapsed().as_secs_f64() * 1000.0,
                );
            }
            Err(err) => {
                error!(error = ?err, "graph prune redundant requires failed");
                log_scalar(
                    &self.rerun,
                    "metrics/prune_redundant/duration_ms",
                    start.elapsed().as_secs_f64() * 1000.0,
                );
                log_scalar(&self.rerun, "metrics/prune_redundant/edges", 0.0);
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Cli {
    pub rerun_mode:                  RerunMode,
    pub rerun_file:                  PathBuf,
    pub graph_snapshot_path:         PathBuf,
    pub graph_autosave_secs:         u64,
    pub graph_course_commit:         Option<String>,
    pub graph_strict_quality:        bool,
    pub graph_prune_requires_s:      Option<u64>,
    pub skip_demo:                   bool,
    pub graph_validation_timeout_ms: u64,
    pub dedup_interval_secs:         u64,
    pub dedup_auto_merge_threshold:  f64,
    pub skip_dedup_on_insert:        bool,
    pub interactive:                 bool,
    pub chapters_pattern:            Option<String>,
    pub chapters_dir:                Option<PathBuf>,
    pub workspace:                   PathBuf,
}

#[derive(Clone)]
pub struct RuntimeOptions {
    pub gateway_mode:          GatewayMode,
    pub min_autosave_secs:     u64,
    pub trigger_initial_save:  bool,
    pub trigger_shutdown_save: bool,
    pub on_started:            Option<AppHook>,
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        Self {
            gateway_mode:          GatewayMode::Real,
            min_autosave_secs:     5,
            trigger_initial_save:  true,
            trigger_shutdown_save: true,
            on_started:            None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum GatewayMode {
    Real,
    Stub,
}

#[derive(Clone)]
pub struct AppHandles {
    pub graph:     ActorRef<GraphManager>,
    pub gateway:   ActorRef<LLMGateway>,
    pub rerun:     Option<ActorRef<RerunSink>>,
    pub scheduler: ActorRef<Scheduler>,
    pub dedup:     ActorRef<DeduplicationAgent>,
}

pub type AppHook = Arc<dyn Fn(AppHandles) -> BoxFuture<'static, Result<()>> + Send + Sync>;

fn git_head_hash() -> Option<String> {
    Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
            } else {
                None
            }
        })
}

fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create directory {}", parent.display()))?;
    }
    Ok(())
}

fn chapter_scope_tag(workspace_root: &Path, chapter_path: &Path) -> String {
    let relative = if chapter_path.is_absolute() {
        pathdiff::diff_paths(chapter_path, workspace_root)
            .unwrap_or_else(|| chapter_path.to_path_buf())
    } else {
        chapter_path.to_path_buf()
    };

    let normalized = relative
        .components()
        .filter_map(|component| match component {
            Component::Normal(part) => Some(part.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/");

    if normalized.is_empty() {
        chapter_path.display().to_string()
    } else {
        normalized
    }
}

/// Orchestrate two-phase autonomous graph construction: harvest nodes, run
/// dedup, then weave edges.
pub async fn run_two_phase_construction(cli: Cli, chapters: Vec<PathBuf>) -> Result<()> {
    if chapters.is_empty() {
        bail!("--chapters matched no files; provide at least one chapter");
    }

    let mut modified_cli = cli.clone();
    modified_cli.skip_demo = true;
    modified_cli.skip_dedup_on_insert = true;

    let workspace_root = modified_cli.workspace.clone();
    let dedup_threshold = modified_cli.dedup_auto_merge_threshold;
    let chapters_for_hook = chapters.clone();

    let hook: AppHook = Arc::new(move |handles: AppHandles| -> BoxFuture<'static, Result<()>> {
        let workspace_root = workspace_root.clone();
        let chapters = chapters_for_hook.clone();
        Box::pin(async move {
            let course_commit = handles
                .graph
                .ask(GetCourseCommit)
                .await
                .unwrap_or_else(|_| String::new());
            let metrics = handles
                .gateway
                .ask(GetGatewayMetrics)
                .await
                .unwrap_or_else(|_| Arc::new(GatewayMetrics::default()));

            let graph = handles.graph.clone();
            let gateway = handles.gateway.clone();
            let rerun = handles.rerun.clone();
            let dedup_agent = handles.dedup.clone();
            let harvester_focus = HarvesterFocus::All;
            let weaver_focus = WeaverFocus::All;

            info!(chapters = chapters.len(), "Phase 1: harvesting nodes");
            let course_commit_for_tasks = course_commit.clone();
            let harvester_tasks = chapters.iter().map(|chapter_path| {
                let chapter_path = chapter_path.clone();
                let chapter_tag = chapter_scope_tag(&workspace_root, &chapter_path);
                let tag = format!("source:{chapter_tag}");
                let graph_for_reader = graph.clone();
                let graph_for_check = graph.clone();
                let gateway_ref = gateway.clone();
                let metrics_ref = Arc::clone(&metrics);
                let dedup_ref = dedup_agent.clone();
                let rerun_ref = rerun.clone();
                let workspace = workspace_root.clone();
                let course_commit = course_commit_for_tasks.clone();

                async move {
                    let actor = HarvesterReader::from_env(
                        workspace,
                        gateway_ref,
                        metrics_ref,
                        graph_for_reader,
                        dedup_ref,
                        rerun_ref,
                        course_commit,
                    )?;
                    let reader = HarvesterReader::spawn(actor);
                    let prompt = format!(
                        "PHASE 1: Harvest EVERY node from {chapter}\n\n- Extract all concepts, \
                         facts, procedures, strategies, learning outcomes, teaching steps, \
                         assessments, and worked examples.\n- Do NOT create edges.\n- Tag every \
                         node with {tag} and add req:/sup:/ref: hints in tags when you see \
                         dependencies.\n- Focus: {harvester_focus} \
                         ({harvester_focus_directive}).\n- Aim for 80-150 nodes from this \
                         chapter; completeness is more important than brevity.",
                        chapter = chapter_path.display(),
                        tag = tag,
                        harvester_focus = harvester_focus,
                        harvester_focus_directive = harvester_focus.directive(),
                    );
                    reader.ask(FileReaderQuery { prompt }).await?;

                    let tagged = graph_for_check
                        .ask(ListNodesByTag { tag: tag.clone() })
                        .await
                        .map_err(anyhow::Error::from)?;
                    if tagged.is_empty() {
                        bail!(
                            "Phase 1 produced no nodes tagged {tag} for chapter {}",
                            chapter_path.display()
                        );
                    }

                    Ok::<_, anyhow::Error>((chapter_path, chapter_tag, tag, tagged.len()))
                }
            });
            let harvester_results: Vec<Result<(PathBuf, String, String, usize)>> =
                join_all(harvester_tasks).await;

            let mut successful_chapters = Vec::new();
            let mut failed_harvests = Vec::new();

            for result in harvester_results {
                match result {
                    Ok((chapter_path, chapter_tag, tag, tagged_count)) => {
                        info!(
                            chapter = %chapter_path.display(),
                            %tag,
                            tagged_count,
                            "Harvest complete for chapter"
                        );
                        successful_chapters.push((chapter_path, chapter_tag, tag));
                    }
                    Err(err) => {
                        warn!(error = ?err, "Harvest failed for chapter");
                        failed_harvests.push(err);
                    }
                }
            }

            if successful_chapters.is_empty() {
                let errors = failed_harvests
                    .into_iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join("; ");
                bail!("Harvest failed for all chapters: {errors}");
            }

            if !failed_harvests.is_empty() {
                warn!(
                    successful = successful_chapters.len(),
                    failed = failed_harvests.len(),
                    total = chapters.len(),
                    "Continuing with successfully harvested chapters only"
                );
            }

            info!("Deduplication barrier starting");
            let dedup_report = dedup_agent
                .ask(RunDeduplication {
                    auto_merge_threshold: dedup_threshold,
                    dry_run:              false,
                })
                .await
                .map_err(anyhow::Error::from)?;
            info!(
                clusters = dedup_report.clusters_analyzed,
                auto_merged = dedup_report.auto_merged.len(),
                pending_review = dedup_report.pending_review.len(),
                "Deduplication complete"
            );

            graph
                .ask(AuditInvariants)
                .await
                .map_err(anyhow::Error::from)?;
            graph
                .ask(PersistSnapshot)
                .await
                .map_err(anyhow::Error::from)?;

            info!(chapters = successful_chapters.len(), "Phase 2: weaving edges");
            let course_commit_for_weavers = course_commit.clone();
            let weaver_tasks = successful_chapters.iter().map(
                |(chapter_path, _chapter_tag, tag): &(PathBuf, String, String)| {
                    let chapter_path = chapter_path.clone();
                    let tag = tag.clone();
                    let graph_ref = graph.clone();
                    let gateway_ref = gateway.clone();
                    let metrics_ref = Arc::clone(&metrics);
                    let rerun_ref = rerun.clone();
                    let dedup_ref = dedup_agent.clone();
                    let workspace = workspace_root.clone();
                    let course_commit = course_commit_for_weavers.clone();

                    async move {
                        let actor = WeaverReader::from_env(
                            workspace,
                            gateway_ref,
                            metrics_ref,
                            graph_ref,
                            dedup_ref,
                            rerun_ref,
                            course_commit,
                        )?;
                        let reader = WeaverReader::spawn(actor);
                        let prompt = format!(
                            "PHASE 2: Connect ALL nodes for {chapter}\n\n- Start with \
                             graph_list_nodes_by_tag {tag} to scope the inventory.\n- Use \
                             graph_search_nodes when slugs are fuzzy; avoid creating new \
                             nodes.\n- Create requires/supports/assesses/precedes/anchors edges \
                             with strong rationales and evidence refs.\n- Focus: {weaver_focus} \
                             ({weaver_focus_directive}).\n- Run graph_gap_summary, \
                             graph_lo_alignment_summary, and graph_dag_check near the end. Target \
                             200-400 edges.",
                            chapter = chapter_path.display(),
                            tag = tag,
                            weaver_focus = weaver_focus,
                            weaver_focus_directive = weaver_focus.directive(),
                        );
                        reader
                            .ask(FileReaderQuery { prompt })
                            .await
                            .map(|_| ())
                            .map_err(anyhow::Error::from)
                    }
                },
            );
            let weaver_results: Vec<Result<()>> = join_all(weaver_tasks).await;
            let total_weaves = weaver_results.len();
            let mut failed_weaves = Vec::new();
            for ((chapter_path, tag), result) in successful_chapters
                .into_iter()
                .map(|(chapter_path, _tag, tag)| (chapter_path, tag))
                .zip(weaver_results)
            {
                if let Err(err) = result {
                    warn!(
                        chapter = %chapter_path.display(),
                        %tag,
                        error = ?err,
                        "Weaving failed for chapter"
                    );
                    failed_weaves.push(err);
                }
            }

            if !failed_weaves.is_empty() {
                warn!(failed = failed_weaves.len(), "Phase 2 completed with weaving failures");
            }
            if total_weaves > 0 && failed_weaves.len() == total_weaves {
                bail!("Weaving failed for all harvested chapters");
            }

            graph
                .ask(AuditInvariants)
                .await
                .map_err(anyhow::Error::from)?;
            info!("Two-phase construction completed");
            Ok(())
        })
    });

    let runtime = RuntimeOptions {
        on_started: Some(hook),
        ..RuntimeOptions::default()
    };

    run_app(modified_cli, runtime).await
}

pub fn cli() -> OptionParser<Cli> {
    let rerun_mode = long("rerun-mode")
        .help("Rerun sink mode: grpc | file | both | none (default grpc)")
        .argument::<String>("mode")
        .parse(|s| match s.as_str() {
            "grpc" => Ok(RerunMode::Grpc),
            "file" => Ok(RerunMode::File),
            "both" => Ok(RerunMode::Both),
            "none" => Ok(RerunMode::None),
            other => Err(format!("invalid rerun mode: {other}")),
        })
        .fallback(RerunMode::Grpc);
    let rerun_file = long("rerun-file")
        .help("Path to write Rerun .rrd when rerun-mode includes file (default weaver.rrd)")
        .argument::<PathBuf>("path")
        .fallback(PathBuf::from("weaver.rrd"));
    let graph_snapshot_path = long("graph-snapshot-path")
        .help("Path for the legacy JSON graph snapshot (default graph_snapshot.json)")
        .argument::<PathBuf>("path")
        .fallback(PathBuf::from("graph_snapshot.json"));
    let graph_autosave_secs = long("graph-autosave-secs")
        .help(
            "Autosave interval in seconds (min 1; clamped to a safe floor at runtime, default 300)",
        )
        .argument::<u64>("secs")
        .fallback(300);
    let graph_validation_timeout_ms = long("graph-validation-timeout-ms")
        .help("Timeout in milliseconds for graph invariant audits (default 2000ms)")
        .argument::<u64>("ms")
        .fallback(2_000);
    let graph_course_commit = long("graph-course-commit")
        .help(
            "Course commit hash to embed in snapshots (defaults to snapshot commit when present; \
             override only when you need to force a new revision)",
        )
        .argument::<String>("hash")
        .optional();
    let graph_strict_quality = long("graph-strict-quality")
        .short('q')
        .help(
            "Enable strict graph quality checks (reachability, coverage/purity, practice, \
             discourse) by promoting warnings to errors; recommended for CI",
        )
        .switch();
    let graph_prune_requires_s = long("graph-prune-requires-secs")
        .help("Optional interval (seconds) to prune redundant requires edges; omit to disable")
        .argument::<u64>("secs")
        .optional();
    let dedup_interval_secs = long("dedup-interval-secs")
        .help("Interval in seconds for background deduplication (set 0 to disable)")
        .argument::<u64>("secs")
        .fallback(3_600);
    let dedup_auto_merge_threshold = long("dedup-auto-merge-threshold")
        .help("Similarity threshold (0.0-1.0) for auto-merging duplicates (default 0.95)")
        .argument::<f64>("threshold")
        .fallback(0.95);
    let skip_dedup_on_insert = long("skip-dedup-on-insert")
        .help("Skip insert-time duplicate checks (useful for bulk imports)")
        .switch();
    let skip_demo = long("skip-demo")
        .help("Skip the startup FileReader demo (useful for tests or headless runs)")
        .switch();
    let interactive = long("interactive")
        .help("Run the legacy interactive FileReader demo instead of two-phase automation")
        .switch();
    let chapters_pattern = long("chapters")
        .help(
            "Glob pattern for chapter files (e.g., 'source/sec-*.ptx'); required with --two-phase",
        )
        .argument::<String>("pattern")
        .optional();
    let chapters_dir = long("chapters-dir")
        .help("Directory containing chapter files; expands to all *.ptx within (recursive)")
        .argument::<PathBuf>("dir")
        .optional();
    let workspace = positional::<PathBuf>("workspace")
        .help("Workspace root to expose to the assistant")
        .fallback(PathBuf::from(PRETEXT_SUBDIR));

    construct! {
        Cli {
            rerun_mode,
            rerun_file,
            graph_snapshot_path,
            graph_autosave_secs,
            graph_course_commit,
            graph_strict_quality,
            graph_prune_requires_s,
            skip_demo,
            graph_validation_timeout_ms,
            dedup_interval_secs,
            dedup_auto_merge_threshold,
            skip_dedup_on_insert,
            interactive,
            chapters_pattern,
            chapters_dir,
            workspace,
        }
    }
    .to_options()
}

async fn persist_once(
    graph: &ActorRef<GraphManager>,
    gateway: &ActorRef<LLMGateway>,
    rerun: &Option<ActorRef<RerunSink>>,
    autosave_path: &Path,
    label: &str,
) -> Result<()> {
    let report = run_persistence_flow(graph, Some(gateway), autosave_path).await;

    report.log_labeled_metrics(rerun, label);

    if let Some(err) = report.audit_error {
        warn!(label = label, error = %err, "graph invariant audit failed during persistence");
        return Err(err);
    }
    if let Some(err) = report.persist_error {
        warn!(label = label, error = %err, "failed to persist graph state");
        return Err(err);
    }
    if let Some(err) = report.save_error {
        warn!(label = label, error = %err, "failed to write legacy snapshot");
        return Err(err);
    }
    if let Some(err) = report.gateway_error {
        warn!(label = label, error = %err, "failed to persist gateway state");
        return Err(err);
    }

    info!(label = label, "persisted graph and gateway state");
    Ok(())
}

async fn reconcile_course_commit(
    graph_actor: &ActorRef<GraphManager>,
    desired: Option<String>,
    strict_quality: bool,
    validation_timeout_ms: u64,
) -> Result<String> {
    let current_commit = graph_actor
        .ask(GetCourseCommit)
        .await
        .unwrap_or_else(|_| String::new());
    let target_commit = desired.clone().unwrap_or_else(|| current_commit.clone());

    if let Some(ref cli_commit) = desired
        && !current_commit.is_empty()
        && *cli_commit != current_commit
    {
        warn!(
            snapshot_course_commit = %current_commit,
            cli_course_commit = %cli_commit,
            "Snapshot course_commit differs; running with CLI override. \
             Regenerate the snapshot or pass --graph-course-commit={} to align.",
            current_commit
        );
    }

    match graph_actor
        .ask(ApplyRuntimeConfig {
            course_commit: target_commit.clone(),
            strict_quality,
            validation_timeout_ms,
        })
        .await
    {
        Ok(()) => Ok(target_commit),
        Err(err) => {
            if desired.is_some() && target_commit != current_commit {
                warn!(
                    error = %err,
                    course_commit = %target_commit,
                    fallback_course_commit = %current_commit,
                    "course_commit override failed validation; falling back to snapshot commit"
                );
                graph_actor
                    .ask(ApplyRuntimeConfig {
                        course_commit: current_commit.clone(),
                        strict_quality,
                        validation_timeout_ms,
                    })
                    .await
                    .map_err(|e| anyhow!(e))?;
                Ok(current_commit)
            } else {
                Err(err.into())
            }
        }
    }
}

async fn spawn_gateway(
    state_url: Url,
    mode: GatewayMode,
) -> Result<(ActorRef<LLMGateway>, Arc<GatewayMetrics>)> {
    match LLMGateway::respawn_persistent(state_url.clone()).await {
        Ok(actor) => {
            debug!(path = %state_url, "restored LLM gateway from persistent snapshot");
            let metrics = actor.ask(GetGatewayMetrics).await.unwrap_or_else(|err| {
                error!(error = ?err, "failed to fetch gateway metrics after restore");
                Arc::new(GatewayMetrics::default())
            });
            Ok((actor, metrics))
        }
        Err(err) => {
            debug!(error = %err, path = %state_url, "gateway state restore unavailable");
            match mode {
                GatewayMode::Real => {
                    let instance = LLMGateway::from_env()?;
                    let metrics = instance.metrics();
                    let actor = LLMGateway::spawn_persistent(state_url.clone(), instance).await?;
                    Ok((actor, metrics))
                }
                GatewayMode::Stub => {
                    let instance = LLMGateway::from(LLMGatewayState::new(
                        GatewayMetrics::default().to_state(),
                    ));
                    let metrics = instance.metrics();
                    let actor = LLMGateway::spawn_persistent(state_url.clone(), instance).await?;
                    Ok((actor, metrics))
                }
            }
        }
    }
}

pub async fn run_app(cli: Cli, runtime: RuntimeOptions) -> Result<()> {
    if cli.graph_autosave_secs == 0 {
        anyhow::bail!("--graph-autosave-secs must be at least 1 second");
    }
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    if let Err(err) = tracing_subscriber::fmt()
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::new("%Y-%m-%d %H:%M:%S%.3f".into()))
        .with_env_filter(filter)
        .try_init()
    {
        debug!(error = %err, "tracing subscriber already initialized; continuing");
    }

    let effective_autosave_secs = cli
        .graph_autosave_secs
        .max(runtime.min_autosave_secs.max(1));
    info!(
        requested = cli.graph_autosave_secs,
        effective = effective_autosave_secs,
        min = runtime.min_autosave_secs,
        "configured autosave interval"
    );

    let mut graph_config = GraphConfig {
        course_commit:         String::new(),
        autosave_path:         cli.graph_snapshot_path.clone(),
        autosave_secs:         effective_autosave_secs,
        strict_quality:        cli.graph_strict_quality,
        validation_timeout_ms: cli.graph_validation_timeout_ms,
        skip_dedup_on_insert:  cli.skip_dedup_on_insert,
    };

    let snapshot_path: PathBuf = graph_config.autosave_path.clone();
    let state_root = if snapshot_path.extension().is_some() {
        snapshot_path.with_extension("state")
    } else {
        snapshot_path.clone()
    };
    let graph_state_dir = state_root.join("graph_manager");
    let gateway_state_dir = state_root.join("llm_gateway");
    let persisted_graph_index = graph_state_dir.join("index.bin");

    let cwd = std::env::current_dir()?;
    let graph_state_url = Url::from_directory_path(cwd.join(&graph_state_dir))
        .map_err(|_| anyhow!("invalid graph state path {}", graph_state_dir.display()))?;
    let gateway_state_url = Url::from_directory_path(cwd.join(&gateway_state_dir))
        .map_err(|_| anyhow!("invalid gateway state path {}", gateway_state_dir.display()))?;
    ensure_parent_dir(&graph_config.autosave_path)?;
    fs::create_dir_all(&graph_state_dir)
        .with_context(|| format!("create graph state dir {}", graph_state_dir.display()))?;
    fs::create_dir_all(&gateway_state_dir)
        .with_context(|| format!("create gateway state dir {}", gateway_state_dir.display()))?;
    ensure_parent_dir(&cli.rerun_file)?;

    let git_head = git_head_hash();
    let desired_course_commit = cli.graph_course_commit.clone();

    let graph_actor = match GraphManager::respawn_persistent(graph_state_url.clone()).await {
        Ok(actor) => {
            debug!(path = %graph_state_dir.display(), "restored graph manager from persistent snapshot");
            actor
        }
        Err(err) => {
            if persisted_graph_index.exists() {
                error!(
                    error = %err,
                    path = %persisted_graph_index.display(),
                    "persisted graph state exists but failed to restore; exiting to avoid data loss"
                );
                std::process::exit(1);
            }
            debug!(
                error = %err,
                path = %graph_state_dir.display(),
                "graph state restore unavailable; falling back to legacy snapshot or empty graph"
            );
            let state = if snapshot_path.exists() {
                match persist::load_graph(&snapshot_path).await {
                    Ok(snapshot) => {
                        debug!(path = %snapshot_path.display(), "loaded legacy graph snapshot");
                        graph_config.course_commit = snapshot.course_commit.clone();
                        GraphManagerState::new(
                            snapshot.graph,
                            graph_config.course_commit.clone(),
                            graph_config.strict_quality,
                            snapshot.graph_version,
                            graph_config.validation_timeout_ms,
                            graph_config.skip_dedup_on_insert,
                        )
                    }
                    Err(err) => {
                        error!(
                            error = %err,
                            path = %snapshot_path.display(),
                            "failed to load graph snapshot; starting with empty graph"
                        );
                        let fallback_commit = desired_course_commit
                            .clone()
                            .or_else(|| git_head.clone())
                            .unwrap_or_default();
                        graph_config.course_commit = fallback_commit.clone();
                        GraphManagerState::new(
                            CurriculumGraph::default(),
                            fallback_commit,
                            graph_config.strict_quality,
                            0,
                            graph_config.validation_timeout_ms,
                            graph_config.skip_dedup_on_insert,
                        )
                    }
                }
            } else {
                let fallback_commit = desired_course_commit
                    .clone()
                    .or_else(|| git_head.clone())
                    .unwrap_or_default();
                graph_config.course_commit = fallback_commit.clone();
                GraphManagerState::new(
                    CurriculumGraph::default(),
                    fallback_commit,
                    graph_config.strict_quality,
                    0,
                    graph_config.validation_timeout_ms,
                    graph_config.skip_dedup_on_insert,
                )
            };
            GraphManager::spawn_persistent(graph_state_url.clone(), state).await?
        }
    };
    if let Err(err) = graph_actor.wait_for_startup_result().await {
        error!(error = ?err, "graph manager failed during startup");
        std::process::exit(1);
    }

    let (gateway, metrics) = spawn_gateway(gateway_state_url.clone(), runtime.gateway_mode).await?;

    let scheduler = Scheduler::spawn(Scheduler::new());

    let mut rerun_targets = Vec::new();
    if matches!(cli.rerun_mode, RerunMode::Grpc | RerunMode::Both) {
        rerun_targets.push(RerunTarget::Grpc {
            name: "weaver".into(),
        });
    }
    if matches!(cli.rerun_mode, RerunMode::File | RerunMode::Both) {
        rerun_targets.push(RerunTarget::File {
            name: "weaver".into(),
            path: cli.rerun_file.clone(),
        });
    }
    let rerun_actor = if rerun_targets.is_empty() {
        None
    } else {
        Some(RerunSink::spawn(rerun_targets))
    };
    let rerun_viz_actor = rerun_actor.as_ref().map(|rerun| {
        GraphVisualizer::spawn(GraphVisualizer::new(
            graph_actor.clone(),
            rerun.clone(),
            GraphVizConfig::default(),
        ))
    });
    if let Some(rerun) = &rerun_actor {
        let mut sinks: Vec<crate::graph::audit::SharedMutationSink> =
            vec![Arc::new(RerunMutationSink::new(rerun.clone()))];
        if let Some(viz) = &rerun_viz_actor {
            sinks.push(Arc::new(GraphVizSink::new(viz.clone())));
        }
        let sink: crate::graph::audit::SharedMutationSink =
            Arc::new(FanoutMutationSink::new(sinks));
        let _ = graph_actor.tell(SetAuditSink { sink }).await;
        if let Some(viz) = &rerun_viz_actor {
            let viz = viz.clone();
            tokio::spawn(async move {
                let _ = viz.tell(PrimeRender).await;
            });
        }
    }

    let autosave_worker = AutosaveWorker {
        graph:         graph_actor.clone(),
        gateway:       Some(gateway.clone()),
        rerun:         rerun_actor.clone(),
        autosave_path: graph_config.autosave_path.clone(),
    };
    let autosave_ref = AutosaveWorker::spawn(autosave_worker);
    let autosave_interval = SetInterval::new(
        autosave_ref.downgrade(),
        Duration::from_secs(graph_config.autosave_secs),
        AutosaveTick,
    );
    scheduler
        .tell(autosave_interval)
        .await
        .expect("scheduler actor not running");

    if let Some(prune_secs) = cli.graph_prune_requires_s.filter(|value| *value > 0) {
        let prune_worker = PruneWorker {
            graph: graph_actor.clone(),
            rerun: rerun_actor.clone(),
        };
        let prune_ref = PruneWorker::spawn(prune_worker);
        let prune_interval =
            SetInterval::new(prune_ref.downgrade(), Duration::from_secs(prune_secs), PruneTick);
        scheduler
            .tell(prune_interval)
            .await
            .expect("scheduler actor not running");
    }

    let dedup_agent = DeduplicationAgent::spawn(DeduplicationAgent::new(
        graph_actor.clone(),
        rerun_actor.clone(),
    ));

    if cli.dedup_interval_secs > 0 {
        let dedup_task = SetInterval::new(
            dedup_agent.downgrade(),
            Duration::from_secs(cli.dedup_interval_secs),
            RunDeduplication {
                auto_merge_threshold: cli.dedup_auto_merge_threshold,
                dry_run:              false,
            },
        );
        scheduler
            .tell(dedup_task)
            .await
            .expect("scheduler actor not running");
    }

    let resolved_commit = reconcile_course_commit(
        &graph_actor,
        desired_course_commit.clone(),
        cli.graph_strict_quality,
        graph_config.validation_timeout_ms,
    )
    .await?;
    debug!(course_commit = %resolved_commit, "runtime course_commit resolved");

    if let Some(hook) = &runtime.on_started {
        hook(AppHandles {
            graph:     graph_actor.clone(),
            gateway:   gateway.clone(),
            rerun:     rerun_actor.clone(),
            scheduler: scheduler.clone(),
            dedup:     dedup_agent.clone(),
        })
        .await?;
    }

    if runtime.trigger_initial_save {
        persist_once(
            &graph_actor,
            &gateway,
            &rerun_actor,
            &graph_config.autosave_path,
            "initial_autosave",
        )
        .await?;
    }

    if !cli.skip_demo {
        let actor = match FileReader::from_env(
            cli.workspace.clone(),
            gateway.clone(),
            Arc::clone(&metrics),
            graph_actor.clone(),
            dedup_agent.clone(),
            rerun_actor.clone(),
            resolved_commit.clone(),
        ) {
            Ok(actor) => actor,
            Err(err) => {
                error!(
                    error = %err,
                    "Failed to initialize FileReader; set OPENAI_MODEL to enable LLM tools"
                );
                return Err(err);
            }
        };

        let tool_names = FileReader::tool_identifiers().map_err(|err| {
            error!(error = %err, "Failed to build LLM tool registry");
            err
        })?;

        debug!(
            workspace = %actor.workspace_root().display(),
            tools = ?tool_names,
            "Initialized FileReader with LLM tool bridge for the PreTeXt project"
        );

        let reader = FileReader::spawn(actor);

        let prompt = "Summarize the key goals of the UNCC CS2 PreTeXt project. Highlight any \
                      modules in the `source/` tree that look important. Please do make effective \
                      use of the `delegate_tasks` tools for all tasks, in parallel if possible.";
        debug!(prompt, "Dispatching FileReaderQuery with LLM tool access");

        match reader
            .ask(FileReaderQuery {
                prompt: prompt.to_string(),
            })
            .await
        {
            Ok(content) => {
                println!("{content}");
            }
            Err(err) => {
                eprintln!("FileReader query failed: {err}");
            }
        }
    } else {
        info!("FileReader demo skipped by flag");
    }

    metrics.log_summary();

    if runtime.trigger_shutdown_save {
        persist_once(
            &graph_actor,
            &gateway,
            &rerun_actor,
            &graph_config.autosave_path,
            "shutdown_persist",
        )
        .await?;
    }

    Ok(())
}
