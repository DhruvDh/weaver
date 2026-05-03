use std::{
    convert::Infallible,
    fs,
    io::{IsTerminal, Write, stderr},
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::{Context as _, Result, anyhow, bail};
use bpaf::{OptionParser, Parser, construct, long, positional};
use chrono::Local;
use futures::future::BoxFuture;
use kameo::{error::SendError, prelude::*};
use kameo_actors::scheduler::{Scheduler, SetInterval};
use kameo_persistence::PersistentActor;
use tokio::{
    sync::mpsc,
    task::JoinHandle,
    time::{sleep, timeout},
};
use tracing::{debug, error, info, warn};
use tracing_subscriber::{
    EnvFilter,
    fmt::writer::{BoxMakeWriter, MakeWriterExt},
};
use url::Url;

use crate::{
    agents::deduplication::{DeduplicationAgent, RunDeduplication},
    constants::PRETEXT_SUBDIR,
    file_reader::{AgentMode, AnalystReader, FileReader, FileReaderQuery, ModeSpec, Reader},
    graph::{
        CurriculumGraph, GraphConfig,
        audit::{FanoutMutationSink, RerunMutationSink},
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
    ui::{BackendEvent, UiAction},
};

pub mod two_phase;

pub use two_phase::run_two_phase_construction;

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
    pub rerun_file:                  Option<PathBuf>,
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
    pub interactive_writable:        bool,
    pub analyst:                     bool,
    pub harvest_timeout_hours:       Option<f64>,
    pub weave_timeout_hours:         Option<f64>,
    pub max_concurrent_chapters:     usize,
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

struct EventLogWriter {
    tx:              mpsc::UnboundedSender<BackendEvent>,
    also_stderr:     bool,
    stderr_fallback: std::io::Stderr,
}

impl std::io::Write for EventLogWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if let Ok(text) = std::str::from_utf8(buf) {
            for line in text.split('\n') {
                if !line.trim().is_empty() {
                    let _ = self.tx.send(BackendEvent::Log(line.trim_end().to_string()));
                    if self.also_stderr {
                        let _ = writeln!(self.stderr_fallback, "{line}");
                    }
                }
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if self.also_stderr {
            self.stderr_fallback.flush()
        } else {
            Ok(())
        }
    }
}

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

fn default_rerun_file() -> PathBuf {
    let now = Local::now();
    let pid = std::process::id();
    PathBuf::from(format!("weaver-{}-p{pid}.rrd", now.format("%Y%m%d-%H%M%S")))
}

pub fn cli() -> OptionParser<Cli> {
    let rerun_file = long("rerun-file")
        .help(
            "Path to write the Rerun .rrd (live viewer always enabled; default \
             weaver-YYYYMMDD-HHMMSS-pppp.rrd)",
        )
        .argument::<PathBuf>("path")
        .optional();
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
        .help("Skip startup interactive initialization (useful for tests or headless runs)")
        .switch();
    let interactive = long("interactive")
        .help("Launch the interactive TUI (read-only analyst mode by default)")
        .switch();
    let interactive_writable = long("interactive-writable")
        .help(
            "Enable writable interactive mode instead of the read-only analyst (use with \
             --interactive)",
        )
        .switch();
    let analyst = long("analyst")
        .help("Shortcut to launch the interactive analyst TUI (same as --interactive)")
        .switch();
    let harvest_timeout_hours = long("harvest-timeout-hours")
        .help("Two-phase: maximum hours to spend harvesting before proceeding to weaving")
        .argument::<f64>("hours")
        .optional();
    let weave_timeout_hours = long("weave-timeout-hours")
        .help("Two-phase: maximum hours to spend weaving before finishing")
        .argument::<f64>("hours")
        .optional();
    let max_concurrent_chapters = long("max-concurrent-chapters")
        .help("Two-phase: maximum chapters to process in parallel (default 4)")
        .argument::<usize>("count")
        .fallback(4);
    let chapters_pattern = long("chapters")
        .help(
            "Glob pattern for chapter entrypoints (defaults to source/*/toctree.ptx when omitted)",
        )
        .argument::<String>("pattern")
        .optional();
    let chapters_dir = long("chapters-dir")
        .help("Directory containing chapter files; expands to all toctree.ptx within (recursive)")
        .argument::<PathBuf>("dir")
        .optional();
    let workspace = positional::<PathBuf>("workspace")
        .help("Workspace root to expose to the assistant")
        .fallback(PathBuf::from(PRETEXT_SUBDIR));

    construct! {
        Cli {
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
            interactive_writable,
            analyst,
            harvest_timeout_hours,
            weave_timeout_hours,
            max_concurrent_chapters,
            chapters_pattern,
            chapters_dir,
            workspace,
        }
    }
    .to_options()
}

pub fn interactive_session_mode(cli: &Cli) -> Option<AgentMode> {
    if !(cli.interactive || cli.analyst) {
        return None;
    }

    if cli.interactive_writable {
        Some(AgentMode::Interactive)
    } else {
        Some(AgentMode::Analyst)
    }
}

fn mode_label(mode: AgentMode) -> &'static str {
    match mode {
        AgentMode::Analyst => "Analyst",
        AgentMode::Interactive => "FileReader",
        AgentMode::Harvester => "Harvester",
        AgentMode::Weaver => "Weaver",
    }
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
    source_root: Option<PathBuf>,
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
            source_root: source_root.clone(),
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
                        source_root: source_root.clone(),
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
    skip_restore: bool,
) -> Result<(ActorRef<LLMGateway>, Arc<GatewayMetrics>)> {
    if !skip_restore && let Ok(actor) = LLMGateway::respawn_persistent(state_url.clone()).await {
        debug!(path = %state_url, "restored LLM gateway from persistent snapshot");
        let metrics = actor.ask(GetGatewayMetrics).await.unwrap_or_else(|err| {
            error!(error = ?err, "failed to fetch gateway metrics after restore");
            Arc::new(GatewayMetrics::default())
        });
        return Ok((actor, metrics));
    }

    debug!(
        skip_restore,
        path = %state_url,
        "gateway state restore unavailable or skipped; starting fresh"
    );
    match mode {
        GatewayMode::Real => {
            let instance = LLMGateway::from_env()?;
            let metrics = instance.metrics();
            let actor = LLMGateway::spawn_persistent(state_url.clone(), instance).await?;
            Ok((actor, metrics))
        }
        GatewayMode::Stub => {
            let instance =
                LLMGateway::from(LLMGatewayState::new(GatewayMetrics::default().to_state()));
            let metrics = instance.metrics();
            let actor = LLMGateway::spawn_persistent(state_url.clone(), instance).await?;
            Ok((actor, metrics))
        }
    }
}

async fn stream_reply_chunks(tx: mpsc::UnboundedSender<BackendEvent>, content: String) {
    if content.trim().is_empty() {
        let _ = tx.send(BackendEvent::RequestComplete);
        return;
    }

    for chunk in content.split_inclusive(|c: char| c.is_whitespace()) {
        let _ = tx.send(BackendEvent::TokenChunk(chunk.to_string()));
        sleep(Duration::from_millis(18)).await;
    }
    let _ = tx.send(BackendEvent::RequestComplete);
}

async fn interactive_backend<M: ModeSpec>(
    reader: ActorRef<Reader<M>>,
    mut action_rx: mpsc::UnboundedReceiver<UiAction>,
    event_tx: mpsc::UnboundedSender<BackendEvent>,
) {
    let mut inflight: Option<JoinHandle<()>> = None;

    while let Some(action) = action_rx.recv().await {
        match action {
            UiAction::SendMessage(prompt) => {
                if let Some(handle) = inflight.take() {
                    handle.abort();
                }
                let reader = reader.clone();
                let tx = event_tx.clone();
                let handle = tokio::spawn(async move {
                    let result =
                        timeout(Duration::from_secs(60), reader.ask(FileReaderQuery { prompt }))
                            .await;
                    match result {
                        Ok(Ok(content)) => stream_reply_chunks(tx, content).await,
                        Ok(Err(err)) => {
                            let _ = tx.send(BackendEvent::Error(err.to_string()));
                        }
                        Err(_) => {
                            let _ = tx.send(BackendEvent::Error(
                                "Request timed out after 60s; try again or simplify.".into(),
                            ));
                        }
                    }
                });
                inflight = Some(handle);
            }
            UiAction::CancelRequest => {
                if let Some(handle) = inflight.take() {
                    handle.abort();
                }
            }
            UiAction::Exit => {
                if let Some(handle) = inflight.take() {
                    handle.abort();
                }
                break;
            }
        }
    }

    if let Some(handle) = inflight {
        handle.abort();
    }
}

pub async fn run_app(cli: Cli, runtime: RuntimeOptions) -> Result<()> {
    if cli.graph_autosave_secs == 0 {
        bail!("--graph-autosave-secs must be at least 1 second");
    }
    let session_mode = interactive_session_mode(&cli);
    let interactive_requested = session_mode.is_some();
    let interactive_tui = interactive_requested && std::io::stdout().is_terminal();
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let log_tui_to_stderr = std::env::var("WEAVER_TUI_LOG_STDERR").is_ok();
    let (action_tx, action_rx) = mpsc::unbounded_channel();
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let event_tx_for_logs = event_tx.clone();
    let log_path = {
        let candidate = PathBuf::from("logs/weaver.log");
        let resolved = if let Some(parent) = candidate.parent() {
            if parent.exists() && !parent.is_dir() {
                PathBuf::from("weaver.log")
            } else {
                candidate
            }
        } else {
            candidate
        };
        ensure_parent_dir(&resolved)?;
        resolved
    };
    let make_file_writer = |path: PathBuf| -> BoxMakeWriter {
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .or_else(|err| {
                let _ = writeln!(
                    stderr(),
                    "failed to open log file {}: {err}; retrying create",
                    path.display()
                );
                fs::File::create(&path)
            })
            .or_else(|err| {
                let fallback = std::env::temp_dir().join("weaver.log");
                let _ = writeln!(
                    stderr(),
                    "logging fallback to {} after error {err}",
                    fallback.display()
                );
                fs::File::create(fallback)
            });

        let file = match file {
            Ok(f) => f,
            Err(err) => {
                let _ = writeln!(stderr(), "logging to stderr only: {err}");
                // Return a writer factory that always writes to stderr.
                return BoxMakeWriter::new(|| -> Box<dyn Write + Send + Sync> {
                    Box::new(stderr())
                });
            }
        };

        let shared = Arc::new(Mutex::new(file));
        BoxMakeWriter::new(move || -> Box<dyn Write + Send + Sync> {
            match shared.lock() {
                Ok(guard) => match guard.try_clone() {
                    Ok(cloned) => Box::new(cloned),
                    Err(err) => {
                        let _ = writeln!(
                            stderr(),
                            "failed to clone log file {}: {err}; logging to stderr",
                            path.display()
                        );
                        Box::new(stderr())
                    }
                },
                Err(poisoned) => {
                    let _ = writeln!(stderr(), "log writer mutex poisoned; logging to stderr");
                    drop(poisoned);
                    Box::new(stderr())
                }
            }
        })
    };

    if interactive_tui {
        let file_writer = make_file_writer(log_path.clone());
        let event_writer = BoxMakeWriter::new(move || EventLogWriter {
            tx:              event_tx_for_logs.clone(),
            also_stderr:     log_tui_to_stderr,
            stderr_fallback: stderr(),
        });
        let writer = event_writer.and(file_writer);
        let _ = tracing_subscriber::fmt()
            .with_timer(tracing_subscriber::fmt::time::ChronoLocal::new(
                "%Y-%m-%d %H:%M:%S%.3f".into(),
            ))
            .with_env_filter(filter.clone())
            .with_writer(writer)
            .try_init();
    } else if let Err(err) = tracing_subscriber::fmt()
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::new("%Y-%m-%d %H:%M:%S%.3f".into()))
        .with_env_filter(filter)
        .with_writer(BoxMakeWriter::new(stderr).and(make_file_writer(log_path.clone())))
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
    let rerun_file = cli.rerun_file.clone().unwrap_or_else(default_rerun_file);
    ensure_parent_dir(&rerun_file)?;

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
                        .with_source_root(cli.workspace.clone())
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
                        .with_source_root(cli.workspace.clone())
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
                .with_source_root(cli.workspace.clone())
            };
            GraphManager::spawn_persistent(graph_state_url.clone(), state).await?
        }
    };
    if let Err(err) = graph_actor.wait_for_startup_result().await {
        error!(error = ?err, "graph manager failed during startup");
        std::process::exit(1);
    }

    let (gateway, metrics) =
        spawn_gateway(gateway_state_url.clone(), runtime.gateway_mode, false).await?;

    let scheduler = Scheduler::spawn(Scheduler::new());

    let rerun_targets = vec![
        RerunTarget::Grpc {
            name: "weaver".into(),
        },
        RerunTarget::File {
            name: "weaver".into(),
            path: rerun_file.clone(),
        },
    ];
    let rerun_actor = Some(RerunSink::spawn(rerun_targets));
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
    if let Err(err) = scheduler.tell(autosave_interval).await {
        warn!(error = ?err, "scheduler actor not running; autosave disabled");
    }

    if let Some(prune_secs) = cli.graph_prune_requires_s.filter(|value| *value > 0) {
        let prune_worker = PruneWorker {
            graph: graph_actor.clone(),
            rerun: rerun_actor.clone(),
        };
        let prune_ref = PruneWorker::spawn(prune_worker);
        let prune_interval =
            SetInterval::new(prune_ref.downgrade(), Duration::from_secs(prune_secs), PruneTick);
        if let Err(err) = scheduler.tell(prune_interval).await {
            warn!(error = ?err, "scheduler actor not running; skipping prune task");
        }
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
        if let Err(err) = scheduler.tell(dedup_task).await {
            warn!(error = ?err, "scheduler actor not running; skipping dedup task");
        }
    }

    let resolved_commit = reconcile_course_commit(
        &graph_actor,
        desired_course_commit.clone(),
        cli.graph_strict_quality,
        graph_config.validation_timeout_ms,
        Some(cli.workspace.clone()),
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

    if runtime.trigger_initial_save
        && let Err(err) = persist_once(
            &graph_actor,
            &gateway,
            &rerun_actor,
            &graph_config.autosave_path,
            "initial_autosave",
        )
        .await
    {
        log_scalar(&rerun_actor, "metrics/initial_autosave/failure", 1.0);
        eprintln!(
            "initial autosave failed (continuing): {err}. Try rerunning with --skip-demo or \
             --graph-strict-quality for more detail."
        );
        warn!(
            error = %err,
            "initial autosave failed; continuing without persisted snapshot"
        );
    }

    if interactive_tui {
        let Some(mode) = session_mode else {
            let err = anyhow!(
                "interactive TUI requested but session mode could not be determined; pass \
                 --interactive or --analyst"
            );
            error!(error = %err, "interactive mode missing");
            return Err(err);
        };
        match mode {
            AgentMode::Analyst => {
                let actor = match AnalystReader::from_env_with_limit(
                    cli.workspace.clone(),
                    gateway.clone(),
                    Arc::clone(&metrics),
                    graph_actor.clone(),
                    dedup_agent.clone(),
                    rerun_actor.clone(),
                    0,
                    resolved_commit.clone(),
                ) {
                    Ok(actor) => actor,
                    Err(err) => {
                        error!(
                            error = %err,
                            "Failed to initialize Analyst; set OPENAI_MODEL to enable LLM tools"
                        );
                        return Err(err);
                    }
                };

                let tool_names = AnalystReader::tool_identifiers().map_err(|err| {
                    error!(error = %err, "Failed to build LLM tool registry");
                    err
                })?;

                debug!(
                    mode = mode_label(AgentMode::Analyst),
                    workspace = %actor.workspace_root().display(),
                    tools = ?tool_names,
                    "Initialized interactive analyst with LLM tool bridge for the PreTeXt project"
                );

                let reader = AnalystReader::spawn(actor);
                let backend =
                    tokio::spawn(interactive_backend(reader, action_rx, event_tx.clone()));

                let ui_result = {
                    let terminal = ratatui::init();
                    let mut terminal = terminal;
                    terminal.clear()?;
                    let result =
                        crate::ui::tui::run(terminal, action_tx.clone(), event_rx, None).await;
                    ratatui::restore();
                    result
                };

                let _ = action_tx.send(UiAction::Exit);
                if let Err(err) = backend.await {
                    warn!(error = %err, "interactive analyst backend task failed");
                }

                ui_result?;
            }
            AgentMode::Interactive => {
                let actor = match FileReader::from_env_with_limit(
                    cli.workspace.clone(),
                    gateway.clone(),
                    Arc::clone(&metrics),
                    graph_actor.clone(),
                    dedup_agent.clone(),
                    rerun_actor.clone(),
                    0,
                    resolved_commit.clone(),
                ) {
                    Ok(actor) => actor,
                    Err(err) => {
                        error!(
                            error = %err,
                            "Failed to initialize writable interactive reader; set OPENAI_MODEL to \
                             enable LLM tools"
                        );
                        return Err(err);
                    }
                };

                let tool_names = FileReader::tool_identifiers().map_err(|err| {
                    error!(error = %err, "Failed to build LLM tool registry");
                    err
                })?;

                debug!(
                    mode = mode_label(AgentMode::Interactive),
                    workspace = %actor.workspace_root().display(),
                    tools = ?tool_names,
                    "Initialized interactive reader with LLM tool bridge for the PreTeXt project"
                );

                let reader = FileReader::spawn(actor);
                let backend =
                    tokio::spawn(interactive_backend(reader, action_rx, event_tx.clone()));

                let ui_result = {
                    let terminal = ratatui::init();
                    let mut terminal = terminal;
                    terminal.clear()?;
                    let result =
                        crate::ui::tui::run(terminal, action_tx.clone(), event_rx, None).await;
                    ratatui::restore();
                    result
                };

                let _ = action_tx.send(UiAction::Exit);
                if let Err(err) = backend.await {
                    warn!(error = %err, "interactive backend task failed");
                }

                ui_result?;
            }
            _ => unreachable!("only analyst or interactive modes are valid for the TUI"),
        }
    } else if interactive_requested && !cli.skip_demo {
        // Preserve validation behavior: interactive without a TTY still validates
        // the configuration so missing OPENAI_MODEL fails fast.
        match session_mode {
            Some(AgentMode::Analyst) => {
                AnalystReader::from_env(
                    cli.workspace.clone(),
                    gateway.clone(),
                    Arc::clone(&metrics),
                    graph_actor.clone(),
                    dedup_agent.clone(),
                    rerun_actor.clone(),
                    resolved_commit.clone(),
                )?;
                info!(
                    mode = mode_label(AgentMode::Analyst),
                    "Interactive analyst mode requested but stdout is not a TTY; skipping TUI \
                     session"
                );
            }
            Some(AgentMode::Interactive) => {
                FileReader::from_env(
                    cli.workspace.clone(),
                    gateway.clone(),
                    Arc::clone(&metrics),
                    graph_actor.clone(),
                    dedup_agent.clone(),
                    rerun_actor.clone(),
                    resolved_commit.clone(),
                )?;
                info!(
                    mode = mode_label(AgentMode::Interactive),
                    "Writable interactive mode requested but stdout is not a TTY; skipping TUI \
                     session"
                );
            }
            _ => {}
        }
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
