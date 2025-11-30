use std::{
    convert::Infallible,
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
    time::Duration,
};

use anyhow::{Context as _, Result, anyhow};
use bpaf::{OptionParser, Parser, construct, long, positional};
use kameo::{error::SendError, prelude::*};
use kameo_actors::scheduler::{Scheduler, SetInterval};
use kameo_persistence::PersistentActor;
use state::InitCell;
use tracing::{debug, error};
use tracing_subscriber::EnvFilter;
use url::Url;
use weaver::{
    constants::PRETEXT_SUBDIR,
    file_reader::{FileReader, FileReaderQuery},
    graph::{
        CurriculumGraph, GraphConfig,
        manager::{
            ApplyRuntimeConfig, GraphManager, GraphManagerState, PersistSnapshot, RedundantRequires,
        },
        persist,
    },
    llm_gateway::{GatewayMetrics, GetGatewayMetrics, LLMGateway, PersistGatewaySnapshot},
    rerun_sink::{RerunSink, RerunTarget},
};

fn log_scalar(rerun: &Option<ActorRef<RerunSink>>, path: impl Into<String>, value: f64) {
    if let Some(sink) = rerun {
        let msg = weaver::rerun_sink::LogScalar {
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

#[derive(Clone)]
struct AutosaveTick;

#[derive(Clone)]
struct PruneTick;

#[derive(Clone, Copy, Debug)]
enum RerunMode {
    Grpc,
    File,
    Both,
    None,
}

#[derive(Clone)]
struct AutosaveWorker {
    graph:   ActorRef<GraphManager>,
    gateway: Option<ActorRef<LLMGateway>>,
    rerun:   Option<ActorRef<RerunSink>>,
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
        let start = std::time::Instant::now();
        let persist: Result<(), anyhow::Error> =
            self.graph.ask(PersistSnapshot).await.map_err(|e| match e {
                SendError::HandlerError(err) => err,
                other => anyhow!(other),
            });
        let gateway_persist: Result<(), anyhow::Error> = if let Some(gateway) = &self.gateway {
            match gateway.ask(PersistGatewaySnapshot).await {
                Ok(()) => Ok(()),
                Err(SendError::HandlerError(e)) => Err(e),
                Err(e) => Err(anyhow!(e)),
            }
        } else {
            Ok(())
        };
        let ok = persist.is_ok() && gateway_persist.is_ok();

        if ok {
            debug!("graph autosave completed (persistent state only)");
        } else {
            error!(
                persist_error = persist
                    .as_ref()
                    .err()
                    .map(|e: &anyhow::Error| e.to_string()),
                gateway_error = gateway_persist
                    .as_ref()
                    .err()
                    .map(|e: &anyhow::Error| e.to_string()),
                "graph autosave failed"
            );
        }
        log_scalar(
            &self.rerun,
            "metrics/autosave/duration_ms",
            start.elapsed().as_secs_f64() * 1000.0,
        );
        log_scalar(&self.rerun, "metrics/autosave/success", if ok { 1.0 } else { 0.0 });
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
struct Cli {
    rerun_mode:             RerunMode,
    rerun_file:             PathBuf,
    workspace:              PathBuf,
    graph_snapshot_path:    PathBuf,
    graph_autosave_secs:    u64,
    graph_course_commit:    String,
    graph_strict_quality:   bool,
    graph_prune_requires_s: Option<u64>,
}

#[derive(Clone, Debug)]
struct AppConfig {
    rerun_mode:             RerunMode,
    rerun_file:             PathBuf,
    workspace:              PathBuf,
    graph_snapshot_path:    PathBuf,
    graph_autosave_secs:    u64,
    graph_course_commit:    String,
    graph_strict_quality:   bool,
    graph_prune_requires_s: Option<u64>,
}

static APP_CONFIG: InitCell<AppConfig> = InitCell::new();

fn app_config() -> &'static AppConfig {
    APP_CONFIG.get()
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

fn cli() -> OptionParser<Cli> {
    let workspace = positional::<PathBuf>("workspace")
        .help("Workspace root to expose to the assistant")
        .fallback(PathBuf::from(PRETEXT_SUBDIR));
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
        .help("Autosave interval in seconds (default 300)")
        .argument::<u64>("secs")
        .fallback(300);
    let graph_course_commit = long("graph-course-commit")
        .help("Course commit hash to embed in snapshots (default empty)")
        .argument::<String>("hash")
        .fallback(String::new());
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

    construct! {
        Cli {
            rerun_mode,
            rerun_file,
            workspace,
            graph_snapshot_path,
            graph_autosave_secs,
            graph_course_commit,
            graph_strict_quality,
            graph_prune_requires_s,
        }
    }
    .to_options()
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let Cli {
        rerun_mode,
        rerun_file,
        workspace,
        graph_snapshot_path,
        graph_autosave_secs,
        mut graph_course_commit,
        graph_strict_quality,
        graph_prune_requires_s,
    } = cli().run();

    if graph_course_commit.is_empty()
        && let Some(head) = git_head_hash()
    {
        graph_course_commit = head;
    }

    let app_cfg = AppConfig {
        rerun_mode,
        rerun_file,
        workspace,
        graph_snapshot_path,
        graph_autosave_secs,
        graph_course_commit,
        graph_strict_quality,
        graph_prune_requires_s,
    };
    APP_CONFIG.set(app_cfg.clone());
    let app_cfg = app_config().clone();

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::fmt()
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::new("%Y-%m-%d %H:%M:%S%.3f".into()))
        .with_env_filter(filter)
        .try_init()
        .map_err(|err| anyhow!("failed to initialize tracing subscriber: {err}"))?;

    debug!("FileReader demo starting");

    let mut graph_config = GraphConfig {
        course_commit:  app_cfg.graph_course_commit.clone(),
        autosave_path:  app_cfg.graph_snapshot_path.clone(),
        autosave_secs:  app_cfg.graph_autosave_secs,
        strict_quality: app_cfg.graph_strict_quality,
    };

    let snapshot_path: PathBuf = graph_config.autosave_path.clone();
    let state_root = if snapshot_path.extension().is_some() {
        snapshot_path.with_extension("state")
    } else {
        snapshot_path.clone()
    };
    let graph_state_dir = state_root.join("graph_manager");
    let gateway_state_dir = state_root.join("llm_gateway");

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
    ensure_parent_dir(&app_cfg.rerun_file)?;

    let graph_actor = match GraphManager::respawn_persistent(graph_state_url.clone()).await {
        Ok(actor) => {
            debug!(path = %graph_state_dir.display(), "restored graph manager from persistent snapshot");
            // Apply current CLI config to ensure strict_quality/course_commit match this
            // run.
            if let Err(err) = actor
                .ask(ApplyRuntimeConfig {
                    course_commit:  graph_config.course_commit.clone(),
                    strict_quality: graph_config.strict_quality,
                })
                .await
            {
                error!(error = ?err, "failed to apply runtime graph config after restore");
                std::process::exit(1);
            }
            // Persist the updated settings so subsequent restarts align.
            if let Err(err) = actor.ask(PersistSnapshot).await {
                error!(error = ?err, "failed to persist graph manager after applying runtime config");
            }
            actor
        }
        Err(err) => {
            debug!(
                error = %err,
                path = %graph_state_dir.display(),
                "graph state restore unavailable; falling back to legacy snapshot or empty graph"
            );
            let state = if snapshot_path.exists() {
                match persist::load_graph(&snapshot_path).await {
                    Ok(snapshot) => {
                        debug!(path = %snapshot_path.display(), "loaded legacy graph snapshot");
                        let course_commit = if graph_config.course_commit.is_empty() {
                            snapshot.course_commit.clone()
                        } else {
                            graph_config.course_commit.clone()
                        };
                        graph_config.course_commit = course_commit.clone();
                        GraphManagerState::new(
                            snapshot.graph,
                            course_commit,
                            graph_config.strict_quality,
                            snapshot.graph_version,
                        )
                    }
                    Err(err) => {
                        error!(
                            error = %err,
                            path = %snapshot_path.display(),
                            "failed to load graph snapshot; starting with empty graph"
                        );
                        GraphManagerState::new(
                            CurriculumGraph::default(),
                            graph_config.course_commit.clone(),
                            graph_config.strict_quality,
                            0,
                        )
                    }
                }
            } else {
                GraphManagerState::new(
                    CurriculumGraph::default(),
                    graph_config.course_commit.clone(),
                    graph_config.strict_quality,
                    0,
                )
            };
            GraphManager::spawn_persistent(graph_state_url.clone(), state).await?
        }
    };

    let (gateway, metrics) = match LLMGateway::respawn_persistent(gateway_state_url.clone()).await {
        Ok(actor) => {
            debug!(path = %gateway_state_dir.display(), "restored LLM gateway from persistent snapshot");
            let metrics = actor.ask(GetGatewayMetrics).await.unwrap_or_else(|err| {
                error!(error = ?err, "failed to fetch gateway metrics after restore");
                Arc::new(GatewayMetrics::default())
            });
            (actor, metrics)
        }
        Err(err) => {
            debug!(
                error = %err,
                path = %gateway_state_dir.display(),
                "gateway state restore unavailable; creating new gateway"
            );
            let instance = LLMGateway::from_env()?;
            let metrics = instance.metrics();
            let actor = LLMGateway::spawn_persistent(gateway_state_url.clone(), instance).await?;
            (actor, metrics)
        }
    };

    let scheduler = Scheduler::spawn(Scheduler::new());

    let mut rerun_targets = Vec::new();
    if matches!(app_cfg.rerun_mode, RerunMode::Grpc | RerunMode::Both) {
        rerun_targets.push(RerunTarget::Grpc {
            name: "weaver".into(),
        });
    }
    if matches!(app_cfg.rerun_mode, RerunMode::File | RerunMode::Both) {
        rerun_targets.push(RerunTarget::File {
            name: "weaver".into(),
            path: app_cfg.rerun_file.clone(),
        });
    }
    let rerun_actor = if rerun_targets.is_empty() {
        None
    } else {
        Some(RerunSink::spawn(rerun_targets))
    };

    // Autosave the graph periodically via scheduler to keep interval logic inside
    // the actor system.
    let autosave_worker = AutosaveWorker {
        graph:   graph_actor.clone(),
        gateway: Some(gateway.clone()),
        rerun:   rerun_actor.clone(),
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

    // Optional periodic prune of redundant requires edges.
    if let Some(prune_secs) = app_cfg.graph_prune_requires_s.filter(|value| *value > 0) {
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

    let actor = match FileReader::from_env(
        app_cfg.workspace.clone(),
        gateway.clone(),
        Arc::clone(&metrics),
        graph_actor.clone(),
        rerun_actor.clone(),
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

    let tool_names = FileReader::tool_identifiers();

    debug!(
        workspace = %actor.workspace_root().display(),
        tools = ?tool_names,
        "Initialized FileReader with LLM tool bridge for the PreTeXt project"
    );

    let reader = FileReader::spawn(actor);

    let prompt = "Summarize the key goals of the UNCC CS2 PreTeXt project. Highlight any modules \
                  in the `source/` tree that look important. Please do make effective use of the \
                  `delegate_tasks` tools for all tasks, in parallel if possible.";
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

    metrics.log_summary();

    Ok(())
}
