use std::{convert::Infallible, env, path::PathBuf, sync::Arc, time::Duration};

use anyhow::{Result, anyhow};
use bpaf::{OptionParser, Parser, construct, long, positional};
use kameo::prelude::*;
use kameo_actors::scheduler::{Scheduler, SetInterval};
use tracing::{debug, error};
use tracing_subscriber::EnvFilter;
use weaver::{
    constants::PRETEXT_SUBDIR,
    file_reader::{FileReader, FileReaderQuery},
    graph::{
        GraphConfig, GraphService,
        manager::{GraphManager, RedundantRequires, SaveSnapshot},
        persist,
    },
    llm_gateway::LLMGateway,
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
    graph: ActorRef<GraphManager>,
    path:  PathBuf,
    rerun: Option<ActorRef<RerunSink>>,
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
        let path = self.path.clone();
        match self.graph.ask(SaveSnapshot { path }).await {
            Ok(()) => {
                debug!("graph autosave completed");
                log_scalar(
                    &self.rerun,
                    "metrics/autosave/duration_ms",
                    start.elapsed().as_secs_f64() * 1000.0,
                );
                log_scalar(&self.rerun, "metrics/autosave/success", 1.0);
            }
            Err(err) => {
                error!(error = ?err, "graph autosave failed");
                log_scalar(
                    &self.rerun,
                    "metrics/autosave/duration_ms",
                    start.elapsed().as_secs_f64() * 1000.0,
                );
                log_scalar(&self.rerun, "metrics/autosave/success", 0.0);
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
struct Cli {
    rerun_mode: RerunMode,
    rerun_file: PathBuf,
    workspace:  PathBuf,
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
    construct! { Cli { rerun_mode, rerun_file, workspace } }.to_options()
}

fn env_override_mode(default: RerunMode) -> RerunMode {
    match env::var("WEAVER_RERUN_MODE").ok().as_deref() {
        Some("grpc") => RerunMode::Grpc,
        Some("file") => RerunMode::File,
        Some("both") => RerunMode::Both,
        Some("none") => RerunMode::None,
        _ => default,
    }
}

fn env_override_file(default: PathBuf) -> PathBuf {
    env::var("WEAVER_RERUN_FILE")
        .ok()
        .map(PathBuf::from)
        .unwrap_or(default)
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let Cli {
        rerun_mode,
        rerun_file,
        workspace,
    } = cli().run();
    // Env vars act as defaults; CLI takes precedence when provided.
    let rerun_mode = env_override_mode(rerun_mode);
    let rerun_file = env_override_file(rerun_file);
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::fmt()
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::new("%Y-%m-%d %H:%M:%S%.3f".into()))
        .with_env_filter(filter)
        .try_init()
        .map_err(|err| anyhow!("failed to initialize tracing subscriber: {err}"))?;

    debug!("FileReader demo starting");

    let course_commit = env::var("GRAPH_COURSE_COMMIT").unwrap_or_default();
    let autosave_path =
        env::var("GRAPH_SNAPSHOT_PATH").unwrap_or_else(|_| "graph_snapshot.json".to_string());
    let autosave_secs = env::var("GRAPH_AUTOSAVE_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);
    let mut graph_config = GraphConfig {
        course_commit: course_commit.clone(),
        autosave_path: autosave_path.clone().into(),
        autosave_secs,
    };

    // Load snapshot first (if present) to avoid autosaving an empty graph.
    let snapshot_path: PathBuf = autosave_path.clone().into();
    let loaded_snapshot = if snapshot_path.exists() {
        match persist::load_graph(&snapshot_path).await {
            Ok(snapshot) => {
                debug!(path = %snapshot_path.display(), "loaded existing graph snapshot");
                Some(snapshot)
            }
            Err(err) => {
                error!(
                    error = %err,
                    path = %snapshot_path.display(),
                    "failed to load graph snapshot; starting with empty graph"
                );
                None
            }
        }
    } else {
        None
    };

    let service = if let Some(snapshot) = loaded_snapshot {
        graph_config.course_commit = snapshot.course_commit.clone();
        GraphService::from_graph(snapshot.graph)
    } else {
        GraphService::new()
    };

    let gateway_instance = LLMGateway::from_env()?;
    let metrics = gateway_instance.metrics();
    let gateway = LLMGateway::spawn(gateway_instance);
    let graph_actor = GraphManager::spawn(GraphManager::new(service, graph_config.clone()));
    let scheduler = Scheduler::spawn(Scheduler::new());

    let mut rerun_targets = Vec::new();
    if matches!(rerun_mode, RerunMode::Grpc | RerunMode::Both) {
        rerun_targets.push(RerunTarget::Grpc {
            name: "weaver".into(),
        });
    }
    if matches!(rerun_mode, RerunMode::File | RerunMode::Both) {
        rerun_targets.push(RerunTarget::File {
            name: "weaver".into(),
            path: rerun_file.clone(),
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
        graph: graph_actor.clone(),
        path:  graph_config.autosave_path.clone(),
        rerun: rerun_actor.clone(),
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
    if let Some(prune_secs) = env::var("GRAPH_PRUNE_REQUIRES_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
    {
        if prune_secs > 0 {
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
    }

    let actor = match FileReader::from_env(
        workspace,
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
