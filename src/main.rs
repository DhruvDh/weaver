use std::{env, path::PathBuf, sync::Arc, time::Duration};

use anyhow::{Result, anyhow};
use bpaf::{OptionParser, Parser, construct, positional};
use kameo::prelude::*;
use tracing::{debug, error};
use tracing_subscriber::EnvFilter;
use weaver::{
    constants::PRETEXT_SUBDIR,
    file_reader::{FileReader, FileReaderQuery},
    graph::{
        GraphConfig, GraphService,
        manager::{GraphManager, SaveSnapshot},
        persist,
    },
    llm_gateway::LLMGateway,
};

#[derive(Clone, Debug)]
struct Cli {
    workspace: PathBuf,
}

fn cli() -> OptionParser<Cli> {
    let workspace = positional::<PathBuf>("workspace")
        .help("Workspace root to expose to the assistant")
        .fallback(PathBuf::from(PRETEXT_SUBDIR));
    construct! { Cli { workspace } }.to_options()
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let Cli { workspace } = cli().run();
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

    // Autosave the graph periodically to avoid data loss.
    let graph_actor_for_save = graph_actor.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(graph_config.autosave_secs));
        let path = graph_config.autosave_path.clone();
        loop {
            interval.tick().await;
            match graph_actor_for_save
                .ask(SaveSnapshot { path: path.clone() })
                .await
            {
                Ok(()) => debug!("graph autosave completed"),
                Err(send_err) => error!(error = ?send_err, "graph autosave failed"),
            }
        }
    });

    let actor = match FileReader::from_env(
        workspace,
        gateway.clone(),
        Arc::clone(&metrics),
        graph_actor.clone(),
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
