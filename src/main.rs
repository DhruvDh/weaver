mod adder;
mod edge_synth;
mod graph;
mod llm;
mod model;
mod node_synth;
mod summary;
mod viz;

use std::{env, fmt, path::PathBuf};

use adder::{AddEdges, AddNodes, ExportDot, GraphAdder, Inventory, Summarize};
use edge_synth::{EdgeGenerator, EdgeGeneratorConfig, GenerateEdges};
use graph::GraphStore;
use kameo::Actor;
use node_synth::{GenerateNodes, NodeGenerator, NodeGeneratorConfig};
use tokio::sync::mpsc;
use tracing::info;
use viz::Viz;

type DynError = Box<dyn std::error::Error + Send + Sync + 'static>;

#[derive(Debug)]
struct CliError(String);

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for CliError {}

#[derive(Debug, Clone)]
struct RunConfig {
    topic:             String,
    concepts:          usize,
    learning_outcomes: usize,
    target_edges:      usize,
    use_llm:           bool,
    export_dot:        Option<PathBuf>,
}

fn usage() -> &'static str {
    "Usage: weaver mvp run [--topic TEXT] [--concepts N] [--los N] [--edges N] [--use-llm \
     true|false] [--export-dot PATH]"
}

fn parse_args() -> Result<RunConfig, CliError> {
    let mut args = env::args().skip(1);

    let Some(command) = args.next() else {
        return Err(CliError(usage().to_string()));
    };

    if command != "mvp" {
        return Err(CliError(format!("unknown command '{command}'. {}", usage())));
    }

    let Some(sub) = args.next() else {
        return Err(CliError(format!("missing subcommand. {}", usage())));
    };

    if sub != "run" {
        return Err(CliError(format!("unknown subcommand '{sub}'. {}", usage())));
    }

    let mut config = RunConfig {
        topic:             "Design Recipe".to_string(),
        concepts:          25,
        learning_outcomes: 5,
        target_edges:      40,
        use_llm:           false,
        export_dot:        None,
    };

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--topic" => {
                config.topic = args
                    .next()
                    .ok_or_else(|| CliError(format!("missing value for --topic. {}", usage())))?;
            }
            "--concepts" => {
                config.concepts = parse_usize(args.next(), "--concepts")?;
            }
            "--los" => {
                config.learning_outcomes = parse_usize(args.next(), "--los")?;
            }
            "--edges" => {
                config.target_edges = parse_usize(args.next(), "--edges")?;
            }
            "--use-llm" => {
                let value = args
                    .next()
                    .ok_or_else(|| CliError(format!("missing value for --use-llm. {}", usage())))?;
                config.use_llm = parse_bool(&value)
                    .ok_or_else(|| CliError(format!("invalid boolean '{value}' for --use-llm")))?;
            }
            "--export-dot" => {
                let value = args.next().ok_or_else(|| {
                    CliError(format!("missing value for --export-dot. {}", usage()))
                })?;
                config.export_dot = Some(PathBuf::from(value));
            }
            other => {
                return Err(CliError(format!("unknown flag '{other}'. {}", usage())));
            }
        }
    }

    Ok(config)
}

fn parse_usize(value: Option<String>, flag: &str) -> Result<usize, CliError> {
    let Some(raw) = value else {
        return Err(CliError(format!("missing value for {flag}. {}", usage())));
    };
    raw.parse::<usize>()
        .map_err(|_| CliError(format!("invalid integer '{raw}' for {flag}")))
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" => Some(true),
        "false" | "0" | "no" => Some(false),
        _ => None,
    }
}

#[tokio::main]
async fn main() -> Result<(), DynError> {
    let config = match parse_args() {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("{err}");
            eprintln!("{}", usage());
            std::process::exit(1);
        }
    };

    run_mvp(config).await
}

async fn run_mvp(config: RunConfig) -> Result<(), DynError> {
    info!(topic = %config.topic, use_llm = config.use_llm, "starting run");

    let (event_tx, event_rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        Viz::new(true).run(event_rx).await;
    });

    let graph_store = GraphStore::new();
    let adder_ref = GraphAdder::spawn(GraphAdder::with_event_sender(graph_store, Some(event_tx)));

    let node_generator_ref = NodeGenerator::spawn(NodeGenerator::new(NodeGeneratorConfig {
        use_llm:                   config.use_llm,
        default_concepts:          config.concepts,
        default_learning_outcomes: config.learning_outcomes,
    }));

    let nodes = node_generator_ref
        .ask(GenerateNodes {
            concepts:          config.concepts,
            learning_outcomes: config.learning_outcomes,
        })
        .await
        .map_err(|err| -> DynError {
            Box::new(CliError(format!("failed to generate nodes: {err}")))
        })?;

    let node_decisions = adder_ref
        .ask(AddNodes(nodes))
        .await
        .map_err(|err| -> DynError { Box::new(CliError(format!("failed to add nodes: {err}"))) })?;

    let accepted_nodes = node_decisions.iter().filter(|d| d.accepted).count();
    let rejected_nodes = node_decisions.len() - accepted_nodes;
    println!(
        "Nodes accepted: {} / {} (rejected {})",
        accepted_nodes,
        node_decisions.len(),
        rejected_nodes
    );

    let inventory = adder_ref.ask(Inventory).await.map_err(|err| -> DynError {
        Box::new(CliError(format!("failed to fetch inventory: {err}")))
    })?;

    let edge_generator_ref = EdgeGenerator::spawn(EdgeGenerator::new(EdgeGeneratorConfig {
        use_llm:              config.use_llm,
        default_target_edges: config.target_edges,
    }));

    let edges = edge_generator_ref
        .ask(GenerateEdges {
            inventory:    inventory.clone(),
            target_edges: config.target_edges,
        })
        .await
        .map_err(|err| -> DynError {
            Box::new(CliError(format!("failed to generate edges: {err}")))
        })?;

    let edge_decisions = adder_ref
        .ask(AddEdges(edges))
        .await
        .map_err(|err| -> DynError { Box::new(CliError(format!("failed to add edges: {err}"))) })?;

    let accepted_edges = edge_decisions.iter().filter(|d| d.accepted).count();
    let rejected_edges = edge_decisions.len() - accepted_edges;
    println!(
        "Edges accepted: {} / {} (rejected {})",
        accepted_edges,
        edge_decisions.len(),
        rejected_edges
    );

    let summary = adder_ref.ask(Summarize).await.map_err(|err| -> DynError {
        Box::new(CliError(format!("failed to compute summary: {err}")))
    })?;

    println!(
        "Nodes: {} (concept={}, learning_outcome={})",
        summary.total_nodes, summary.concepts, summary.learning_outcomes
    );
    println!(
        "Edges: {} (prerequisite_for={}, supports={})",
        summary.total_edges, summary.prerequisite_edges, summary.supports_edges
    );
    println!(
        "Prerequisite DAG: {}",
        if summary.prerequisite_dag_ok {
            "OK"
        } else {
            "Cycle detected"
        }
    );

    if !summary.top_learning_outcomes.is_empty() {
        println!("Top learning outcomes by incoming supports:");
        for entry in summary.top_learning_outcomes.iter().take(5) {
            println!("  {} ({} supports) - {}", entry.id, entry.supports, entry.text);
        }
    }

    if let Some(path) = &config.export_dot {
        let dot = adder_ref.ask(ExportDot).await.map_err(|err| -> DynError {
            Box::new(CliError(format!("failed to export DOT: {err}")))
        })?;
        tokio::fs::write(path, dot).await?;
        println!("DOT graph written to {}", path.display());
    }

    edge_generator_ref.stop_gracefully().await.ok();
    node_generator_ref.stop_gracefully().await.ok();
    adder_ref.stop_gracefully().await.ok();

    Ok(())
}
