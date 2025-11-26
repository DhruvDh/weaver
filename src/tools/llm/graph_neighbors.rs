use async_trait::async_trait;
use bon::Builder;
use petgraph::visit::EdgeRef;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{
    CallState, ToolExecutionError, ToolInputError, ToolInputResult, ToolInstance, ToolOutput,
    ToolPrototype, schema_for_args,
};
use crate::{
    graph::EdgeKind,
    tools::llm::graph_tools::{GRAPH, map_graph_err},
};

const IDENTIFIER: &str = "graph_neighbors";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NeighborsArgs {
    pub slug:      String,
    #[serde(default)]
    pub edge_kind: Option<String>, // requires, supports, assesses, precedes, anchors
    #[serde(default)]
    pub direction: Option<String>, // outgoing, incoming, both
}

pub(super) fn neighbors_meta() -> ToolPrototype {
    ToolPrototype {
        id:          IDENTIFIER,
        description: "List neighbors of a node filtered by edge kind and direction.",
        schema:      schema_for_args::<NeighborsArgs>(),
        parse:       parse_neighbors,
    }
}

fn parse_neighbors(raw: Value, _state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: NeighborsArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    IDENTIFIER,
            message: err.to_string(),
        })?;
    Ok(Box::new(NeighborsTool { args }))
}

struct NeighborsTool {
    args: NeighborsArgs,
}

#[async_trait]
impl ToolInstance for NeighborsTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        use petgraph::Direction;
        let graph = GRAPH.read();
        let node = graph
            .node_by_slug(&self.args.slug)
            .map_err(|e| map_graph_err(e, IDENTIFIER))?;

        let kinds_filter = if let Some(kind) = self.args.edge_kind.as_deref() {
            let k = kind.to_ascii_lowercase();
            if ["requires", "supports", "assesses", "precedes", "anchors"].contains(&k.as_str()) {
                Some(k)
            } else {
                return Err(ToolExecutionError::Input(ToolInputError::InvalidPayload {
                    tool:    IDENTIFIER,
                    message: format!("unknown edge_kind `{}`", kind),
                }));
            }
        } else {
            None
        };
        let dir = match self.args.direction.as_deref() {
            Some("incoming") => Some(Direction::Incoming),
            Some("outgoing") => Some(Direction::Outgoing),
            _ => None, // both
        };

        let mut neighbors = Vec::new();
        let directions = match dir {
            Some(d) => vec![d],
            None => vec![Direction::Incoming, Direction::Outgoing],
        };

        for d in directions {
            for edge in graph.graph().edges_directed(node, d) {
                if let Some(kind_str) = kinds_filter.as_deref()
                    && !edge_kind_matches(kind_str, &edge.weight().kind)
                {
                    continue;
                }
                let other = if d == Direction::Outgoing {
                    edge.target()
                } else {
                    edge.source()
                };
                neighbors.push(json!({
                    "neighbor_slug": graph.graph()[other].slug,
                    "edge_kind": edge_kind_name(&edge.weight().kind),
                    "direction": if d == Direction::Outgoing { "outgoing" } else { "incoming" },
                }));
            }
        }

        Ok(ToolOutput::new(json!({"status": "ok", "neighbors": neighbors})))
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
