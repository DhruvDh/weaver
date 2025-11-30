use async_trait::async_trait;
use bon::Builder;
use petgraph::{
    algo::{
        articulation_points, bridges, dijkstra, greedy_feedback_arc_set, page_rank, tarjan_scc,
    },
    visit::EdgeRef,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::info;

use crate::{
    graph::{EdgeKind, traversal},
    tools::llm::{
        CallState, ToolExecutionError, ToolInputError, ToolInputResult, ToolInstance, ToolOutput,
        ToolPrototype, schema_for_args,
    },
};

const REQUIRES_CYCLES: &str = "graph_requires_cycles";
const REQUIRES_PAGERANK: &str = "graph_requires_pagerank";
const REQUIRES_BRIDGES: &str = "graph_requires_bridges";
const REQUIRES_ARTICULATION: &str = "graph_requires_articulation";
const REQUIRES_FEEDBACK: &str = "graph_requires_feedback_arcs";
const REQUIRES_SHORTEST_PATH: &str = "graph_requires_shortest_path";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CyclesArgs {
    #[serde(default)]
    #[schemars(description = "Maximum components to return (default 200, max 500).")]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PageRankArgs {
    #[serde(default = "default_damping")]
    #[schemars(description = "Damping factor (0..1), default 0.85.")]
    pub damping:    f64,
    #[serde(default = "default_iterations")]
    #[schemars(description = "Number of iterations, default 20.")]
    pub iterations: usize,
    #[serde(default)]
    #[schemars(description = "Maximum nodes to return, default 50, max 500.")]
    pub limit:      Option<usize>,
}

const fn default_damping() -> f64 {
    0.85
}
const fn default_iterations() -> usize {
    20
}

pub(super) fn tool_prototypes() -> Vec<ToolPrototype> {
    vec![
        ToolPrototype {
            id:          REQUIRES_CYCLES,
            description: "Detect cycles in the requires layer (returns SCCs > size 1).",
            schema:      schema_for_args::<CyclesArgs>(),
            parse:       parse_cycles,
        },
        ToolPrototype {
            id:          REQUIRES_PAGERANK,
            description: "PageRank over the requires layer (influence of knowledge nodes).",
            schema:      schema_for_args::<PageRankArgs>(),
            parse:       parse_pagerank,
        },
        ToolPrototype {
            id:          REQUIRES_BRIDGES,
            description: "Bridges (cut edges) in the requires layer.",
            schema:      schema_for_args::<CyclesArgs>(),
            parse:       parse_bridges,
        },
        ToolPrototype {
            id:          REQUIRES_ARTICULATION,
            description: "Articulation points (cut nodes) in the requires layer.",
            schema:      schema_for_args::<CyclesArgs>(),
            parse:       parse_articulation,
        },
        ToolPrototype {
            id:          REQUIRES_FEEDBACK,
            description: "Greedy feedback arc set suggestions to break requires cycles.",
            schema:      schema_for_args::<CyclesArgs>(),
            parse:       parse_feedback,
        },
        ToolPrototype {
            id:          REQUIRES_SHORTEST_PATH,
            description: "Dijkstra shortest path (requires-only, unit weights) between two slugs.",
            schema:      schema_for_args::<ShortestPathArgs>(),
            parse:       parse_shortest_path,
        },
    ]
}

fn parse_cycles(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: CyclesArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    REQUIRES_CYCLES,
            message: err.to_string(),
        })?;
    Ok(Box::new(CyclesTool {
        args,
        state: state.clone(),
    }))
}

fn parse_pagerank(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let mut args: PageRankArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    REQUIRES_PAGERANK,
            message: err.to_string(),
        })?;
    if !(0.0..=1.0).contains(&args.damping) {
        return Err(ToolInputError::InvalidPayload {
            tool:    REQUIRES_PAGERANK,
            message: "damping must be between 0 and 1".into(),
        });
    }
    if args.iterations == 0 {
        args.iterations = default_iterations();
    }
    Ok(Box::new(PageRankTool {
        args,
        state: state.clone(),
    }))
}

fn parse_bridges(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: CyclesArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    REQUIRES_BRIDGES,
            message: err.to_string(),
        })?;
    Ok(Box::new(BridgesTool {
        args,
        state: state.clone(),
    }))
}

fn parse_articulation(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: CyclesArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    REQUIRES_ARTICULATION,
            message: err.to_string(),
        })?;
    Ok(Box::new(ArticulationTool {
        args,
        state: state.clone(),
    }))
}

fn parse_feedback(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: CyclesArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    REQUIRES_FEEDBACK,
            message: err.to_string(),
        })?;
    Ok(Box::new(FeedbackTool {
        args,
        state: state.clone(),
    }))
}

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ShortestPathArgs {
    pub from_slug: String,
    pub to_slug:   String,
}

fn parse_shortest_path(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: ShortestPathArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    REQUIRES_SHORTEST_PATH,
            message: err.to_string(),
        })?;
    Ok(Box::new(ShortestPathTool {
        args,
        state: state.clone(),
    }))
}

struct CyclesTool {
    args:  CyclesArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for CyclesTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = self
            .state
            .graph
            .ask(crate::graph::manager::GetGraph)
            .await
            .map_err(super::common::map_send_err_inf)?;

        let view = traversal::requires_view(&graph);
        let mut sccs: Vec<Vec<_>> = tarjan_scc(&view)
            .into_iter()
            .filter(|c| c.len() > 1)
            .collect();
        sccs.sort_by_key(|component| std::cmp::Reverse(component.len()));
        let limit = self.args.limit.unwrap_or(200).min(500);
        let items: Vec<_> = sccs
            .into_iter()
            .take(limit)
            .map(|comp| {
                json!({
                    "size": comp.len(),
                    "slugs": comp.into_iter().map(|n| graph[n].slug.clone()).collect::<Vec<_>>()
                })
            })
            .collect();

        info!(tool = REQUIRES_CYCLES, count = items.len(), "graph requires cycles");

        Ok(ToolOutput::new(json!({
            "type": "graph_analysis",
            "tool": REQUIRES_CYCLES,
            "components": items,
        })))
    }
}

struct PageRankTool {
    args:  PageRankArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for PageRankTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = self
            .state
            .graph
            .ask(crate::graph::manager::GetGraph)
            .await
            .map_err(super::common::map_send_err_inf)?;

        let view = traversal::requires_view(&graph);
        let scores = page_rank(&view, self.args.damping, self.args.iterations);
        let mut items: Vec<_> = scores
            .iter()
            .enumerate()
            .map(|(idx, score)| (graph[petgraph::graph::NodeIndex::new(idx)].slug.clone(), *score))
            .collect();
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let limit = self.args.limit.unwrap_or(50).min(500);
        let payload: Vec<_> = items
            .into_iter()
            .take(limit)
            .map(|(slug, score)| json!({ "slug": slug, "score": score }))
            .collect();

        info!(
            tool = REQUIRES_PAGERANK,
            limit = limit,
            iter = self.args.iterations,
            damping = self.args.damping,
            "graph requires pagerank"
        );

        Ok(ToolOutput::new(json!({
            "type": "graph_analysis",
            "tool": REQUIRES_PAGERANK,
            "items": payload,
        })))
    }
}

struct BridgesTool {
    args:  CyclesArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for BridgesTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = self
            .state
            .graph
            .ask(crate::graph::manager::GetGraph)
            .await
            .map_err(super::common::map_send_err_inf)?;
        // Build an unweighted temp graph to satisfy trait bounds.
        let mut temp = petgraph::graph::Graph::<(), (), petgraph::Directed>::with_capacity(
            graph.node_count(),
            graph.edge_count(),
        );
        let mut idx_map = Vec::with_capacity(graph.node_count());
        for n in graph.node_indices() {
            let idx = temp.add_node(());
            idx_map.push((n, idx));
        }
        for e in graph.edge_indices() {
            if let EdgeKind::Requires(_) = graph[e].kind
                && let Some((u, v)) = graph.edge_endpoints(e)
            {
                let u_idx = idx_map.iter().find(|(orig, _)| *orig == u).unwrap().1;
                let v_idx = idx_map.iter().find(|(orig, _)| *orig == v).unwrap().1;
                temp.add_edge(u_idx, v_idx, ());
            }
        }

        let mut list: Vec<_> = bridges(&temp)
            .map(|e| {
                let (u, v) = (e.source(), e.target());
                let from_slug = graph[idx_map.iter().find(|(_, idx)| *idx == u).unwrap().0]
                    .slug
                    .clone();
                let to_slug = graph[idx_map.iter().find(|(_, idx)| *idx == v).unwrap().0]
                    .slug
                    .clone();
                json!({"from": from_slug, "to": to_slug})
            })
            .collect();
        let limit = self.args.limit.unwrap_or(200).min(500);
        if list.len() > limit {
            list.truncate(limit);
        }
        info!(tool = REQUIRES_BRIDGES, count = list.len(), "graph requires bridges");
        Ok(ToolOutput::new(
            json!({"type": "graph_analysis","tool": REQUIRES_BRIDGES,"edges": list}),
        ))
    }
}

struct ArticulationTool {
    args:  CyclesArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for ArticulationTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = self
            .state
            .graph
            .ask(crate::graph::manager::GetGraph)
            .await
            .map_err(super::common::map_send_err_inf)?;
        let mut temp = petgraph::graph::Graph::<(), (), petgraph::Directed>::with_capacity(
            graph.node_count(),
            graph.edge_count(),
        );
        let mut idx_map = Vec::with_capacity(graph.node_count());
        for n in graph.node_indices() {
            let idx = temp.add_node(());
            idx_map.push((n, idx));
        }
        for e in graph.edge_indices() {
            if let EdgeKind::Requires(_) = graph[e].kind
                && let Some((u, v)) = graph.edge_endpoints(e)
            {
                let u_idx = idx_map.iter().find(|(orig, _)| *orig == u).unwrap().1;
                let v_idx = idx_map.iter().find(|(orig, _)| *orig == v).unwrap().1;
                temp.add_edge(u_idx, v_idx, ());
            }
        }

        let mut nodes: Vec<_> = articulation_points::articulation_points(&temp)
            .into_iter()
            .map(|n| {
                let orig = idx_map.iter().find(|(_, idx)| *idx == n).unwrap().0;
                graph[orig].slug.clone()
            })
            .collect();
        let limit = self.args.limit.unwrap_or(200).min(500);
        if nodes.len() > limit {
            nodes.truncate(limit);
        }
        info!(tool = REQUIRES_ARTICULATION, count = nodes.len(), "graph requires articulation");
        Ok(ToolOutput::new(
            json!({"type": "graph_analysis","tool": REQUIRES_ARTICULATION,"nodes": nodes}),
        ))
    }
}

struct FeedbackTool {
    args:  CyclesArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for FeedbackTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = self
            .state
            .graph
            .ask(crate::graph::manager::GetGraph)
            .await
            .map_err(super::common::map_send_err_inf)?;
        let view = traversal::requires_view(&graph);
        let set = greedy_feedback_arc_set(&view);
        let mut edges: Vec<_> = set
            .into_iter()
            .map(|e| {
                let (u, v) = (e.source(), e.target());
                json!({"from": graph[u].slug.clone(), "to": graph[v].slug.clone()})
            })
            .collect();
        let limit = self.args.limit.unwrap_or(200).min(500);
        if edges.len() > limit {
            edges.truncate(limit);
        }
        info!(tool = REQUIRES_FEEDBACK, count = edges.len(), "graph requires feedback arcs");
        Ok(ToolOutput::new(
            json!({"type": "graph_analysis","tool": REQUIRES_FEEDBACK,"edges": edges}),
        ))
    }
}

struct ShortestPathTool {
    args:  ShortestPathArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for ShortestPathTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph_ref = self
            .state
            .graph
            .ask(crate::graph::manager::GetGraph)
            .await
            .map_err(super::common::map_send_err_inf)?;
        // resolve slugs
        let from = graph_ref
            .node_indices()
            .find(|&n| graph_ref[n].slug == self.args.from_slug)
            .ok_or_else(|| {
                ToolExecutionError::Input(ToolInputError::InvalidPayload {
                    tool:    REQUIRES_SHORTEST_PATH,
                    message: format!("slug `{}` not found", self.args.from_slug),
                })
            })?;
        let to = graph_ref
            .node_indices()
            .find(|&n| graph_ref[n].slug == self.args.to_slug)
            .ok_or_else(|| {
                ToolExecutionError::Input(ToolInputError::InvalidPayload {
                    tool:    REQUIRES_SHORTEST_PATH,
                    message: format!("slug `{}` not found", self.args.to_slug),
                })
            })?;

        let view = traversal::requires_view(&graph_ref);
        let dist = dijkstra(&view, from, Some(to), |_| 1usize);
        let cost = dist.get(&to).copied();

        let path = traversal::requires_one_path(&graph_ref, from, to)
            .unwrap_or_default()
            .into_iter()
            .map(|n| graph_ref[n].slug.clone())
            .collect::<Vec<_>>();

        info!(
            tool = REQUIRES_SHORTEST_PATH,
            from = %self.args.from_slug,
            to = %self.args.to_slug,
            cost = ?cost,
            "graph requires shortest path"
        );

        Ok(ToolOutput::new(json!({
            "type": "graph_analysis",
            "tool": REQUIRES_SHORTEST_PATH,
            "cost": cost,
            "path": path,
        })))
    }
}
