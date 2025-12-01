use std::sync::Arc;

use anyhow::{Result as AnyResult, anyhow};
use async_trait::async_trait;
use bon::Builder;
use kameo::prelude::ActorRef;
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

use super::common::{attach_meta, graph_meta, parse_args_with_builder};
use crate::{
    graph::{CurriculumGraph, EdgeKind, traversal},
    tools::llm::{
        CallState, ToolExecutionError, ToolInputError, ToolInputResult, ToolInstance, ToolOutput,
        ToolPayloadMode, ToolPrototype,
        analysis_cache::{AnalysisCacheKey, AnalysisKind},
        apply_preview_cost, build_cost_preview, estimate_tokens_from_characters,
        payload_size_bytes, prepare_payload_estimates, schema_for_args,
    },
};

async fn load_graph_with_version(
    graph: &ActorRef<crate::graph::manager::GraphManager>,
    cache: &crate::tools::llm::graph_tools::analysis_cache::AnalysisCache,
) -> Result<(Arc<CurriculumGraph>, u64), ToolExecutionError> {
    let (g, version): (Arc<CurriculumGraph>, u64) = graph
        .ask(crate::graph::manager::GetGraphWithVersion)
        .await
        .map_err(super::common::map_send_err_inf)?;
    cache.prune_for_version(version);
    Ok((g, version))
}

async fn join_blocking_json(
    handle: tokio::task::JoinHandle<AnyResult<Value>>,
    tool: &'static str,
) -> Result<Value, ToolExecutionError> {
    match handle.await {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(err)) => Err(ToolExecutionError::Internal(err)),
        Err(join_err) => Err(ToolExecutionError::execution(anyhow!(
            "blocking analysis `{tool}` panicked or was cancelled: {join_err}"
        ))),
    }
}

async fn respond_with_envelope(
    tool: &'static str,
    payload: Value,
    fetch_body: bool,
    state: &CallState,
) -> Result<ToolOutput, ToolExecutionError> {
    let meta = graph_meta(&state.graph).await?;
    let approx_bytes = payload_size_bytes(&payload);
    let estimates = prepare_payload_estimates(&state.metrics, state.model.as_str(), approx_bytes);
    let mode = ToolPayloadMode::from_fetch_flag(fetch_body);
    match mode {
        ToolPayloadMode::Preview => {
            let mut preview = build_cost_preview(
                tool,
                approx_bytes,
                estimates.safe_tokens,
                vec![
                    "Set fetch_body=true to stream results.".to_string(),
                    "Use limit to bound output volume.".to_string(),
                ],
            );
            let preview_bytes = payload_size_bytes(&preview);
            let preview_tokens = estimate_tokens_from_characters(preview_bytes as usize);
            apply_preview_cost(
                &mut preview,
                &state.metrics,
                state.model.as_str(),
                state.conversation_id.as_str(),
                preview_tokens,
                estimates.safe_tokens,
            );
            let with_meta = attach_meta(preview, &meta);
            let hint_bytes = payload_size_bytes(&with_meta);
            Ok(ToolOutput::with_byte_hint(with_meta, hint_bytes))
        }
        ToolPayloadMode::Body => {
            let with_meta = attach_meta(payload, &meta);
            let size = payload_size_bytes(&with_meta);
            Ok(ToolOutput::with_byte_hint(with_meta, size))
        }
    }
}

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
    pub limit:      Option<usize>,
    #[serde(default)]
    #[schemars(
        description = "When true, return the full body; otherwise respond with a preview/cost \
                       envelope."
    )]
    pub fetch_body: bool,
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
    #[serde(default)]
    #[schemars(
        description = "When true, return the full body; otherwise respond with a preview/cost \
                       envelope."
    )]
    pub fetch_body: bool,
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
    let args = parse_args_with_builder(REQUIRES_CYCLES, raw, |args: CyclesArgs| Ok(args))?;
    Ok(Box::new(CyclesTool {
        args,
        state: state.clone(),
    }))
}

fn parse_pagerank(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let mut args = parse_args_with_builder(REQUIRES_PAGERANK, raw, |args: PageRankArgs| Ok(args))?;
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
    let args = parse_args_with_builder(REQUIRES_BRIDGES, raw, |args: CyclesArgs| Ok(args))?;
    Ok(Box::new(BridgesTool {
        args,
        state: state.clone(),
    }))
}

fn parse_articulation(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args = parse_args_with_builder(REQUIRES_ARTICULATION, raw, |args: CyclesArgs| Ok(args))?;
    Ok(Box::new(ArticulationTool {
        args,
        state: state.clone(),
    }))
}

fn parse_feedback(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args = parse_args_with_builder(REQUIRES_FEEDBACK, raw, |args: CyclesArgs| Ok(args))?;
    Ok(Box::new(FeedbackTool {
        args,
        state: state.clone(),
    }))
}

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ShortestPathArgs {
    pub from_slug:  String,
    pub to_slug:    String,
    #[serde(default)]
    #[schemars(
        description = "When true, return the path immediately; otherwise respond with a \
                       preview/cost envelope."
    )]
    pub fetch_body: bool,
}

fn parse_shortest_path(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args =
        parse_args_with_builder(REQUIRES_SHORTEST_PATH, raw, |args: ShortestPathArgs| Ok(args))?;
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
        let (graph, graph_version) =
            load_graph_with_version(&self.state.graph, &self.state.analysis_cache).await?;
        let cache_key = AnalysisCacheKey {
            graph_version,
            kind: AnalysisKind::RequiresCycles,
        };

        let cached = self
            .state
            .analysis_cache
            .get_or_try_insert_with_async(cache_key, || {
                let graph = Arc::clone(&graph);
                async move {
                    let handle = tokio::task::spawn_blocking(move || -> AnyResult<Value> {
                        let view = traversal::requires_view(&graph);
                        let mut sccs: Vec<Vec<_>> = tarjan_scc(&view)
                            .into_iter()
                            .filter(|c| c.len() > 1)
                            .collect();
                        sccs.sort_by_key(|component| std::cmp::Reverse(component.len()));
                        let items: Vec<_> = sccs
                            .into_iter()
                            .take(500)
                            .map(|comp| {
                                json!({
                                    "size": comp.len(),
                                    "slugs": comp.into_iter().map(|n| graph[n].slug.clone()).collect::<Vec<_>>()
                                })
                            })
                            .collect();
                        Ok(json!({
                            "components": items,
                        }))
                    });
                    join_blocking_json(handle, REQUIRES_CYCLES).await
                }
            })
            .await?;

        let mut payload = cached.payload.clone();
        let limit = self.args.limit.unwrap_or(200).min(500);
        if let Some(arr) = payload["components"].as_array().cloned() {
            let mut trimmed = arr;
            if trimmed.len() > limit {
                trimmed.truncate(limit);
            }
            payload["components"] = json!(trimmed);
        }

        info!(
            tool = REQUIRES_CYCLES,
            count = payload["components"]
                .as_array()
                .map(|v| v.len())
                .unwrap_or(0),
            "graph requires cycles"
        );

        respond_with_envelope(
            REQUIRES_CYCLES,
            json!({
                "type": "graph_analysis",
                "tool": REQUIRES_CYCLES,
                "components": payload["components"].clone(),
            }),
            self.args.fetch_body,
            &self.state,
        )
        .await
    }
}

struct PageRankTool {
    args:  PageRankArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for PageRankTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, graph_version) =
            load_graph_with_version(&self.state.graph, &self.state.analysis_cache).await?;

        let cache_key = AnalysisCacheKey {
            graph_version,
            kind: AnalysisKind::RequiresPagerank {
                damping_bits: self.args.damping.to_bits(),
                iterations:   self.args.iterations,
            },
        };

        let cached = self
            .state
            .analysis_cache
            .get_or_try_insert_with_async(cache_key, || {
                let graph = Arc::clone(&graph);
                let damping = self.args.damping;
                let iterations = self.args.iterations;
                async move {
                    let handle = tokio::task::spawn_blocking(move || -> AnyResult<Value> {
                        let view = traversal::requires_view(&graph);
                        let scores = page_rank(&view, damping, iterations);
                        let mut items: Vec<_> = scores
                            .iter()
                            .enumerate()
                            .map(|(idx, score)| {
                                (graph[petgraph::graph::NodeIndex::new(idx)].slug.clone(), *score)
                            })
                            .collect();
                        items.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let top = items
                            .into_iter()
                            .take(500)
                            .map(|(slug, score)| json!({ "slug": slug, "score": score }))
                            .collect::<Vec<_>>();
                        Ok(json!({ "items": top }))
                    });
                    join_blocking_json(handle, REQUIRES_PAGERANK).await
                }
            })
            .await?;

        let mut items = cached
            .payload
            .get("items")
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default();
        let limit = self.args.limit.unwrap_or(50).min(500);
        if items.len() > limit {
            items.truncate(limit);
        }

        info!(
            tool = REQUIRES_PAGERANK,
            limit = limit,
            iter = self.args.iterations,
            damping = self.args.damping,
            "graph requires pagerank"
        );

        respond_with_envelope(
            REQUIRES_PAGERANK,
            json!({
                "type": "graph_analysis",
                "tool": REQUIRES_PAGERANK,
                "items": items,
            }),
            self.args.fetch_body,
            &self.state,
        )
        .await
    }
}

struct BridgesTool {
    args:  CyclesArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for BridgesTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, graph_version) =
            load_graph_with_version(&self.state.graph, &self.state.analysis_cache).await?;
        let cache_key = AnalysisCacheKey {
            graph_version,
            kind: AnalysisKind::RequiresBridges,
        };

        let cached = self
            .state
            .analysis_cache
            .get_or_try_insert_with_async(cache_key, || {
                let graph = Arc::clone(&graph);
                async move {
                    let handle = tokio::task::spawn_blocking(move || -> AnyResult<Value> {
                        let mut temp =
                            petgraph::graph::Graph::<(), (), petgraph::Directed>::with_capacity(
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

                        let list: Vec<_> = bridges(&temp)
                            .map(|e| {
                                let (u, v) = (e.source(), e.target());
                                let from_slug = graph
                                    [idx_map.iter().find(|(_, idx)| *idx == u).unwrap().0]
                                    .slug
                                    .clone();
                                let to_slug = graph
                                    [idx_map.iter().find(|(_, idx)| *idx == v).unwrap().0]
                                    .slug
                                    .clone();
                                json!({"from": from_slug, "to": to_slug})
                            })
                            .collect();
                        Ok(json!({"edges": list}))
                    });
                    join_blocking_json(handle, REQUIRES_BRIDGES).await
                }
            })
            .await?;

        let limit = self.args.limit.unwrap_or(200).min(500);
        let mut edges = cached
            .payload
            .get("edges")
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default();
        if edges.len() > limit {
            edges.truncate(limit);
        }
        info!(tool = REQUIRES_BRIDGES, count = edges.len(), "graph requires bridges");
        respond_with_envelope(
            REQUIRES_BRIDGES,
            json!({"type": "graph_analysis","tool": REQUIRES_BRIDGES,"edges": edges}),
            self.args.fetch_body,
            &self.state,
        )
        .await
    }
}

struct ArticulationTool {
    args:  CyclesArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for ArticulationTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, graph_version) =
            load_graph_with_version(&self.state.graph, &self.state.analysis_cache).await?;
        let cache_key = AnalysisCacheKey {
            graph_version,
            kind: AnalysisKind::RequiresArticulation,
        };

        let cached = self
            .state
            .analysis_cache
            .get_or_try_insert_with_async(cache_key, || {
                let graph = Arc::clone(&graph);
                async move {
                    let handle = tokio::task::spawn_blocking(move || -> AnyResult<Value> {
                        let mut temp =
                            petgraph::graph::Graph::<(), (), petgraph::Directed>::with_capacity(
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

                        let nodes: Vec<_> = articulation_points::articulation_points(&temp)
                            .into_iter()
                            .map(|n| {
                                let orig = idx_map.iter().find(|(_, idx)| *idx == n).unwrap().0;
                                graph[orig].slug.clone()
                            })
                            .collect();
                        Ok(json!({"nodes": nodes}))
                    });
                    join_blocking_json(handle, REQUIRES_ARTICULATION).await
                }
            })
            .await?;

        let limit = self.args.limit.unwrap_or(200).min(500);
        let mut nodes = cached
            .payload
            .get("nodes")
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default();
        if nodes.len() > limit {
            nodes.truncate(limit);
        }
        info!(tool = REQUIRES_ARTICULATION, count = nodes.len(), "graph requires articulation");
        respond_with_envelope(
            REQUIRES_ARTICULATION,
            json!({"type": "graph_analysis","tool": REQUIRES_ARTICULATION,"nodes": nodes}),
            self.args.fetch_body,
            &self.state,
        )
        .await
    }
}

struct FeedbackTool {
    args:  CyclesArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for FeedbackTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, graph_version) =
            load_graph_with_version(&self.state.graph, &self.state.analysis_cache).await?;
        let cache_key = AnalysisCacheKey {
            graph_version,
            kind: AnalysisKind::RequiresFeedback,
        };

        let cached = self
            .state
            .analysis_cache
            .get_or_try_insert_with_async(cache_key, || {
                let graph = Arc::clone(&graph);
                async move {
                    let handle = tokio::task::spawn_blocking(move || -> AnyResult<Value> {
                        let view = traversal::requires_view(&graph);
                        let set = greedy_feedback_arc_set(&view);
                        let edges: Vec<_> = set
                            .into_iter()
                            .map(|e| {
                                let (u, v) = (e.source(), e.target());
                                json!({"from": graph[u].slug.clone(), "to": graph[v].slug.clone()})
                            })
                            .collect();
                        Ok(json!({"edges": edges}))
                    });
                    join_blocking_json(handle, REQUIRES_FEEDBACK).await
                }
            })
            .await?;

        let limit = self.args.limit.unwrap_or(200).min(500);
        let mut edges = cached
            .payload
            .get("edges")
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default();
        if edges.len() > limit {
            edges.truncate(limit);
        }
        info!(tool = REQUIRES_FEEDBACK, count = edges.len(), "graph requires feedback arcs");
        respond_with_envelope(
            REQUIRES_FEEDBACK,
            json!({"type": "graph_analysis","tool": REQUIRES_FEEDBACK,"edges": edges}),
            self.args.fetch_body,
            &self.state,
        )
        .await
    }
}

struct ShortestPathTool {
    args:  ShortestPathArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for ShortestPathTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph_ref, graph_version) =
            load_graph_with_version(&self.state.graph, &self.state.analysis_cache).await?;

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

        let cache_key = AnalysisCacheKey {
            graph_version,
            kind: AnalysisKind::RequiresShortestPath {
                from: self.args.from_slug.clone(),
                to:   self.args.to_slug.clone(),
            },
        };

        let cached = self
            .state
            .analysis_cache
            .get_or_try_insert_with_async(cache_key, || {
                let graph_ref = Arc::clone(&graph_ref);
                async move {
                    let handle = tokio::task::spawn_blocking(move || -> AnyResult<Value> {
                        let view = traversal::requires_view(&graph_ref);
                        let dist = dijkstra(&view, from, Some(to), |_| 1usize);
                        let cost = dist.get(&to).copied();

                        let path = traversal::requires_one_path(&graph_ref, from, to)
                            .unwrap_or_default()
                            .into_iter()
                            .map(|n| graph_ref[n].slug.clone())
                            .collect::<Vec<_>>();
                        Ok(json!({ "cost": cost, "path": path }))
                    });
                    join_blocking_json(handle, REQUIRES_SHORTEST_PATH).await
                }
            })
            .await?;

        let cost = cached
            .payload
            .get("cost")
            .and_then(|v| v.as_u64().map(|c| c as usize));
        let path = cached
            .payload
            .get("path")
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default();

        info!(
            tool = REQUIRES_SHORTEST_PATH,
            from = %self.args.from_slug,
            to = %self.args.to_slug,
            cost,
            "graph requires shortest path"
        );

        respond_with_envelope(
            REQUIRES_SHORTEST_PATH,
            json!({
                "type": "graph_analysis",
                "tool": REQUIRES_SHORTEST_PATH,
                "cost": cost,
                "path": path,
            }),
            self.args.fetch_body,
            &self.state,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn join_blocking_panics_return_execution_error() {
        let handle = tokio::task::spawn_blocking(|| -> AnyResult<Value> {
            panic!("boom");
        });

        let err = join_blocking_json(handle, "test_tool").await.unwrap_err();
        assert!(matches!(err, ToolExecutionError::Execution(_)));
    }
}
