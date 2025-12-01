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

use super::common::{graph_meta, parse_args_with_builder};
use crate::{
    graph::{CurriculumGraph, EdgeKind, traversal},
    tools::llm::{
        CallState, ToolExecutionError, ToolInputError, ToolInstance, ToolOutput, ToolPayloadMode,
        ToolPrototype,
        analysis_cache::{AnalysisCacheKey, AnalysisKind, with_cached_analysis_result},
        common::{Page, ToolRunPayload, ToolRunner},
        payload_size_bytes, require_string,
    },
};

async fn load_graph_with_version(
    graph: &ActorRef<crate::graph::manager::GraphManager>,
    cache: &crate::tools::llm::graph_tools::analysis_cache::AnalysisCache,
) -> Result<(Arc<CurriculumGraph>, u64), ToolExecutionError> {
    let (g, version): (Arc<CurriculumGraph>, u64) = graph
        .ask(crate::graph::commands::GetGraphWithVersion)
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

const REQUIRES_CYCLES: &str = "graph_requires_cycles";
const REQUIRES_PAGERANK: &str = "graph_requires_pagerank";
const REQUIRES_BRIDGES: &str = "graph_requires_bridges";
const REQUIRES_ARTICULATION: &str = "graph_requires_articulation";
const REQUIRES_FEEDBACK: &str = "graph_requires_feedback_arcs";
const REQUIRES_SHORTEST_PATH: &str = "graph_requires_shortest_path";
pub(crate) const ALGORITHM_TOOL_IDS: &[&str] = &[
    REQUIRES_CYCLES,
    REQUIRES_PAGERANK,
    REQUIRES_BRIDGES,
    REQUIRES_ARTICULATION,
    REQUIRES_FEEDBACK,
    REQUIRES_SHORTEST_PATH,
];

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

crate::analysis_tool!(
    requires_cycles_meta,
    id: REQUIRES_CYCLES,
    description: "Detect cycles in the requires layer (returns SCCs > size 1).",
    args: CyclesArgs,
    prepare: |raw| parse_args_with_builder(REQUIRES_CYCLES, raw, |args: CyclesArgs| Ok(args)),
    runner: |args: CyclesArgs, state: &CallState| CyclesTool {
        args,
        state: state.clone(),
    }
);

crate::analysis_tool!(
    requires_pagerank_meta,
    id: REQUIRES_PAGERANK,
    description: "PageRank over the requires layer (influence of knowledge nodes).",
    args: PageRankArgs,
    prepare: |raw| {
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
        Ok(args)
    },
    runner: |args: PageRankArgs, state: &CallState| PageRankTool {
        args,
        state: state.clone(),
    }
);

crate::analysis_tool!(
    requires_bridges_meta,
    id: REQUIRES_BRIDGES,
    description: "Bridges (cut edges) in the requires layer.",
    args: CyclesArgs,
    prepare: |raw| parse_args_with_builder(REQUIRES_BRIDGES, raw, |args: CyclesArgs| Ok(args)),
    runner: |args: CyclesArgs, state: &CallState| BridgesTool {
        args,
        state: state.clone(),
    }
);

crate::analysis_tool!(
    requires_articulation_meta,
    id: REQUIRES_ARTICULATION,
    description: "Articulation points (cut nodes) in the requires layer.",
    args: CyclesArgs,
    prepare: |raw| parse_args_with_builder(REQUIRES_ARTICULATION, raw, |args: CyclesArgs| Ok(args)),
    runner: |args: CyclesArgs, state: &CallState| ArticulationTool {
        args,
        state: state.clone(),
    }
);

crate::analysis_tool!(
    requires_feedback_meta,
    id: REQUIRES_FEEDBACK,
    description: "Greedy feedback arc set suggestions to break requires cycles.",
    args: CyclesArgs,
    prepare: |raw| parse_args_with_builder(REQUIRES_FEEDBACK, raw, |args: CyclesArgs| Ok(args)),
    runner: |args: CyclesArgs, state: &CallState| FeedbackTool {
        args,
        state: state.clone(),
    }
);

crate::analysis_tool!(
    requires_shortest_path_meta,
    id: REQUIRES_SHORTEST_PATH,
    description: "Dijkstra shortest path (requires-only, unit weights) between two slugs.",
    args: ShortestPathArgs,
    prepare: |raw| parse_args_with_builder(
        REQUIRES_SHORTEST_PATH,
        raw,
        |mut args: ShortestPathArgs| {
            args.from_slug = require_string(args.from_slug, REQUIRES_SHORTEST_PATH, "from_slug")?;
            args.to_slug = require_string(args.to_slug, REQUIRES_SHORTEST_PATH, "to_slug")?;
            Ok(args)
        },
    ),
    runner: |args: ShortestPathArgs, state: &CallState| ShortestPathTool {
        args,
        state: state.clone(),
    }
);

pub(super) fn tool_prototypes() -> Vec<ToolPrototype> {
    vec![
        requires_cycles_meta(),
        requires_pagerank_meta(),
        requires_bridges_meta(),
        requires_articulation_meta(),
        requires_feedback_meta(),
        requires_shortest_path_meta(),
    ]
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

struct CyclesTool {
    args:  CyclesArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for CyclesTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let mode = ToolPayloadMode::from_fetch_flag(self.args.fetch_body);
        let (graph, graph_version) =
            load_graph_with_version(&self.state.graph, &self.state.analysis_cache).await?;
        let meta = graph_meta(&self.state.graph).await?;
        let cache_key = AnalysisCacheKey {
            graph_version,
            kind: AnalysisKind::RequiresCycles,
        };

        let payload = with_cached_analysis_result(
            &self.state.analysis_cache,
            cache_key,
            graph_version,
            || {
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
                        Ok(json!({ "components": items }))
                    });
                    join_blocking_json(handle, REQUIRES_CYCLES).await
                }
            },
        )
        .await?;

        let mut components = payload["components"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let total = components.len();
        let limit = self.args.limit.unwrap_or(200).min(500);
        if components.len() > limit {
            components.truncate(limit);
        }
        let has_more = total > limit;
        let body = json!({
            "type": "graph_analysis",
            "tool": REQUIRES_CYCLES,
            "components": components,
        });
        info!(tool = REQUIRES_CYCLES, count = total, limit, has_more, "graph requires cycles");

        let approx = payload_size_bytes(&body);
        let page = Page {
            offset: 0,
            limit,
            has_more,
        };

        ToolRunner::new(REQUIRES_CYCLES, &self.state)
            .with_mode(mode)
            .with_meta(meta)
            .hints(vec![
                "Set fetch_body=true to stream results.".to_string(),
                "Use limit to bound output volume.".to_string(),
            ])
            .run(move |_| async move {
                Ok(ToolRunPayload {
                    body,
                    approx_bytes: Some(approx),
                    preview: None,
                    preview_hints: Vec::new(),
                    page: Some(page),
                })
            })
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
        let mode = ToolPayloadMode::from_fetch_flag(self.args.fetch_body);
        let (graph, graph_version) =
            load_graph_with_version(&self.state.graph, &self.state.analysis_cache).await?;
        let meta = graph_meta(&self.state.graph).await?;

        let cache_key = AnalysisCacheKey {
            graph_version,
            kind: AnalysisKind::RequiresPagerank {
                damping_bits: self.args.damping.to_bits(),
                iterations:   self.args.iterations,
            },
        };

        let payload = with_cached_analysis_result(
            &self.state.analysis_cache,
            cache_key,
            graph_version,
            || {
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
            },
        )
        .await?;

        let mut items = payload
            .get("items")
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default();
        let total = items.len();
        let limit = self.args.limit.unwrap_or(50).min(500);
        if items.len() > limit {
            items.truncate(limit);
        }
        let has_more = total > limit;

        info!(
            tool = REQUIRES_PAGERANK,
            limit,
            iter = self.args.iterations,
            damping = self.args.damping,
            total,
            has_more,
            "graph requires pagerank"
        );

        let body = json!({
            "type": "graph_analysis",
            "tool": REQUIRES_PAGERANK,
            "items": items,
        });
        let approx = payload_size_bytes(&body);
        let page = Page {
            offset: 0,
            limit,
            has_more,
        };

        ToolRunner::new(REQUIRES_PAGERANK, &self.state)
            .with_mode(mode)
            .with_meta(meta)
            .hints(vec![
                "Set fetch_body=true to stream results.".to_string(),
                "Use limit to bound output volume.".to_string(),
            ])
            .run(move |_| async move {
                Ok(ToolRunPayload {
                    body,
                    approx_bytes: Some(approx),
                    preview: None,
                    preview_hints: Vec::new(),
                    page: Some(page),
                })
            })
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
        let mode = ToolPayloadMode::from_fetch_flag(self.args.fetch_body);
        let (graph, graph_version) =
            load_graph_with_version(&self.state.graph, &self.state.analysis_cache).await?;
        let meta = graph_meta(&self.state.graph).await?;
        let cache_key = AnalysisCacheKey {
            graph_version,
            kind: AnalysisKind::RequiresBridges,
        };

        let payload = with_cached_analysis_result(
            &self.state.analysis_cache,
            cache_key,
            graph_version,
            || {
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
            },
        )
        .await?;

        let mut edges = payload
            .get("edges")
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default();
        let total = edges.len();
        let limit = self.args.limit.unwrap_or(200).min(500);
        if edges.len() > limit {
            edges.truncate(limit);
        }
        let has_more = total > limit;
        info!(
            tool = REQUIRES_BRIDGES,
            count = total,
            limit,
            has_more,
            "graph requires bridges"
        );

        let body = json!({"type": "graph_analysis","tool": REQUIRES_BRIDGES,"edges": edges});
        let approx = payload_size_bytes(&body);
        let page = Page {
            offset: 0,
            limit,
            has_more,
        };

        ToolRunner::new(REQUIRES_BRIDGES, &self.state)
            .with_mode(mode)
            .with_meta(meta)
            .hints(vec![
                "Set fetch_body=true to stream results.".to_string(),
                "Use limit to bound output volume.".to_string(),
            ])
            .run(move |_| async move {
                Ok(ToolRunPayload {
                    body,
                    approx_bytes: Some(approx),
                    preview: None,
                    preview_hints: Vec::new(),
                    page: Some(page),
                })
            })
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
        let mode = ToolPayloadMode::from_fetch_flag(self.args.fetch_body);
        let (graph, graph_version) =
            load_graph_with_version(&self.state.graph, &self.state.analysis_cache).await?;
        let meta = graph_meta(&self.state.graph).await?;
        let cache_key = AnalysisCacheKey {
            graph_version,
            kind: AnalysisKind::RequiresArticulation,
        };

        let payload = with_cached_analysis_result(
            &self.state.analysis_cache,
            cache_key,
            graph_version,
            || {
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
            },
        )
        .await?;

        let mut nodes = payload
            .get("nodes")
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default();
        let total = nodes.len();
        let limit = self.args.limit.unwrap_or(200).min(500);
        if nodes.len() > limit {
            nodes.truncate(limit);
        }
        let has_more = total > limit;
        info!(
            tool = REQUIRES_ARTICULATION,
            count = total,
            limit,
            has_more,
            "graph requires articulation"
        );

        let body = json!({"type": "graph_analysis","tool": REQUIRES_ARTICULATION,"nodes": nodes});
        let approx = payload_size_bytes(&body);
        let page = Page {
            offset: 0,
            limit,
            has_more,
        };

        ToolRunner::new(REQUIRES_ARTICULATION, &self.state)
            .with_mode(mode)
            .with_meta(meta)
            .hints(vec![
                "Set fetch_body=true to stream results.".to_string(),
                "Use limit to bound output volume.".to_string(),
            ])
            .run(move |_| async move {
                Ok(ToolRunPayload {
                    body,
                    approx_bytes: Some(approx),
                    preview: None,
                    preview_hints: Vec::new(),
                    page: Some(page),
                })
            })
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
        let mode = ToolPayloadMode::from_fetch_flag(self.args.fetch_body);
        let (graph, graph_version) =
            load_graph_with_version(&self.state.graph, &self.state.analysis_cache).await?;
        let meta = graph_meta(&self.state.graph).await?;
        let cache_key = AnalysisCacheKey {
            graph_version,
            kind: AnalysisKind::RequiresFeedback,
        };

        let payload = with_cached_analysis_result(
            &self.state.analysis_cache,
            cache_key,
            graph_version,
            || {
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
            },
        )
        .await?;

        let mut edges = payload
            .get("edges")
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default();
        let total = edges.len();
        let limit = self.args.limit.unwrap_or(200).min(500);
        if edges.len() > limit {
            edges.truncate(limit);
        }
        let has_more = total > limit;
        info!(
            tool = REQUIRES_FEEDBACK,
            count = total,
            limit,
            has_more,
            "graph requires feedback arcs"
        );

        let body = json!({"type": "graph_analysis","tool": REQUIRES_FEEDBACK,"edges": edges});
        let approx = payload_size_bytes(&body);
        let page = Page {
            offset: 0,
            limit,
            has_more,
        };

        ToolRunner::new(REQUIRES_FEEDBACK, &self.state)
            .with_mode(mode)
            .with_meta(meta)
            .hints(vec![
                "Set fetch_body=true to stream results.".to_string(),
                "Use limit to bound output volume.".to_string(),
            ])
            .run(move |_| async move {
                Ok(ToolRunPayload {
                    body,
                    approx_bytes: Some(approx),
                    preview: None,
                    preview_hints: Vec::new(),
                    page: Some(page),
                })
            })
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
        let mode = ToolPayloadMode::from_fetch_flag(self.args.fetch_body);
        let (graph_ref, graph_version) =
            load_graph_with_version(&self.state.graph, &self.state.analysis_cache).await?;
        let meta = graph_meta(&self.state.graph).await?;

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

        let payload = with_cached_analysis_result(
            &self.state.analysis_cache,
            cache_key,
            graph_version,
            || {
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
            },
        )
        .await?;

        let cost = payload
            .get("cost")
            .and_then(|v| v.as_u64().map(|c| c as usize));
        let path = payload
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

        let body = json!({
            "type": "graph_analysis",
            "tool": REQUIRES_SHORTEST_PATH,
            "cost": cost,
            "path": path,
        });
        let approx = payload_size_bytes(&body);

        ToolRunner::new(REQUIRES_SHORTEST_PATH, &self.state)
            .with_mode(mode)
            .with_meta(meta)
            .hints(vec!["Set fetch_body=true to stream results.".to_string()])
            .run(move |_| async move {
                Ok(ToolRunPayload {
                    body,
                    approx_bytes: Some(approx),
                    preview: None,
                    preview_hints: Vec::new(),
                    page: None,
                })
            })
            .await
    }
}
