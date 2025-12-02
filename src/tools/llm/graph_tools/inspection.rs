use async_trait::async_trait;
use bon::Builder;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::common::{map_send_err, map_send_err_inf, parse_args_with_builder};
use crate::{
    graph::{
        NodeKind,
        commands::{
            EdgeKindFilter, ListNodesByKind, ListNodesByTag, ListTags, NodeKindSelector,
            NodeSearchResult, NodeSummary, SearchNodes,
        },
        manager::GetCourseCommit,
    },
    tools::llm::{
        CallState, ToolExecutionError, ToolInstance, ToolOutput, ToolPayloadMode, ToolPrototype,
        common::{ToolRunPayload, ToolRunner},
        require_string, require_usize_min,
    },
};

fn render_node_summary(summary: NodeSummary) -> serde_json::Value {
    json!({
        "slug": summary.slug,
        "title": summary.title,
        "kind": summary.kind,
        "knowledge_type": summary.knowledge_type,
        "tags": summary.tags,
    })
}

fn render_search_result(result: NodeSearchResult) -> serde_json::Value {
    json!({
        "slug": result.slug,
        "title": result.title,
        "kind": result.kind,
        "knowledge_type": result.knowledge_type,
        "tags": result.tags,
        "score": result.score,
    })
}

// ---------- Course commit ----------

const COURSE_COMMIT: &str = "graph_course_commit";

#[derive(Debug, Clone, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct CourseCommitArgs {}

crate::analysis_tool!(
    course_commit_meta,
    id: COURSE_COMMIT,
    description: "Return the course commit hash currently configured for the graph.",
    args: CourseCommitArgs,
    prepare: |raw| parse_args_with_builder(COURSE_COMMIT, raw, |_input: CourseCommitArgs| {
        Ok(CourseCommitArgs {})
    }),
    runner: |_args: CourseCommitArgs, state: &CallState| CourseCommitTool {
        state: state.clone(),
    }
);

struct CourseCommitTool {
    state: CallState,
}

#[async_trait]
impl ToolInstance for CourseCommitTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let commit = self
            .state
            .graph
            .ask(GetCourseCommit)
            .await
            .map_err(map_send_err_inf)?;
        let meta = super::common::graph_meta(&self.state.graph).await?;

        ToolRunner::new(COURSE_COMMIT, &self.state)
            .with_mode(ToolPayloadMode::Body)
            .with_meta(meta)
            .run(|_| async move {
                Ok(ToolRunPayload::new(json!({
                    "type": "graph_view",
                    "tool": COURSE_COMMIT,
                    "course_commit": commit,
                })))
            })
            .await
    }
}

// ---------- Neighbors ----------

const GRAPH_NEIGHBORS: &str = "graph_neighbors";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NeighborsArgs {
    #[schemars(description = "Existing node slug whose neighborhood you want to inspect.")]
    #[builder(with = |value: String| -> crate::tools::llm::ToolInputResult<_> {
        require_string(value, GRAPH_NEIGHBORS, "slug")
    })]
    pub slug:      String,
    #[serde(default)]
    #[schemars(
        description = "Optional edge layer filter: requires | supports | assesses | precedes | \
                       anchors. Leave empty to see all kinds."
    )]
    pub edge_kind: Option<EdgeKindFilter>,
    #[serde(default)]
    #[schemars(
        description = "Direction relative to the node: incoming | outgoing | both (default)."
    )]
    pub direction: Option<NeighborDirectionArg>,
    #[serde(default)]
    #[schemars(description = "Maximum neighbors to return (default 50, max 200).")]
    pub limit:     Option<usize>,
    #[serde(default)]
    #[schemars(description = "Offset into the neighbor list (default 0).")]
    pub offset:    Option<usize>,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum NeighborDirectionArg {
    Incoming,
    Outgoing,
    Both,
}

crate::analysis_tool!(
    neighbors_meta,
    id: GRAPH_NEIGHBORS,
    description: "Inspect local graph structure: list neighbors with edge_kind and direction \
                  (requires/supports/assesses/precedes/anchors). Use to read prerequisites, \
                  scaffolds, assessment links, and discourse anchors around a node.",
    args: NeighborsArgs,
    prepare: |raw| parse_args_with_builder(GRAPH_NEIGHBORS, raw, |mut input: NeighborsArgs| {
        input.slug = require_string(input.slug, GRAPH_NEIGHBORS, "slug")?;
        Ok(input)
    }),
    runner: |args: NeighborsArgs, state: &CallState| NeighborsTool {
        args,
        state: state.clone(),
    }
);

struct NeighborsTool {
    args:  NeighborsArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for NeighborsTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        use crate::graph::commands::{NeighborDirection, Neighbors};

        let direction = match self.args.direction {
            Some(NeighborDirectionArg::Incoming) => Some(NeighborDirection::Incoming),
            Some(NeighborDirectionArg::Outgoing) => Some(NeighborDirection::Outgoing),
            Some(NeighborDirectionArg::Both) | None => Some(NeighborDirection::Both),
        };

        let neighbors = self
            .state
            .graph
            .ask(Neighbors {
                slug: self.args.slug.clone(),
                edge_kind: self.args.edge_kind,
                direction,
            })
            .await
            .map_err(|e| map_send_err(e, GRAPH_NEIGHBORS))?;

        let (page, page_meta) = crate::paginate!(neighbors, self.args.limit, self.args.offset);

        let rendered: Vec<_> = page
            .iter()
            .map(|n| {
                json!({
                    "neighbor_slug": n.neighbor_slug,
                    "edge_kind": n.edge_kind,
                    "direction": n.direction,
                })
            })
            .collect();

        let meta = super::common::graph_meta(&self.state.graph).await?;
        let slug = self.args.slug.clone();
        let neighbors = rendered;

        ToolRunner::new(GRAPH_NEIGHBORS, &self.state)
            .with_mode(ToolPayloadMode::Body)
            .with_meta(meta)
            .run(move |_| async move {
                Ok(ToolRunPayload {
                    body:          json!({
                        "type": "graph_view",
                        "tool": GRAPH_NEIGHBORS,
                        "slug": slug,
                        "offset": page_meta.offset,
                        "limit": page_meta.limit,
                        "has_more": page_meta.has_more,
                        "neighbors": neighbors,
                    }),
                    approx_bytes:  None,
                    preview:       None,
                    preview_hints: Vec::new(),
                    page:          Some(page_meta),
                })
            })
            .await
    }
}

// ---------- Get node ----------

const GET_NODE: &str = "graph_get_node";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GetNodeArgs {
    #[schemars(
        description = "Existing node slug in the curriculum graph. Use graph_neighbors or prior \
                       tools to discover slugs."
    )]
    #[builder(with = |value: String| -> crate::tools::llm::ToolInputResult<_> {
        require_string(value, GET_NODE, "slug")
    })]
    pub slug: String,
}

crate::analysis_tool!(
    get_node_meta,
    id: GET_NODE,
    description: "Fetch a node payload by slug (kind, statement, rubric/construct-irrelevant \
                  data, grain/load/scope, source_refs, tags). Use this before proposing edits \
                  or edges.",
    args: GetNodeArgs,
    prepare: |raw| parse_args_with_builder(GET_NODE, raw, |mut input: GetNodeArgs| {
        input.slug = require_string(input.slug, GET_NODE, "slug")?;
        Ok(input)
    }),
    runner: |args: GetNodeArgs, state: &CallState| GetNodeTool {
        args,
        state: state.clone(),
    }
);

struct GetNodeTool {
    args:  GetNodeArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for GetNodeTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let payload = self
            .state
            .graph
            .ask(crate::graph::commands::GetNode {
                slug: self.args.slug.clone(),
            })
            .await
            .map_err(|e| map_send_err(e, GET_NODE))?;
        let meta = super::common::graph_meta(&self.state.graph).await?;
        let value = match &payload.kind {
            NodeKind::Knowledge(k) => json!({
                "slug": payload.slug,
                "logical_id": payload.logical_id,
                "kind": "knowledge",
                "knowledge_type": k.knowledge_type,
                "title": k.title,
                "statement": k.statement,
                "confidence": k.confidence,
                "rubric_criteria": k.rubric_criteria,
                "construct_irrelevant_demands": k.construct_irrelevant_demands,
                "grain_level": k.grain_level,
                "intrinsic_load": k.intrinsic_load,
                "introduction_scope": k.introduction_scope,
                "source_refs": k.source_refs,
                "tags": payload.tags,
            }),
            NodeKind::TeachingStep(ts) => json!({
                "slug": payload.slug,
                "logical_id": payload.logical_id,
                "kind": "teaching_step",
                "title": ts.title,
                "statement": ts.statement,
                "purpose": ts.purpose,
                "episode": ts.episode,
                "method_tags": ts.method_tags,
                "source_refs": ts.source_refs,
                "rationale": ts.rationale,
                "tags": payload.tags,
            }),
        };
        ToolRunner::new(GET_NODE, &self.state)
            .with_mode(ToolPayloadMode::Body)
            .with_meta(meta)
            .run(|_| async move {
                Ok(ToolRunPayload::new(json!({
                    "type": "graph_view",
                    "tool": GET_NODE,
                    "node": value,
                })))
            })
            .await
    }
}

// ---------- List nodes by tag ----------

const LIST_NODES_BY_TAG: &str = "graph_list_nodes_by_tag";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ListNodesByTagArgs {
    #[schemars(description = "Filter nodes that include this tag (case-insensitive match).")]
    #[builder(with = |value: String| -> crate::tools::llm::ToolInputResult<_> {
        require_string(value, LIST_NODES_BY_TAG, "tag")
    })]
    pub tag:    String,
    #[serde(default)]
    #[schemars(description = "Maximum nodes to return (default 50, max 200).")]
    pub limit:  Option<usize>,
    #[serde(default)]
    #[schemars(description = "Offset into the matching list (default 0).")]
    pub offset: Option<usize>,
}

crate::analysis_tool!(
    list_nodes_by_tag_meta,
    id: LIST_NODES_BY_TAG,
    description: "List nodes carrying a specific tag. Use chapter tags like `source:<chapter>` \
                  to scope work for harvesters/weavers.",
    args: ListNodesByTagArgs,
    prepare: |raw| parse_args_with_builder(LIST_NODES_BY_TAG, raw, |mut input: ListNodesByTagArgs| {
        input.tag = require_string(input.tag, LIST_NODES_BY_TAG, "tag")?;
        Ok(input)
    }),
    runner: |args: ListNodesByTagArgs, state: &CallState| ListNodesByTagTool {
        args,
        state: state.clone(),
    }
);

struct ListNodesByTagTool {
    args:  ListNodesByTagArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for ListNodesByTagTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let nodes = self
            .state
            .graph
            .ask(ListNodesByTag {
                tag: self.args.tag.clone(),
            })
            .await
            .map_err(|e| map_send_err(e, LIST_NODES_BY_TAG))?;

        let (page, page_meta) = crate::paginate!(nodes, self.args.limit, self.args.offset);
        let rendered: Vec<_> = page.into_iter().map(render_node_summary).collect();
        let meta = super::common::graph_meta(&self.state.graph).await?;
        let tag = self.args.tag.clone();

        ToolRunner::new(LIST_NODES_BY_TAG, &self.state)
            .with_mode(ToolPayloadMode::Body)
            .with_meta(meta)
            .run(move |_| async move {
                Ok(ToolRunPayload {
                    body:          json!({
                        "type": "graph_view",
                        "tool": LIST_NODES_BY_TAG,
                        "tag": tag,
                        "offset": page_meta.offset,
                        "limit": page_meta.limit,
                        "has_more": page_meta.has_more,
                        "nodes": rendered,
                    }),
                    approx_bytes:  None,
                    preview:       None,
                    preview_hints: Vec::new(),
                    page:          Some(page_meta),
                })
            })
            .await
    }
}

// ---------- List nodes by kind ----------

const LIST_NODES_BY_KIND: &str = "graph_list_nodes_by_kind";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ListNodesByKindArgs {
    #[schemars(
        description = "Select knowledge_type to filter knowledge nodes or choose \
                       teaching_step/any_knowledge."
    )]
    pub selector: NodeKindSelector,
    #[serde(default)]
    #[schemars(description = "Maximum nodes to return (default 50, max 200).")]
    pub limit:    Option<usize>,
    #[serde(default)]
    #[schemars(description = "Offset into the matching list (default 0).")]
    pub offset:   Option<usize>,
}

crate::analysis_tool!(
    list_nodes_by_kind_meta,
    id: LIST_NODES_BY_KIND,
    description: "List nodes by type: pick a knowledge_type, any_knowledge, or teaching_step.",
    args: ListNodesByKindArgs,
    prepare: |raw| parse_args_with_builder(LIST_NODES_BY_KIND, raw, |input: ListNodesByKindArgs| {
        Ok(input)
    }),
    runner: |args: ListNodesByKindArgs, state: &CallState| ListNodesByKindTool {
        args,
        state: state.clone(),
    }
);

struct ListNodesByKindTool {
    args:  ListNodesByKindArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for ListNodesByKindTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let nodes = self
            .state
            .graph
            .ask(ListNodesByKind {
                selector: self.args.selector.clone(),
            })
            .await
            .map_err(|e| map_send_err(e, LIST_NODES_BY_KIND))?;

        let (page, page_meta) = crate::paginate!(nodes, self.args.limit, self.args.offset);
        let rendered: Vec<_> = page.into_iter().map(render_node_summary).collect();
        let meta = super::common::graph_meta(&self.state.graph).await?;

        ToolRunner::new(LIST_NODES_BY_KIND, &self.state)
            .with_mode(ToolPayloadMode::Body)
            .with_meta(meta)
            .run(move |_| async move {
                Ok(ToolRunPayload {
                    body:          json!({
                        "type": "graph_view",
                        "tool": LIST_NODES_BY_KIND,
                        "offset": page_meta.offset,
                        "limit": page_meta.limit,
                        "has_more": page_meta.has_more,
                        "nodes": rendered,
                    }),
                    approx_bytes:  None,
                    preview:       None,
                    preview_hints: Vec::new(),
                    page:          Some(page_meta),
                })
            })
            .await
    }
}

// ---------- List tags ----------

const LIST_TAGS: &str = "graph_list_tags";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct ListTagsArgs {}

crate::analysis_tool!(
    list_tags_meta,
    id: LIST_TAGS,
    description: "List all tags present in the graph (deduplicated).",
    args: ListTagsArgs,
    prepare: |raw| parse_args_with_builder(LIST_TAGS, raw, |_input: ListTagsArgs| Ok(ListTagsArgs {})),
    runner: |args: ListTagsArgs, state: &CallState| ListTagsTool {
        _args: args,
        state: state.clone(),
    }
);

struct ListTagsTool {
    _args: ListTagsArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for ListTagsTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let tags = self
            .state
            .graph
            .ask(ListTags)
            .await
            .map_err(map_send_err_inf)?;
        let meta = super::common::graph_meta(&self.state.graph).await?;

        ToolRunner::new(LIST_TAGS, &self.state)
            .with_mode(ToolPayloadMode::Body)
            .with_meta(meta)
            .run(move |_| async move {
                Ok(ToolRunPayload::new(json!({
                    "type": "graph_view",
                    "tool": LIST_TAGS,
                    "tags": tags,
                })))
            })
            .await
    }
}

// ---------- Search nodes ----------

const SEARCH_NODES: &str = "graph_search_nodes";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SearchNodesArgs {
    #[schemars(description = "Query to match against slug/title/statement (case-insensitive).")]
    #[builder(with = |value: String| -> crate::tools::llm::ToolInputResult<_> {
        require_string(value, SEARCH_NODES, "query")
    })]
    pub query: String,
    #[serde(default)]
    #[schemars(description = "Maximum matches to return (default 20, max 200).")]
    pub limit: Option<usize>,
}

crate::analysis_tool!(
    search_nodes_meta,
    id: SEARCH_NODES,
    description: "Fuzzy search nodes by slug/title/statement. Uses substring + Jaro-Winkler \
                  scoring. Helpful when slugs are unknown or dedup changed them.",
    args: SearchNodesArgs,
    prepare: |raw| parse_args_with_builder(SEARCH_NODES, raw, |mut input: SearchNodesArgs| {
        input.query = require_string(input.query, SEARCH_NODES, "query")?;
        if let Some(limit) = input.limit {
            input.limit = Some(require_usize_min(limit, 1, SEARCH_NODES, "limit")?);
        }
        Ok(input)
    }),
    runner: |args: SearchNodesArgs, state: &CallState| SearchNodesTool {
        args,
        state: state.clone(),
    }
);

struct SearchNodesTool {
    args:  SearchNodesArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for SearchNodesTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let limit = self.args.limit.unwrap_or(20).clamp(1, 200);
        let matches = self
            .state
            .graph
            .ask(SearchNodes {
                query: self.args.query.clone(),
                limit,
            })
            .await
            .map_err(|e| map_send_err(e, SEARCH_NODES))?;
        let rendered: Vec<_> = matches.into_iter().map(render_search_result).collect();
        let meta = super::common::graph_meta(&self.state.graph).await?;
        let query = self.args.query.clone();

        ToolRunner::new(SEARCH_NODES, &self.state)
            .with_mode(ToolPayloadMode::Body)
            .with_meta(meta)
            .run(move |_| async move {
                Ok(ToolRunPayload::new(json!({
                    "type": "graph_view",
                    "tool": SEARCH_NODES,
                    "query": query,
                    "limit": limit,
                    "matches": rendered,
                })))
            })
            .await
    }
}

pub(super) fn tool_prototypes() -> Vec<ToolPrototype> {
    vec![
        neighbors_meta(),
        get_node_meta(),
        list_nodes_by_tag_meta(),
        list_nodes_by_kind_meta(),
        list_tags_meta(),
        search_nodes_meta(),
        course_commit_meta(),
    ]
}
