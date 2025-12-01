use async_trait::async_trait;
use bon::Builder;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::common::{map_send_err, parse_args_with_builder};
use crate::{
    graph::{NodeKind, commands::EdgeKindFilter},
    tools::llm::{
        CallState, ToolExecutionError, ToolInstance, ToolOutput, ToolPayloadMode, ToolPrototype,
        common::{ToolRunPayload, ToolRunner},
        require_string,
    },
};

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

pub(super) fn tool_prototypes() -> Vec<ToolPrototype> {
    vec![neighbors_meta(), get_node_meta()]
}
