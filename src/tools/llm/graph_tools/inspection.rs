use async_trait::async_trait;
use bon::Builder;
use kameo::prelude::ActorRef;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use super::common::{map_send_err, paginate};
use crate::{
    graph::{NodeKind, manager::EdgeKindFilter},
    tools::llm::{
        CallState, ToolExecutionError, ToolInputError, ToolInputResult, ToolInstance, ToolOutput,
        ToolPrototype, payload_size_bytes, require_string, schema_for_args,
    },
};

// ---------- Neighbors ----------

const GRAPH_NEIGHBORS: &str = "graph_neighbors";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NeighborsArgs {
    #[builder(with = |value: String| -> crate::tools::llm::ToolInputResult<_> {
        require_string(value, GRAPH_NEIGHBORS, "slug")
    })]
    #[schemars(description = "Existing node slug whose neighborhood you want to inspect.")]
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
    pub direction: Option<String>, // outgoing, incoming, both
    #[serde(default)]
    #[schemars(description = "Maximum neighbors to return (default 50, max 200).")]
    pub limit:     Option<usize>,
    #[serde(default)]
    #[schemars(description = "Offset into the neighbor list (default 0).")]
    pub offset:    Option<usize>,
}

pub(super) fn neighbors_meta() -> ToolPrototype {
    ToolPrototype {
        id:          GRAPH_NEIGHBORS,
        description: "Inspect local graph structure: list neighbors with edge_kind and direction \
                      (requires/supports/assesses/precedes/anchors). Use to read prerequisites, \
                      scaffolds, assessment links, and discourse anchors around a node.",
        schema:      schema_for_args::<NeighborsArgs>(),
        parse:       parse_neighbors,
    }
}

fn parse_neighbors(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let mut args: NeighborsArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    GRAPH_NEIGHBORS,
            message: err.to_string(),
        })?;
    if let Some(dir) = args.direction.as_deref()
        && !matches!(dir, "incoming" | "outgoing" | "both")
    {
        return Err(ToolInputError::InvalidPayload {
            tool:    GRAPH_NEIGHBORS,
            message: "direction must be incoming|outgoing|both".into(),
        });
    }
    args.slug = require_string(args.slug.clone(), GRAPH_NEIGHBORS, "slug")?;
    Ok(Box::new(NeighborsTool {
        args,
        graph: state.graph.clone(),
    }))
}

struct NeighborsTool {
    args:  NeighborsArgs,
    graph: ActorRef<crate::graph::manager::GraphManager>,
}

#[async_trait]
impl ToolInstance for NeighborsTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        use crate::graph::manager::{NeighborDirection, Neighbors};

        let direction = match self.args.direction.as_deref() {
            Some("incoming") => Some(NeighborDirection::Incoming),
            Some("outgoing") => Some(NeighborDirection::Outgoing),
            _ => Some(NeighborDirection::Both),
        };

        let neighbors = self
            .graph
            .ask(Neighbors {
                slug: self.args.slug.clone(),
                edge_kind: self.args.edge_kind,
                direction,
            })
            .await
            .map_err(|e| map_send_err(e, GRAPH_NEIGHBORS))?;

        let (page, offset, limit, has_more) =
            paginate(neighbors, self.args.limit, self.args.offset);

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

        let payload = json!({
            "type": "graph_view",
            "tool": GRAPH_NEIGHBORS,
            "slug": self.args.slug,
            "offset": offset,
            "limit": limit,
            "has_more": has_more,
            "neighbors": rendered,
        });

        Ok(ToolOutput::with_byte_hint(
            payload.clone(),
            crate::tools::llm::payload_size_bytes(&payload),
        ))
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
    pub slug: String,
}

pub(super) fn get_node_meta() -> ToolPrototype {
    ToolPrototype {
        id:          GET_NODE,
        description: "Fetch a node payload by slug (kind, statement, rubric/construct-irrelevant \
                      data, grain/load/scope, source_refs, tags). Use this before proposing edits \
                      or edges.",
        schema:      schema_for_args::<GetNodeArgs>(),
        parse:       parse_get_node,
    }
}

fn parse_get_node(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let mut args: GetNodeArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    GET_NODE,
            message: err.to_string(),
        })?;
    args.slug = require_string(args.slug.clone(), GET_NODE, "slug")?;
    Ok(Box::new(GetNodeTool {
        args,
        graph: state.graph.clone(),
    }))
}

struct GetNodeTool {
    args:  GetNodeArgs,
    graph: ActorRef<crate::graph::manager::GraphManager>,
}

#[async_trait]
impl ToolInstance for GetNodeTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let payload = self
            .graph
            .ask(crate::graph::manager::GetNode {
                slug: self.args.slug.clone(),
            })
            .await
            .map_err(|e| map_send_err(e, GET_NODE))?;
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
                "tags": payload.tags,
            }),
        };
        let payload = json!({
            "type": "graph_view",
            "tool": GET_NODE,
            "node": value,
        });
        Ok(ToolOutput::with_byte_hint(payload.clone(), payload_size_bytes(&payload)))
    }
}

pub(super) fn tool_prototypes() -> Vec<ToolPrototype> {
    vec![neighbors_meta(), get_node_meta()]
}
