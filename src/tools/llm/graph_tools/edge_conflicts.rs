use async_trait::async_trait;
use bon::Builder;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::common::{MaybeApply, map_send_err, parse_args_with_builder};
use crate::{
    graph::{
        EdgeKind,
        commands::{GetEdgeConflicts, ResolveEdgeConflict},
    },
    tools::llm::{
        CallState, ToolExecutionError, ToolInstance, ToolOutput, ToolPayloadMode, ToolPrototype,
        common::{ToolRunPayload, ToolRunner},
    },
};

const GRAPH_EDGE_CONFLICTS: &str = "graph_edge_conflicts";
const GRAPH_RESOLVE_EDGE_CONFLICT: &str = "graph_resolve_edge_conflict";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GetEdgeConflictsArgs {
    #[serde(default)]
    #[schemars(description = "Maximum conflicts to return (default 50, max 200).")]
    pub limit:  Option<usize>,
    #[serde(default)]
    #[schemars(description = "Offset into the conflict list (default 0).")]
    pub offset: Option<usize>,
}

crate::analysis_tool!(
    edge_conflicts_meta,
    id: GRAPH_EDGE_CONFLICTS,
    description: "List edges carrying conflicts (duplicate edge payloads that could not be merged \
                  deterministically). Returns from/to slugs, current edge payload, and queued \
                  conflicting payloads.",
    args: GetEdgeConflictsArgs,
    prepare: |raw| parse_args_with_builder(GRAPH_EDGE_CONFLICTS, raw, |input: GetEdgeConflictsArgs| {
        Ok(input)
    }),
    runner: |args: GetEdgeConflictsArgs, state: &CallState| EdgeConflictsTool {
        args,
        state: state.clone(),
    }
);

struct EdgeConflictsTool {
    args:  GetEdgeConflictsArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for EdgeConflictsTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let conflicts = self
            .state
            .graph
            .ask(GetEdgeConflicts)
            .await
            .map_err(|e| map_send_err(e, GRAPH_EDGE_CONFLICTS))?;

        let (page, page_meta) = crate::paginate!(conflicts, self.args.limit, self.args.offset);
        let rendered: Vec<_> = page
            .iter()
            .map(|entry| {
                json!({
                    "edge_id": entry.edge_id,
                    "from_slug": entry.from_slug,
                    "to_slug": entry.to_slug,
                    "kind": entry.kind,
                    "confidence": entry.confidence,
                    "conflicts": entry.conflicts,
                })
            })
            .collect();

        let meta = super::common::graph_meta(&self.state.graph).await?;
        let payload = rendered;

        ToolRunner::new(GRAPH_EDGE_CONFLICTS, &self.state)
            .with_mode(ToolPayloadMode::Body)
            .with_meta(meta)
            .run(move |_| async move {
                Ok(ToolRunPayload {
                    body:          json!({
                        "type": "graph_view",
                        "tool": GRAPH_EDGE_CONFLICTS,
                        "offset": page_meta.offset,
                        "limit": page_meta.limit,
                        "has_more": page_meta.has_more,
                        "conflicts": payload,
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

#[derive(Debug, Clone, Builder, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResolveEdgeConflictArgs {
    #[schemars(description = "Edge id (index) of the conflict to resolve.")]
    pub edge_id:         u32,
    #[schemars(description = "Merged edge payload to apply.")]
    pub resolved_kind:   EdgeKind,
    #[serde(default)]
    #[schemars(
        description = "Optional confidence to set on the merged edge (defaults to existing)."
    )]
    pub confidence:      Option<f32>,
    #[serde(default = "crate::tools::llm::default_false")]
    #[schemars(description = "Set apply=true to commit; default false previews the change.")]
    pub apply:           bool,
    #[serde(default = "default_clear_conflicts")]
    #[schemars(description = "Whether to clear queued conflicts after applying the resolution.")]
    pub clear_conflicts: bool,
}

const fn default_clear_conflicts() -> bool {
    true
}

impl MaybeApply for ResolveEdgeConflictArgs {
    fn apply_flag(&self) -> bool {
        self.apply
    }
}

crate::graph_action_tool!(
    resolve_edge_conflict_meta,
    id: GRAPH_RESOLVE_EDGE_CONFLICT,
    description: "Resolve a conflicting edge payload by supplying the merged EdgeKind payload. \
                  Clears queued conflicts when apply=true (default).",
    args: ResolveEdgeConflictArgs,
    prepare: |raw| parse_args_with_builder(GRAPH_RESOLVE_EDGE_CONFLICT, raw, |input: ResolveEdgeConflictArgs| {
        Ok(input)
    }),
    build: |args: &ResolveEdgeConflictArgs| ResolveEdgeConflict {
        edge_id: args.edge_id,
        resolved_kind: args.resolved_kind.clone(),
        confidence: args.confidence,
        clear_conflicts: args.clear_conflicts,
    },
    ok: |args: &ResolveEdgeConflictArgs, _| {
        json!({
            "type": "graph_command",
            "tool": GRAPH_RESOLVE_EDGE_CONFLICT,
            "status": "ok",
            "edge_id": args.edge_id,
            "cleared_conflicts": args.clear_conflicts,
        })
    },
    map_err: |e| map_send_err(e, GRAPH_RESOLVE_EDGE_CONFLICT)
);

pub fn tool_prototypes() -> Vec<ToolPrototype> {
    vec![edge_conflicts_meta(), resolve_edge_conflict_meta()]
}
