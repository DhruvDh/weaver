use async_trait::async_trait;
use bon::Builder;
use kameo::prelude::ActorRef;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use tracing::info;

use super::common::{graph_meta, map_send_err, parse_args_with_builder};
use crate::{
    graph::manager::RedundantRequires,
    tools::llm::{
        CallState, ToolExecutionError, ToolInstance, ToolOutput, ToolPayloadMode, ToolPrototype,
        common::{ToolRunPayload, ToolRunner},
        payload_size_bytes,
    },
};

const REDUNDANT_REQUIRES: &str = "graph_redundant_requires";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RedundantRequiresArgs {
    #[serde(default)]
    #[schemars(description = "When true, also prune the redundant requires edges.")]
    pub prune:  bool,
    #[serde(default)]
    #[schemars(description = "Maximum items to return (default 200, max 500).")]
    pub limit:  Option<usize>,
    #[serde(default)]
    #[schemars(description = "Offset into the result set (default 0).")]
    pub offset: Option<usize>,
    #[serde(default)]
    #[builder(default = false)]
    #[schemars(
        description = "Set apply=true to actually prune when prune=true; default false previews \
                       the list only."
    )]
    pub apply:  bool,
}

pub(super) fn tool_prototypes() -> Vec<ToolPrototype> {
    vec![redundant_requires_meta()]
}

crate::analysis_tool!(
    redundant_requires_meta,
    id: REDUNDANT_REQUIRES,
    description: "List redundant requires edges (edges removable without changing reachability). Optionally prune them.",
    args: RedundantRequiresArgs,
    prepare: |raw| parse_args_with_builder(REDUNDANT_REQUIRES, raw, |args: RedundantRequiresArgs| Ok(args)),
    runner: |args: RedundantRequiresArgs, state: &CallState| RedundantRequiresTool {
        args,
        graph: state.graph.clone(),
        state: state.clone(),
    }
);

struct RedundantRequiresTool {
    args:  RedundantRequiresArgs,
    graph: ActorRef<crate::graph::manager::GraphManager>,
    state: CallState,
}

#[async_trait]
impl ToolInstance for RedundantRequiresTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let edges: Vec<(String, String)> = self
            .graph
            .ask(RedundantRequires {
                prune: self.args.prune && self.args.apply,
            })
            .await
            .map_err(|e| map_send_err(e, REDUNDANT_REQUIRES))?;

        let total = edges.len();
        let (page, page_meta) = crate::paginate!(edges, self.args.limit, self.args.offset);
        let edges = page
            .into_iter()
            .map(|(u, v)| json!({ "from": u, "to": v }))
            .collect::<Vec<_>>();

        let meta = graph_meta(&self.graph).await?;
        let pruned = self.args.prune && self.args.apply;

        info!(
            tool = REDUNDANT_REQUIRES,
            prune = pruned,
            count = total,
            offset = page_meta.offset,
            limit = page_meta.limit,
            has_more = page_meta.has_more,
            "graph redundant requires"
        );

        let payload = json!({
            "type": "graph_view",
            "tool": REDUNDANT_REQUIRES,
            "pruned": pruned,
            "total": total,
            "offset": page_meta.offset,
            "limit": page_meta.limit,
            "has_more": page_meta.has_more,
            "edges": edges,
        });
        let approx = payload_size_bytes(&payload);

        ToolRunner::new(REDUNDANT_REQUIRES, &self.state)
            .with_mode(ToolPayloadMode::Body)
            .with_meta(meta)
            .hints(vec![format!(
                "Set apply=true with prune=true to remove redundant edges ({} total).",
                total
            )])
            .run(move |_| async move {
                Ok(ToolRunPayload {
                    body:          payload,
                    approx_bytes:  Some(approx),
                    preview:       None,
                    preview_hints: Vec::new(),
                    page:          Some(page_meta),
                })
            })
            .await
    }
}
