use async_trait::async_trait;
use bon::Builder;
use kameo::prelude::ActorRef;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::info;

use super::common::{attach_meta, graph_meta, map_send_err, parse_args_with_builder};
use crate::{
    graph::manager::RedundantRequires,
    tools::llm::{
        CallState, ToolExecutionError, ToolInputResult, ToolInstance, ToolPrototype,
        schema_for_args,
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
    vec![ToolPrototype {
        id:          REDUNDANT_REQUIRES,
        description: "List redundant requires edges (edges removable without changing \
                      reachability). Optionally prune them.",
        schema:      schema_for_args::<RedundantRequiresArgs>(),
        parse:       parse_redundant_requires,
    }]
}

fn parse_redundant_requires(
    raw: Value,
    state: &CallState,
) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args =
        parse_args_with_builder(REDUNDANT_REQUIRES, raw, |args: RedundantRequiresArgs| Ok(args))?;
    Ok(Box::new(RedundantRequiresTool {
        args,
        graph: state.graph.clone(),
    }))
}

struct RedundantRequiresTool {
    args:  RedundantRequiresArgs,
    graph: ActorRef<crate::graph::manager::GraphManager>,
}

#[async_trait]
impl ToolInstance for RedundantRequiresTool {
    async fn execute(&self) -> Result<crate::tools::llm::ToolOutput, ToolExecutionError> {
        let edges: Vec<(String, String)> = self
            .graph
            .ask(RedundantRequires {
                prune: self.args.prune && self.args.apply,
            })
            .await
            .map_err(|e| map_send_err(e, REDUNDANT_REQUIRES))?;

        let limit = self.args.limit.unwrap_or(200).min(500);
        let offset = self.args.offset.unwrap_or(0).min(edges.len());
        let slice = edges
            .iter()
            .skip(offset)
            .take(limit)
            .map(|(u, v)| json!({"from": u, "to": v}))
            .collect::<Vec<_>>();

        let meta = graph_meta(&self.graph).await?;
        if self.args.prune && !self.args.apply {
            let preview = attach_meta(
                json!({
                    "type": "graph_view",
                    "tool": REDUNDANT_REQUIRES,
                    "status": "preview",
                    "pruned": false,
                    "apply_hint": "Set apply=true to prune redundant requires edges.",
                    "total": edges.len(),
                    "offset": offset,
                    "limit": limit,
                    "edges": slice,
                }),
                &meta,
            );
            return Ok(crate::tools::llm::ToolOutput::new(preview));
        }

        info!(
            tool = REDUNDANT_REQUIRES,
            prune = self.args.prune,
            count = edges.len(),
            offset,
            limit,
            "graph redundant requires"
        );

        let payload = attach_meta(
            json!({
                "type": "graph_view",
                "tool": REDUNDANT_REQUIRES,
                "pruned": self.args.prune && self.args.apply,
                "total": edges.len(),
                "offset": offset,
                "limit": limit,
                "edges": slice,
            }),
            &meta,
        );

        Ok(crate::tools::llm::ToolOutput::new(payload))
    }
}
