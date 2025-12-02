use std::sync::Arc;

use async_trait::async_trait;
use kameo::prelude::ActorRef;
use serde_json::json;

use super::{
    analysis_cache::AnalysisCache,
    common::{graph_meta, parse_args_with_builder},
};
use crate::tools::llm::{
    CallState, ToolExecutionError, ToolInstance, ToolOutput, ToolPayloadMode, ToolPrototype,
    common::{ToolRunPayload, ToolRunner},
    payload_size_bytes,
};

const CLEAR_CACHE: &str = "graph_analysis_cache_clear";

#[derive(Debug, Clone, serde::Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct ClearCacheArgs {}

pub(super) fn tool_prototypes() -> Vec<ToolPrototype> {
    vec![clear_cache_meta()]
}

crate::analysis_tool!(
    clear_cache_meta,
    id: CLEAR_CACHE,
    description: "Clear all analysis cache entries (ops/admin tool).",
    args: ClearCacheArgs,
    prepare: |raw| parse_args_with_builder(CLEAR_CACHE, raw, |args: ClearCacheArgs| Ok(args)),
    runner: |_: ClearCacheArgs, state: &CallState| ClearCacheTool {
        analysis_cache: Arc::clone(&state.analysis_cache),
        graph: state.graph.clone(),
        state: state.clone(),
    }
);

struct ClearCacheTool {
    analysis_cache: Arc<AnalysisCache>,
    graph:          ActorRef<crate::graph::manager::GraphManager>,
    state:          CallState,
}

#[async_trait]
impl ToolInstance for ClearCacheTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let before = self.analysis_cache.len() as u64;
        self.analysis_cache.clear_all();
        let after = self.analysis_cache.len() as u64;

        tracing::info!(
            target: "weaver.analysis_cache",
            cleared = before,
            remaining = after,
            "analysis cache cleared via admin tool"
        );

        let meta = graph_meta(&self.graph).await?;
        let body = json!({
            "type": "graph_admin",
            "tool": CLEAR_CACHE,
            "cleared": before,
            "remaining": after,
        });
        let approx = payload_size_bytes(&body);

        ToolRunner::new(CLEAR_CACHE, &self.state)
            .with_mode(ToolPayloadMode::Body)
            .with_meta(meta)
            .hints(vec![format!("Cleared {before} entries; {after} remain.")])
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
