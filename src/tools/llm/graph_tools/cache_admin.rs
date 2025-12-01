use std::sync::Arc;

use async_trait::async_trait;
use kameo::prelude::ActorRef;
use serde_json::{Value, json};

use super::{
    analysis_cache::AnalysisCache,
    common::{attach_meta, graph_meta, parse_args_with_builder},
};
use crate::tools::llm::{
    CallState, ToolExecutionError, ToolInputResult, ToolInstance, ToolOutput, ToolPrototype,
    schema_for_args,
};

const CLEAR_CACHE: &str = "graph_analysis_cache_clear";

#[derive(Debug, Clone, serde::Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct ClearCacheArgs;

pub(super) fn tool_prototypes() -> Vec<ToolPrototype> {
    vec![ToolPrototype {
        id:          CLEAR_CACHE,
        description: "Clear all analysis cache entries (ops/admin tool).",
        schema:      schema_for_args::<ClearCacheArgs>(),
        parse:       parse_clear_cache,
    }]
}

fn parse_clear_cache(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_args_with_builder(CLEAR_CACHE, raw, |args: ClearCacheArgs| Ok(args))?;
    Ok(Box::new(ClearCacheTool {
        analysis_cache: Arc::clone(&state.analysis_cache),
        graph:          state.graph.clone(),
    }))
}

struct ClearCacheTool {
    analysis_cache: Arc<AnalysisCache>,
    graph:          ActorRef<crate::graph::manager::GraphManager>,
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
        let payload = attach_meta(
            json!({
                "type": "graph_admin",
                "tool": CLEAR_CACHE,
                "cleared": before,
                "remaining": after,
            }),
            &meta,
        );

        Ok(ToolOutput::new(payload))
    }
}
