use std::sync::Arc;

use async_trait::async_trait;
use bon::Builder;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::info;

use super::load_graph_with_version;
use crate::{
    analysis,
    tools::llm::{
        CallState, ToolExecutionError, ToolInstance, ToolOutput, ToolPayloadMode, ToolPrototype,
        common::{ToolRunPayload, ToolRunner},
        graph_tools::analysis_cache::{
            AnalysisCacheKey, AnalysisKind, with_cached_analysis_result,
        },
        payload_size_bytes, require_string,
    },
};

const DISCOURSE_ORPHANS: &str = "graph_discourse_orphans";
const BORROW_AHEAD: &str = "graph_borrow_ahead";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DiscourseOrphansArgs {
    #[serde(default)]
    #[schemars(description = "Optional episode filter; when set, only check this episode.")]
    pub episode: Option<String>,
}

crate::analysis_tool!(
    discourse_orphans_meta,
    id: DISCOURSE_ORPHANS,
    description: "Discourse continuity: list TeachingSteps with no precedes links in their \
                  episode (orphans). Optionally filter by episode.",
    args: DiscourseOrphansArgs,
    prepare: |raw| crate::tools::llm::common::parse_args(DISCOURSE_ORPHANS, raw),
    runner: |args: DiscourseOrphansArgs, state: &CallState| DiscourseOrphansTool {
        args,
        state: state.clone(),
    }
);

struct DiscourseOrphansTool {
    args:  DiscourseOrphansArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for DiscourseOrphansTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, graph_version) =
            load_graph_with_version(&self.state.graph, &self.state.analysis_cache).await?;
        let meta = super::common::graph_meta(&self.state.graph).await?;

        let cache_key = AnalysisCacheKey {
            graph_version,
            kind: AnalysisKind::DiscourseOrphans {
                episode: self.args.episode.clone(),
            },
        };

        let payload = with_cached_analysis_result(
            &self.state.analysis_cache,
            cache_key,
            graph_version,
            || {
                let graph = Arc::clone(&graph);
                let episode = self.args.episode.clone();
                async move {
                    let list = analysis::discourse_orphans(&graph, episode.as_deref());
                    let slugs: Vec<_> = list.into_iter().map(|n| graph[n].slug.clone()).collect();
                    Ok::<Value, ToolExecutionError>(json!({
                        "type": "graph_analysis",
                        "tool": DISCOURSE_ORPHANS,
                        "episode": episode,
                        "orphans": slugs,
                    }))
                }
            },
        )
        .await?;

        let orphan_count = payload["orphans"]
            .as_array()
            .map(|v: &Vec<_>| v.len())
            .unwrap_or(0);
        info!(
            tool = DISCOURSE_ORPHANS,
            episode = ?self.args.episode,
            orphan_count,
            "graph discourse orphans"
        );
        let approx = payload_size_bytes(&payload);

        ToolRunner::new(DISCOURSE_ORPHANS, &self.state)
            .with_mode(ToolPayloadMode::Body)
            .with_meta(meta)
            .run(move |_| async move {
                Ok(ToolRunPayload {
                    body:          payload,
                    approx_bytes:  Some(approx),
                    preview:       None,
                    preview_hints: Vec::new(),
                    page:          None,
                })
            })
            .await
    }
}

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BorrowAheadArgs {
    #[schemars(description = "Episode identifier to scan for borrow-ahead uses.")]
    pub episode: String,
}

crate::analysis_tool!(
    borrow_ahead_meta,
    id: BORROW_AHEAD,
    description: "Borrow-ahead detection within an episode: steps that use knowledge before \
                  introduction in-scope, classified by introduction_scope and ordering \
                  severity.",
    args: BorrowAheadArgs,
    prepare: |raw| super::common::parse_args_with_builder(
        BORROW_AHEAD,
        raw,
        |mut input: BorrowAheadArgs| {
            input.episode = require_string(input.episode, BORROW_AHEAD, "episode")?;
            Ok(input)
        },
    ),
    runner: |args: BorrowAheadArgs, state: &CallState| BorrowAheadTool {
        args,
        state: state.clone(),
    }
);

struct BorrowAheadTool {
    args:  BorrowAheadArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for BorrowAheadTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, graph_version) =
            load_graph_with_version(&self.state.graph, &self.state.analysis_cache).await?;
        let meta = super::common::graph_meta(&self.state.graph).await?;

        let cache_key = AnalysisCacheKey {
            graph_version,
            kind: AnalysisKind::BorrowAhead {
                episode: self.args.episode.clone(),
            },
        };

        let payload = with_cached_analysis_result(
            &self.state.analysis_cache,
            cache_key,
            graph_version,
            || {
                let graph = Arc::clone(&graph);
                let episode = self.args.episode.clone();
                async move {
                    let results = analysis::borrow_ahead(&graph, &episode);
                    let rendered: Vec<_> = results
                        .into_iter()
                        .map(|b| {
                            json!({
                                "step_slug": graph[b.step].slug.clone(),
                                "target_slug": graph[b.target].slug.clone(),
                                "severity": b.severity,
                            })
                        })
                        .collect();
                    Ok::<Value, ToolExecutionError>(json!({
                        "type": "graph_analysis",
                        "tool": BORROW_AHEAD,
                        "episode": episode,
                        "borrow_ahead": rendered,
                    }))
                }
            },
        )
        .await?;

        let count = payload["borrow_ahead"]
            .as_array()
            .map(|v: &Vec<_>| v.len())
            .unwrap_or(0);
        info!(
            tool = BORROW_AHEAD,
            episode = %self.args.episode,
            count,
            "graph borrow ahead"
        );
        let approx = payload_size_bytes(&payload);

        ToolRunner::new(BORROW_AHEAD, &self.state)
            .with_mode(ToolPayloadMode::Body)
            .with_meta(meta)
            .run(move |_| async move {
                Ok(ToolRunPayload {
                    body:          payload,
                    approx_bytes:  Some(approx),
                    preview:       None,
                    preview_hints: Vec::new(),
                    page:          None,
                })
            })
            .await
    }
}

pub(super) fn tool_prototypes() -> Vec<ToolPrototype> {
    vec![discourse_orphans_meta(), borrow_ahead_meta()]
}
