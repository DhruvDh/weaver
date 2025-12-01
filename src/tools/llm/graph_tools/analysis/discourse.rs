use std::sync::Arc;

use async_trait::async_trait;
use bon::Builder;
use kameo::prelude::ActorRef;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::info;

use super::load_graph_with_version;
use crate::{
    analysis,
    graph::manager::GraphManager,
    tools::llm::{
        CallState, ToolExecutionError, ToolInputResult, ToolInstance, ToolOutput, ToolPrototype,
        graph_tools::{
            analysis_cache::{AnalysisCache, AnalysisCacheKey, AnalysisKind},
            common::parse_with,
        },
        payload_size_bytes, require_string, schema_for_args,
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

pub(super) fn discourse_orphans_meta() -> ToolPrototype {
    ToolPrototype {
        id:          DISCOURSE_ORPHANS,
        description: "Discourse continuity: list TeachingSteps with no precedes links in their \
                      episode (orphans). Optionally filter by episode.",
        schema:      schema_for_args::<DiscourseOrphansArgs>(),
        parse:       parse_discourse_orphans,
    }
}

fn parse_discourse_orphans(
    raw: Value,
    state: &CallState,
) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_with(
        DISCOURSE_ORPHANS,
        raw,
        state,
        |args: DiscourseOrphansArgs| Ok(args),
        |args, state| {
            Ok(DiscourseOrphansTool {
                args,
                graph: state.graph.clone(),
                analysis_cache: Arc::clone(&state.analysis_cache),
            })
        },
    )
}

struct DiscourseOrphansTool {
    args:           DiscourseOrphansArgs,
    graph:          ActorRef<GraphManager>,
    analysis_cache: Arc<AnalysisCache>,
}

#[async_trait]
impl ToolInstance for DiscourseOrphansTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, graph_version) =
            load_graph_with_version(&self.graph, &self.analysis_cache).await?;

        let cache_key = AnalysisCacheKey {
            graph_version,
            kind: AnalysisKind::DiscourseOrphans {
                episode: self.args.episode.clone(),
            },
        };

        let cached = self.analysis_cache.get_or_insert_with(cache_key, || {
            let list = analysis::discourse_orphans(&graph, self.args.episode.as_deref());
            let slugs: Vec<_> = list.into_iter().map(|n| graph[n].slug.clone()).collect();
            json!({
                "type": "graph_analysis",
                "tool": DISCOURSE_ORPHANS,
                "episode": self.args.episode,
                "orphans": slugs,
            })
        });
        let payload = cached.payload.clone();
        info!(
            tool = DISCOURSE_ORPHANS,
            episode = ?self.args.episode,
            orphan_count = payload["orphans"].as_array().map(|v: &Vec<_>| v.len()).unwrap_or(0),
            "graph discourse orphans"
        );
        Ok(ToolOutput::with_byte_hint(payload.clone(), payload_size_bytes(&payload)))
    }
}

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BorrowAheadArgs {
    #[schemars(description = "Episode identifier to scan for borrow-ahead uses.")]
    pub episode: String,
}

pub(super) fn borrow_ahead_meta() -> ToolPrototype {
    ToolPrototype {
        id:          BORROW_AHEAD,
        description: "Borrow-ahead detection within an episode: steps that use knowledge before \
                      introduction in-scope, classified by introduction_scope and ordering \
                      severity.",
        schema:      schema_for_args::<BorrowAheadArgs>(),
        parse:       parse_borrow_ahead,
    }
}

fn parse_borrow_ahead(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_with(
        BORROW_AHEAD,
        raw,
        state,
        |mut input: BorrowAheadArgs| {
            input.episode = require_string(input.episode, BORROW_AHEAD, "episode")?;
            Ok(input)
        },
        |args, state| {
            Ok(BorrowAheadTool {
                args,
                graph: state.graph.clone(),
                analysis_cache: Arc::clone(&state.analysis_cache),
            })
        },
    )
}

struct BorrowAheadTool {
    args:           BorrowAheadArgs,
    graph:          ActorRef<GraphManager>,
    analysis_cache: Arc<AnalysisCache>,
}

#[async_trait]
impl ToolInstance for BorrowAheadTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, graph_version) =
            load_graph_with_version(&self.graph, &self.analysis_cache).await?;

        let cache_key = AnalysisCacheKey {
            graph_version,
            kind: AnalysisKind::BorrowAhead {
                episode: self.args.episode.clone(),
            },
        };

        let cached = self.analysis_cache.get_or_insert_with(cache_key, || {
            let results = analysis::borrow_ahead(&graph, &self.args.episode);
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
            json!({
                "type": "graph_analysis",
                "tool": BORROW_AHEAD,
                "episode": self.args.episode,
                "borrow_ahead": rendered,
            })
        });
        let payload = cached.payload.clone();
        info!(
            tool = BORROW_AHEAD,
            episode = %self.args.episode,
            count = payload["borrow_ahead"].as_array().map(|v: &Vec<_>| v.len()).unwrap_or(0),
            "graph borrow ahead"
        );
        Ok(ToolOutput::with_byte_hint(payload.clone(), payload_size_bytes(&payload)))
    }
}

pub(super) fn tool_prototypes() -> Vec<ToolPrototype> {
    vec![discourse_orphans_meta(), borrow_ahead_meta()]
}
