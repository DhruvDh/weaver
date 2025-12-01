use std::sync::Arc;

use async_trait::async_trait;
use bon::Builder;
use kameo::prelude::ActorRef;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::info;

use super::{
    CachedGapBundle, SummaryContext, cache_gap_bundle, decode_cached, load_graph_with_version,
};
use crate::{
    graph::manager::GraphManager,
    llm_gateway::GatewayMetrics,
    tools::llm::{
        CallState, ToolExecutionError, ToolInputResult, ToolInstance, ToolOutput, ToolPrototype,
        graph_tools::{
            analysis_cache::AnalysisCache,
            common,
            common::{paginate, parse_with},
        },
        payload_size_bytes, schema_for_args,
    },
};

const GAP_SUMMARY: &str = "graph_gap_summary";
const EXAMPLE_GAPS_VIEW: &str = "graph_example_gaps_view";
const FADEABILITY_VIEW: &str = "graph_fadeability_view";
const PRACTICE_GAPS_VIEW: &str = "graph_practice_gaps_view";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GapSummaryArgs {
    #[serde(default)]
    #[schemars(
        description = "When true, return the summary payload; otherwise return a cost preview.",
        default = "crate::tools::llm::default_false"
    )]
    pub fetch_body: bool,
}

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GapViewArgs {
    #[serde(default)]
    #[schemars(description = "Maximum items to return (default 50, max 200).")]
    pub limit:  Option<usize>,
    #[serde(default)]
    #[schemars(description = "Offset into the result set (default 0).")]
    pub offset: Option<usize>,
}

pub(super) fn tool_prototypes() -> Vec<ToolPrototype> {
    vec![
        gap_summary_meta(),
        example_gaps_view_meta(),
        fadeability_view_meta(),
        practice_gaps_view_meta(),
    ]
}

pub(super) fn gap_summary_meta() -> ToolPrototype {
    ToolPrototype {
        id:          GAP_SUMMARY,
        description: "Compact summary of example gaps, fadeability issues, and practice gaps with \
                      preview/cost support.",
        schema:      schema_for_args::<GapSummaryArgs>(),
        parse:       parse_gap_summary,
    }
}

pub(super) fn example_gaps_view_meta() -> ToolPrototype {
    ToolPrototype {
        id:          EXAMPLE_GAPS_VIEW,
        description: "View example/variety gaps with pagination.",
        schema:      schema_for_args::<GapViewArgs>(),
        parse:       parse_example_gaps_view,
    }
}

pub(super) fn fadeability_view_meta() -> ToolPrototype {
    ToolPrototype {
        id:          FADEABILITY_VIEW,
        description: "View assessments that fail fadeability (supports acting as hidden \
                      prerequisites).",
        schema:      schema_for_args::<GapViewArgs>(),
        parse:       parse_fadeability_view,
    }
}

pub(super) fn practice_gaps_view_meta() -> ToolPrototype {
    ToolPrototype {
        id:          PRACTICE_GAPS_VIEW,
        description: "View procedural practice gaps with pagination.",
        schema:      schema_for_args::<GapViewArgs>(),
        parse:       parse_practice_gaps_view,
    }
}

fn parse_gap_summary(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_with(
        GAP_SUMMARY,
        raw,
        state,
        |args: GapSummaryArgs| Ok(args),
        |args, state| {
            Ok(GapSummaryTool {
                args,
                graph: state.graph.clone(),
                metrics: Arc::clone(&state.metrics),
                model: Arc::clone(&state.model),
                conversation_id: Arc::clone(&state.conversation_id),
                analysis_cache: Arc::clone(&state.analysis_cache),
            })
        },
    )
}

fn parse_example_gaps_view(
    raw: Value,
    state: &CallState,
) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_with(
        EXAMPLE_GAPS_VIEW,
        raw,
        state,
        |args: GapViewArgs| Ok(args),
        |args, state| {
            Ok(ExampleGapsViewTool {
                args,
                graph: state.graph.clone(),
                analysis_cache: Arc::clone(&state.analysis_cache),
            })
        },
    )
}

fn parse_fadeability_view(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_with(
        FADEABILITY_VIEW,
        raw,
        state,
        |args: GapViewArgs| Ok(args),
        |args, state| {
            Ok(FadeabilityViewTool {
                args,
                graph: state.graph.clone(),
                analysis_cache: Arc::clone(&state.analysis_cache),
            })
        },
    )
}

fn parse_practice_gaps_view(
    raw: Value,
    state: &CallState,
) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_with(
        PRACTICE_GAPS_VIEW,
        raw,
        state,
        |args: GapViewArgs| Ok(args),
        |args, state| {
            Ok(PracticeGapsViewTool {
                args,
                graph: state.graph.clone(),
                analysis_cache: Arc::clone(&state.analysis_cache),
            })
        },
    )
}

struct GapSummaryTool {
    args:            GapSummaryArgs,
    graph:           ActorRef<GraphManager>,
    metrics:         Arc<GatewayMetrics>,
    model:           Arc<String>,
    conversation_id: Arc<String>,
    analysis_cache:  Arc<AnalysisCache>,
}

#[async_trait]
impl ToolInstance for GapSummaryTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, graph_version) =
            load_graph_with_version(&self.graph, &self.analysis_cache).await?;
        let meta = common::graph_meta(&self.graph).await?;

        let cached = cache_gap_bundle(&self.analysis_cache, graph_version, &graph);
        let bundle: CachedGapBundle = decode_cached(&cached.payload)?;

        let payload = json!({
            "type": "graph_analysis",
            "tool": GAP_SUMMARY,
            "summary": {
                "example_gaps": {
                    "count": bundle.example_gaps.len(),
                },
                "fadeability": {
                    "assessments_with_issues": bundle.fadeability.len(),
                },
                "practice_gaps": {
                    "count": bundle.practice_gaps.len(),
                },
            }
        });

        super::finalize_summary_tool(
            GAP_SUMMARY,
            payload,
            self.args.fetch_body,
            SummaryContext {
                metrics:         self.metrics.as_ref(),
                model:           self.model.as_str(),
                conversation_id: self.conversation_id.as_str(),
                hint_prefix:     "Summary is",
                meta:            &meta,
            },
        )
    }
}

struct ExampleGapsViewTool {
    args:           GapViewArgs,
    graph:          ActorRef<GraphManager>,
    analysis_cache: Arc<AnalysisCache>,
}

#[async_trait]
impl ToolInstance for ExampleGapsViewTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, graph_version) =
            load_graph_with_version(&self.graph, &self.analysis_cache).await?;
        let meta = common::graph_meta(&self.graph).await?;

        let cached = cache_gap_bundle(&self.analysis_cache, graph_version, &graph);
        let bundle: CachedGapBundle = decode_cached(&cached.payload)?;
        let gaps = bundle
            .example_gaps
            .into_iter()
            .map(|gap| json!({ "slug": gap.slug, "description": gap.description }))
            .collect::<Vec<_>>();

        let (gaps, offset, limit, has_more) = paginate(gaps, self.args.limit, self.args.offset);

        let payload = json!({
            "type": "graph_view",
            "tool": EXAMPLE_GAPS_VIEW,
            "offset": offset,
            "limit": limit,
            "has_more": has_more,
            "gaps": gaps,
        });

        info!(tool = EXAMPLE_GAPS_VIEW, offset, limit, has_more, "graph example gaps view");

        let payload = common::attach_meta(payload, &meta);
        Ok(ToolOutput::with_byte_hint(payload.clone(), payload_size_bytes(&payload)))
    }
}

struct FadeabilityViewTool {
    args:           GapViewArgs,
    graph:          ActorRef<GraphManager>,
    analysis_cache: Arc<AnalysisCache>,
}

#[async_trait]
impl ToolInstance for FadeabilityViewTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, graph_version) =
            load_graph_with_version(&self.graph, &self.analysis_cache).await?;
        let meta = common::graph_meta(&self.graph).await?;

        let cached = cache_gap_bundle(&self.analysis_cache, graph_version, &graph);
        let bundle: CachedGapBundle = decode_cached(&cached.payload)?;
        let issues = bundle
            .fadeability
            .into_iter()
            .map(|i| {
                let supports: Vec<_> = i
                    .support_edges
                    .into_iter()
                    .map(|e| json!({ "from": e.from, "to": e.to }))
                    .collect();
                json!({
                    "assessment_slug": i.assessment_slug,
                    "support_edges": supports,
                })
            })
            .collect::<Vec<_>>();

        let (issues, offset, limit, has_more) = paginate(issues, self.args.limit, self.args.offset);

        let payload = json!({
            "type": "graph_view",
            "tool": FADEABILITY_VIEW,
            "offset": offset,
            "limit": limit,
            "has_more": has_more,
            "assessments": issues,
        });

        info!(tool = FADEABILITY_VIEW, offset, limit, has_more, "graph fadeability view");

        let payload = common::attach_meta(payload, &meta);
        Ok(ToolOutput::with_byte_hint(payload.clone(), payload_size_bytes(&payload)))
    }
}

struct PracticeGapsViewTool {
    args:           GapViewArgs,
    graph:          ActorRef<GraphManager>,
    analysis_cache: Arc<AnalysisCache>,
}

#[async_trait]
impl ToolInstance for PracticeGapsViewTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, graph_version) =
            load_graph_with_version(&self.graph, &self.analysis_cache).await?;
        let meta = common::graph_meta(&self.graph).await?;

        let cached = cache_gap_bundle(&self.analysis_cache, graph_version, &graph);
        let bundle: CachedGapBundle = decode_cached(&cached.payload)?;
        let gaps = bundle
            .practice_gaps
            .into_iter()
            .map(|g| json!({ "slug": g.slug }))
            .collect::<Vec<_>>();

        let (gaps, offset, limit, has_more) = paginate(gaps, self.args.limit, self.args.offset);

        let payload = json!({
            "type": "graph_view",
            "tool": PRACTICE_GAPS_VIEW,
            "offset": offset,
            "limit": limit,
            "has_more": has_more,
            "gaps": gaps,
        });

        info!(tool = PRACTICE_GAPS_VIEW, offset, limit, has_more, "graph practice gaps view");

        let payload = common::attach_meta(payload, &meta);
        Ok(ToolOutput::with_byte_hint(payload.clone(), payload_size_bytes(&payload)))
    }
}
