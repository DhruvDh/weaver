use async_trait::async_trait;
use bon::Builder;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use tracing::info;

use super::{CachedGapBundle, cache_gap_bundle, decode_cached, load_graph_with_version};
use crate::tools::llm::{
    CallState, ToolExecutionError, ToolInstance, ToolOutput, ToolPayloadMode, ToolPrototype,
    common::{ToolRunPayload, ToolRunner},
    graph_tools::common,
    payload_size_bytes,
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

crate::analysis_tool!(
    gap_summary_meta,
    id: GAP_SUMMARY,
    description: "Compact summary of example gaps, fadeability issues, and practice gaps with \
                  preview/cost support.",
    args: GapSummaryArgs,
    prepare: |raw| crate::tools::llm::common::parse_args(GAP_SUMMARY, raw),
    runner: |args: GapSummaryArgs, state: &CallState| GapSummaryTool {
        args,
        state: state.clone(),
    }
);

crate::analysis_tool!(
    example_gaps_view_meta,
    id: EXAMPLE_GAPS_VIEW,
    description: "View example/variety gaps with pagination.",
    args: GapViewArgs,
    prepare: |raw| crate::tools::llm::common::parse_args(EXAMPLE_GAPS_VIEW, raw),
    runner: |args: GapViewArgs, state: &CallState| ExampleGapsViewTool {
        args,
        state: state.clone(),
    }
);

crate::analysis_tool!(
    fadeability_view_meta,
    id: FADEABILITY_VIEW,
    description: "View assessments that fail fadeability (supports acting as hidden \
                  prerequisites).",
    args: GapViewArgs,
    prepare: |raw| crate::tools::llm::common::parse_args(FADEABILITY_VIEW, raw),
    runner: |args: GapViewArgs, state: &CallState| FadeabilityViewTool {
        args,
        state: state.clone(),
    }
);

crate::analysis_tool!(
    practice_gaps_view_meta,
    id: PRACTICE_GAPS_VIEW,
    description: "View procedural practice gaps with pagination.",
    args: GapViewArgs,
    prepare: |raw| crate::tools::llm::common::parse_args(PRACTICE_GAPS_VIEW, raw),
    runner: |args: GapViewArgs, state: &CallState| PracticeGapsViewTool {
        args,
        state: state.clone(),
    }
);

struct GapSummaryTool {
    args:  GapSummaryArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for GapSummaryTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, graph_version) =
            load_graph_with_version(&self.state.graph, &self.state.analysis_cache).await?;
        let meta = common::graph_meta(&self.state.graph).await?;

        let cached = cache_gap_bundle(&self.state.analysis_cache, graph_version, &graph);
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

        let mode = ToolPayloadMode::from_fetch_flag(self.args.fetch_body);
        let approx = crate::tools::llm::payload_size_bytes(&payload);
        ToolRunner::new(GAP_SUMMARY, &self.state)
            .with_mode(mode)
            .with_meta(meta)
            .hints(vec![format!(
                "Summary is ~{} bytes; set fetch_body=true to retrieve it.",
                approx
            )])
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

struct ExampleGapsViewTool {
    args:  GapViewArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for ExampleGapsViewTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, graph_version) =
            load_graph_with_version(&self.state.graph, &self.state.analysis_cache).await?;
        let meta = common::graph_meta(&self.state.graph).await?;

        let cached = cache_gap_bundle(&self.state.analysis_cache, graph_version, &graph);
        let bundle: CachedGapBundle = decode_cached(&cached.payload)?;
        let gaps = bundle
            .example_gaps
            .into_iter()
            .map(|gap| json!({ "slug": gap.slug, "description": gap.description }))
            .collect::<Vec<_>>();

        let (gaps, page) = crate::paginate!(gaps, self.args.limit, self.args.offset);

        let payload = json!({
            "type": "graph_view",
            "tool": EXAMPLE_GAPS_VIEW,
            "offset": page.offset,
            "limit": page.limit,
            "has_more": page.has_more,
            "gaps": gaps,
        });

        info!(
            tool = EXAMPLE_GAPS_VIEW,
            offset = page.offset,
            limit = page.limit,
            has_more = page.has_more,
            "graph example gaps view"
        );
        let approx = payload_size_bytes(&payload);
        ToolRunner::new(EXAMPLE_GAPS_VIEW, &self.state)
            .with_mode(ToolPayloadMode::Body)
            .with_meta(meta)
            .run(move |_| async move {
                Ok(ToolRunPayload {
                    body:          payload,
                    approx_bytes:  Some(approx),
                    preview:       None,
                    preview_hints: Vec::new(),
                    page:          Some(page),
                })
            })
            .await
    }
}

struct FadeabilityViewTool {
    args:  GapViewArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for FadeabilityViewTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, graph_version) =
            load_graph_with_version(&self.state.graph, &self.state.analysis_cache).await?;
        let meta = common::graph_meta(&self.state.graph).await?;

        let cached = cache_gap_bundle(&self.state.analysis_cache, graph_version, &graph);
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

        let (issues, page) = crate::paginate!(issues, self.args.limit, self.args.offset);

        let payload = json!({
            "type": "graph_view",
            "tool": FADEABILITY_VIEW,
            "offset": page.offset,
            "limit": page.limit,
            "has_more": page.has_more,
            "assessments": issues,
        });

        info!(
            tool = FADEABILITY_VIEW,
            offset = page.offset,
            limit = page.limit,
            has_more = page.has_more,
            "graph fadeability view"
        );
        let approx = payload_size_bytes(&payload);
        ToolRunner::new(FADEABILITY_VIEW, &self.state)
            .with_mode(ToolPayloadMode::Body)
            .with_meta(meta)
            .run(move |_| async move {
                Ok(ToolRunPayload {
                    body:          payload,
                    approx_bytes:  Some(approx),
                    preview:       None,
                    preview_hints: Vec::new(),
                    page:          Some(page),
                })
            })
            .await
    }
}

struct PracticeGapsViewTool {
    args:  GapViewArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for PracticeGapsViewTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, graph_version) =
            load_graph_with_version(&self.state.graph, &self.state.analysis_cache).await?;
        let meta = common::graph_meta(&self.state.graph).await?;

        let cached = cache_gap_bundle(&self.state.analysis_cache, graph_version, &graph);
        let bundle: CachedGapBundle = decode_cached(&cached.payload)?;
        let gaps = bundle
            .practice_gaps
            .into_iter()
            .map(|g| json!({ "slug": g.slug }))
            .collect::<Vec<_>>();

        let (gaps, page) = crate::paginate!(gaps, self.args.limit, self.args.offset);

        let payload = json!({
            "type": "graph_view",
            "tool": PRACTICE_GAPS_VIEW,
            "offset": page.offset,
            "limit": page.limit,
            "has_more": page.has_more,
            "gaps": gaps,
        });

        info!(
            tool = PRACTICE_GAPS_VIEW,
            offset = page.offset,
            limit = page.limit,
            has_more = page.has_more,
            "graph practice gaps view"
        );
        let approx = payload_size_bytes(&payload);
        ToolRunner::new(PRACTICE_GAPS_VIEW, &self.state)
            .with_mode(ToolPayloadMode::Body)
            .with_meta(meta)
            .run(move |_| async move {
                Ok(ToolRunPayload {
                    body:          payload,
                    approx_bytes:  Some(approx),
                    preview:       None,
                    preview_hints: Vec::new(),
                    page:          Some(page),
                })
            })
            .await
    }
}
