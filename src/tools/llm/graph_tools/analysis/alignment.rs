use std::{collections::HashSet, sync::Arc};

use async_trait::async_trait;
use bon::Builder;
use petgraph::visit::EdgeRef;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::info;

use super::{
    cache_lo_bundle, decode_cached, load_graph_with_version, load_lo_with_graph,
    load_lo_with_graph_only,
};
use crate::{
    analysis,
    graph::CurriculumGraph,
    schema::types::KnowledgeType,
    tools::llm::{
        CallState, ToolExecutionError, ToolInstance, ToolOutput, ToolPayloadMode, ToolPrototype,
        common::{
            AssessmentItem, LearningOutcome, Slug, ToolRunPayload, ToolRunner, resolve_typed,
        },
        graph_tools::{
            analysis::{CachedLoBundle, NoArgs},
            analysis_cache::{
                AnalysisCacheKey, AnalysisKind, with_cached_analysis, with_cached_analysis_result,
            },
            common,
            common::{map_send_err_inf, resolve_slugs},
        },
        payload_size_bytes, require_string,
    },
};

// ---------- DAG check ----------

const DAG_CHECK: &str = "graph_dag_check";

crate::analysis_tool!(
    dag_check_meta,
    id: DAG_CHECK,
    description: "Verify that 'requires' edges form an acyclic DAG (Directed Acyclic Graph). Returns is_dag=true if valid, false if cycles exist. ALWAYS run after adding requires edges. If cycles detected, identify and remove the weakest edge (prefer removing 'helpful' before 'strong', 'strong' before 'necessary').",
    args: NoArgs,
    prepare: |raw| crate::tools::llm::common::parse_args(DAG_CHECK, raw),
    runner: |_: NoArgs, state: &CallState| DAGCheckTool {
        state: state.clone(),
    }
);

struct DAGCheckTool {
    state: CallState,
}

#[async_trait]
impl ToolInstance for DAGCheckTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, graph_version) =
            load_graph_with_version(&self.state.graph, &self.state.analysis_cache).await?;
        let meta = common::graph_meta(&self.state.graph).await?;

        let cache_key = AnalysisCacheKey {
            graph_version,
            kind: AnalysisKind::DagCheck,
        };

        let graph = Arc::clone(&graph);
        let payload = with_cached_analysis(
            &self.state.analysis_cache,
            cache_key,
            graph_version,
            move || async move {
                let dag = analysis::requires_is_dag(&graph);
                let topo = analysis::requires_toposort(&graph).ok();
                json!({
                    "type": "graph_analysis",
                    "tool": DAG_CHECK,
                    "is_dag": dag,
                    "topo_order_count": topo.as_ref().map(|v| v.len()),
                })
            },
        )
        .await;

        info!(
            tool = DAG_CHECK,
            is_dag = payload["is_dag"].as_bool().unwrap_or(false),
            topo_order_count = payload["topo_order_count"].as_u64(),
            "graph requires dag check"
        );

        ToolRunner::new(DAG_CHECK, &self.state)
            .with_mode(crate::tools::llm::ToolPayloadMode::Body)
            .with_meta(meta)
            .run(|_| async move { Ok(ToolRunPayload::new(payload)) })
            .await
    }
}

// ---------- First principles ----------

const FIRST_PRINCIPLES_VIEW: &str = "graph_first_principles"; // backward-compatible id
const FIRST_PRINCIPLES_SUMMARY: &str = "graph_first_principles_summary";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FirstPrinciplesViewArgs {
    #[serde(default)]
    #[schemars(description = "Max results (default 50, max 200).")]
    pub limit:  Option<usize>,
    #[serde(default)]
    #[schemars(description = "Pagination offset.")]
    pub offset: Option<usize>,
}

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FirstPrinciplesSummaryArgs {
    #[serde(default)]
    #[schemars(
        description = "Return payload if true.",
        default = "crate::tools::llm::default_false"
    )]
    pub fetch_body: bool,
}

crate::analysis_tool!(
    first_principles_meta,
    id: FIRST_PRINCIPLES_VIEW,
    description: "List 'first principles'—knowledge nodes with no incoming requires edges. These are the foundational concepts that don't depend on other course content. They form the starting points for learning paths. All assessments should be reachable from first principles via requires edges.",
    args: FirstPrinciplesViewArgs,
    prepare: |raw| crate::tools::llm::common::parse_args(FIRST_PRINCIPLES_VIEW, raw),
    runner: |args: FirstPrinciplesViewArgs, state: &CallState| FirstPrinciplesViewTool {
        args,
        state: state.clone(),
    }
);

crate::analysis_tool!(
    first_principles_summary_meta,
    id: FIRST_PRINCIPLES_SUMMARY,
    description: "Summary statistics of first-principle nodes by knowledge type. Shows how many factual, conceptual, procedural, and metacognitive nodes have no prerequisites. Useful for understanding the foundation layer of the curriculum.",
    args: FirstPrinciplesSummaryArgs,
    prepare: |raw| crate::tools::llm::common::parse_args(FIRST_PRINCIPLES_SUMMARY, raw),
    runner: |args: FirstPrinciplesSummaryArgs, state: &CallState| FirstPrinciplesSummaryTool {
        args,
        state: state.clone(),
    }
);

struct FirstPrinciplesViewTool {
    args:  FirstPrinciplesViewArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for FirstPrinciplesViewTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph: Arc<CurriculumGraph> = self
            .state
            .graph
            .ask(crate::graph::commands::GetGraph)
            .await
            .map_err(map_send_err_inf)?;
        let meta = common::graph_meta(&self.state.graph).await?;
        let mode = ToolPayloadMode::Body;
        let fps = analysis::first_principles(&graph);
        let items: Vec<_> = fps
            .iter()
            .map(|id| {
                let kt = match &graph[*id].kind {
                    crate::graph::NodeKind::Knowledge(k) => k.knowledge_type,
                    _ => KnowledgeType::Conceptual,
                };
                json!({ "slug": graph[*id].slug, "knowledge_type": kt })
            })
            .collect();

        let (items, page) = crate::paginate!(items, self.args.limit, self.args.offset);

        let payload = json!({
            "type": "graph_view",
            "tool": FIRST_PRINCIPLES_VIEW,
            "offset": page.offset,
            "limit": page.limit,
            "has_more": page.has_more,
            "items": items,
        });

        info!(
            tool = FIRST_PRINCIPLES_VIEW,
            offset = page.offset,
            limit = page.limit,
            has_more = page.has_more,
            "graph first_principles view"
        );

        let approx = payload_size_bytes(&payload);
        ToolRunner::new(FIRST_PRINCIPLES_VIEW, &self.state)
            .with_mode(mode)
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

struct FirstPrinciplesSummaryTool {
    args:  FirstPrinciplesSummaryArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for FirstPrinciplesSummaryTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph: Arc<CurriculumGraph> = self
            .state
            .graph
            .ask(crate::graph::commands::GetGraph)
            .await
            .map_err(map_send_err_inf)?;
        let meta = common::graph_meta(&self.state.graph).await?;

        let fps = analysis::first_principles(&graph);
        let mut counts = std::collections::HashMap::new();
        for id in &fps {
            if let crate::graph::NodeKind::Knowledge(k) = &graph[*id].kind {
                *counts.entry(k.knowledge_type).or_insert(0usize) += 1;
            }
        }

        let payload = json!({
            "type": "graph_analysis",
            "tool": FIRST_PRINCIPLES_SUMMARY,
            "summary": {
                "total_instructional_knowledge": graph
                    .node_indices()
                    .filter(|n| matches!(&graph[*n].kind, crate::graph::NodeKind::Knowledge(k) if k.knowledge_type.is_instructional_knowledge()))
                    .count(),
                "first_principles_count": fps.len(),
                "by_knowledge_type": counts,
            }
        });

        let mode = ToolPayloadMode::from_fetch_flag(self.args.fetch_body);
        let approx = payload_size_bytes(&payload);
        ToolRunner::new(FIRST_PRINCIPLES_SUMMARY, &self.state)
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

// ---------- LO reachability / coverage / alignment ----------

const LO_REACH: &str = "graph_lo_reachability";
const LO_ALIGNMENT: &str = "graph_lo_alignment_summary";
const COVERAGE: &str = "graph_lo_coverage";
const LO_ASSESSMENTS_VIEW: &str = "graph_lo_assessments_view";
const LO_MISSING_CRITERIA_VIEW: &str = "graph_lo_missing_criteria_view";
const LO_ANCHORS_VIEW: &str = "graph_lo_anchors_view";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LoReachArgs {
    #[schemars(description = "LO slug.")]
    pub lo_slug: String,
}

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LoAssessmentsViewArgs {
    #[schemars(description = "LO slug.")]
    pub lo_slug:        String,
    #[serde(default)]
    #[schemars(description = "true = reachable only; false = unreachable only.")]
    pub reachable_only: Option<bool>,
    #[serde(default)]
    #[schemars(description = "Max results (default 50, max 200).")]
    pub limit:          Option<usize>,
    #[serde(default)]
    #[schemars(description = "Pagination offset.")]
    pub offset:         Option<usize>,
}

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LoMissingCriteriaViewArgs {
    #[schemars(description = "LO slug.")]
    pub lo_slug: String,
    #[serde(default)]
    #[schemars(description = "Max results (default 50, max 200).")]
    pub limit:   Option<usize>,
    #[serde(default)]
    #[schemars(description = "Pagination offset.")]
    pub offset:  Option<usize>,
}

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LoAnchorsViewArgs {
    #[schemars(description = "LO slug.")]
    pub lo_slug: String,
    #[serde(default)]
    #[schemars(description = "Max results (default 50, max 200).")]
    pub limit:   Option<usize>,
    #[serde(default)]
    #[schemars(description = "Pagination offset.")]
    pub offset:  Option<usize>,
}

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LoAlignmentArgs {
    #[schemars(description = "LO slug.")]
    pub lo_slug:    String,
    #[serde(default)]
    #[schemars(
        description = "Return payload if true.",
        default = "crate::tools::llm::default_false"
    )]
    pub fetch_body: bool,
}
crate::analysis_tool!(
    lo_reach_meta,
    id: LO_REACH,
    description: "Check if a Learning Outcome's assessments are reachable from first principles via requires edges. Each assessment shows whether it can be reached. If not reachable, the LO may not be properly aligned (students couldn't learn prerequisites to attempt assessment).",
    args: LoReachArgs,
    prepare: |raw| super::common::parse_args_with_builder(
        LO_REACH,
        raw,
        |mut input: LoReachArgs| {
            input.lo_slug = require_string(input.lo_slug, LO_REACH, "lo_slug")?;
            Ok(input)
        },
    ),
    runner: |args: LoReachArgs, state: &CallState| LoReachTool {
        args,
        state: state.clone(),
    }
);

crate::analysis_tool!(
    lo_alignment_meta,
    id: LO_ALIGNMENT,
    description: "Comprehensive alignment summary for a Learning Outcome. Shows: (1) assessment reachability from first principles, (2) rubric criteria coverage by observation_features, (3) TeachingStep anchors with target impact. Use to verify an LO is properly connected and measurable.",
    args: LoAlignmentArgs,
    prepare: |raw| super::common::parse_args_with_builder(
        LO_ALIGNMENT,
        raw,
        |mut input: LoAlignmentArgs| {
            input.lo_slug = require_string(input.lo_slug, LO_ALIGNMENT, "lo_slug")?;
            Ok(input)
        },
    ),
    runner: |args: LoAlignmentArgs, state: &CallState| LoAlignmentTool {
        args,
        state: state.clone(),
    }
);

crate::analysis_tool!(
    lo_assessments_view_meta,
    id: LO_ASSESSMENTS_VIEW,
    description: "List all AssessmentItems linked to a LearningOutcome via assesses edges. Shows which assessments are reachable from first principles. Filter by reachable_only=true/false. Every LO should have at least one reachable assessment with scope=target.",
    args: LoAssessmentsViewArgs,
    prepare: |raw| super::common::parse_args_with_builder(
        LO_ASSESSMENTS_VIEW,
        raw,
        |mut input: LoAssessmentsViewArgs| {
            input.lo_slug = require_string(input.lo_slug, LO_ASSESSMENTS_VIEW, "lo_slug")?;
            Ok(input)
        },
    ),
    runner: |args: LoAssessmentsViewArgs, state: &CallState| LoAssessmentsViewTool {
        args,
        state: state.clone(),
    }
);

crate::analysis_tool!(
    lo_missing_criteria_view_meta,
    id: LO_MISSING_CRITERIA_VIEW,
    description: "List uncovered rubric criteria for LO.",
    args: LoMissingCriteriaViewArgs,
    prepare: |raw| super::common::parse_args_with_builder(
        LO_MISSING_CRITERIA_VIEW,
        raw,
        |mut input: LoMissingCriteriaViewArgs| {
            input.lo_slug = require_string(input.lo_slug, LO_MISSING_CRITERIA_VIEW, "lo_slug")?;
            Ok(input)
        },
    ),
    runner: |args: LoMissingCriteriaViewArgs, state: &CallState| LoMissingCriteriaViewTool {
        args,
        state: state.clone(),
    }
);

crate::analysis_tool!(
    lo_anchors_view_meta,
    id: LO_ANCHORS_VIEW,
    description: "List TeachingSteps anchoring to LO.",
    args: LoAnchorsViewArgs,
    prepare: |raw| super::common::parse_args_with_builder(
        LO_ANCHORS_VIEW,
        raw,
        |mut input: LoAnchorsViewArgs| {
            input.lo_slug = require_string(input.lo_slug, LO_ANCHORS_VIEW, "lo_slug")?;
            Ok(input)
        },
    ),
    runner: |args: LoAnchorsViewArgs, state: &CallState| LoAnchorsViewTool {
        args,
        state: state.clone(),
    }
);

crate::analysis_tool!(
    coverage_meta,
    id: COVERAGE,
    description: "Show rubric coverage for a LearningOutcome. For each rubric_criterion, shows which assesses edges' observation_features cover it. Missing coverage means that criterion isn't being measured by any assessment. Add observation_features to existing assesses edges or create new assessments.",
    args: LoReachArgs,
    prepare: |raw| super::common::parse_args_with_builder(
        COVERAGE,
        raw,
        |mut input: LoReachArgs| {
            input.lo_slug = require_string(input.lo_slug, COVERAGE, "lo_slug")?;
            Ok(input)
        },
    ),
    runner: |args: LoReachArgs, state: &CallState| LoCoverageTool {
        args,
        state: state.clone(),
    }
);

struct LoReachTool {
    args:  LoReachArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for LoReachTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let mode = ToolPayloadMode::Body;
        let (graph, graph_version, lo) = load_lo_with_graph(
            &self.state.graph,
            &self.state.analysis_cache,
            &self.args.lo_slug,
            LO_REACH,
        )
        .await?;
        let meta = common::graph_meta(&self.state.graph).await?;

        let cached = cache_lo_bundle(
            &self.state.analysis_cache,
            graph_version,
            &graph,
            lo,
            &self.args.lo_slug,
        );
        let bundle: CachedLoBundle = decode_cached(&cached.payload)?;

        let assessments = bundle
            .assessments
            .into_iter()
            .map(|a| {
                json!({
                    "assessment_slug": a.assessment_slug,
                    "reachable_from_first_principle": a.reachable_from_first_principle,
                })
            })
            .collect::<Vec<_>>();
        let payload = json!({
            "type": "graph_analysis",
            "tool": LO_REACH,
            "lo_slug": self.args.lo_slug,
            "assessments": assessments,
        });
        info!(tool = LO_REACH, lo_slug = %self.args.lo_slug, "graph lo reachability");
        let approx = payload_size_bytes(&payload);

        ToolRunner::new(LO_REACH, &self.state)
            .with_mode(mode)
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

struct LoCoverageTool {
    args:  LoReachArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for LoCoverageTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let mode = ToolPayloadMode::Body;
        let (graph, graph_version, lo) = load_lo_with_graph(
            &self.state.graph,
            &self.state.analysis_cache,
            &self.args.lo_slug,
            COVERAGE,
        )
        .await?;
        let meta = common::graph_meta(&self.state.graph).await?;

        let cached = cache_lo_bundle(
            &self.state.analysis_cache,
            graph_version,
            &graph,
            lo,
            &self.args.lo_slug,
        );
        let bundle: CachedLoBundle = decode_cached(&cached.payload)?;

        let payload = json!({
            "type": "graph_analysis",
            "tool": COVERAGE,
            "lo_slug": self.args.lo_slug,
            "covered": bundle.coverage.covered,
            "missing": bundle.coverage.missing,
            "unused_observation_features": bundle.coverage.unused_observation_features,
        });
        info!(tool = COVERAGE, lo_slug = %self.args.lo_slug, "graph lo coverage");
        let approx = payload_size_bytes(&payload);

        ToolRunner::new(COVERAGE, &self.state)
            .with_mode(mode)
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

struct LoAlignmentTool {
    args:  LoAlignmentArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for LoAlignmentTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let mode = ToolPayloadMode::from_fetch_flag(self.args.fetch_body);
        let (graph, graph_version, lo) = load_lo_with_graph(
            &self.state.graph,
            &self.state.analysis_cache,
            &self.args.lo_slug,
            LO_ALIGNMENT,
        )
        .await?;
        let meta = common::graph_meta(&self.state.graph).await?;

        let cached = cache_lo_bundle(
            &self.state.analysis_cache,
            graph_version,
            &graph,
            lo,
            &self.args.lo_slug,
        );
        let bundle: CachedLoBundle = decode_cached(&cached.payload)?;

        let reachable: Vec<_> = bundle
            .assessments
            .iter()
            .filter(|a| a.reachable_from_first_principle)
            .map(|a| a.assessment_slug.clone())
            .collect();
        let unreachable: Vec<_> = bundle
            .assessments
            .iter()
            .filter(|a| !a.reachable_from_first_principle)
            .map(|a| a.assessment_slug.clone())
            .collect();

        let sample = |list: &Vec<String>| list.iter().take(3).cloned().collect::<Vec<_>>();

        let payload = json!({
            "type": "graph_analysis",
            "tool": LO_ALIGNMENT,
            "lo_slug": self.args.lo_slug,
            "assessments": {
                "total": bundle.assessments.len(),
                "reachable": reachable.len(),
                "unreachable": unreachable.len(),
                "sample_reachable": sample(&reachable),
                "sample_unreachable": sample(&unreachable),
            },
            "coverage": {
                "total_criteria": bundle.coverage.covered.len() + bundle.coverage.missing.len(),
                "covered": bundle.coverage.covered.len(),
                "missing": bundle.coverage.missing.len(),
                "missing_sample": sample(&bundle.coverage.missing),
                "unused_observation_features": bundle.coverage.unused_observation_features.len(),
                "unused_sample": sample(&bundle.coverage.unused_observation_features),
            },
            "target_anchors": {
                "total": bundle.target_anchors.len(),
                "sample": sample(&bundle.target_anchors),
            }
        });

        let approx = payload_size_bytes(&payload);
        ToolRunner::new(LO_ALIGNMENT, &self.state)
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

struct LoAssessmentsViewTool {
    args:  LoAssessmentsViewArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for LoAssessmentsViewTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, graph_version, lo) = load_lo_with_graph(
            &self.state.graph,
            &self.state.analysis_cache,
            &self.args.lo_slug,
            LO_ASSESSMENTS_VIEW,
        )
        .await?;
        let meta = common::graph_meta(&self.state.graph).await?;

        let cached = cache_lo_bundle(
            &self.state.analysis_cache,
            graph_version,
            &graph,
            lo,
            &self.args.lo_slug,
        );
        let bundle: CachedLoBundle = decode_cached(&cached.payload)?;

        let items: Vec<_> = bundle
            .assessments
            .into_iter()
            .filter_map(|a| {
                if let Some(filter) = self.args.reachable_only
                    && filter != a.reachable_from_first_principle
                {
                    return None;
                }
                Some(json!({
                    "assessment_slug": a.assessment_slug,
                    "reachable_from_first_principle": a.reachable_from_first_principle,
                }))
            })
            .collect();

        let (items, page) = crate::paginate!(items, self.args.limit, self.args.offset);

        let payload = json!({
            "type": "graph_view",
            "tool": LO_ASSESSMENTS_VIEW,
            "lo_slug": self.args.lo_slug,
            "offset": page.offset,
            "limit": page.limit,
            "has_more": page.has_more,
            "assessments": items,
        });

        info!(
            tool = LO_ASSESSMENTS_VIEW,
            lo_slug = %self.args.lo_slug,
            offset = page.offset,
            limit = page.limit,
            has_more = page.has_more,
            "graph lo assessments view"
        );

        let approx = payload_size_bytes(&payload);
        ToolRunner::new(LO_ASSESSMENTS_VIEW, &self.state)
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

struct LoMissingCriteriaViewTool {
    args:  LoMissingCriteriaViewArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for LoMissingCriteriaViewTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, graph_version, lo) = load_lo_with_graph(
            &self.state.graph,
            &self.state.analysis_cache,
            &self.args.lo_slug,
            LO_MISSING_CRITERIA_VIEW,
        )
        .await?;
        let meta = common::graph_meta(&self.state.graph).await?;

        let cached = cache_lo_bundle(
            &self.state.analysis_cache,
            graph_version,
            &graph,
            lo,
            &self.args.lo_slug,
        );
        let bundle: CachedLoBundle = decode_cached(&cached.payload)?;
        let missing = bundle.coverage.missing;

        let (missing, page) = crate::paginate!(missing, self.args.limit, self.args.offset);

        let payload = json!({
            "type": "graph_view",
            "tool": LO_MISSING_CRITERIA_VIEW,
            "lo_slug": self.args.lo_slug,
            "offset": page.offset,
            "limit": page.limit,
            "has_more": page.has_more,
            "missing": missing,
        });

        info!(
            tool = LO_MISSING_CRITERIA_VIEW,
            lo_slug = %self.args.lo_slug,
            offset = page.offset,
            limit = page.limit,
            has_more = page.has_more,
            "graph lo missing criteria view"
        );

        let approx = payload_size_bytes(&payload);
        ToolRunner::new(LO_MISSING_CRITERIA_VIEW, &self.state)
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

struct LoAnchorsViewTool {
    args:  LoAnchorsViewArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for LoAnchorsViewTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, lo) =
            load_lo_with_graph_only(&self.state.graph, &self.args.lo_slug, LO_ANCHORS_VIEW).await?;
        let meta = common::graph_meta(&self.state.graph).await?;

        let anchors: Vec<_> = graph
            .edges_directed(lo, petgraph::Direction::Incoming)
            .filter_map(|e| match &e.weight().kind {
                crate::graph::EdgeKind::Anchors(attrs) => Some(json!({
                    "teaching_step_slug": graph[e.source()].slug.clone(),
                    "impact": attrs.impact,
                })),
                _ => None,
            })
            .collect();

        let (anchors, page) = crate::paginate!(anchors, self.args.limit, self.args.offset);

        let payload = json!({
            "type": "graph_view",
            "tool": LO_ANCHORS_VIEW,
            "lo_slug": self.args.lo_slug,
            "offset": page.offset,
            "limit": page.limit,
            "has_more": page.has_more,
            "anchors": anchors,
        });

        info!(
            tool = LO_ANCHORS_VIEW,
            lo_slug = %self.args.lo_slug,
            offset = page.offset,
            limit = page.limit,
            has_more = page.has_more,
            "graph lo anchors view"
        );

        let approx = payload_size_bytes(&payload);
        ToolRunner::new(LO_ANCHORS_VIEW, &self.state)
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

// ---------- Keystone ----------

const KEYSTONE: &str = "graph_keystone";

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct KeystoneArgs {
    pub limit:      Option<usize>,
    pub offset:     Option<usize>,
    #[serde(default = "default_true")]
    pub fetch_body: bool,
}

crate::analysis_tool!(
    keystone_meta,
    id: KEYSTONE,
    description: "Identify 'keystone' nodes that are critical to the graph structure (high betweenness centrality). These nodes appear on many paths between first principles and assessments. Keystones are high-priority for scaffolding—if students struggle here, many downstream concepts are affected.",
    args: KeystoneArgs,
    prepare: |raw| crate::tools::llm::common::parse_args(KEYSTONE, raw),
    runner: |args: KeystoneArgs, state: &CallState| KeystoneTool {
        args,
        state: state.clone(),
    }
);

struct KeystoneTool {
    args:  KeystoneArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for KeystoneTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let mode = ToolPayloadMode::from_fetch_flag(self.args.fetch_body);
        let (graph, graph_version) =
            load_graph_with_version(&self.state.graph, &self.state.analysis_cache).await?;
        let meta = common::graph_meta(&self.state.graph).await?;

        let cache_key = AnalysisCacheKey {
            graph_version,
            kind: AnalysisKind::Keystone,
        };

        let payload = with_cached_analysis_result(
            &self.state.analysis_cache,
            cache_key,
            graph_version,
            || {
                let graph = Arc::clone(&graph);
                async move {
                    let scores = analysis::keystone_scores(&graph);
                    let rendered: Vec<_> = scores
                        .into_iter()
                        .map(|score| {
                            json!({
                                "slug": graph[score.node].slug,
                                "score": score.score,
                                "in_reach": score.in_reach,
                                "out_reach": score.out_reach,
                            })
                        })
                        .collect();

                    Ok::<Value, ToolExecutionError>(json!({
                        "type": "graph_analysis",
                        "tool": KEYSTONE,
                        "total_ranked": rendered.len(),
                        "scores": rendered,
                    }))
                }
            },
        )
        .await?;

        let all_scores = payload
            .get("scores")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let total_ranked = all_scores.len();
        let limit = self.args.limit.or(Some(20));
        let (scores, page) = crate::paginate!(all_scores, limit, self.args.offset);
        let payload = json!({
            "type": "graph_analysis",
            "tool": KEYSTONE,
            "total_ranked": total_ranked,
            "scores": scores,
        });

        info!(
            tool = KEYSTONE,
            total_ranked,
            limit = page.limit,
            offset = page.offset,
            has_more = page.has_more,
            "graph keystone scores"
        );

        let approx = payload_size_bytes(&payload);
        ToolRunner::new(KEYSTONE, &self.state)
            .with_mode(mode)
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

// ---------- Practice gaps / Extraneous / Alignment ----------

const EXTRANEOUS: &str = "graph_extraneous";
const ALIGNMENT_GAPS: &str = "graph_assessment_gaps";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExtraneousArgs {
    #[schemars(description = "AssessmentItem slug.")]
    pub assessment_slug: String,
    #[schemars(description = "LO slug.")]
    pub lo_slug:         String,
    #[serde(default)]
    #[schemars(description = "Optional intended prereqs to compare.")]
    pub intended_slugs:  Vec<String>,
}

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AlignmentGapsArgs {
    #[serde(default)]
    #[schemars(
        description = "Return payload if true.",
        default = "crate::tools::llm::default_false"
    )]
    pub fetch_body: bool,
}
crate::analysis_tool!(
    extraneous_meta,
    id: EXTRANEOUS,
    description: "Identify construct-irrelevant demands for an AssessmentItem—skills the assessment requires but doesn't intend to measure. For example, an assessment of 'write a function with docstring' might have construct-irrelevant demands of 'prose writing ability'. These should be minimized to improve construct validity.",
    args: ExtraneousArgs,
    prepare: |raw| super::common::parse_args_with_builder(
        EXTRANEOUS,
        raw,
        |mut input: ExtraneousArgs| {
            input.assessment_slug = require_string(input.assessment_slug, EXTRANEOUS, "assessment_slug")?;
            input.lo_slug = require_string(input.lo_slug, EXTRANEOUS, "lo_slug")?;
            Ok(input)
        },
    ),
    runner: |args: ExtraneousArgs, state: &CallState| ExtraneousTool {
        args,
        state: state.clone(),
    }
);

struct ExtraneousTool {
    args:  ExtraneousArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for ExtraneousTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let assessment = resolve_typed::<AssessmentItem>(
            &self.state.graph,
            Slug::new(self.args.assessment_slug.clone()),
            EXTRANEOUS,
        )
        .await?;
        let lo = resolve_typed::<LearningOutcome>(
            &self.state.graph,
            Slug::new(self.args.lo_slug.clone()),
            EXTRANEOUS,
        )
        .await?;
        let graph: Arc<CurriculumGraph> = self
            .state
            .graph
            .ask(crate::graph::commands::GetGraph)
            .await
            .map_err(map_send_err_inf)?;
        let meta = common::graph_meta(&self.state.graph).await?;

        let mut intended = HashSet::new();
        if !self.args.intended_slugs.is_empty() {
            let ids =
                resolve_slugs(&self.state.graph, self.args.intended_slugs.clone(), EXTRANEOUS)
                    .await?;
            intended.extend(ids);
        }
        if intended.is_empty() {
            let derived = crate::analysis::requires_ancestors(&graph, lo);
            intended.extend(derived);
        }

        let report = analysis::extraneous_report(&graph, assessment, lo, &intended);
        let extraneous_slugs: Vec<_> = report
            .extraneous_nodes
            .iter()
            .map(|n| graph[*n].slug.clone())
            .collect();

        let declared_cid = match &graph[assessment].kind {
            crate::graph::NodeKind::Knowledge(k) => k.construct_irrelevant_demands.clone(),
            _ => Vec::new(),
        };

        let payload = json!({
            "type": "graph_analysis",
            "tool": EXTRANEOUS,
            "assessment_slug": graph[assessment].slug,
            "lo_slug": graph[lo].slug,
            "extraneous_slugs": extraneous_slugs,
            "declared_construct_irrelevant_demands": declared_cid,
        });
        info!(
            tool = EXTRANEOUS,
            assessment_slug = %self.args.assessment_slug,
            lo_slug = %self.args.lo_slug,
            "graph extraneous report"
        );
        let approx = payload_size_bytes(&payload);
        ToolRunner::new(EXTRANEOUS, &self.state)
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
crate::analysis_tool!(
    alignment_gaps_meta,
    id: ALIGNMENT_GAPS,
    description: "Scan all LearningOutcomes for alignment issues: LOs without assessments, assessments not reachable from first principles, and missing rubric coverage. Returns a prioritized list of problems to fix. Use graph_lo_alignment_summary for detailed per-LO analysis.",
    args: AlignmentGapsArgs,
    prepare: |raw| crate::tools::llm::common::parse_args(ALIGNMENT_GAPS, raw),
    runner: |args: AlignmentGapsArgs, state: &CallState| AlignmentGapsTool {
        args,
        state: state.clone(),
    }
);

struct AlignmentGapsTool {
    args:  AlignmentGapsArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for AlignmentGapsTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let mode = ToolPayloadMode::from_fetch_flag(self.args.fetch_body);
        let (graph, graph_version) =
            load_graph_with_version(&self.state.graph, &self.state.analysis_cache).await?;
        let meta = common::graph_meta(&self.state.graph).await?;

        let cache_key = AnalysisCacheKey {
            graph_version,
            kind: AnalysisKind::AssessmentGaps,
        };

        let payload = with_cached_analysis_result(
            &self.state.analysis_cache,
            cache_key,
            graph_version,
            || {
                let graph = Arc::clone(&graph);
                async move {
                    let los = analysis::lo_missing_target_assessments(&graph);
                    let orphan = analysis::orphan_assessments(&graph);
                    let unreachable = analysis::unreachable_assessments(&graph);
                    Ok::<Value, ToolExecutionError>(json!({
                        "type": "graph_analysis",
                        "tool": ALIGNMENT_GAPS,
                        "los_missing_target_assessment": los.into_iter().map(|n| graph[n].slug.clone()).collect::<Vec<_>>(),
                        "assessments_without_lo": orphan.into_iter().map(|n| graph[n].slug.clone()).collect::<Vec<_>>(),
                        "assessments_unreachable": unreachable.into_iter().map(|n| graph[n].slug.clone()).collect::<Vec<_>>(),
                    }))
                }
            },
        )
        .await?;

        let approx = payload_size_bytes(&payload);
        ToolRunner::new(ALIGNMENT_GAPS, &self.state)
            .with_mode(mode)
            .with_meta(meta)
            .hints(vec![format!(
                "Payload is ~{} bytes; set fetch_body=true to retrieve it.",
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

pub(super) fn tool_prototypes() -> Vec<ToolPrototype> {
    vec![
        dag_check_meta(),
        first_principles_meta(),
        first_principles_summary_meta(),
        lo_reach_meta(),
        coverage_meta(),
        lo_alignment_meta(),
        lo_assessments_view_meta(),
        lo_missing_criteria_view_meta(),
        lo_anchors_view_meta(),
        keystone_meta(),
        extraneous_meta(),
        alignment_gaps_meta(),
    ]
}
