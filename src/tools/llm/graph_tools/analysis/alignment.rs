use std::sync::Arc;

use async_trait::async_trait;
use bon::Builder;
use kameo::prelude::ActorRef;
use petgraph::visit::EdgeRef;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::info;

use super::{
    SummaryContext, cache_lo_bundle, decode_cached, finalize_summary_tool, load_graph_with_version,
    load_lo_with_graph, load_lo_with_graph_only,
};
use crate::{
    analysis,
    graph::CurriculumGraph,
    schema::types::KnowledgeType,
    tools::llm::{
        CallState, ToolExecutionError, ToolInputResult, ToolInstance, ToolOutput, ToolPrototype,
        graph_tools::{
            analysis::{CachedLoBundle, NoArgs},
            analysis_cache::{AnalysisCache, AnalysisCacheKey, AnalysisKind},
            common,
            common::{
                ensure_knowledge_type, map_send_err_inf, paginate, parse_with, resolve_slug,
                resolve_slugs,
            },
        },
        payload_size_bytes, require_string, schema_for_args,
    },
};

// ---------- DAG check ----------

const DAG_CHECK: &str = "graph_dag_check";

pub(super) fn dag_check_meta() -> ToolPrototype {
    ToolPrototype {
        id:          DAG_CHECK,
        description: "Check the Requires DAG invariant: ensures all requires edges are acyclic \
                      (Knowledge Space Theory). Returns is_dag and topological order length.",
        schema:      schema_for_args::<NoArgs>(),
        parse:       parse_dag_check,
    }
}

fn parse_dag_check(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_with(
        DAG_CHECK,
        raw,
        state,
        |args: NoArgs| Ok(args),
        |_, state| {
            Ok(DAGCheckTool {
                graph:          state.graph.clone(),
                analysis_cache: Arc::clone(&state.analysis_cache),
            })
        },
    )
}

struct DAGCheckTool {
    graph:          ActorRef<crate::graph::manager::GraphManager>,
    analysis_cache: Arc<AnalysisCache>,
}

#[async_trait]
impl ToolInstance for DAGCheckTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, graph_version) =
            load_graph_with_version(&self.graph, &self.analysis_cache).await?;
        let meta = common::graph_meta(&self.graph).await?;

        let cache_key = AnalysisCacheKey {
            graph_version,
            kind: AnalysisKind::DagCheck,
        };

        let cached = self.analysis_cache.get_or_insert_with(cache_key, || {
            let dag = analysis::requires_is_dag(&graph);
            let topo = analysis::requires_toposort(&graph).ok();
            json!({
                    "type": "graph_analysis",
                    "tool": DAG_CHECK,
                "is_dag": dag,
                "topo_order_count": topo.as_ref().map(|v| v.len()),
            })
        });

        let payload = common::attach_meta(cached.payload.clone(), &meta);
        info!(
            tool = DAG_CHECK,
            is_dag = payload["is_dag"].as_bool().unwrap_or(false),
            topo_order_count = payload["topo_order_count"].as_u64(),
            "graph requires dag check"
        );
        Ok(ToolOutput::with_byte_hint(payload.clone(), payload_size_bytes(&payload)))
    }
}

// ---------- First principles ----------

const FIRST_PRINCIPLES_VIEW: &str = "graph_first_principles"; // backward-compatible id
const FIRST_PRINCIPLES_SUMMARY: &str = "graph_first_principles_summary";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FirstPrinciplesViewArgs {
    #[serde(default)]
    #[schemars(description = "Maximum items to return (default 50, max 200).")]
    pub limit:  Option<usize>,
    #[serde(default)]
    #[schemars(description = "Offset into the first-principles list (default 0).")]
    pub offset: Option<usize>,
}

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FirstPrinciplesSummaryArgs {
    #[serde(default)]
    #[schemars(
        description = "When true, return the summary payload; otherwise return a cost preview.",
        default = "crate::tools::llm::default_false"
    )]
    pub fetch_body: bool,
}

pub(super) fn first_principles_meta() -> ToolPrototype {
    ToolPrototype {
        id:          FIRST_PRINCIPLES_VIEW,
        description: "View first-principles (requires in-degree 0 instructional knowledge) with \
                      pagination.",
        schema:      schema_for_args::<FirstPrinciplesViewArgs>(),
        parse:       parse_first_principles_view,
    }
}

pub(super) fn first_principles_summary_meta() -> ToolPrototype {
    ToolPrototype {
        id:          FIRST_PRINCIPLES_SUMMARY,
        description: "Summary of first-principles counts by knowledge type; returns preview \
                      unless fetch_body=true.",
        schema:      schema_for_args::<FirstPrinciplesSummaryArgs>(),
        parse:       parse_first_principles_summary,
    }
}

fn parse_first_principles_view(
    raw: Value,
    state: &CallState,
) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_with(
        FIRST_PRINCIPLES_VIEW,
        raw,
        state,
        |args: FirstPrinciplesViewArgs| Ok(args),
        |args, state| {
            Ok(FirstPrinciplesViewTool {
                args,
                graph: state.graph.clone(),
            })
        },
    )
}

fn parse_first_principles_summary(
    raw: Value,
    state: &CallState,
) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_with(
        FIRST_PRINCIPLES_SUMMARY,
        raw,
        state,
        |args: FirstPrinciplesSummaryArgs| Ok(args),
        |args, state| {
            Ok(FirstPrinciplesSummaryTool {
                args,
                graph: state.graph.clone(),
                metrics: Arc::clone(&state.metrics),
                model: Arc::clone(&state.model),
                conversation_id: Arc::clone(&state.conversation_id),
            })
        },
    )
}

struct FirstPrinciplesViewTool {
    args:  FirstPrinciplesViewArgs,
    graph: ActorRef<crate::graph::manager::GraphManager>,
}

#[async_trait]
impl ToolInstance for FirstPrinciplesViewTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph: Arc<CurriculumGraph> = self
            .graph
            .ask(crate::graph::commands::GetGraph)
            .await
            .map_err(map_send_err_inf)?;

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

        let (items, offset, limit, has_more) = paginate(items, self.args.limit, self.args.offset);

        let payload = json!({
            "type": "graph_view",
            "tool": FIRST_PRINCIPLES_VIEW,
            "offset": offset,
            "limit": limit,
            "has_more": has_more,
            "items": items,
        });

        info!(
            tool = FIRST_PRINCIPLES_VIEW,
            offset, limit, has_more, "graph first_principles view"
        );

        Ok(ToolOutput::with_byte_hint(payload.clone(), payload_size_bytes(&payload)))
    }
}

struct FirstPrinciplesSummaryTool {
    args:            FirstPrinciplesSummaryArgs,
    graph:           ActorRef<crate::graph::manager::GraphManager>,
    metrics:         Arc<crate::llm_gateway::GatewayMetrics>,
    model:           Arc<String>,
    conversation_id: Arc<String>,
}

#[async_trait]
impl ToolInstance for FirstPrinciplesSummaryTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph: Arc<CurriculumGraph> = self
            .graph
            .ask(crate::graph::commands::GetGraph)
            .await
            .map_err(map_send_err_inf)?;
        let meta = common::graph_meta(&self.graph).await?;

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

        finalize_summary_tool(
            FIRST_PRINCIPLES_SUMMARY,
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
    #[schemars(
        description = "Learning Outcome slug to inspect. Must already exist in the graph and be \
                       knowledge_type = learning_outcome."
    )]
    pub lo_slug: String,
}

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LoAssessmentsViewArgs {
    #[schemars(description = "Learning Outcome slug to inspect.")]
    pub lo_slug:        String,
    #[serde(default)]
    #[schemars(
        description = "Return only reachable assessments when true; only unreachable when false; \
                       default includes all."
    )]
    pub reachable_only: Option<bool>,
    #[serde(default)]
    #[schemars(description = "Maximum items to return (default 50, max 200).")]
    pub limit:          Option<usize>,
    #[serde(default)]
    #[schemars(description = "Offset into the assessments list (default 0).")]
    pub offset:         Option<usize>,
}

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LoMissingCriteriaViewArgs {
    #[schemars(description = "Learning Outcome slug to inspect.")]
    pub lo_slug: String,
    #[serde(default)]
    #[schemars(description = "Maximum items to return (default 50, max 200).")]
    pub limit:   Option<usize>,
    #[serde(default)]
    #[schemars(description = "Offset into the missing criteria list (default 0).")]
    pub offset:  Option<usize>,
}

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LoAnchorsViewArgs {
    #[schemars(description = "Learning Outcome slug to inspect.")]
    pub lo_slug: String,
    #[serde(default)]
    #[schemars(description = "Maximum items to return (default 50, max 200).")]
    pub limit:   Option<usize>,
    #[serde(default)]
    #[schemars(description = "Offset into the anchors list (default 0).")]
    pub offset:  Option<usize>,
}

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LoAlignmentArgs {
    #[schemars(description = "Learning Outcome slug to summarize.")]
    pub lo_slug:    String,
    #[serde(default)]
    #[schemars(
        description = "When true, return the summary payload; otherwise return a cost preview.",
        default = "crate::tools::llm::default_false"
    )]
    pub fetch_body: bool,
}

pub(super) fn lo_reach_meta() -> ToolPrototype {
    ToolPrototype {
        id:          LO_REACH,
        description: "Constructive Alignment predicate for a Learning Outcome: which assessments \
                      are reachable from first principles via requires* paths and target this LO.",
        schema:      schema_for_args::<LoReachArgs>(),
        parse:       parse_lo_reach,
    }
}

pub(super) fn lo_alignment_meta() -> ToolPrototype {
    ToolPrototype {
        id:          LO_ALIGNMENT,
        description: "High-level alignment summary for a Learning Outcome: reachability from \
                      first principles via assessments (constructive alignment), rubric coverage \
                      vs observation_features, and target anchors. Use this first when checking \
                      LO alignment; use reachability/coverage tools for deeper debugging.",
        schema:      schema_for_args::<LoAlignmentArgs>(),
        parse:       parse_lo_alignment,
    }
}

pub(super) fn lo_assessments_view_meta() -> ToolPrototype {
    ToolPrototype {
        id:          LO_ASSESSMENTS_VIEW,
        description: "View assessments linked to an LO with reachability flags (paged).",
        schema:      schema_for_args::<LoAssessmentsViewArgs>(),
        parse:       parse_lo_assessments_view,
    }
}

pub(super) fn lo_missing_criteria_view_meta() -> ToolPrototype {
    ToolPrototype {
        id:          LO_MISSING_CRITERIA_VIEW,
        description: "View missing rubric criteria for an LO (paged).",
        schema:      schema_for_args::<LoMissingCriteriaViewArgs>(),
        parse:       parse_lo_missing_criteria_view,
    }
}

pub(super) fn lo_anchors_view_meta() -> ToolPrototype {
    ToolPrototype {
        id:          LO_ANCHORS_VIEW,
        description: "View anchors (teaching steps) targeting an LO (paged).",
        schema:      schema_for_args::<LoAnchorsViewArgs>(),
        parse:       parse_lo_anchors_view,
    }
}

pub(super) fn coverage_meta() -> ToolPrototype {
    ToolPrototype {
        id:          COVERAGE,
        description: "Coverage report for a Learning Outcome: compares rubric_criteria to \
                      observation_features on target assesses edges; use to find rubric coverage \
                      gaps.",
        schema:      schema_for_args::<LoReachArgs>(),
        parse:       parse_lo_coverage,
    }
}

fn parse_lo_reach(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_with(
        LO_REACH,
        raw,
        state,
        |mut input: LoReachArgs| {
            input.lo_slug = require_string(input.lo_slug, LO_REACH, "lo_slug")?;
            Ok(input)
        },
        |args, state| {
            Ok(LoReachTool {
                args,
                graph: state.graph.clone(),
                analysis_cache: Arc::clone(&state.analysis_cache),
            })
        },
    )
}

fn parse_lo_alignment(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_with(
        LO_ALIGNMENT,
        raw,
        state,
        |mut input: LoAlignmentArgs| {
            input.lo_slug = require_string(input.lo_slug, LO_ALIGNMENT, "lo_slug")?;
            Ok(input)
        },
        |args, state| {
            Ok(LoAlignmentTool {
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

fn parse_lo_coverage(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_with(
        COVERAGE,
        raw,
        state,
        |mut input: LoReachArgs| {
            input.lo_slug = require_string(input.lo_slug, COVERAGE, "lo_slug")?;
            Ok(input)
        },
        |args, state| {
            Ok(LoCoverageTool {
                args,
                graph: state.graph.clone(),
                analysis_cache: Arc::clone(&state.analysis_cache),
            })
        },
    )
}

struct LoReachTool {
    args:           LoReachArgs,
    graph:          ActorRef<crate::graph::manager::GraphManager>,
    analysis_cache: Arc<AnalysisCache>,
}

#[async_trait]
impl ToolInstance for LoReachTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, graph_version, lo) =
            load_lo_with_graph(&self.graph, &self.analysis_cache, &self.args.lo_slug, LO_REACH)
                .await?;
        let meta = common::graph_meta(&self.graph).await?;

        let cached =
            cache_lo_bundle(&self.analysis_cache, graph_version, &graph, lo, &self.args.lo_slug);
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
        let payload = common::attach_meta(payload, &meta);
        Ok(ToolOutput::with_byte_hint(payload.clone(), payload_size_bytes(&payload)))
    }
}

struct LoCoverageTool {
    args:           LoReachArgs,
    graph:          ActorRef<crate::graph::manager::GraphManager>,
    analysis_cache: Arc<AnalysisCache>,
}

#[async_trait]
impl ToolInstance for LoCoverageTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, graph_version, lo) =
            load_lo_with_graph(&self.graph, &self.analysis_cache, &self.args.lo_slug, COVERAGE)
                .await?;
        let meta = common::graph_meta(&self.graph).await?;

        let cached =
            cache_lo_bundle(&self.analysis_cache, graph_version, &graph, lo, &self.args.lo_slug);
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
        let payload = common::attach_meta(payload, &meta);
        Ok(ToolOutput::with_byte_hint(payload.clone(), payload_size_bytes(&payload)))
    }
}

struct LoAlignmentTool {
    args:            LoAlignmentArgs,
    graph:           ActorRef<crate::graph::manager::GraphManager>,
    metrics:         Arc<crate::llm_gateway::GatewayMetrics>,
    model:           Arc<String>,
    conversation_id: Arc<String>,
    analysis_cache:  Arc<AnalysisCache>,
}

#[async_trait]
impl ToolInstance for LoAlignmentTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, graph_version, lo) =
            load_lo_with_graph(&self.graph, &self.analysis_cache, &self.args.lo_slug, LO_ALIGNMENT)
                .await?;
        let meta = common::graph_meta(&self.graph).await?;

        let cached =
            cache_lo_bundle(&self.analysis_cache, graph_version, &graph, lo, &self.args.lo_slug);
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

        finalize_summary_tool(
            LO_ALIGNMENT,
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

fn parse_lo_assessments_view(
    raw: Value,
    state: &CallState,
) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_with(
        LO_ASSESSMENTS_VIEW,
        raw,
        state,
        |mut input: LoAssessmentsViewArgs| {
            input.lo_slug = require_string(input.lo_slug, LO_ASSESSMENTS_VIEW, "lo_slug")?;
            Ok(input)
        },
        |args, state| {
            Ok(LoAssessmentsViewTool {
                args,
                graph: state.graph.clone(),
                analysis_cache: Arc::clone(&state.analysis_cache),
            })
        },
    )
}

fn parse_lo_missing_criteria_view(
    raw: Value,
    state: &CallState,
) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_with(
        LO_MISSING_CRITERIA_VIEW,
        raw,
        state,
        |mut input: LoMissingCriteriaViewArgs| {
            input.lo_slug = require_string(input.lo_slug, LO_MISSING_CRITERIA_VIEW, "lo_slug")?;
            Ok(input)
        },
        |args, state| {
            Ok(LoMissingCriteriaViewTool {
                args,
                graph: state.graph.clone(),
                analysis_cache: Arc::clone(&state.analysis_cache),
            })
        },
    )
}

fn parse_lo_anchors_view(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_with(
        LO_ANCHORS_VIEW,
        raw,
        state,
        |mut input: LoAnchorsViewArgs| {
            input.lo_slug = require_string(input.lo_slug, LO_ANCHORS_VIEW, "lo_slug")?;
            Ok(input)
        },
        |args, state| {
            Ok(LoAnchorsViewTool {
                args,
                graph: state.graph.clone(),
            })
        },
    )
}

struct LoAssessmentsViewTool {
    args:           LoAssessmentsViewArgs,
    graph:          ActorRef<crate::graph::manager::GraphManager>,
    analysis_cache: Arc<AnalysisCache>,
}

#[async_trait]
impl ToolInstance for LoAssessmentsViewTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, graph_version, lo) = load_lo_with_graph(
            &self.graph,
            &self.analysis_cache,
            &self.args.lo_slug,
            LO_ASSESSMENTS_VIEW,
        )
        .await?;
        let meta = common::graph_meta(&self.graph).await?;

        let cached =
            cache_lo_bundle(&self.analysis_cache, graph_version, &graph, lo, &self.args.lo_slug);
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

        let (items, offset, limit, has_more) = paginate(items, self.args.limit, self.args.offset);

        let payload = json!({
            "type": "graph_view",
            "tool": LO_ASSESSMENTS_VIEW,
            "lo_slug": self.args.lo_slug,
            "offset": offset,
            "limit": limit,
            "has_more": has_more,
            "assessments": items,
        });

        info!(
            tool = LO_ASSESSMENTS_VIEW,
            lo_slug = %self.args.lo_slug,
            offset,
            limit,
            has_more,
            "graph lo assessments view"
        );

        let payload = common::attach_meta(payload, &meta);
        Ok(ToolOutput::with_byte_hint(payload.clone(), payload_size_bytes(&payload)))
    }
}

struct LoMissingCriteriaViewTool {
    args:           LoMissingCriteriaViewArgs,
    graph:          ActorRef<crate::graph::manager::GraphManager>,
    analysis_cache: Arc<AnalysisCache>,
}

#[async_trait]
impl ToolInstance for LoMissingCriteriaViewTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, graph_version, lo) = load_lo_with_graph(
            &self.graph,
            &self.analysis_cache,
            &self.args.lo_slug,
            LO_MISSING_CRITERIA_VIEW,
        )
        .await?;
        let meta = common::graph_meta(&self.graph).await?;

        let cached =
            cache_lo_bundle(&self.analysis_cache, graph_version, &graph, lo, &self.args.lo_slug);
        let bundle: CachedLoBundle = decode_cached(&cached.payload)?;
        let missing = bundle.coverage.missing;

        let (missing, offset, limit, has_more) =
            paginate(missing, self.args.limit, self.args.offset);

        let payload = json!({
            "type": "graph_view",
            "tool": LO_MISSING_CRITERIA_VIEW,
            "lo_slug": self.args.lo_slug,
            "offset": offset,
            "limit": limit,
            "has_more": has_more,
            "missing": missing,
        });

        info!(
            tool = LO_MISSING_CRITERIA_VIEW,
            lo_slug = %self.args.lo_slug,
            offset,
            limit,
            has_more,
            "graph lo missing criteria view"
        );

        let payload = common::attach_meta(payload, &meta);
        Ok(ToolOutput::with_byte_hint(payload.clone(), payload_size_bytes(&payload)))
    }
}

struct LoAnchorsViewTool {
    args:  LoAnchorsViewArgs,
    graph: ActorRef<crate::graph::manager::GraphManager>,
}

#[async_trait]
impl ToolInstance for LoAnchorsViewTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, lo) =
            load_lo_with_graph_only(&self.graph, &self.args.lo_slug, LO_ANCHORS_VIEW).await?;
        let meta = common::graph_meta(&self.graph).await?;

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

        let (anchors, offset, limit, has_more) =
            paginate(anchors, self.args.limit, self.args.offset);

        let payload = json!({
            "type": "graph_view",
            "tool": LO_ANCHORS_VIEW,
            "lo_slug": self.args.lo_slug,
            "offset": offset,
            "limit": limit,
            "has_more": has_more,
            "anchors": anchors,
        });

        info!(
            tool = LO_ANCHORS_VIEW,
            lo_slug = %self.args.lo_slug,
            offset,
            limit,
            has_more,
            "graph lo anchors view"
        );

        let payload = common::attach_meta(payload, &meta);
        Ok(ToolOutput::with_byte_hint(payload.clone(), payload_size_bytes(&payload)))
    }
}

// ---------- Keystone ----------

const KEYSTONE: &str = "graph_keystone";

pub(super) fn keystone_meta() -> ToolPrototype {
    ToolPrototype {
        id:          KEYSTONE,
        description: "Compute keystone scores (in_reach * out_reach) for knowledge nodes to \
                      identify structurally critical concepts in the requires DAG (betweenness \
                      approximation). Higher scores imply more reasoning paths depend on the \
                      node. Returns top nodes only to keep output bounded.",
        schema:      schema_for_args::<NoArgs>(),
        parse:       parse_keystone,
    }
}

fn parse_keystone(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_with(
        KEYSTONE,
        raw,
        state,
        |args: NoArgs| Ok(args),
        |_, state| {
            Ok(KeystoneTool {
                graph:          state.graph.clone(),
                analysis_cache: Arc::clone(&state.analysis_cache),
            })
        },
    )
}

struct KeystoneTool {
    graph:          ActorRef<crate::graph::manager::GraphManager>,
    analysis_cache: Arc<AnalysisCache>,
}

#[async_trait]
impl ToolInstance for KeystoneTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, graph_version) =
            load_graph_with_version(&self.graph, &self.analysis_cache).await?;

        let cache_key = AnalysisCacheKey {
            graph_version,
            kind: AnalysisKind::Keystone,
        };

        let cached = self.analysis_cache.get_or_insert_with(cache_key, || {
            let scores = analysis::keystone_scores(&graph);
            let top_n = 20usize;
            let rendered: Vec<_> = scores
                .into_iter()
                .take(top_n)
                .map(|score| {
                    json!({
                        "slug": graph[score.node].slug,
                        "score": score.score,
                        "in_reach": score.in_reach,
                        "out_reach": score.out_reach,
                    })
                })
                .collect();

            json!({
                "type": "graph_analysis",
                "tool": KEYSTONE,
                "total_ranked": rendered.len(),
                "scores": rendered,
            })
        });

        let payload = cached.payload.clone();

        info!(
            tool = KEYSTONE,
            total_ranked = payload["total_ranked"].as_u64().unwrap_or(0),
            "graph keystone scores"
        );

        Ok(ToolOutput::with_byte_hint(payload.clone(), payload_size_bytes(&payload)))
    }
}

// ---------- Practice gaps / Extraneous / Alignment ----------

const EXTRANEOUS: &str = "graph_extraneous";
const ALIGNMENT_GAPS: &str = "graph_assessment_gaps";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExtraneousArgs {
    pub assessment_slug: String,
    pub lo_slug:         String,
    #[serde(default)]
    pub intended_slugs:  Vec<String>,
}

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AlignmentGapsArgs {
    #[serde(default)]
    #[schemars(
        description = "When true, return the full payload; otherwise return a cost preview.",
        default = "crate::tools::llm::default_false"
    )]
    pub fetch_body: bool,
}

pub(super) fn extraneous_meta() -> ToolPrototype {
    ToolPrototype {
        id:          EXTRANEOUS,
        description: "Extraneous(A,L): knowledge required by an assessment via requires* but not \
                      intended for the target LO. Compares to construct_irrelevant_demands to \
                      surface construct-irrelevant demands.",
        schema:      schema_for_args::<ExtraneousArgs>(),
        parse:       parse_extraneous,
    }
}

fn parse_extraneous(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_with(
        EXTRANEOUS,
        raw,
        state,
        |mut input: ExtraneousArgs| {
            input.assessment_slug =
                require_string(input.assessment_slug, EXTRANEOUS, "assessment_slug")?;
            input.lo_slug = require_string(input.lo_slug, EXTRANEOUS, "lo_slug")?;
            Ok(input)
        },
        |args, state| {
            Ok(ExtraneousTool {
                args,
                graph: state.graph.clone(),
            })
        },
    )
}

struct ExtraneousTool {
    args:  ExtraneousArgs,
    graph: ActorRef<crate::graph::manager::GraphManager>,
}

#[async_trait]
impl ToolInstance for ExtraneousTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let assessment =
            resolve_slug(&self.graph, self.args.assessment_slug.clone(), EXTRANEOUS).await?;
        let lo = resolve_slug(&self.graph, self.args.lo_slug.clone(), EXTRANEOUS).await?;
        let graph: Arc<CurriculumGraph> = self
            .graph
            .ask(crate::graph::commands::GetGraph)
            .await
            .map_err(map_send_err_inf)?;
        ensure_knowledge_type(
            &graph,
            assessment,
            &self.args.assessment_slug,
            KnowledgeType::AssessmentItem,
            EXTRANEOUS,
        )?;
        ensure_knowledge_type(
            &graph,
            lo,
            &self.args.lo_slug,
            KnowledgeType::LearningOutcome,
            EXTRANEOUS,
        )?;

        let mut intended = std::collections::HashSet::new();
        if !self.args.intended_slugs.is_empty() {
            let ids =
                resolve_slugs(&self.graph, self.args.intended_slugs.clone(), EXTRANEOUS).await?;
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
        Ok(ToolOutput::with_byte_hint(payload.clone(), payload_size_bytes(&payload)))
    }
}

pub(super) fn alignment_gaps_meta() -> ToolPrototype {
    ToolPrototype {
        id:          ALIGNMENT_GAPS,
        description: "Course-wide alignment gaps: LOs with no target assessments, assessments \
                      with no LO, and assessments unreachable from first principles (constructive \
                      alignment scan).",
        schema:      schema_for_args::<AlignmentGapsArgs>(),
        parse:       parse_alignment_gaps,
    }
}

fn parse_alignment_gaps(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_with(
        ALIGNMENT_GAPS,
        raw,
        state,
        |args: AlignmentGapsArgs| Ok(args),
        |args, state| {
            Ok(AlignmentGapsTool {
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

struct AlignmentGapsTool {
    args:            AlignmentGapsArgs,
    graph:           ActorRef<crate::graph::manager::GraphManager>,
    metrics:         Arc<crate::llm_gateway::GatewayMetrics>,
    model:           Arc<String>,
    conversation_id: Arc<String>,
    analysis_cache:  Arc<AnalysisCache>,
}

#[async_trait]
impl ToolInstance for AlignmentGapsTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let (graph, graph_version) =
            load_graph_with_version(&self.graph, &self.analysis_cache).await?;
        let meta = common::graph_meta(&self.graph).await?;

        let cache_key = AnalysisCacheKey {
            graph_version,
            kind: AnalysisKind::AssessmentGaps,
        };

        let cached = self.analysis_cache.get_or_insert_with(cache_key, || {
            let los = analysis::lo_missing_target_assessments(&graph);
            let orphan = analysis::orphan_assessments(&graph);
            let unreachable = analysis::unreachable_assessments(&graph);
            json!({
                "type": "graph_analysis",
                "tool": ALIGNMENT_GAPS,
                "los_missing_target_assessment": los.into_iter().map(|n| graph[n].slug.clone()).collect::<Vec<_>>(),
                "assessments_without_lo": orphan.into_iter().map(|n| graph[n].slug.clone()).collect::<Vec<_>>(),
                "assessments_unreachable": unreachable.into_iter().map(|n| graph[n].slug.clone()).collect::<Vec<_>>(),
            })
        });

        finalize_summary_tool(
            ALIGNMENT_GAPS,
            cached.payload.clone(),
            self.args.fetch_body,
            SummaryContext {
                metrics:         self.metrics.as_ref(),
                model:           self.model.as_str(),
                conversation_id: self.conversation_id.as_str(),
                hint_prefix:     "Payload is",
                meta:            &meta,
            },
        )
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
