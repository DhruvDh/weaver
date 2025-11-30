use std::sync::Arc;

use async_trait::async_trait;
use bon::Builder;
use kameo::prelude::ActorRef;
use petgraph::visit::EdgeRef;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::info;

use super::common::{
    ensure_knowledge_type, map_send_err_inf, paginate, resolve_slug, resolve_slugs,
};
use crate::{
    analysis,
    schema::types::KnowledgeType,
    tools::llm::{
        CallState, ToolExecutionError, ToolInputError, ToolInputResult, ToolInstance, ToolOutput,
        ToolPayloadMode, ToolPrototype, apply_preview_cost, build_cost_preview,
        estimate_tokens_from_characters, payload_size_bytes, prepare_payload_estimates,
        require_string, schema_for_args,
    },
};

// Shared empty args
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NoArgs {}

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
    let _: NoArgs = serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
        tool:    DAG_CHECK,
        message: err.to_string(),
    })?;
    Ok(Box::new(DAGCheckTool {
        graph: state.graph.clone(),
    }))
}

struct DAGCheckTool {
    graph: ActorRef<crate::graph::manager::GraphManager>,
}

#[async_trait]
impl ToolInstance for DAGCheckTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = self
            .graph
            .ask(crate::graph::manager::GetGraph)
            .await
            .map_err(map_send_err_inf)?;
        let dag = analysis::requires_is_dag(&graph);
        let topo = analysis::requires_toposort(&graph).ok();
        let payload = json!({
            "type": "graph_analysis",
            "tool": DAG_CHECK,
            "is_dag": dag,
            "topo_order_count": topo.as_ref().map(|v| v.len()),
        });
        info!(
            tool = DAG_CHECK,
            is_dag = dag,
            topo_order_count = payload["topo_order_count"]
                .as_u64()
                .map(|v| v as usize)
                .unwrap_or(0),
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
    let args: FirstPrinciplesViewArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    FIRST_PRINCIPLES_VIEW,
            message: err.to_string(),
        })?;
    Ok(Box::new(FirstPrinciplesViewTool {
        args,
        graph: state.graph.clone(),
    }))
}

fn parse_first_principles_summary(
    raw: Value,
    state: &CallState,
) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: FirstPrinciplesSummaryArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    FIRST_PRINCIPLES_SUMMARY,
            message: err.to_string(),
        })?;
    Ok(Box::new(FirstPrinciplesSummaryTool {
        args,
        graph: state.graph.clone(),
        metrics: Arc::clone(&state.metrics),
        model: Arc::clone(&state.model),
        conversation_id: Arc::clone(&state.conversation_id),
    }))
}

struct FirstPrinciplesViewTool {
    args:  FirstPrinciplesViewArgs,
    graph: ActorRef<crate::graph::manager::GraphManager>,
}

#[async_trait]
impl ToolInstance for FirstPrinciplesViewTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = self
            .graph
            .ask(crate::graph::manager::GetGraph)
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
        let graph = self
            .graph
            .ask(crate::graph::manager::GetGraph)
            .await
            .map_err(map_send_err_inf)?;

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

        let approx_bytes = payload_size_bytes(&payload);
        let estimates = prepare_payload_estimates(&self.metrics, self.model.as_str(), approx_bytes);
        let mode = ToolPayloadMode::from_fetch_flag(self.args.fetch_body);
        info!(
            tool = FIRST_PRINCIPLES_SUMMARY,
            mode = mode.as_str(),
            approx_bytes,
            "graph first_principles summary"
        );

        match mode {
            ToolPayloadMode::Preview => {
                let hints = vec![format!(
                    "Summary is ~{} bytes; set fetch_body=true to retrieve it.",
                    approx_bytes
                )];
                let mut preview = build_cost_preview(
                    FIRST_PRINCIPLES_SUMMARY,
                    approx_bytes,
                    estimates.safe_tokens,
                    hints,
                );
                let preview_bytes = payload_size_bytes(&preview);
                let preview_tokens = estimate_tokens_from_characters(preview_bytes as usize);
                apply_preview_cost(
                    &mut preview,
                    &self.metrics,
                    self.model.as_str(),
                    self.conversation_id.as_str(),
                    preview_tokens,
                    estimates.safe_tokens,
                );
                Ok(ToolOutput::with_byte_hint(preview, preview_bytes))
            }
            ToolPayloadMode::Body => Ok(ToolOutput::with_byte_hint(payload.clone(), approx_bytes)),
        }
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
    let mut args: LoReachArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    LO_REACH,
            message: err.to_string(),
        })?;
    args.lo_slug = require_string(args.lo_slug.clone(), LO_REACH, "lo_slug")?;
    Ok(Box::new(LoReachTool {
        args,
        graph: state.graph.clone(),
    }))
}

fn parse_lo_alignment(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let mut args: LoAlignmentArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    LO_ALIGNMENT,
            message: err.to_string(),
        })?;
    args.lo_slug = require_string(args.lo_slug.clone(), LO_ALIGNMENT, "lo_slug")?;
    Ok(Box::new(LoAlignmentTool {
        args,
        graph: state.graph.clone(),
        metrics: Arc::clone(&state.metrics),
        model: Arc::clone(&state.model),
        conversation_id: Arc::clone(&state.conversation_id),
    }))
}

fn parse_lo_coverage(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let mut args: LoReachArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    COVERAGE,
            message: err.to_string(),
        })?;
    args.lo_slug = require_string(args.lo_slug.clone(), COVERAGE, "lo_slug")?;
    Ok(Box::new(LoCoverageTool {
        args,
        graph: state.graph.clone(),
    }))
}

struct LoReachTool {
    args:  LoReachArgs,
    graph: ActorRef<crate::graph::manager::GraphManager>,
}

#[async_trait]
impl ToolInstance for LoReachTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let lo = resolve_slug(&self.graph, self.args.lo_slug.clone(), LO_REACH).await?;
        let graph = self
            .graph
            .ask(crate::graph::manager::GetGraph)
            .await
            .map_err(map_send_err_inf)?;
        ensure_knowledge_type(
            &graph,
            lo,
            &self.args.lo_slug,
            KnowledgeType::LearningOutcome,
            LO_REACH,
        )?;
        let fps = analysis::first_principles(&graph);
        let report = analysis::lo_reachability(&graph, lo, &fps);
        let assessments = report
            .assessments
            .into_iter()
            .map(|a| {
                json!({
                    "assessment_slug": graph[a.assessment].slug,
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
        Ok(ToolOutput::with_byte_hint(payload.clone(), payload_size_bytes(&payload)))
    }
}

struct LoCoverageTool {
    args:  LoReachArgs,
    graph: ActorRef<crate::graph::manager::GraphManager>,
}

#[async_trait]
impl ToolInstance for LoCoverageTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let lo = resolve_slug(&self.graph, self.args.lo_slug.clone(), COVERAGE).await?;
        let graph = self
            .graph
            .ask(crate::graph::manager::GetGraph)
            .await
            .map_err(map_send_err_inf)?;
        ensure_knowledge_type(
            &graph,
            lo,
            &self.args.lo_slug,
            KnowledgeType::LearningOutcome,
            COVERAGE,
        )?;
        let report = analysis::coverage_report(&graph, lo);
        let payload = json!({
            "type": "graph_analysis",
            "tool": COVERAGE,
            "lo_slug": self.args.lo_slug,
            "covered": report.covered_criteria,
            "missing": report.missing_criteria,
            "unused_observation_features": report.unused_observation_features,
        });
        info!(tool = COVERAGE, lo_slug = %self.args.lo_slug, "graph lo coverage");
        Ok(ToolOutput::with_byte_hint(payload.clone(), payload_size_bytes(&payload)))
    }
}

struct LoAlignmentTool {
    args:            LoAlignmentArgs,
    graph:           ActorRef<crate::graph::manager::GraphManager>,
    metrics:         Arc<crate::llm_gateway::GatewayMetrics>,
    model:           Arc<String>,
    conversation_id: Arc<String>,
}

#[async_trait]
impl ToolInstance for LoAlignmentTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let lo = resolve_slug(&self.graph, self.args.lo_slug.clone(), LO_ALIGNMENT).await?;
        let graph = self
            .graph
            .ask(crate::graph::manager::GetGraph)
            .await
            .map_err(map_send_err_inf)?;
        ensure_knowledge_type(
            &graph,
            lo,
            &self.args.lo_slug,
            KnowledgeType::LearningOutcome,
            LO_ALIGNMENT,
        )?;

        let fps = analysis::first_principles(&graph);
        let reach = analysis::lo_reachability(&graph, lo, &fps);
        let coverage = analysis::coverage_report(&graph, lo);

        let mut reachable = Vec::new();
        let mut unreachable = Vec::new();
        for a in &reach.assessments {
            let slug = graph[a.assessment].slug.clone();
            if a.reachable_from_first_principle {
                reachable.push(slug);
            } else {
                unreachable.push(slug);
            }
        }

        let target_anchors: Vec<_> = graph
            .edges_directed(lo, petgraph::Direction::Incoming)
            .filter_map(|e| match &e.weight().kind {
                crate::graph::EdgeKind::Anchors(attrs)
                    if matches!(attrs.impact, crate::graph::AnchorImpact::Target) =>
                {
                    Some(graph[e.source()].slug.clone())
                }
                _ => None,
            })
            .collect();

        let sample = |list: &Vec<String>| list.iter().take(3).cloned().collect::<Vec<_>>();

        let payload = json!({
            "type": "graph_analysis",
            "tool": LO_ALIGNMENT,
            "lo_slug": self.args.lo_slug,
            "assessments": {
                "total": reach.assessments.len(),
                "reachable": reachable.len(),
                "unreachable": unreachable.len(),
                "sample_reachable": sample(&reachable),
                "sample_unreachable": sample(&unreachable),
            },
            "coverage": {
                "total_criteria": coverage.covered_criteria.len() + coverage.missing_criteria.len(),
                "covered": coverage.covered_criteria.len(),
                "missing": coverage.missing_criteria.len(),
                "missing_sample": sample(&coverage.missing_criteria),
                "unused_observation_features": coverage.unused_observation_features.len(),
                "unused_sample": sample(&coverage.unused_observation_features),
            },
            "target_anchors": {
                "total": target_anchors.len(),
                "sample": sample(&target_anchors),
            }
        });

        let approx_bytes = payload_size_bytes(&payload);
        let estimates = prepare_payload_estimates(&self.metrics, self.model.as_str(), approx_bytes);
        let mode = ToolPayloadMode::from_fetch_flag(self.args.fetch_body);
        info!(
            tool = LO_ALIGNMENT,
            lo_slug = %self.args.lo_slug,
            mode = mode.as_str(),
            approx_bytes,
            "graph lo alignment summary"
        );

        match mode {
            ToolPayloadMode::Preview => {
                let hints = vec![format!(
                    "Summary is ~{} bytes; set fetch_body=true to retrieve it.",
                    approx_bytes
                )];
                let mut preview =
                    build_cost_preview(LO_ALIGNMENT, approx_bytes, estimates.safe_tokens, hints);
                let preview_bytes = payload_size_bytes(&preview);
                let preview_tokens = estimate_tokens_from_characters(preview_bytes as usize);
                apply_preview_cost(
                    &mut preview,
                    &self.metrics,
                    self.model.as_str(),
                    self.conversation_id.as_str(),
                    preview_tokens,
                    estimates.safe_tokens,
                );
                Ok(ToolOutput::with_byte_hint(preview, preview_bytes))
            }
            ToolPayloadMode::Body => Ok(ToolOutput::with_byte_hint(payload.clone(), approx_bytes)),
        }
    }
}

fn parse_lo_assessments_view(
    raw: Value,
    state: &CallState,
) -> ToolInputResult<Box<dyn ToolInstance>> {
    let mut args: LoAssessmentsViewArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    LO_ASSESSMENTS_VIEW,
            message: err.to_string(),
        })?;
    args.lo_slug = require_string(args.lo_slug.clone(), LO_ASSESSMENTS_VIEW, "lo_slug")?;
    Ok(Box::new(LoAssessmentsViewTool {
        args,
        graph: state.graph.clone(),
    }))
}

fn parse_lo_missing_criteria_view(
    raw: Value,
    state: &CallState,
) -> ToolInputResult<Box<dyn ToolInstance>> {
    let mut args: LoMissingCriteriaViewArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    LO_MISSING_CRITERIA_VIEW,
            message: err.to_string(),
        })?;
    args.lo_slug = require_string(args.lo_slug.clone(), LO_MISSING_CRITERIA_VIEW, "lo_slug")?;
    Ok(Box::new(LoMissingCriteriaViewTool {
        args,
        graph: state.graph.clone(),
    }))
}

fn parse_lo_anchors_view(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let mut args: LoAnchorsViewArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    LO_ANCHORS_VIEW,
            message: err.to_string(),
        })?;
    args.lo_slug = require_string(args.lo_slug.clone(), LO_ANCHORS_VIEW, "lo_slug")?;
    Ok(Box::new(LoAnchorsViewTool {
        args,
        graph: state.graph.clone(),
    }))
}

struct LoAssessmentsViewTool {
    args:  LoAssessmentsViewArgs,
    graph: ActorRef<crate::graph::manager::GraphManager>,
}

#[async_trait]
impl ToolInstance for LoAssessmentsViewTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let lo = resolve_slug(&self.graph, self.args.lo_slug.clone(), LO_ASSESSMENTS_VIEW).await?;
        let graph = self
            .graph
            .ask(crate::graph::manager::GetGraph)
            .await
            .map_err(map_send_err_inf)?;
        ensure_knowledge_type(
            &graph,
            lo,
            &self.args.lo_slug,
            KnowledgeType::LearningOutcome,
            LO_ASSESSMENTS_VIEW,
        )?;

        let fps = analysis::first_principles(&graph);
        let report = analysis::lo_reachability(&graph, lo, &fps);

        let items: Vec<_> = report
            .assessments
            .into_iter()
            .filter_map(|a| {
                if let Some(filter) = self.args.reachable_only
                    && filter != a.reachable_from_first_principle
                {
                    return None;
                }
                Some(json!({
                    "assessment_slug": graph[a.assessment].slug.clone(),
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

        Ok(ToolOutput::with_byte_hint(payload.clone(), payload_size_bytes(&payload)))
    }
}

struct LoMissingCriteriaViewTool {
    args:  LoMissingCriteriaViewArgs,
    graph: ActorRef<crate::graph::manager::GraphManager>,
}

#[async_trait]
impl ToolInstance for LoMissingCriteriaViewTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let lo =
            resolve_slug(&self.graph, self.args.lo_slug.clone(), LO_MISSING_CRITERIA_VIEW).await?;
        let graph = self
            .graph
            .ask(crate::graph::manager::GetGraph)
            .await
            .map_err(map_send_err_inf)?;
        ensure_knowledge_type(
            &graph,
            lo,
            &self.args.lo_slug,
            KnowledgeType::LearningOutcome,
            LO_MISSING_CRITERIA_VIEW,
        )?;

        let coverage = analysis::coverage_report(&graph, lo);
        let missing = coverage.missing_criteria;

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
        let lo = resolve_slug(&self.graph, self.args.lo_slug.clone(), LO_ANCHORS_VIEW).await?;
        let graph = self
            .graph
            .ask(crate::graph::manager::GetGraph)
            .await
            .map_err(map_send_err_inf)?;
        ensure_knowledge_type(
            &graph,
            lo,
            &self.args.lo_slug,
            KnowledgeType::LearningOutcome,
            LO_ANCHORS_VIEW,
        )?;

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

        Ok(ToolOutput::with_byte_hint(payload.clone(), payload_size_bytes(&payload)))
    }
}

// ---------- Gap summary + views ----------

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
    let args: GapSummaryArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    GAP_SUMMARY,
            message: err.to_string(),
        })?;
    Ok(Box::new(GapSummaryTool {
        args,
        graph: state.graph.clone(),
        metrics: Arc::clone(&state.metrics),
        model: Arc::clone(&state.model),
        conversation_id: Arc::clone(&state.conversation_id),
    }))
}

fn parse_example_gaps_view(
    raw: Value,
    state: &CallState,
) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: GapViewArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    EXAMPLE_GAPS_VIEW,
            message: err.to_string(),
        })?;
    Ok(Box::new(ExampleGapsViewTool {
        args,
        graph: state.graph.clone(),
    }))
}

fn parse_fadeability_view(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: GapViewArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    FADEABILITY_VIEW,
            message: err.to_string(),
        })?;
    Ok(Box::new(FadeabilityViewTool {
        args,
        graph: state.graph.clone(),
    }))
}

fn parse_practice_gaps_view(
    raw: Value,
    state: &CallState,
) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: GapViewArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    PRACTICE_GAPS_VIEW,
            message: err.to_string(),
        })?;
    Ok(Box::new(PracticeGapsViewTool {
        args,
        graph: state.graph.clone(),
    }))
}

struct GapSummaryTool {
    args:            GapSummaryArgs,
    graph:           ActorRef<crate::graph::manager::GraphManager>,
    metrics:         Arc<crate::llm_gateway::GatewayMetrics>,
    model:           Arc<String>,
    conversation_id: Arc<String>,
}

#[async_trait]
impl ToolInstance for GapSummaryTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = self
            .graph
            .ask(crate::graph::manager::GetGraph)
            .await
            .map_err(map_send_err_inf)?;

        let example = analysis::example_gaps(&graph);
        let fade = analysis::fadeability_issues(&graph);
        let practice = analysis::procedural_practice_gaps(&graph);

        let payload = json!({
            "type": "graph_analysis",
            "tool": GAP_SUMMARY,
            "summary": {
                "example_gaps": {
                    "count": example.len(),
                },
                "fadeability": {
                    "assessments_with_issues": fade.len(),
                },
                "practice_gaps": {
                    "count": practice.len(),
                },
            }
        });

        let approx_bytes = payload_size_bytes(&payload);
        let estimates = prepare_payload_estimates(&self.metrics, self.model.as_str(), approx_bytes);
        let mode = ToolPayloadMode::from_fetch_flag(self.args.fetch_body);
        info!(tool = GAP_SUMMARY, mode = mode.as_str(), approx_bytes, "graph gap summary");

        match mode {
            ToolPayloadMode::Preview => {
                let hints = vec![format!(
                    "Summary is ~{} bytes; set fetch_body=true to retrieve it.",
                    approx_bytes
                )];
                let mut preview =
                    build_cost_preview(GAP_SUMMARY, approx_bytes, estimates.safe_tokens, hints);
                let preview_bytes = payload_size_bytes(&preview);
                let preview_tokens = estimate_tokens_from_characters(preview_bytes as usize);
                apply_preview_cost(
                    &mut preview,
                    &self.metrics,
                    self.model.as_str(),
                    self.conversation_id.as_str(),
                    preview_tokens,
                    estimates.safe_tokens,
                );
                Ok(ToolOutput::with_byte_hint(preview, preview_bytes))
            }
            ToolPayloadMode::Body => Ok(ToolOutput::with_byte_hint(payload.clone(), approx_bytes)),
        }
    }
}

struct ExampleGapsViewTool {
    args:  GapViewArgs,
    graph: ActorRef<crate::graph::manager::GraphManager>,
}

#[async_trait]
impl ToolInstance for ExampleGapsViewTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = self
            .graph
            .ask(crate::graph::manager::GetGraph)
            .await
            .map_err(map_send_err_inf)?;
        let gaps = analysis::example_gaps(&graph)
            .into_iter()
            .map(|gap| {
                json!({
                    "slug": graph[gap.node].slug.clone(),
                    "description": gap.description,
                })
            })
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

        Ok(ToolOutput::with_byte_hint(payload.clone(), payload_size_bytes(&payload)))
    }
}

struct FadeabilityViewTool {
    args:  GapViewArgs,
    graph: ActorRef<crate::graph::manager::GraphManager>,
}

#[async_trait]
impl ToolInstance for FadeabilityViewTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = self
            .graph
            .ask(crate::graph::manager::GetGraph)
            .await
            .map_err(map_send_err_inf)?;
        let issues = analysis::fadeability_issues(&graph)
            .into_iter()
            .map(|i| {
                let supports: Vec<_> = i
                    .support_edges
                    .into_iter()
                    .filter_map(|e| graph.edge_endpoints(e))
                    .map(|(u, v)| {
                        json!({
                            "from": graph[u].slug.clone(),
                            "to": graph[v].slug.clone(),
                        })
                    })
                    .collect();
                json!({
                    "assessment_slug": graph[i.assessment].slug.clone(),
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

        Ok(ToolOutput::with_byte_hint(payload.clone(), payload_size_bytes(&payload)))
    }
}

struct PracticeGapsViewTool {
    args:  GapViewArgs,
    graph: ActorRef<crate::graph::manager::GraphManager>,
}

#[async_trait]
impl ToolInstance for PracticeGapsViewTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = self
            .graph
            .ask(crate::graph::manager::GetGraph)
            .await
            .map_err(map_send_err_inf)?;
        let gaps = analysis::procedural_practice_gaps(&graph)
            .into_iter()
            .map(|g| json!({ "slug": graph[g.node].slug.clone() }))
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
    let _: NoArgs = serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
        tool:    KEYSTONE,
        message: err.to_string(),
    })?;
    Ok(Box::new(KeystoneTool {
        graph: state.graph.clone(),
    }))
}

struct KeystoneTool {
    graph: ActorRef<crate::graph::manager::GraphManager>,
}

#[async_trait]
impl ToolInstance for KeystoneTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = self
            .graph
            .ask(crate::graph::manager::GetGraph)
            .await
            .map_err(map_send_err_inf)?;
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

        let payload = json!({
            "type": "graph_analysis",
            "tool": KEYSTONE,
            "total_ranked": rendered.len(),
            "scores": rendered,
        });

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
    let mut args: ExtraneousArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    EXTRANEOUS,
            message: err.to_string(),
        })?;
    args.assessment_slug =
        require_string(args.assessment_slug.clone(), EXTRANEOUS, "assessment_slug")?;
    args.lo_slug = require_string(args.lo_slug.clone(), EXTRANEOUS, "lo_slug")?;
    Ok(Box::new(ExtraneousTool {
        args,
        graph: state.graph.clone(),
    }))
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
        let graph = self
            .graph
            .ask(crate::graph::manager::GetGraph)
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
    let args: AlignmentGapsArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    ALIGNMENT_GAPS,
            message: err.to_string(),
        })?;
    Ok(Box::new(AlignmentGapsTool {
        args,
        graph: state.graph.clone(),
        metrics: Arc::clone(&state.metrics),
        model: Arc::clone(&state.model),
        conversation_id: Arc::clone(&state.conversation_id),
    }))
}

struct AlignmentGapsTool {
    args:            AlignmentGapsArgs,
    graph:           ActorRef<crate::graph::manager::GraphManager>,
    metrics:         Arc<crate::llm_gateway::GatewayMetrics>,
    model:           Arc<String>,
    conversation_id: Arc<String>,
}

#[async_trait]
impl ToolInstance for AlignmentGapsTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = self
            .graph
            .ask(crate::graph::manager::GetGraph)
            .await
            .map_err(map_send_err_inf)?;
        let los = analysis::lo_missing_target_assessments(&graph);
        let orphan = analysis::orphan_assessments(&graph);
        let unreachable = analysis::unreachable_assessments(&graph);
        let payload = json!({
            "type": "graph_analysis",
            "tool": ALIGNMENT_GAPS,
            "los_missing_target_assessment": los.into_iter().map(|n| graph[n].slug.clone()).collect::<Vec<_>>(),
            "assessments_without_lo": orphan.into_iter().map(|n| graph[n].slug.clone()).collect::<Vec<_>>(),
            "assessments_unreachable": unreachable.into_iter().map(|n| graph[n].slug.clone()).collect::<Vec<_>>(),
        });
        let approx_bytes = payload_size_bytes(&payload);
        let estimates = prepare_payload_estimates(&self.metrics, self.model.as_str(), approx_bytes);
        let mode = ToolPayloadMode::from_fetch_flag(self.args.fetch_body);
        info!(
            tool = ALIGNMENT_GAPS,
            mode = mode.as_str(),
            approx_bytes,
            "graph assessment gaps"
        );
        match mode {
            ToolPayloadMode::Preview => {
                let hints = vec![format!(
                    "Payload is ~{} bytes; set fetch_body=true to retrieve it.",
                    approx_bytes
                )];
                let mut preview =
                    build_cost_preview(ALIGNMENT_GAPS, approx_bytes, estimates.safe_tokens, hints);
                let preview_bytes = payload_size_bytes(&preview);
                let preview_tokens = estimate_tokens_from_characters(preview_bytes as usize);
                apply_preview_cost(
                    &mut preview,
                    &self.metrics,
                    self.model.as_str(),
                    self.conversation_id.as_str(),
                    preview_tokens,
                    estimates.safe_tokens,
                );
                Ok(ToolOutput::with_byte_hint(preview, preview_bytes))
            }
            ToolPayloadMode::Body => Ok(ToolOutput::with_byte_hint(payload.clone(), approx_bytes)),
        }
    }
}

// ---------- Discourse analyses ----------

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
    let args: DiscourseOrphansArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    DISCOURSE_ORPHANS,
            message: err.to_string(),
        })?;
    Ok(Box::new(DiscourseOrphansTool {
        args,
        graph: state.graph.clone(),
    }))
}

struct DiscourseOrphansTool {
    args:  DiscourseOrphansArgs,
    graph: ActorRef<crate::graph::manager::GraphManager>,
}

#[async_trait]
impl ToolInstance for DiscourseOrphansTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = self
            .graph
            .ask(crate::graph::manager::GetGraph)
            .await
            .map_err(map_send_err_inf)?;
        let list = analysis::discourse_orphans(&graph, self.args.episode.as_deref());
        let slugs: Vec<_> = list.into_iter().map(|n| graph[n].slug.clone()).collect();
        let payload = json!({
            "type": "graph_analysis",
            "tool": DISCOURSE_ORPHANS,
            "episode": self.args.episode,
            "orphans": slugs,
        });
        info!(
            tool = DISCOURSE_ORPHANS,
            episode = ?self.args.episode,
            orphan_count = payload["orphans"].as_array().map(|v| v.len()).unwrap_or(0),
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
    let mut args: BorrowAheadArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    BORROW_AHEAD,
            message: err.to_string(),
        })?;
    args.episode = require_string(args.episode.clone(), BORROW_AHEAD, "episode")?;
    Ok(Box::new(BorrowAheadTool {
        args,
        graph: state.graph.clone(),
    }))
}

struct BorrowAheadTool {
    args:  BorrowAheadArgs,
    graph: ActorRef<crate::graph::manager::GraphManager>,
}

#[async_trait]
impl ToolInstance for BorrowAheadTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = self
            .graph
            .ask(crate::graph::manager::GetGraph)
            .await
            .map_err(map_send_err_inf)?;
        let results = analysis::borrow_ahead(&graph, &self.args.episode);
        let rendered: Vec<_> = results
            .into_iter()
            .map(|b| {
                json!({
                    "step_slug": graph[b.step].slug,
                    "target_slug": graph[b.target].slug,
                    "severity": b.severity,
                })
            })
            .collect();
        let payload = json!({
            "type": "graph_analysis",
            "tool": BORROW_AHEAD,
            "episode": self.args.episode,
            "borrow_ahead": rendered,
        });
        info!(
            tool = BORROW_AHEAD,
            episode = %self.args.episode,
            count = payload["borrow_ahead"].as_array().map(|v| v.len()).unwrap_or(0),
            "graph borrow ahead"
        );
        Ok(ToolOutput::with_byte_hint(payload.clone(), payload_size_bytes(&payload)))
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
        gap_summary_meta(),
        example_gaps_view_meta(),
        fadeability_view_meta(),
        practice_gaps_view_meta(),
        keystone_meta(),
        extraneous_meta(),
        alignment_gaps_meta(),
        discourse_orphans_meta(),
        borrow_ahead_meta(),
    ]
}
