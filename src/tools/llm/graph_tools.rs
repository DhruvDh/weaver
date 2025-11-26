use std::{env, sync::Arc, time::Duration};

use async_trait::async_trait;
use bon::Builder;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use petgraph::visit::EdgeRef;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::time;
use tracing::warn;

use crate::{
    analysis,
    graph::{
        AnchorImpact, AssessesAttrs, CaseTag, GraphError, GraphService, IntroductionScope,
        KnowledgeNode, NodeKind, RequiresAttrs, SupportsAttrs, TeachingPurpose, TeachingStepNode,
    },
    schema::types::{
        AssessmentScope, EvidenceLink, IntendedEffect, KnowledgeType, SourceRef, Strength,
        SupportKind,
    },
    tools::llm::{
        CallState, ToolExecutionError, ToolInputError, ToolInputResult, ToolInstance, ToolOutput,
        ToolPrototype, graph_neighbors::neighbors_meta, schema_for_args,
    },
};

/// Shared in-memory graph for tool calls in this process.
pub(crate) static GRAPH: Lazy<Arc<RwLock<GraphService>>> = Lazy::new(|| {
    let svc = Arc::new(RwLock::new(GraphService::new()));
    start_autosave(Arc::clone(&svc));
    svc
});

// ---------- Helpers ----------

pub(crate) fn map_graph_err(err: GraphError, tool: &'static str) -> ToolExecutionError {
    match err {
        GraphError::MissingSlug(_) => ToolExecutionError::Input(ToolInputError::InvalidPath {
            tool,
            path: "".into(),
            message: err.to_string(),
        }),
        GraphError::InvalidEndpoints { .. } | GraphError::RequiresCycle | GraphError::Schema(_) => {
            ToolExecutionError::Input(ToolInputError::InvalidPayload {
                tool,
                message: err.to_string(),
            })
        }
    }
}

fn start_autosave(service: Arc<RwLock<GraphService>>) {
    // Requires a Tokio runtime; if unavailable, skip autosave.
    if tokio::runtime::Handle::try_current().is_err() {
        warn!("graph autosave not started: no Tokio runtime");
        return;
    }
    let path =
        env::var("GRAPH_SNAPSHOT_PATH").unwrap_or_else(|_| "graph_snapshot.json".to_string());
    let interval_secs = env::var("GRAPH_AUTOSAVE_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300u64);
    tokio::spawn(async move {
        let mut ticker = time::interval(Duration::from_secs(interval_secs));
        loop {
            ticker.tick().await;
            let graph_clone = { service.read().snapshot_graph() };
            if let Err(err) = crate::graph::persist::save_graph(&graph_clone, &path).await {
                warn!(error = ?err, "graph autosave failed");
            }
        }
    });
}

// ---------- Insert knowledge node ----------

const INSERT_KNOWLEDGE: &str = "graph_insert_knowledge";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InsertKnowledgeArgs {
    pub slug: String,
    pub title: String,
    pub statement: String,
    pub knowledge_type: KnowledgeType,
    #[serde(default)]
    pub rubric_criteria: Vec<String>,
    #[serde(default)]
    pub construct_irrelevant_demands: Vec<String>,
    #[serde(default)]
    pub source_refs: Vec<SourceRef>,
    #[serde(default)]
    pub confidence: f32,
    #[serde(default)]
    pub grain_level: Option<crate::graph::GrainLevel>,
    #[serde(default)]
    pub intrinsic_load: Option<crate::graph::IntrinsicLoad>,
    #[serde(default)]
    pub introduction_scope: Option<IntroductionScope>,
    #[serde(default)]
    pub tags: Vec<String>,
}

pub(super) fn insert_knowledge_meta() -> ToolPrototype {
    ToolPrototype {
        id:          INSERT_KNOWLEDGE,
        description: "Insert or update a knowledge/LO/assessment node in the in-memory graph.",
        schema:      schema_for_args::<InsertKnowledgeArgs>(),
        parse:       parse_insert_knowledge,
    }
}

fn parse_insert_knowledge(
    raw: Value,
    _state: &CallState,
) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: InsertKnowledgeArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    INSERT_KNOWLEDGE,
            message: err.to_string(),
        })?;
    Ok(Box::new(InsertKnowledgeTool { args }))
}

struct InsertKnowledgeTool {
    args: InsertKnowledgeArgs,
}

#[async_trait]
impl ToolInstance for InsertKnowledgeTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let mut graph = GRAPH.write();
        let payload = KnowledgeNode {
            title: self.args.title.clone(),
            statement: self.args.statement.clone(),
            knowledge_type: self.args.knowledge_type,
            source_refs: self.args.source_refs.clone(),
            confidence: self.args.confidence,
            rubric_criteria: self.args.rubric_criteria.clone(),
            construct_irrelevant_demands: self.args.construct_irrelevant_demands.clone(),
            grain_level: self.args.grain_level,
            intrinsic_load: self.args.intrinsic_load,
            introduction_scope: self
                .args
                .introduction_scope
                .unwrap_or(IntroductionScope::InCourse),
        };
        let id = graph
            .add_knowledge_node(self.args.slug.clone(), payload, self.args.tags.clone())
            .map_err(|e| map_graph_err(e, INSERT_KNOWLEDGE))?;
        Ok(ToolOutput::new(json!({
            "status": "ok",
            "node_id": id.index(),
        })))
    }
}

// ---------- Insert teaching step ----------

const INSERT_TEACHING: &str = "graph_insert_teaching_step";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InsertTeachingArgs {
    pub slug:        String,
    pub title:       String,
    pub statement:   String,
    pub purpose:     TeachingPurpose,
    #[serde(default)]
    pub method_tags: Vec<String>,
    pub episode:     String,
    #[serde(default)]
    pub source_refs: Vec<SourceRef>,
    #[serde(default)]
    pub tags:        Vec<String>,
}

pub(super) fn insert_teaching_meta() -> ToolPrototype {
    ToolPrototype {
        id:          INSERT_TEACHING,
        description: "Insert a TeachingStep node.",
        schema:      schema_for_args::<InsertTeachingArgs>(),
        parse:       parse_insert_teaching,
    }
}

fn parse_insert_teaching(raw: Value, _state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: InsertTeachingArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    INSERT_TEACHING,
            message: err.to_string(),
        })?;
    Ok(Box::new(InsertTeachingTool { args }))
}

struct InsertTeachingTool {
    args: InsertTeachingArgs,
}

#[async_trait]
impl ToolInstance for InsertTeachingTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let mut graph = GRAPH.write();
        let payload = TeachingStepNode {
            title:       self.args.title.clone(),
            statement:   self.args.statement.clone(),
            purpose:     self.args.purpose,
            method_tags: self.args.method_tags.clone(),
            episode:     self.args.episode.clone(),
            source_refs: self.args.source_refs.clone(),
        };
        let id = graph
            .add_teaching_step(self.args.slug.clone(), payload, self.args.tags.clone())
            .map_err(|e| map_graph_err(e, INSERT_TEACHING))?;
        Ok(ToolOutput::new(json!({"status": "ok", "node_id": id.index()})))
    }
}

// ---------- add requires ----------

const ADD_REQUIRES: &str = "graph_add_requires";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AddRequiresArgs {
    pub from_slug:     String,
    pub to_slug:       String,
    pub strength:      Strength,
    pub rationale:     String,
    #[serde(default)]
    pub evidence_refs: Vec<SourceRef>,
    #[serde(default = "default_confidence")]
    pub confidence:    f32,
}

fn default_confidence() -> f32 {
    1.0
}

pub(super) fn add_requires_meta() -> ToolPrototype {
    ToolPrototype {
        id:          ADD_REQUIRES,
        description: "Add a requires edge (guarded by DAG check).",
        schema:      schema_for_args::<AddRequiresArgs>(),
        parse:       parse_add_requires,
    }
}

fn parse_add_requires(raw: Value, _state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: AddRequiresArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    ADD_REQUIRES,
            message: err.to_string(),
        })?;
    Ok(Box::new(AddRequiresTool { args }))
}

struct AddRequiresTool {
    args: AddRequiresArgs,
}

#[async_trait]
impl ToolInstance for AddRequiresTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let mut graph = GRAPH.write();
        let from = graph
            .node_by_slug(&self.args.from_slug)
            .map_err(|e| map_graph_err(e, ADD_REQUIRES))?;
        let to = graph
            .node_by_slug(&self.args.to_slug)
            .map_err(|e| map_graph_err(e, ADD_REQUIRES))?;
        graph
            .add_requires_edge(
                from,
                to,
                RequiresAttrs {
                    strength:      self.args.strength,
                    rationale:     self.args.rationale.clone(),
                    evidence_refs: self.args.evidence_refs.clone(),
                },
                self.args.confidence,
            )
            .map_err(|e| map_graph_err(e, ADD_REQUIRES))?;
        Ok(ToolOutput::new(json!({"status": "ok"})))
    }
}

// ---------- add supports ----------

const ADD_SUPPORTS: &str = "graph_add_supports";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AddSupportsArgs {
    pub from_slug:       String,
    pub to_slug:         String,
    pub support_kind:    SupportKind,
    pub intended_effect: IntendedEffect,
    #[serde(default)]
    pub case_tag:        Option<CaseTag>,
    #[serde(default)]
    pub coverage_tags:   Vec<String>,
    #[serde(default)]
    pub evidence_refs:   Vec<SourceRef>,
    #[serde(default = "default_confidence")]
    pub confidence:      f32,
}

pub(super) fn add_supports_meta() -> ToolPrototype {
    ToolPrototype {
        id:          ADD_SUPPORTS,
        description: "Add a supports edge.",
        schema:      schema_for_args::<AddSupportsArgs>(),
        parse:       parse_add_supports,
    }
}

fn parse_add_supports(raw: Value, _state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: AddSupportsArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    ADD_SUPPORTS,
            message: err.to_string(),
        })?;
    Ok(Box::new(AddSupportsTool { args }))
}

struct AddSupportsTool {
    args: AddSupportsArgs,
}

#[async_trait]
impl ToolInstance for AddSupportsTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let mut graph = GRAPH.write();
        let from = graph
            .node_by_slug(&self.args.from_slug)
            .map_err(|e| map_graph_err(e, ADD_SUPPORTS))?;
        let to = graph
            .node_by_slug(&self.args.to_slug)
            .map_err(|e| map_graph_err(e, ADD_SUPPORTS))?;
        graph
            .add_supports_edge(
                from,
                to,
                SupportsAttrs {
                    support_kind:    self.args.support_kind,
                    intended_effect: self.args.intended_effect,
                    case_tag:        self.args.case_tag,
                    coverage_tags:   self.args.coverage_tags.clone(),
                    evidence_refs:   self.args.evidence_refs.clone(),
                },
                self.args.confidence,
            )
            .map_err(|e| map_graph_err(e, ADD_SUPPORTS))?;
        Ok(ToolOutput::new(json!({"status": "ok"})))
    }
}

// ---------- add assesses ----------

const ADD_ASSESSES: &str = "graph_add_assesses";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AddAssessesArgs {
    pub from_slug:            String,
    pub to_slug:              String,
    pub claim:                String,
    #[serde(default)]
    pub observation_features: Vec<String>,
    pub scope:                AssessmentScope,
    #[serde(default = "default_confidence")]
    pub confidence:           f32,
}

pub(super) fn add_assesses_meta() -> ToolPrototype {
    ToolPrototype {
        id:          ADD_ASSESSES,
        description: "Add an assesses edge (assessment -> LO).",
        schema:      schema_for_args::<AddAssessesArgs>(),
        parse:       parse_add_assesses,
    }
}

fn parse_add_assesses(raw: Value, _state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: AddAssessesArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    ADD_ASSESSES,
            message: err.to_string(),
        })?;
    Ok(Box::new(AddAssessesTool { args }))
}

struct AddAssessesTool {
    args: AddAssessesArgs,
}

#[async_trait]
impl ToolInstance for AddAssessesTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let mut graph = GRAPH.write();
        let from = graph
            .node_by_slug(&self.args.from_slug)
            .map_err(|e| map_graph_err(e, ADD_ASSESSES))?;
        let to = graph
            .node_by_slug(&self.args.to_slug)
            .map_err(|e| map_graph_err(e, ADD_ASSESSES))?;
        graph
            .add_assesses_edge(
                from,
                to,
                AssessesAttrs {
                    evidence_link: EvidenceLink {
                        claim:                self.args.claim.clone(),
                        observation_features: self.args.observation_features.clone(),
                        scope:                self.args.scope,
                    },
                },
                self.args.confidence,
            )
            .map_err(|e| map_graph_err(e, ADD_ASSESSES))?;
        Ok(ToolOutput::new(json!({"status": "ok"})))
    }
}

// ---------- add precedes ----------

const ADD_PRECEDES: &str = "graph_add_precedes";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AddPrecedesArgs {
    pub from_slug:  String,
    pub to_slug:    String,
    pub episode:    String,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

pub(super) fn add_precedes_meta() -> ToolPrototype {
    ToolPrototype {
        id:          ADD_PRECEDES,
        description: "Add a precedes edge between TeachingSteps.",
        schema:      schema_for_args::<AddPrecedesArgs>(),
        parse:       parse_add_precedes,
    }
}

fn parse_add_precedes(raw: Value, _state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: AddPrecedesArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    ADD_PRECEDES,
            message: err.to_string(),
        })?;
    Ok(Box::new(AddPrecedesTool { args }))
}

struct AddPrecedesTool {
    args: AddPrecedesArgs,
}

#[async_trait]
impl ToolInstance for AddPrecedesTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let mut graph = GRAPH.write();
        let from = graph
            .node_by_slug(&self.args.from_slug)
            .map_err(|e| map_graph_err(e, ADD_PRECEDES))?;
        let to = graph
            .node_by_slug(&self.args.to_slug)
            .map_err(|e| map_graph_err(e, ADD_PRECEDES))?;
        graph
            .add_precedes_edge(from, to, self.args.episode.clone(), self.args.confidence)
            .map_err(|e| map_graph_err(e, ADD_PRECEDES))?;
        Ok(ToolOutput::new(json!({"status": "ok"})))
    }
}

// ---------- add anchors ----------

const ADD_ANCHORS: &str = "graph_add_anchors";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AddAnchorsArgs {
    pub from_slug:  String,
    pub to_slug:    String,
    pub impact:     AnchorImpact,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

pub(super) fn add_anchors_meta() -> ToolPrototype {
    ToolPrototype {
        id:          ADD_ANCHORS,
        description: "Add an anchors edge (TeachingStep -> Knowledge/LO/Assessment).",
        schema:      schema_for_args::<AddAnchorsArgs>(),
        parse:       parse_add_anchors,
    }
}

fn parse_add_anchors(raw: Value, _state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: AddAnchorsArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    ADD_ANCHORS,
            message: err.to_string(),
        })?;
    Ok(Box::new(AddAnchorsTool { args }))
}

struct AddAnchorsTool {
    args: AddAnchorsArgs,
}

#[async_trait]
impl ToolInstance for AddAnchorsTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let mut graph = GRAPH.write();
        let from = graph
            .node_by_slug(&self.args.from_slug)
            .map_err(|e| map_graph_err(e, ADD_ANCHORS))?;
        let to = graph
            .node_by_slug(&self.args.to_slug)
            .map_err(|e| map_graph_err(e, ADD_ANCHORS))?;
        graph
            .add_anchors_edge(from, to, self.args.impact, self.args.confidence)
            .map_err(|e| map_graph_err(e, ADD_ANCHORS))?;
        Ok(ToolOutput::new(json!({"status": "ok"})))
    }
}

// ---------- Inspection ----------

const GET_NODE: &str = "graph_get_node";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GetNodeArgs {
    pub slug: String,
}

pub(super) fn get_node_meta() -> ToolPrototype {
    ToolPrototype {
        id:          GET_NODE,
        description: "Fetch a node payload by slug.",
        schema:      schema_for_args::<GetNodeArgs>(),
        parse:       parse_get_node,
    }
}

fn parse_get_node(raw: Value, _state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: GetNodeArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    GET_NODE,
            message: err.to_string(),
        })?;
    Ok(Box::new(GetNodeTool { args }))
}

struct GetNodeTool {
    args: GetNodeArgs,
}

#[async_trait]
impl ToolInstance for GetNodeTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = GRAPH.read();
        let id = graph
            .node_by_slug(&self.args.slug)
            .map_err(|e| map_graph_err(e, GET_NODE))?;
        let payload = &graph.graph()[id];
        let value = match &payload.kind {
            NodeKind::Knowledge(k) => json!({
                "slug": payload.slug,
                "logical_id": payload.logical_id,
                "kind": "knowledge",
                "knowledge_type": k.knowledge_type,
                "title": k.title,
                "statement": k.statement,
                "confidence": k.confidence,
                "rubric_criteria": k.rubric_criteria,
                "construct_irrelevant_demands": k.construct_irrelevant_demands,
                "grain_level": k.grain_level,
                "intrinsic_load": k.intrinsic_load,
                "introduction_scope": k.introduction_scope,
                "source_refs": k.source_refs,
                "tags": payload.tags,
            }),
            NodeKind::TeachingStep(ts) => json!({
                "slug": payload.slug,
                "logical_id": payload.logical_id,
                "kind": "teaching_step",
                "title": ts.title,
                "statement": ts.statement,
                "purpose": ts.purpose,
                "episode": ts.episode,
                "method_tags": ts.method_tags,
                "source_refs": ts.source_refs,
                "tags": payload.tags,
            }),
        };
        Ok(ToolOutput::new(json!({"status": "ok", "node": value})))
    }
}

// ---------- Analyses ----------

const DAG_CHECK: &str = "graph_dag_check";

pub(super) fn dag_check_meta() -> ToolPrototype {
    ToolPrototype {
        id:          DAG_CHECK,
        description: "Check whether requires layer is a DAG; returns optional cycle indicator.",
        schema:      schema_for_args::<NoArgs>(),
        parse:       parse_dag_check,
    }
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct NoArgs {}

fn parse_dag_check(raw: Value, _state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let _: NoArgs = serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
        tool:    DAG_CHECK,
        message: err.to_string(),
    })?;
    Ok(Box::new(DAGCheckTool))
}

#[derive(Default)]
struct DAGCheckTool;

#[async_trait]
impl ToolInstance for DAGCheckTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = GRAPH.read();
        let dag = analysis::requires_is_dag(graph.graph());
        let topo = analysis::requires_toposort(graph.graph()).ok();
        Ok(ToolOutput::new(json!({
            "status": "ok",
            "is_dag": dag,
            "topo_order_count": topo.as_ref().map(|v| v.len()),
        })))
    }
}

const FIRST_PRINCIPLES: &str = "graph_first_principles";

pub(super) fn first_principles_meta() -> ToolPrototype {
    ToolPrototype {
        id:          FIRST_PRINCIPLES,
        description: "List slugs of first-principle nodes (requires in-degree 0).",
        schema:      schema_for_args::<NoArgs>(),
        parse:       parse_first_principles,
    }
}

#[derive(Default)]
struct FirstPrinciplesTool;

fn parse_first_principles(
    raw: Value,
    _state: &CallState,
) -> ToolInputResult<Box<dyn ToolInstance>> {
    let _: NoArgs = serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
        tool:    FIRST_PRINCIPLES,
        message: err.to_string(),
    })?;
    Ok(Box::new(FirstPrinciplesTool))
}

#[async_trait]
impl ToolInstance for FirstPrinciplesTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = GRAPH.read();
        let fps = analysis::first_principles(graph.graph());
        let slugs: Vec<_> = fps
            .iter()
            .map(|id| graph.graph()[*id].slug.clone())
            .collect();
        Ok(ToolOutput::new(json!({"status": "ok", "first_principles": slugs})))
    }
}

const LO_REACH: &str = "graph_lo_reachability";
const LO_ALIGNMENT: &str = "graph_lo_alignment_summary";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LoReachArgs {
    pub lo_slug: String,
}

pub(super) fn lo_reach_meta() -> ToolPrototype {
    ToolPrototype {
        id:          LO_REACH,
        description: "Reachability report for a Learning Outcome.",
        schema:      schema_for_args::<LoReachArgs>(),
        parse:       parse_lo_reach,
    }
}

pub(super) fn lo_alignment_meta() -> ToolPrototype {
    ToolPrototype {
        id:          LO_ALIGNMENT,
        description: "Alignment summary for a Learning Outcome: reachability, coverage, target \
                      anchors.",
        schema:      schema_for_args::<LoReachArgs>(),
        parse:       parse_lo_alignment,
    }
}

fn parse_lo_reach(raw: Value, _state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: LoReachArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    LO_REACH,
            message: err.to_string(),
        })?;
    Ok(Box::new(LoReachTool { args }))
}

struct LoReachTool {
    args: LoReachArgs,
}

#[async_trait]
impl ToolInstance for LoReachTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = GRAPH.read();
        let lo = graph
            .node_by_slug(&self.args.lo_slug)
            .map_err(|e| map_graph_err(e, LO_REACH))?;
        let fps = analysis::first_principles(graph.graph());
        let report = analysis::lo_reachability(graph.graph(), lo, &fps);
        let assessments = report
            .assessments
            .into_iter()
            .map(|a| {
                json!({
                    "assessment_slug": graph.graph()[a.assessment].slug,
                    "reachable_from_first_principle": a.reachable_from_first_principle,
                })
            })
            .collect::<Vec<_>>();
        Ok(ToolOutput::new(json!({"status": "ok", "assessments": assessments})))
    }
}

const COVERAGE: &str = "graph_lo_coverage";

pub(super) fn coverage_meta() -> ToolPrototype {
    ToolPrototype {
        id:          COVERAGE,
        description: "Coverage report for a Learning Outcome.",
        schema:      schema_for_args::<LoReachArgs>(),
        parse:       parse_lo_coverage,
    }
}

fn parse_lo_coverage(raw: Value, _state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: LoReachArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    COVERAGE,
            message: err.to_string(),
        })?;
    Ok(Box::new(LoCoverageTool { args }))
}

struct LoCoverageTool {
    args: LoReachArgs,
}

#[async_trait]
impl ToolInstance for LoCoverageTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = GRAPH.read();
        let lo = graph
            .node_by_slug(&self.args.lo_slug)
            .map_err(|e| map_graph_err(e, COVERAGE))?;
        let report = analysis::coverage_report(graph.graph(), lo);
        Ok(ToolOutput::new(json!({
            "status": "ok",
            "covered": report.covered_criteria,
            "missing": report.missing_criteria,
            "unused_observation_features": report.unused_observation_features,
        })))
    }
}

fn parse_lo_alignment(raw: Value, _state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: LoReachArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    LO_ALIGNMENT,
            message: err.to_string(),
        })?;
    Ok(Box::new(LoAlignmentTool { args }))
}

struct LoAlignmentTool {
    args: LoReachArgs,
}

#[async_trait]
impl ToolInstance for LoAlignmentTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = GRAPH.read();
        let lo = graph
            .node_by_slug(&self.args.lo_slug)
            .map_err(|e| map_graph_err(e, LO_ALIGNMENT))?;

        // Reachability
        let fps = analysis::first_principles(graph.graph());
        let reach = analysis::lo_reachability(graph.graph(), lo, &fps);
        let assessments = reach
            .assessments
            .into_iter()
            .map(|a| {
                json!({
                    "assessment_slug": graph.graph()[a.assessment].slug,
                    "reachable_from_first_principle": a.reachable_from_first_principle,
                })
            })
            .collect::<Vec<_>>();

        // Coverage
        let coverage = analysis::coverage_report(graph.graph(), lo);

        // Anchors with impact=target
        let target_anchors: Vec<_> = graph
            .graph()
            .edges_directed(lo, petgraph::Direction::Incoming)
            .filter_map(|e| match &e.weight().kind {
                crate::graph::EdgeKind::Anchors(attrs)
                    if matches!(attrs.impact, crate::graph::AnchorImpact::Target) =>
                {
                    Some(graph.graph()[e.source()].slug.clone())
                }
                _ => None,
            })
            .collect();

        Ok(ToolOutput::new(json!({
            "status": "ok",
            "assessments": assessments,
            "coverage": {
                "covered": coverage.covered_criteria,
                "missing": coverage.missing_criteria,
                "unused_observation_features": coverage.unused_observation_features,
            },
            "target_anchors": target_anchors,
        })))
    }
}

const EXAMPLE_GAPS: &str = "graph_example_gaps";

pub(super) fn example_gaps_meta() -> ToolPrototype {
    ToolPrototype {
        id:          EXAMPLE_GAPS,
        description: "List nodes missing required examples per policy.",
        schema:      schema_for_args::<NoArgs>(),
        parse:       parse_example_gaps,
    }
}

#[derive(Default)]
struct ExampleGapsTool;

fn parse_example_gaps(raw: Value, _state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let _: NoArgs = serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
        tool:    EXAMPLE_GAPS,
        message: err.to_string(),
    })?;
    Ok(Box::new(ExampleGapsTool))
}

#[async_trait]
impl ToolInstance for ExampleGapsTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = GRAPH.read();
        let gaps = analysis::example_gaps(graph.graph());
        let rendered: Vec<_> = gaps
            .into_iter()
            .map(|gap| {
                json!({
                    "slug": graph.graph()[gap.node].slug,
                    "description": gap.description,
                })
            })
            .collect();
        Ok(ToolOutput::new(json!({"status": "ok", "gaps": rendered})))
    }
}

const KEYSTONE: &str = "graph_keystone";

pub(super) fn keystone_meta() -> ToolPrototype {
    ToolPrototype {
        id:          KEYSTONE,
        description: "Compute keystone scores (in_reach * out_reach).",
        schema:      schema_for_args::<NoArgs>(),
        parse:       parse_keystone,
    }
}

#[derive(Default)]
struct KeystoneTool;

fn parse_keystone(raw: Value, _state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let _: NoArgs = serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
        tool:    KEYSTONE,
        message: err.to_string(),
    })?;
    Ok(Box::new(KeystoneTool))
}

#[async_trait]
impl ToolInstance for KeystoneTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = GRAPH.read();
        let scores = analysis::keystone_scores(graph.graph());
        let rendered: Vec<_> = scores
            .into_iter()
            .map(|score| {
                json!({
                    "slug": graph.graph()[score.node].slug,
                    "score": score.score,
                    "in_reach": score.in_reach,
                    "out_reach": score.out_reach,
                })
            })
            .collect();
        Ok(ToolOutput::new(json!({"status": "ok", "scores": rendered})))
    }
}

const BORROW_AHEAD: &str = "graph_borrow_ahead";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BorrowAheadArgs {
    pub episode: String,
}

const FADEABILITY: &str = "graph_fadeability";

pub(super) fn fadeability_meta() -> ToolPrototype {
    ToolPrototype {
        id:          FADEABILITY,
        description: "Detect assessments reachable only when supports edges act as prerequisites.",
        schema:      schema_for_args::<NoArgs>(),
        parse:       parse_fadeability,
    }
}

fn parse_fadeability(raw: Value, _state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let _: NoArgs = serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
        tool:    FADEABILITY,
        message: err.to_string(),
    })?;
    Ok(Box::new(FadeabilityTool))
}

struct FadeabilityTool;

#[async_trait]
impl ToolInstance for FadeabilityTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = GRAPH.read();
        let issues = analysis::fadeability_issues(graph.graph());
        let rendered: Vec<_> = issues
            .into_iter()
            .map(|i| json!({ "assessment_slug": graph.graph()[i.assessment].slug }))
            .collect();
        Ok(ToolOutput::new(json!({"status": "ok", "issues": rendered})))
    }
}

const PRACTICE_GAPS: &str = "graph_procedural_practice_gaps";
const EXTRANEOUS: &str = "graph_extraneous";
const ALIGNMENT_GAPS: &str = "graph_assessment_gaps";
const DISCOURSE_ORPHANS: &str = "graph_discourse_orphans";

pub(super) fn practice_gaps_meta() -> ToolPrototype {
    ToolPrototype {
        id:          PRACTICE_GAPS,
        description: "List procedural nodes that are not on a requires-path to any assessment \
                      targeting an LO.",
        schema:      schema_for_args::<NoArgs>(),
        parse:       parse_practice_gaps,
    }
}

fn parse_practice_gaps(raw: Value, _state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let _: NoArgs = serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
        tool:    PRACTICE_GAPS,
        message: err.to_string(),
    })?;
    Ok(Box::new(PracticeGapsTool))
}

struct PracticeGapsTool;

#[async_trait]
impl ToolInstance for PracticeGapsTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = GRAPH.read();
        let gaps = analysis::procedural_practice_gaps(graph.graph());
        let rendered: Vec<_> = gaps
            .into_iter()
            .map(|g| json!({ "slug": graph.graph()[g.node].slug }))
            .collect();
        Ok(ToolOutput::new(json!({"status": "ok", "gaps": rendered})))
    }
}

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExtraneousArgs {
    pub assessment_slug: String,
    pub lo_slug:         String,
    #[serde(default)]
    pub intended_slugs:  Vec<String>,
}

pub(super) fn extraneous_meta() -> ToolPrototype {
    ToolPrototype {
        id:          EXTRANEOUS,
        description: "Compute extraneous knowledge for an assessment-LO pair. Provide \
                      intended_slugs to declare intended knowledge; otherwise intended set is \
                      empty.",
        schema:      schema_for_args::<ExtraneousArgs>(),
        parse:       parse_extraneous,
    }
}

fn parse_extraneous(raw: Value, _state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: ExtraneousArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    EXTRANEOUS,
            message: err.to_string(),
        })?;
    Ok(Box::new(ExtraneousTool { args }))
}

struct ExtraneousTool {
    args: ExtraneousArgs,
}

#[async_trait]
impl ToolInstance for ExtraneousTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = GRAPH.read();
        let assessment = graph
            .node_by_slug(&self.args.assessment_slug)
            .map_err(|e| map_graph_err(e, EXTRANEOUS))?;
        let lo = graph
            .node_by_slug(&self.args.lo_slug)
            .map_err(|e| map_graph_err(e, EXTRANEOUS))?;

        // Build intended set from provided slugs (optional).
        let mut intended = std::collections::HashSet::new();
        for slug in &self.args.intended_slugs {
            if let Ok(id) = graph.node_by_slug(slug) {
                intended.insert(id);
            }
        }

        let report = analysis::extraneous_report(graph.graph(), assessment, lo, &intended);

        let extraneous_slugs: Vec<_> = report
            .extraneous_nodes
            .iter()
            .map(|n| graph.graph()[*n].slug.clone())
            .collect();

        // Declared construct_irrelevant_demands (as strings) on the assessment node.
        let declared_cid = match &graph.graph()[assessment].kind {
            crate::graph::NodeKind::Knowledge(k) => k.construct_irrelevant_demands.clone(),
            _ => Vec::new(),
        };

        Ok(ToolOutput::new(json!({
            "status": "ok",
            "extraneous_slugs": extraneous_slugs,
            "declared_construct_irrelevant_demands": declared_cid,
        })))
    }
}

pub(super) fn alignment_gaps_meta() -> ToolPrototype {
    ToolPrototype {
        id:          ALIGNMENT_GAPS,
        description: "Report LOs with no target assessments, assessments with no LO, and \
                      assessments unreachable from first principles.",
        schema:      schema_for_args::<NoArgs>(),
        parse:       parse_alignment_gaps,
    }
}

fn parse_alignment_gaps(raw: Value, _state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let _: NoArgs = serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
        tool:    ALIGNMENT_GAPS,
        message: err.to_string(),
    })?;
    Ok(Box::new(AlignmentGapsTool))
}

struct AlignmentGapsTool;

#[async_trait]
impl ToolInstance for AlignmentGapsTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = GRAPH.read();
        let los = analysis::lo_missing_target_assessments(graph.graph());
        let orphan = analysis::orphan_assessments(graph.graph());
        let unreachable = analysis::unreachable_assessments(graph.graph());
        Ok(ToolOutput::new(json!({
            "status": "ok",
            "los_missing_target_assessment": los.into_iter().map(|n| graph.graph()[n].slug.clone()).collect::<Vec<_>>(),
            "assessments_without_lo": orphan.into_iter().map(|n| graph.graph()[n].slug.clone()).collect::<Vec<_>>(),
            "assessments_unreachable": unreachable.into_iter().map(|n| graph.graph()[n].slug.clone()).collect::<Vec<_>>(),
        })))
    }
}

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DiscourseOrphansArgs {
    #[serde(default)]
    pub episode: Option<String>,
}

pub(super) fn discourse_orphans_meta() -> ToolPrototype {
    ToolPrototype {
        id:          DISCOURSE_ORPHANS,
        description: "List TeachingSteps with no precedes links in their episode. Optionally \
                      filter by episode.",
        schema:      schema_for_args::<DiscourseOrphansArgs>(),
        parse:       parse_discourse_orphans,
    }
}

fn parse_discourse_orphans(
    raw: Value,
    _state: &CallState,
) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: DiscourseOrphansArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    DISCOURSE_ORPHANS,
            message: err.to_string(),
        })?;
    Ok(Box::new(DiscourseOrphansTool { args }))
}

struct DiscourseOrphansTool {
    args: DiscourseOrphansArgs,
}

#[async_trait]
impl ToolInstance for DiscourseOrphansTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = GRAPH.read();
        let list = analysis::discourse_orphans(graph.graph(), self.args.episode.as_deref());
        let slugs: Vec<_> = list
            .into_iter()
            .map(|n| graph.graph()[n].slug.clone())
            .collect();
        Ok(ToolOutput::new(json!({"status": "ok", "orphans": slugs})))
    }
}

pub(super) fn borrow_ahead_meta() -> ToolPrototype {
    ToolPrototype {
        id:          BORROW_AHEAD,
        description: "Detect borrow-ahead uses within an episode.",
        schema:      schema_for_args::<BorrowAheadArgs>(),
        parse:       parse_borrow_ahead,
    }
}

fn parse_borrow_ahead(raw: Value, _state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: BorrowAheadArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    BORROW_AHEAD,
            message: err.to_string(),
        })?;
    Ok(Box::new(BorrowAheadTool { args }))
}

struct BorrowAheadTool {
    args: BorrowAheadArgs,
}

#[async_trait]
impl ToolInstance for BorrowAheadTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = GRAPH.read();
        let results = analysis::borrow_ahead(graph.graph(), &self.args.episode);
        let rendered: Vec<_> = results
            .into_iter()
            .map(|b| {
                json!({
                    "step_slug": graph.graph()[b.step].slug,
                    "target_slug": graph.graph()[b.target].slug,
                    "severity": b.severity,
                })
            })
            .collect();
        Ok(ToolOutput::new(json!({"status": "ok", "borrow_ahead": rendered})))
    }
}

// ---------- Registration ----------

pub fn graph_tool_prototypes() -> Vec<ToolPrototype> {
    vec![
        insert_knowledge_meta(),
        insert_teaching_meta(),
        add_requires_meta(),
        add_supports_meta(),
        add_assesses_meta(),
        add_precedes_meta(),
        add_anchors_meta(),
        get_node_meta(),
        dag_check_meta(),
        first_principles_meta(),
        lo_reach_meta(),
        coverage_meta(),
        example_gaps_meta(),
        keystone_meta(),
        borrow_ahead_meta(),
        save_snapshot_meta(),
        load_snapshot_meta(),
        fadeability_meta(),
        practice_gaps_meta(),
        alignment_gaps_meta(),
        discourse_orphans_meta(),
        neighbors_meta(),
        lo_alignment_meta(),
        extraneous_meta(),
    ]
}

// ---------- Persistence tools ----------

const SAVE_SNAPSHOT: &str = "graph_save_now";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SaveSnapshotArgs {
    #[serde(default)]
    pub path: Option<String>,
}

pub(super) fn save_snapshot_meta() -> ToolPrototype {
    ToolPrototype {
        id:          SAVE_SNAPSHOT,
        description: "Persist the in-memory graph to disk immediately. Path defaults to \
                      GRAPH_SNAPSHOT_PATH or graph_snapshot.json.",
        schema:      schema_for_args::<SaveSnapshotArgs>(),
        parse:       parse_save_snapshot,
    }
}

fn parse_save_snapshot(raw: Value, _state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: SaveSnapshotArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    SAVE_SNAPSHOT,
            message: err.to_string(),
        })?;
    Ok(Box::new(SaveSnapshotTool { args }))
}

struct SaveSnapshotTool {
    args: SaveSnapshotArgs,
}

#[async_trait]
impl ToolInstance for SaveSnapshotTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let path = self
            .args
            .path
            .clone()
            .or_else(|| env::var("GRAPH_SNAPSHOT_PATH").ok())
            .unwrap_or_else(|| "graph_snapshot.json".to_string());
        let graph_clone = { GRAPH.read().snapshot_graph() };
        crate::graph::persist::save_graph(&graph_clone, &path)
            .await
            .map_err(ToolExecutionError::from)?;
        Ok(ToolOutput::new(json!({"status": "ok", "path": path})))
    }
}

const LOAD_SNAPSHOT: &str = "graph_load_snapshot";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LoadSnapshotArgs {
    pub path: String,
}

pub(super) fn load_snapshot_meta() -> ToolPrototype {
    ToolPrototype {
        id:          LOAD_SNAPSHOT,
        description: "Load graph snapshot from disk, replacing in-memory graph.",
        schema:      schema_for_args::<LoadSnapshotArgs>(),
        parse:       parse_load_snapshot,
    }
}

fn parse_load_snapshot(raw: Value, _state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: LoadSnapshotArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    LOAD_SNAPSHOT,
            message: err.to_string(),
        })?;
    Ok(Box::new(LoadSnapshotTool { args }))
}

struct LoadSnapshotTool {
    args: LoadSnapshotArgs,
}

#[async_trait]
impl ToolInstance for LoadSnapshotTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let new_graph = crate::graph::persist::load_graph(&self.args.path)
            .await
            .map_err(ToolExecutionError::from)?;
        {
            let mut guard = GRAPH.write();
            guard.replace_graph(new_graph);
        }
        Ok(ToolOutput::new(json!({"status": "ok", "path": self.args.path})))
    }
}
