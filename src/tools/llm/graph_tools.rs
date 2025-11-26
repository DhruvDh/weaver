use std::{env, path::PathBuf};

use anyhow::anyhow;
use async_trait::async_trait;
use bon::Builder;
use kameo::{error::SendError, prelude::ActorRef};
use petgraph::visit::EdgeRef;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    analysis::{self},
    graph::{
        AnchorImpact, AssessesAttrs, CaseTag, CurriculumGraph, GraphError, IntroductionScope,
        KnowledgeNode, NodeId, NodeKind, RequiresAttrs, SupportsAttrs, TeachingPurpose,
        TeachingStepNode,
        manager::{
            AddAnchors, AddAssesses, AddPrecedes, AddRequires, AddSupports, GetGraph, GetNode,
            InsertKnowledge, InsertTeachingStep, LoadSnapshot, SaveSnapshot, UpdateKnowledge,
            UpdateTeachingStep,
        },
    },
    schema::types::{
        AssessmentScope, EvidenceLink, IntendedEffect, KnowledgeType, SourceRef, Strength,
        SupportKind,
    },
    tools::llm::{
        CallState, ToolExecutionError, ToolInputError, ToolInputResult, ToolInstance, ToolOutput,
        ToolPrototype, schema_for_args,
    },
};

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

fn find_node_by_slug(graph: &CurriculumGraph, slug: &str) -> Option<NodeId> {
    graph.node_indices().find(|&n| graph[n].slug == slug)
}

fn map_send_err<M>(err: SendError<M, GraphError>, tool: &'static str) -> ToolExecutionError {
    match err {
        SendError::HandlerError(e) => map_graph_err(e, tool),
        other => ToolExecutionError::Internal(anyhow!("{:?}", other)),
    }
}

fn map_send_err_inf<M>(err: SendError<M, std::convert::Infallible>) -> ToolExecutionError {
    ToolExecutionError::Internal(anyhow!("{:?}", err))
}

fn map_send_err_anyhow<M>(err: SendError<M, anyhow::Error>) -> ToolExecutionError {
    match err {
        SendError::HandlerError(e) => ToolExecutionError::Internal(e),
        other => ToolExecutionError::Internal(anyhow!("{:?}", other)),
    }
}

/// Generic, minimal boilerplate tool wrapper for simple graph commands that are
/// just an actor message + a JSON success payload.
type MsgReply<Msg> = <crate::graph::manager::GraphManager as kameo::message::Message<Msg>>::Reply;

struct GraphCommandTool<Args, Msg>
where
    Msg: Send + 'static,
    crate::graph::manager::GraphManager: kameo::message::Message<Msg>,
{
    args:    Args,
    graph:   ActorRef<crate::graph::manager::GraphManager>,
    build:   fn(&Args) -> Msg,
    map_ok:  fn(&Args, <MsgReply<Msg> as kameo::Reply>::Ok) -> Value,
    map_err: fn(SendError<Msg, <MsgReply<Msg> as kameo::Reply>::Error>) -> ToolExecutionError,
}

impl<Args, Msg> GraphCommandTool<Args, Msg>
where
    Msg: Send + 'static,
    crate::graph::manager::GraphManager: kameo::message::Message<Msg>,
{
    fn new(
        args: Args,
        graph: ActorRef<crate::graph::manager::GraphManager>,
        build: fn(&Args) -> Msg,
        map_ok: fn(&Args, <MsgReply<Msg> as kameo::Reply>::Ok) -> Value,
        map_err: fn(SendError<Msg, <MsgReply<Msg> as kameo::Reply>::Error>) -> ToolExecutionError,
    ) -> Self {
        Self {
            args,
            graph,
            build,
            map_ok,
            map_err,
        }
    }
}

#[async_trait]
impl<Args, Msg> ToolInstance for GraphCommandTool<Args, Msg>
where
    Args: Send + Sync + 'static,
    Msg: Send + 'static,
    crate::graph::manager::GraphManager: kameo::message::Message<Msg>,
{
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let msg = (self.build)(&self.args);
        let reply: <MsgReply<Msg> as kameo::Reply>::Ok =
            self.graph.ask(msg).await.map_err(|e| (self.map_err)(e))?;

        Ok(ToolOutput::new((self.map_ok)(&self.args, reply)))
    }
}

fn parse_graph_command<Args, Msg>(
    tool: &'static str,
    raw: Value,
    state: &CallState,
    build: fn(&Args) -> Msg,
    map_ok: fn(&Args, <MsgReply<Msg> as kameo::Reply>::Ok) -> Value,
    map_err: fn(SendError<Msg, <MsgReply<Msg> as kameo::Reply>::Error>) -> ToolExecutionError,
) -> ToolInputResult<Box<dyn ToolInstance>>
where
    Args: for<'de> Deserialize<'de> + JsonSchema + Clone + Send + Sync + 'static,
    Msg: Send + 'static,
    crate::graph::manager::GraphManager: kameo::message::Message<Msg>,
{
    let args: Args = serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
        tool,
        message: err.to_string(),
    })?;

    Ok(Box::new(GraphCommandTool::new(
        args,
        state.graph.clone(),
        build,
        map_ok,
        map_err,
    )))
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

pub(super) fn update_knowledge_meta() -> ToolPrototype {
    ToolPrototype {
        id:          UPDATE_KNOWLEDGE,
        description: "Update a knowledge/LO/assessment node (revalidates incident edges).",
        schema:      schema_for_args::<InsertKnowledgeArgs>(),
        parse:       parse_update_knowledge,
    }
}

fn build_insert_knowledge(args: &InsertKnowledgeArgs) -> InsertKnowledge {
    let payload = KnowledgeNode {
        title: args.title.clone(),
        statement: args.statement.clone(),
        knowledge_type: args.knowledge_type,
        source_refs: args.source_refs.clone(),
        confidence: args.confidence,
        rubric_criteria: args.rubric_criteria.clone(),
        construct_irrelevant_demands: args.construct_irrelevant_demands.clone(),
        grain_level: args.grain_level,
        intrinsic_load: args.intrinsic_load,
        introduction_scope: args
            .introduction_scope
            .unwrap_or(IntroductionScope::InCourse),
    };
    InsertKnowledge {
        slug: args.slug.clone(),
        payload,
        tags: args.tags.clone(),
    }
}

fn ok_node_id(_: &InsertKnowledgeArgs, id: NodeId) -> Value {
    json!({"status": "ok", "node_id": id.index()})
}

fn parse_insert_knowledge(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_graph_command::<InsertKnowledgeArgs, InsertKnowledge>(
        INSERT_KNOWLEDGE,
        raw,
        state,
        build_insert_knowledge,
        ok_node_id,
        |e| map_send_err(e, INSERT_KNOWLEDGE),
    )
}

fn parse_update_knowledge(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_graph_command::<InsertKnowledgeArgs, UpdateKnowledge>(
        UPDATE_KNOWLEDGE,
        raw,
        state,
        |args| UpdateKnowledge {
            slug:    args.slug.clone(),
            payload: KnowledgeNode {
                title: args.title.clone(),
                statement: args.statement.clone(),
                knowledge_type: args.knowledge_type,
                source_refs: args.source_refs.clone(),
                confidence: args.confidence,
                rubric_criteria: args.rubric_criteria.clone(),
                construct_irrelevant_demands: args.construct_irrelevant_demands.clone(),
                grain_level: args.grain_level,
                intrinsic_load: args.intrinsic_load,
                introduction_scope: args
                    .introduction_scope
                    .unwrap_or(IntroductionScope::InCourse),
            },
            tags:    args.tags.clone(),
        },
        ok_node_id,
        |e| map_send_err(e, UPDATE_KNOWLEDGE),
    )
}

// ---------- Insert teaching step ----------

const INSERT_TEACHING: &str = "graph_insert_teaching_step";
const UPDATE_KNOWLEDGE: &str = "graph_update_knowledge";
const UPDATE_TEACHING: &str = "graph_update_teaching_step";

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

pub(super) fn update_teaching_meta() -> ToolPrototype {
    ToolPrototype {
        id:          UPDATE_TEACHING,
        description: "Update a TeachingStep node (revalidates incident edges).",
        schema:      schema_for_args::<InsertTeachingArgs>(),
        parse:       parse_update_teaching,
    }
}

fn build_insert_teaching(args: &InsertTeachingArgs) -> InsertTeachingStep {
    let payload = TeachingStepNode {
        title:       args.title.clone(),
        statement:   args.statement.clone(),
        purpose:     args.purpose,
        method_tags: args.method_tags.clone(),
        episode:     args.episode.clone(),
        source_refs: args.source_refs.clone(),
    };
    InsertTeachingStep {
        slug: args.slug.clone(),
        payload,
        tags: args.tags.clone(),
    }
}

fn parse_insert_teaching(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_graph_command::<InsertTeachingArgs, InsertTeachingStep>(
        INSERT_TEACHING,
        raw,
        state,
        build_insert_teaching,
        |_, id| json!({"status": "ok", "node_id": id.index()}),
        |e| map_send_err(e, INSERT_TEACHING),
    )
}

fn parse_update_teaching(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_graph_command::<InsertTeachingArgs, UpdateTeachingStep>(
        UPDATE_TEACHING,
        raw,
        state,
        |args| UpdateTeachingStep {
            slug:    args.slug.clone(),
            payload: TeachingStepNode {
                title:       args.title.clone(),
                statement:   args.statement.clone(),
                purpose:     args.purpose,
                method_tags: args.method_tags.clone(),
                episode:     args.episode.clone(),
                source_refs: args.source_refs.clone(),
            },
            tags:    args.tags.clone(),
        },
        |_, id| json!({ "status": "ok", "node_id": id.index() }),
        |e| map_send_err(e, UPDATE_TEACHING),
    )
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

fn build_add_requires(args: &AddRequiresArgs) -> AddRequires {
    AddRequires {
        from:       args.from_slug.clone(),
        to:         args.to_slug.clone(),
        attrs:      RequiresAttrs {
            strength:      args.strength,
            rationale:     args.rationale.clone(),
            evidence_refs: args.evidence_refs.clone(),
        },
        confidence: args.confidence,
    }
}

fn parse_add_requires(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_graph_command::<AddRequiresArgs, AddRequires>(
        ADD_REQUIRES,
        raw,
        state,
        build_add_requires,
        |_, _| json!({"status": "ok"}),
        |e| map_send_err(e, ADD_REQUIRES),
    )
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

fn build_add_supports(args: &AddSupportsArgs) -> AddSupports {
    AddSupports {
        from:       args.from_slug.clone(),
        to:         args.to_slug.clone(),
        attrs:      SupportsAttrs {
            support_kind:    args.support_kind,
            intended_effect: args.intended_effect,
            case_tag:        args.case_tag,
            coverage_tags:   args.coverage_tags.clone(),
            evidence_refs:   args.evidence_refs.clone(),
        },
        confidence: args.confidence,
    }
}

fn parse_add_supports(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_graph_command::<AddSupportsArgs, AddSupports>(
        ADD_SUPPORTS,
        raw,
        state,
        build_add_supports,
        |_, _| json!({"status": "ok"}),
        |e| map_send_err(e, ADD_SUPPORTS),
    )
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

fn build_add_assesses(args: &AddAssessesArgs) -> AddAssesses {
    AddAssesses {
        from:       args.from_slug.clone(),
        to:         args.to_slug.clone(),
        attrs:      AssessesAttrs {
            evidence_link: EvidenceLink {
                claim:                args.claim.clone(),
                observation_features: args.observation_features.clone(),
                scope:                args.scope,
            },
        },
        confidence: args.confidence,
    }
}

fn parse_add_assesses(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_graph_command::<AddAssessesArgs, AddAssesses>(
        ADD_ASSESSES,
        raw,
        state,
        build_add_assesses,
        |_, _| json!({"status": "ok"}),
        |e| map_send_err(e, ADD_ASSESSES),
    )
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

fn build_add_precedes(args: &AddPrecedesArgs) -> AddPrecedes {
    AddPrecedes {
        from:       args.from_slug.clone(),
        to:         args.to_slug.clone(),
        episode:    args.episode.clone(),
        confidence: args.confidence,
    }
}

fn parse_add_precedes(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_graph_command::<AddPrecedesArgs, AddPrecedes>(
        ADD_PRECEDES,
        raw,
        state,
        build_add_precedes,
        |_, _| json!({"status": "ok"}),
        |e| map_send_err(e, ADD_PRECEDES),
    )
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

fn build_add_anchors(args: &AddAnchorsArgs) -> AddAnchors {
    AddAnchors {
        from:       args.from_slug.clone(),
        to:         args.to_slug.clone(),
        impact:     args.impact,
        confidence: args.confidence,
    }
}

fn parse_add_anchors(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_graph_command::<AddAnchorsArgs, AddAnchors>(
        ADD_ANCHORS,
        raw,
        state,
        build_add_anchors,
        |_, _| json!({"status": "ok"}),
        |e| map_send_err(e, ADD_ANCHORS),
    )
}

// ---------- Inspection ----------

const GRAPH_NEIGHBORS: &str = "graph_neighbors";
const RENAME_NODE: &str = "graph_rename_node";
const REMOVE_NODE: &str = "graph_remove_node";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NeighborsArgs {
    pub slug:      String,
    #[serde(default)]
    pub edge_kind: Option<String>, // requires, supports, assesses, precedes, anchors
    #[serde(default)]
    pub direction: Option<String>, // outgoing, incoming, both
}

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RenameNodeArgs {
    pub old_slug: String,
    pub new_slug: String,
}

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RemoveNodeArgs {
    pub slug: String,
}

pub(super) fn neighbors_meta() -> ToolPrototype {
    ToolPrototype {
        id:          GRAPH_NEIGHBORS,
        description: "List neighbors of a node filtered by edge kind and direction.",
        schema:      schema_for_args::<NeighborsArgs>(),
        parse:       parse_neighbors,
    }
}

fn parse_neighbors(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: NeighborsArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    GRAPH_NEIGHBORS,
            message: err.to_string(),
        })?;
    Ok(Box::new(NeighborsTool {
        args,
        graph: state.graph.clone(),
    }))
}

struct NeighborsTool {
    args:  NeighborsArgs,
    graph: ActorRef<crate::graph::manager::GraphManager>,
}

#[async_trait]
impl ToolInstance for NeighborsTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let neighbors = self
            .graph
            .ask(crate::graph::manager::Neighbors {
                slug:      self.args.slug.clone(),
                edge_kind: self.args.edge_kind.clone(),
                direction: self.args.direction.clone(),
            })
            .await
            .map_err(|e| map_send_err(e, GRAPH_NEIGHBORS))?;

        Ok(ToolOutput::new(json!({"status": "ok", "neighbors": neighbors})))
    }
}

pub(super) fn rename_node_meta() -> ToolPrototype {
    ToolPrototype {
        id:          RENAME_NODE,
        description: "Rename a node slug and update dependent assesses claims.",
        schema:      schema_for_args::<RenameNodeArgs>(),
        parse:       parse_rename_node,
    }
}

fn parse_rename_node(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_graph_command::<RenameNodeArgs, crate::graph::manager::RenameNode>(
        RENAME_NODE,
        raw,
        state,
        |args| crate::graph::manager::RenameNode {
            old_slug: args.old_slug.clone(),
            new_slug: args.new_slug.clone(),
        },
        |_, _| json!({"status": "ok"}),
        |e| map_send_err(e, RENAME_NODE),
    )
}

pub(super) fn remove_node_meta() -> ToolPrototype {
    ToolPrototype {
        id:          REMOVE_NODE,
        description: "Delete a node and its incident edges.",
        schema:      schema_for_args::<RemoveNodeArgs>(),
        parse:       parse_remove_node,
    }
}

fn parse_remove_node(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_graph_command::<RemoveNodeArgs, crate::graph::manager::RemoveNode>(
        REMOVE_NODE,
        raw,
        state,
        |args| crate::graph::manager::RemoveNode {
            slug: args.slug.clone(),
        },
        |_, _| json!({"status": "ok"}),
        |e| map_send_err(e, REMOVE_NODE),
    )
}

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

fn parse_get_node(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: GetNodeArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    GET_NODE,
            message: err.to_string(),
        })?;
    Ok(Box::new(GetNodeTool {
        args,
        graph: state.graph.clone(),
    }))
}

struct GetNodeTool {
    args:  GetNodeArgs,
    graph: ActorRef<crate::graph::manager::GraphManager>,
}

#[async_trait]
impl ToolInstance for GetNodeTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let payload = self
            .graph
            .ask(GetNode {
                slug: self.args.slug.clone(),
            })
            .await
            .map_err(|e| map_send_err(e, GET_NODE))?;
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
        let graph = self.graph.ask(GetGraph).await.map_err(map_send_err_inf)?;
        let dag = analysis::requires_is_dag(&graph);
        let topo = analysis::requires_toposort(&graph).ok();
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

struct FirstPrinciplesTool {
    graph: ActorRef<crate::graph::manager::GraphManager>,
}

fn parse_first_principles(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let _: NoArgs = serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
        tool:    FIRST_PRINCIPLES,
        message: err.to_string(),
    })?;
    Ok(Box::new(FirstPrinciplesTool {
        graph: state.graph.clone(),
    }))
}

#[async_trait]
impl ToolInstance for FirstPrinciplesTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = self.graph.ask(GetGraph).await.map_err(map_send_err_inf)?;
        let fps = analysis::first_principles(&graph);
        let slugs: Vec<_> = fps.iter().map(|id| graph[*id].slug.clone()).collect();
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

fn parse_lo_reach(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: LoReachArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    LO_REACH,
            message: err.to_string(),
        })?;
    Ok(Box::new(LoReachTool {
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
        let graph = self.graph.ask(GetGraph).await.map_err(map_send_err_inf)?;
        let lo = find_node_by_slug(&graph, &self.args.lo_slug).ok_or_else(|| {
            ToolExecutionError::Input(ToolInputError::InvalidPath {
                tool:    LO_REACH,
                path:    self.args.lo_slug.clone(),
                message: "slug not found".into(),
            })
        })?;
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

fn parse_lo_coverage(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: LoReachArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    COVERAGE,
            message: err.to_string(),
        })?;
    Ok(Box::new(LoCoverageTool {
        args,
        graph: state.graph.clone(),
    }))
}

struct LoCoverageTool {
    args:  LoReachArgs,
    graph: ActorRef<crate::graph::manager::GraphManager>,
}

#[async_trait]
impl ToolInstance for LoCoverageTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = self.graph.ask(GetGraph).await.map_err(map_send_err_inf)?;
        let lo = find_node_by_slug(&graph, &self.args.lo_slug).ok_or_else(|| {
            ToolExecutionError::Input(ToolInputError::InvalidPath {
                tool:    COVERAGE,
                path:    self.args.lo_slug.clone(),
                message: "slug not found".into(),
            })
        })?;
        let report = analysis::coverage_report(&graph, lo);
        Ok(ToolOutput::new(json!({
            "status": "ok",
            "covered": report.covered_criteria,
            "missing": report.missing_criteria,
            "unused_observation_features": report.unused_observation_features,
        })))
    }
}

fn parse_lo_alignment(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: LoReachArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    LO_ALIGNMENT,
            message: err.to_string(),
        })?;
    Ok(Box::new(LoAlignmentTool {
        args,
        graph: state.graph.clone(),
    }))
}

struct LoAlignmentTool {
    args:  LoReachArgs,
    graph: ActorRef<crate::graph::manager::GraphManager>,
}

#[async_trait]
impl ToolInstance for LoAlignmentTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = self.graph.ask(GetGraph).await.map_err(map_send_err_inf)?;
        let lo = find_node_by_slug(&graph, &self.args.lo_slug).ok_or_else(|| {
            ToolExecutionError::Input(ToolInputError::InvalidPath {
                tool:    LO_ALIGNMENT,
                path:    self.args.lo_slug.clone(),
                message: "slug not found".into(),
            })
        })?;

        // Reachability
        let fps = analysis::first_principles(&graph);
        let reach = analysis::lo_reachability(&graph, lo, &fps);
        let assessments = reach
            .assessments
            .into_iter()
            .map(|a| {
                json!({
                    "assessment_slug": graph[a.assessment].slug,
                    "reachable_from_first_principle": a.reachable_from_first_principle,
                })
            })
            .collect::<Vec<_>>();

        // Coverage
        let coverage = analysis::coverage_report(&graph, lo);

        // Anchors with impact=target
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

struct ExampleGapsTool {
    graph: ActorRef<crate::graph::manager::GraphManager>,
}

fn parse_example_gaps(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let _: NoArgs = serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
        tool:    EXAMPLE_GAPS,
        message: err.to_string(),
    })?;
    Ok(Box::new(ExampleGapsTool {
        graph: state.graph.clone(),
    }))
}

#[async_trait]
impl ToolInstance for ExampleGapsTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = self.graph.ask(GetGraph).await.map_err(map_send_err_inf)?;
        let gaps = analysis::example_gaps(&graph);
        let rendered: Vec<_> = gaps
            .into_iter()
            .map(|gap| {
                json!({
                    "slug": graph[gap.node].slug,
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

struct KeystoneTool {
    graph: ActorRef<crate::graph::manager::GraphManager>,
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

#[async_trait]
impl ToolInstance for KeystoneTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = self.graph.ask(GetGraph).await.map_err(map_send_err_inf)?;
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

fn parse_fadeability(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let _: NoArgs = serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
        tool:    FADEABILITY,
        message: err.to_string(),
    })?;
    Ok(Box::new(FadeabilityTool {
        graph: state.graph.clone(),
    }))
}

struct FadeabilityTool {
    graph: ActorRef<crate::graph::manager::GraphManager>,
}

#[async_trait]
impl ToolInstance for FadeabilityTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = self.graph.ask(GetGraph).await.map_err(map_send_err_inf)?;
        let issues = analysis::fadeability_issues(&graph);
        let rendered: Vec<_> = issues
            .into_iter()
            .map(|i| json!({ "assessment_slug": graph[i.assessment].slug }))
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

fn parse_practice_gaps(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let _: NoArgs = serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
        tool:    PRACTICE_GAPS,
        message: err.to_string(),
    })?;
    Ok(Box::new(PracticeGapsTool {
        graph: state.graph.clone(),
    }))
}

struct PracticeGapsTool {
    graph: ActorRef<crate::graph::manager::GraphManager>,
}

#[async_trait]
impl ToolInstance for PracticeGapsTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = self.graph.ask(GetGraph).await.map_err(map_send_err_inf)?;
        let gaps = analysis::procedural_practice_gaps(&graph);
        let rendered: Vec<_> = gaps
            .into_iter()
            .map(|g| json!({ "slug": graph[g.node].slug }))
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

fn parse_extraneous(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: ExtraneousArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    EXTRANEOUS,
            message: err.to_string(),
        })?;
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
        let graph = self.graph.ask(GetGraph).await.map_err(map_send_err_inf)?;
        let assessment =
            find_node_by_slug(&graph, &self.args.assessment_slug).ok_or_else(|| {
                ToolExecutionError::Input(ToolInputError::InvalidPath {
                    tool:    EXTRANEOUS,
                    path:    self.args.assessment_slug.clone(),
                    message: "slug not found".into(),
                })
            })?;
        let lo = find_node_by_slug(&graph, &self.args.lo_slug).ok_or_else(|| {
            ToolExecutionError::Input(ToolInputError::InvalidPath {
                tool:    EXTRANEOUS,
                path:    self.args.lo_slug.clone(),
                message: "slug not found".into(),
            })
        })?;

        // Build intended set from provided slugs (optional).
        let mut intended = std::collections::HashSet::new();
        for slug in &self.args.intended_slugs {
            if let Some(id) = find_node_by_slug(&graph, slug) {
                intended.insert(id);
            }
        }

        let report = analysis::extraneous_report(&graph, assessment, lo, &intended);

        let extraneous_slugs: Vec<_> = report
            .extraneous_nodes
            .iter()
            .map(|n| graph[*n].slug.clone())
            .collect();

        // Declared construct_irrelevant_demands (as strings) on the assessment node.
        let declared_cid = match &graph[assessment].kind {
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

fn parse_alignment_gaps(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let _: NoArgs = serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
        tool:    ALIGNMENT_GAPS,
        message: err.to_string(),
    })?;
    Ok(Box::new(AlignmentGapsTool {
        graph: state.graph.clone(),
    }))
}

struct AlignmentGapsTool {
    graph: ActorRef<crate::graph::manager::GraphManager>,
}

#[async_trait]
impl ToolInstance for AlignmentGapsTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let graph = self.graph.ask(GetGraph).await.map_err(map_send_err_inf)?;
        let los = analysis::lo_missing_target_assessments(&graph);
        let orphan = analysis::orphan_assessments(&graph);
        let unreachable = analysis::unreachable_assessments(&graph);
        Ok(ToolOutput::new(json!({
            "status": "ok",
            "los_missing_target_assessment": los.into_iter().map(|n| graph[n].slug.clone()).collect::<Vec<_>>(),
            "assessments_without_lo": orphan.into_iter().map(|n| graph[n].slug.clone()).collect::<Vec<_>>(),
            "assessments_unreachable": unreachable.into_iter().map(|n| graph[n].slug.clone()).collect::<Vec<_>>(),
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
        let graph = self.graph.ask(GetGraph).await.map_err(map_send_err_inf)?;
        let list = analysis::discourse_orphans(&graph, self.args.episode.as_deref());
        let slugs: Vec<_> = list.into_iter().map(|n| graph[n].slug.clone()).collect();
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

fn parse_borrow_ahead(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let args: BorrowAheadArgs =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    BORROW_AHEAD,
            message: err.to_string(),
        })?;
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
        let graph = self.graph.ask(GetGraph).await.map_err(map_send_err_inf)?;
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
        Ok(ToolOutput::new(json!({"status": "ok", "borrow_ahead": rendered})))
    }
}

// ---------- Registration ----------

pub fn graph_tool_prototypes() -> Vec<ToolPrototype> {
    vec![
        insert_knowledge_meta(),
        update_knowledge_meta(),
        insert_teaching_meta(),
        update_teaching_meta(),
        add_requires_meta(),
        add_supports_meta(),
        add_assesses_meta(),
        add_precedes_meta(),
        add_anchors_meta(),
        rename_node_meta(),
        remove_node_meta(),
        neighbors_meta(),
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

fn build_save_snapshot(args: &SaveSnapshotArgs) -> SaveSnapshot {
    let path_str = args
        .path
        .clone()
        .or_else(|| env::var("GRAPH_SNAPSHOT_PATH").ok())
        .unwrap_or_else(|| "graph_snapshot.json".to_string());
    SaveSnapshot {
        path: PathBuf::from(path_str),
    }
}

fn ok_save_snapshot(args: &SaveSnapshotArgs, _: ()) -> Value {
    let path = args
        .path
        .clone()
        .or_else(|| env::var("GRAPH_SNAPSHOT_PATH").ok())
        .unwrap_or_else(|| "graph_snapshot.json".to_string());
    json!({"status": "ok", "path": path})
}

fn parse_save_snapshot(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_graph_command::<SaveSnapshotArgs, SaveSnapshot>(
        SAVE_SNAPSHOT,
        raw,
        state,
        build_save_snapshot,
        ok_save_snapshot,
        map_send_err_anyhow,
    )
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

fn build_load_snapshot(args: &LoadSnapshotArgs) -> LoadSnapshot {
    LoadSnapshot {
        path: PathBuf::from(args.path.clone()),
    }
}

fn parse_load_snapshot(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    parse_graph_command::<LoadSnapshotArgs, LoadSnapshot>(
        LOAD_SNAPSHOT,
        raw,
        state,
        build_load_snapshot,
        |args, _| json!({"status": "ok", "path": args.path}),
        map_send_err_anyhow,
    )
}
