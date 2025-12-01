use std::{pin::Pin, sync::Arc};

use bon::Builder;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::info;

use super::common::{MaybeApply, default_confidence, map_send_err};
use crate::{
    graph::{
        AnchorImpact, AnchorsAttrs, AssessesAttrs, CaseTag, IntroductionScope, KnowledgeNode,
        PrecedesAttrs, RequiresAttrs, SupportsAttrs, TeachingPurpose, TeachingStepNode,
        commands::{
            AddAnchors, AddAssesses, AddPrecedes, AddRequires, AddSupports, RemoveNode, RenameNode,
            UpdateKnowledge, UpdateTeachingStep,
        },
    },
    schema::types::{
        AssessmentScope, EvidenceLink, IntendedEffect, KnowledgeType, SourceRef, Strength,
        SupportKind,
    },
    tools::llm::{
        CallState, ToolInputResult, ToolPrototype,
        common::{
            AnyKnowledge, AssessmentItem, LearningOutcome, Slug, TeachingStep, resolve_typed,
        },
        require_string,
    },
};

fn command_ok(tool: &'static str, extra: serde_json::Value) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    map.insert("type".into(), json!("graph_command"));
    map.insert("tool".into(), json!(tool));
    map.insert("status".into(), json!("ok"));
    if let serde_json::Value::Object(obj) = extra {
        for (k, v) in obj {
            map.insert(k, v);
        }
    }
    serde_json::Value::Object(map)
}

// ---------- Insert / update knowledge ----------

const INSERT_KNOWLEDGE: &str = "graph_insert_knowledge";
const UPDATE_KNOWLEDGE: &str = "graph_update_knowledge";

#[derive(Debug, Clone, Builder, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InsertKnowledgeArgs {
    #[schemars(
        description = "Stable slug for this knowledge/LO/assessment node (must be unique)."
    )]
    #[builder(with = |value: String| -> ToolInputResult<_> {
        require_string(value, INSERT_KNOWLEDGE, "slug")
    })]
    pub slug: String,
    #[schemars(description = "Short title for the knowledge/LO/assessment.")]
    #[builder(with = |value: String| -> ToolInputResult<_> {
        require_string(value, INSERT_KNOWLEDGE, "title")
    })]
    pub title: String,
    #[schemars(
        description = "Full statement of the knowledge item, learning outcome, or assessment \
                       target."
    )]
    #[builder(with = |value: String| -> ToolInputResult<_> {
        require_string(value, INSERT_KNOWLEDGE, "statement")
    })]
    pub statement: String,
    #[schemars(
        description = "KnowledgeType: factual, conceptual, procedural, metacognitive, \
                       learning_outcome, or assessment_item."
    )]
    pub knowledge_type: KnowledgeType,
    #[schemars(description = "Rubric criteria (expected for learning_outcome nodes).")]
    #[serde(default)]
    pub rubric_criteria: Vec<String>,
    #[schemars(description = "Known construct-irrelevant demands (assessment_item nodes).")]
    #[serde(default)]
    pub construct_irrelevant_demands: Vec<String>,
    #[schemars(
        description = "Source spans in the repository supporting this node (path + line range, \
                       pinned to repo revision)."
    )]
    #[serde(default)]
    pub source_refs: Vec<SourceRef>,
    #[schemars(
        description = "Confidence in this node (0.0-1.0). Defaults to 1.0 when omitted or 0.0)."
    )]
    #[serde(default)]
    pub confidence: f32,
    #[serde(default)]
    #[schemars(
        description = "Grain level of this knowledge: macro (coarse), mid (default), or micro \
                       (fine)."
    )]
    pub grain_level: Option<crate::graph::GrainLevel>,
    #[serde(default)]
    #[schemars(description = "Intrinsic cognitive load: low, medium, or high.")]
    pub intrinsic_load: Option<crate::graph::IntrinsicLoad>,
    #[serde(default)]
    #[schemars(
        description = "Where this knowledge is introduced: in_course (default), prior, or \
                       external."
    )]
    pub introduction_scope: Option<IntroductionScope>,
    #[serde(default)]
    #[schemars(
        description = "Free-form tags to group/filter knowledge nodes (e.g., principle, \
                       misconception, keystone)."
    )]
    pub tags: Vec<String>,
    #[serde(default)]
    #[builder(default = false)]
    #[schemars(
        description = "Set apply=true to perform the mutation; default false returns a preview."
    )]
    pub apply: bool,
}

impl MaybeApply for InsertKnowledgeArgs {
    fn apply_flag(&self) -> bool {
        self.apply
    }
}

fn build_insert_knowledge(args: &InsertKnowledgeArgs) -> crate::graph::commands::InsertKnowledge {
    crate::graph::commands::InsertKnowledge {
        slug:    args.slug.clone(),
        payload: KnowledgeNode {
            title: args.title.clone(),
            statement: args.statement.clone(),
            knowledge_type: args.knowledge_type,
            source_refs: args.source_refs.clone(),
            confidence: if args.confidence == 0.0 {
                1.0
            } else {
                args.confidence
            },
            rubric_criteria: args.rubric_criteria.clone(),
            construct_irrelevant_demands: args.construct_irrelevant_demands.clone(),
            grain_level: args.grain_level,
            intrinsic_load: args.intrinsic_load,
            introduction_scope: args
                .introduction_scope
                .unwrap_or(crate::graph::IntroductionScope::InCourse),
        },
        tags:    args.tags.clone(),
    }
}

crate::graph_action_tool!(
    insert_knowledge_meta,
    id: INSERT_KNOWLEDGE,
    description: "Insert a new mid-grain Knowledge node \
                  (factual/conceptual/procedural/metacognitive/LO/assessment). Use for a single \
                  assessable idea (Assessable Atom) with stable slug, statement, Bloom-based \
                  knowledge_type, grain level, introduction scope, rubric/construct-irrelevant \
                  fields, and source_refs.",
    args: InsertKnowledgeArgs,
    prepare: |raw| super::common::parse_args_with_builder(
        INSERT_KNOWLEDGE,
        raw,
        |mut input: InsertKnowledgeArgs| {
            input.slug = require_string(input.slug, INSERT_KNOWLEDGE, "slug")?;
            input.title = require_string(input.title, INSERT_KNOWLEDGE, "title")?;
            input.statement = require_string(input.statement, INSERT_KNOWLEDGE, "statement")?;
            Ok(input)
        },
    ),
    build: |args: &InsertKnowledgeArgs| build_insert_knowledge(args),
    ok: |args: &InsertKnowledgeArgs, _| {
        info!(tool = INSERT_KNOWLEDGE, slug = %args.slug, "graph insert knowledge");
        command_ok(INSERT_KNOWLEDGE, json!({"slug": args.slug}))
    },
    map_err: |e| map_send_err(e, INSERT_KNOWLEDGE)
);

crate::graph_action_tool!(
    update_knowledge_meta,
    id: UPDATE_KNOWLEDGE,
    description: "Update an existing Knowledge node in place (statement/metadata/tags) without \
                  changing its identity. Use to refine wording, rubric metadata, \
                  load/grain/scope—not for splitting/merging concepts.",
    args: InsertKnowledgeArgs,
    prepare: |raw| super::common::parse_args_with_builder(
        UPDATE_KNOWLEDGE,
        raw,
        |mut input: InsertKnowledgeArgs| {
            input.slug = require_string(input.slug, UPDATE_KNOWLEDGE, "slug")?;
            input.title = require_string(input.title, UPDATE_KNOWLEDGE, "title")?;
            input.statement = require_string(input.statement, UPDATE_KNOWLEDGE, "statement")?;
            Ok(input)
        },
    ),
    build: |args: &InsertKnowledgeArgs| UpdateKnowledge {
        slug:    args.slug.clone(),
        payload: build_insert_knowledge(args).payload,
        tags:    args.tags.clone(),
    },
    ok: |args: &InsertKnowledgeArgs, _| {
        info!(tool = UPDATE_KNOWLEDGE, slug = %args.slug, "graph update knowledge");
        command_ok(UPDATE_KNOWLEDGE, json!({"slug": args.slug}))
    },
    map_err: |e| map_send_err(e, UPDATE_KNOWLEDGE)
);

// ---------- Insert / update teaching step ----------

const INSERT_TEACHING: &str = "graph_insert_teaching_step";
const UPDATE_TEACHING: &str = "graph_update_teaching_step";

#[derive(Debug, Clone, Builder, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InsertTeachingArgs {
    #[schemars(description = "Stable slug for this teaching step.")]
    #[builder(with = |value: String| -> ToolInputResult<_> {
        require_string(value, INSERT_TEACHING, "slug")
    })]
    pub slug:        String,
    #[schemars(description = "Short title for the teaching step.")]
    #[builder(with = |value: String| -> ToolInputResult<_> {
        require_string(value, INSERT_TEACHING, "title")
    })]
    pub title:       String,
    #[schemars(description = "Concise statement/summary of the step.")]
    #[builder(with = |value: String| -> ToolInputResult<_> {
        require_string(value, INSERT_TEACHING, "statement")
    })]
    pub statement:   String,
    #[schemars(
        description = "Purpose of the step: setup (prepare/motivate), idea (introduce concept), \
                       use (apply/practice), or consolidate (refine/summarize)."
    )]
    pub purpose:     TeachingPurpose,
    #[serde(default)]
    #[schemars(
        description = "Method tags describing pedagogy: e.g., worked-example, naive-first, \
                       analogy, retrieval-practice, socratic-question, reflection."
    )]
    pub method_tags: Vec<String>,
    #[schemars(
        description = "Episode identifier; precedes edges for this step must stay within this \
                       episode."
    )]
    #[builder(with = |value: String| -> ToolInputResult<_> {
        require_string(value, INSERT_TEACHING, "episode")
    })]
    pub episode:     String,
    #[serde(default)]
    #[schemars(description = "Source spans in the repository that define this teaching step.")]
    pub source_refs: Vec<SourceRef>,
    #[serde(default)]
    #[schemars(description = "Free-form tags to group/filter teaching steps.")]
    pub tags:        Vec<String>,
    #[serde(default)]
    #[schemars(description = "Rationale when a step intentionally has no anchors.")]
    pub rationale:   Option<String>,
    #[serde(default)]
    #[builder(default = false)]
    #[schemars(
        description = "Set apply=true to perform the mutation; default false returns a preview."
    )]
    pub apply:       bool,
}

impl MaybeApply for InsertTeachingArgs {
    fn apply_flag(&self) -> bool {
        self.apply
    }
}

fn build_insert_teaching(args: &InsertTeachingArgs) -> crate::graph::commands::InsertTeachingStep {
    crate::graph::commands::InsertTeachingStep {
        slug:    args.slug.clone(),
        payload: TeachingStepNode {
            title:       args.title.clone(),
            statement:   args.statement.clone(),
            purpose:     args.purpose,
            method_tags: args.method_tags.clone(),
            episode:     args.episode.clone(),
            source_refs: args.source_refs.clone(),
            rationale:   args.rationale.clone(),
        },
        tags:    args.tags.clone(),
    }
}

crate::graph_action_tool!(
    insert_teaching_meta,
    id: INSERT_TEACHING,
    description: "Insert a TeachingStep in the discourse layer for a specific episode. A \
                  TeachingStep is an atomic narrative step with purpose \
                  (setup/idea/use/consolidate) and method_tags that will be ordered by precedes \
                  and anchored to knowledge/LOs/assessments.",
    args: InsertTeachingArgs,
    prepare: |raw| super::common::parse_args_with_builder(
        INSERT_TEACHING,
        raw,
        |mut input: InsertTeachingArgs| {
            input.slug = require_string(input.slug, INSERT_TEACHING, "slug")?;
            input.title = require_string(input.title, INSERT_TEACHING, "title")?;
            input.statement = require_string(input.statement, INSERT_TEACHING, "statement")?;
            input.episode = require_string(input.episode, INSERT_TEACHING, "episode")?;
            Ok(input)
        },
    ),
    build: |args: &InsertTeachingArgs| build_insert_teaching(args),
    ok: |args: &InsertTeachingArgs, _| {
        info!(tool = INSERT_TEACHING, slug = %args.slug, "graph insert teaching_step");
        command_ok(INSERT_TEACHING, json!({"slug": args.slug}))
    },
    map_err: |e| map_send_err(e, INSERT_TEACHING)
);

crate::graph_action_tool!(
    update_teaching_meta,
    id: UPDATE_TEACHING,
    description: "Update an existing TeachingStep (statement, purpose, method_tags, episode) \
                  while preserving its identity and discourse links.",
    args: InsertTeachingArgs,
    prepare: |raw| super::common::parse_args_with_builder(
        UPDATE_TEACHING,
        raw,
        |mut input: InsertTeachingArgs| {
            input.slug = require_string(input.slug, UPDATE_TEACHING, "slug")?;
            input.title = require_string(input.title, UPDATE_TEACHING, "title")?;
            input.statement = require_string(input.statement, UPDATE_TEACHING, "statement")?;
            input.episode = require_string(input.episode, UPDATE_TEACHING, "episode")?;
            Ok(input)
        },
    ),
    build: |args: &InsertTeachingArgs| UpdateTeachingStep {
        slug:    args.slug.clone(),
        payload: build_insert_teaching(args).payload,
        tags:    args.tags.clone(),
    },
    ok: |args: &InsertTeachingArgs, _| {
        info!(tool = UPDATE_TEACHING, slug = %args.slug, "graph update teaching_step");
        command_ok(UPDATE_TEACHING, json!({"slug": args.slug}))
    },
    map_err: |e| map_send_err(e, UPDATE_TEACHING)
);

// ---------- Edge tools ----------

const ADD_REQUIRES: &str = "graph_add_requires";
const ADD_SUPPORTS: &str = "graph_add_supports";
const ADD_ASSESSES: &str = "graph_add_assesses";
const ADD_PRECEDES: &str = "graph_add_precedes";
const ADD_ANCHORS: &str = "graph_add_anchors";

#[derive(Debug, Clone, Builder, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AddRequiresArgs {
    #[schemars(description = "Prerequisite knowledge slug (source of the requires edge).")]
    #[builder(with = |v: String| -> ToolInputResult<_> {
        require_string(v, ADD_REQUIRES, "from_slug")
    })]
    pub from_slug:     String,
    #[schemars(description = "Dependent knowledge/assessment slug that requires the source.")]
    #[builder(with = |v: String| -> ToolInputResult<_> {
        require_string(v, ADD_REQUIRES, "to_slug")
    })]
    pub to_slug:       String,
    #[schemars(description = "Strength of dependency: necessary | strong | helpful.")]
    pub strength:      Strength,
    #[schemars(description = "Short rationale explaining the prerequisite link.")]
    #[builder(with = |v: String| -> ToolInputResult<_> {
        require_string(v, ADD_REQUIRES, "rationale")
    })]
    pub rationale:     String,
    #[serde(default)]
    #[schemars(
        description = "Optional source_refs defending this prerequisite (e.g., text anchors)."
    )]
    pub evidence_refs: Vec<SourceRef>,
    #[serde(default = "default_confidence")]
    pub confidence:    f32,
    #[serde(default)]
    #[builder(default = false)]
    pub apply:         bool,
}

impl MaybeApply for AddRequiresArgs {
    fn apply_flag(&self) -> bool {
        self.apply
    }
}

crate::graph_action_tool!(
    add_requires_meta,
    id: ADD_REQUIRES,
    description: "Add a requires edge (prerequisite) between knowledge nodes (cycle-checked; \
                  follows the Knowledge DAG).",
    args: AddRequiresArgs,
    prepare: |raw| super::common::parse_args_with_builder(
        ADD_REQUIRES,
        raw,
        |mut input: AddRequiresArgs| {
            input.from_slug = require_string(input.from_slug, ADD_REQUIRES, "from_slug")?;
            input.to_slug = require_string(input.to_slug, ADD_REQUIRES, "to_slug")?;
            input.rationale = require_string(input.rationale, ADD_REQUIRES, "rationale")?;
            Ok(input)
        },
    ),
    build: |args: &AddRequiresArgs| AddRequires {
        from:       args.from_slug.clone(),
        to:         args.to_slug.clone(),
        attrs:      RequiresAttrs {
            strength:      args.strength,
            rationale:     args.rationale.clone(),
            evidence_refs: args.evidence_refs.clone(),
        },
        confidence: args.confidence,
    },
    ok: |args: &AddRequiresArgs, _| {
        info!(tool = ADD_REQUIRES, from = %args.from_slug, to = %args.to_slug, "graph add requires");
        command_ok(ADD_REQUIRES, json!({"from": args.from_slug, "to": args.to_slug}))
    },
    map_err: |e| map_send_err(e, ADD_REQUIRES),
    preflight: Some(Arc::new(
        |args: &AddRequiresArgs, state: &CallState| -> Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<(), crate::tools::llm::ToolExecutionError>
                    > + Send,
            >,
        > {
            let from = args.from_slug.clone();
            let to = args.to_slug.clone();
            let graph = state.graph.clone();
            Box::pin(async move {
                resolve_typed::<AnyKnowledge>(&graph, Slug::new(from), ADD_REQUIRES).await?;
                resolve_typed::<AnyKnowledge>(&graph, Slug::new(to), ADD_REQUIRES).await?;
                Ok(())
            })
        },
    ))
);

#[derive(Debug, Clone, Builder, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AddSupportsArgs {
    #[schemars(description = "Source knowledge slug providing the support (example/analogy/etc.).")]
    #[builder(with = |v: String| -> ToolInputResult<_> {
        require_string(v, ADD_SUPPORTS, "from_slug")
    })]
    pub from_slug:       String,
    #[schemars(description = "Target knowledge/LO slug receiving the support.")]
    #[builder(with = |v: String| -> ToolInputResult<_> {
        require_string(v, ADD_SUPPORTS, "to_slug")
    })]
    pub to_slug:         String,
    #[schemars(
        description = "Kind of pedagogical support: worked_example | analogy | counterexample | \
                       misconception_fix | strategy_hint | rubric_note."
    )]
    pub support_kind:    SupportKind,
    #[schemars(description = "Intended cognitive effect: reduce_extraneous_load | \
                              increase_germane_load | motivate | contrast.")]
    pub intended_effect: IntendedEffect,
    #[serde(default)]
    #[schemars(
        description = "Case type this support covers: typical | edge | error_case (used for \
                       example variety policies)."
    )]
    pub case_tag:        Option<CaseTag>,
    #[serde(default)]
    #[schemars(
        description = "Coverage tags describing which cases/conditions this support covers."
    )]
    pub coverage_tags:   Vec<String>,
    #[serde(default)]
    #[schemars(
        description = "Source refs for this support (where the example/analogy lives in the text)."
    )]
    pub evidence_refs:   Vec<SourceRef>,
    #[serde(default = "default_confidence")]
    pub confidence:      f32,
    #[serde(default)]
    #[builder(default = false)]
    pub apply:           bool,
}

impl MaybeApply for AddSupportsArgs {
    fn apply_flag(&self) -> bool {
        self.apply
    }
}

crate::graph_action_tool!(
    add_supports_meta,
    id: ADD_SUPPORTS,
    description: "Add a supports edge (worked example / analogy / counterexample / misconception \
                  fix / strategy hint / rubric note) from one knowledge node to another \
                  knowledge/LO. Supports are scaffolds (Cognitive Load Theory) and must remain \
                  fadeable.",
    args: AddSupportsArgs,
    prepare: |raw| super::common::parse_args_with_builder(
        ADD_SUPPORTS,
        raw,
        |mut input: AddSupportsArgs| {
            input.from_slug = require_string(input.from_slug, ADD_SUPPORTS, "from_slug")?;
            input.to_slug = require_string(input.to_slug, ADD_SUPPORTS, "to_slug")?;
            Ok(input)
        },
    ),
    build: |args: &AddSupportsArgs| AddSupports {
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
    },
    ok: |args: &AddSupportsArgs, _| {
        info!(tool = ADD_SUPPORTS, from = %args.from_slug, to = %args.to_slug, "graph add supports");
        command_ok(ADD_SUPPORTS, json!({"from": args.from_slug, "to": args.to_slug}))
    },
    map_err: |e| map_send_err(e, ADD_SUPPORTS),
    preflight: Some(Arc::new(
        |args: &AddSupportsArgs, state: &CallState| -> Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<(), crate::tools::llm::ToolExecutionError>
                    > + Send,
            >,
        > {
            let from = args.from_slug.clone();
            let to = args.to_slug.clone();
            let graph = state.graph.clone();
            Box::pin(async move {
                resolve_typed::<AnyKnowledge>(&graph, Slug::new(from), ADD_SUPPORTS).await?;
                resolve_typed::<AnyKnowledge>(&graph, Slug::new(to), ADD_SUPPORTS).await?;
                Ok(())
            })
        },
    ))
);

#[derive(Debug, Clone, Builder, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AddAssessesArgs {
    #[schemars(description = "Assessment item slug (must be an assessment_item node).")]
    #[builder(with = |v: String| -> ToolInputResult<_> {
        require_string(v, ADD_ASSESSES, "from_slug")
    })]
    pub from_slug:            String,
    #[schemars(description = "Learning outcome slug being assessed.")]
    #[builder(with = |v: String| -> ToolInputResult<_> {
        require_string(v, ADD_ASSESSES, "to_slug")
    })]
    pub to_slug:              String,
    #[schemars(
        description = "Scope of the evidence link: target (assesses LO directly) or enabling \
                       (assesses a supporting sub-skill)."
    )]
    pub scope:                AssessmentScope,
    #[serde(default)]
    #[schemars(
        description = "Observable features/scoring dimensions this item elicits; should cover the \
                       LO's rubric_criteria for target scope."
    )]
    pub observation_features: Vec<String>,
    #[serde(default = "default_confidence")]
    pub confidence:           f32,
    #[serde(default)]
    #[builder(default = false)]
    pub apply:                bool,
}

impl MaybeApply for AddAssessesArgs {
    fn apply_flag(&self) -> bool {
        self.apply
    }
}

crate::graph_action_tool!(
    add_assesses_meta,
    id: ADD_ASSESSES,
    description: "Add an assesses edge (assessment_item -> learning_outcome). Claim is set to the \
                  target LO slug automatically.",
    args: AddAssessesArgs,
    prepare: |raw| super::common::parse_args_with_builder(
        ADD_ASSESSES,
        raw,
        |mut input: AddAssessesArgs| {
            input.from_slug = require_string(input.from_slug, ADD_ASSESSES, "from_slug")?;
            input.to_slug = require_string(input.to_slug, ADD_ASSESSES, "to_slug")?;
            Ok(input)
        },
    ),
    build: |args: &AddAssessesArgs| AddAssesses {
        from:       args.from_slug.clone(),
        to:         args.to_slug.clone(),
        attrs:      AssessesAttrs {
            evidence_link: EvidenceLink {
                claim:                args.to_slug.clone(),
                observation_features: args.observation_features.clone(),
                scope:                args.scope,
            },
        },
        confidence: args.confidence,
    },
    ok: |args: &AddAssessesArgs, _| {
        info!(tool = ADD_ASSESSES, from = %args.from_slug, to = %args.to_slug, "graph add assesses");
        command_ok(ADD_ASSESSES, json!({"from": args.from_slug, "to": args.to_slug}))
    },
    map_err: |e| map_send_err(e, ADD_ASSESSES),
    preflight: Some(Arc::new(
        |args: &AddAssessesArgs, state: &CallState| -> Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<(), crate::tools::llm::ToolExecutionError>
                    > + Send,
            >,
        > {
            let from = args.from_slug.clone();
            let to = args.to_slug.clone();
            let graph = state.graph.clone();
            Box::pin(async move {
                resolve_typed::<AssessmentItem>(&graph, Slug::new(from), ADD_ASSESSES).await?;
                resolve_typed::<LearningOutcome>(&graph, Slug::new(to), ADD_ASSESSES).await?;
                Ok(())
            })
        },
    ))
);

#[derive(Debug, Clone, Builder, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AddPrecedesArgs {
    #[schemars(description = "Slug of earlier teaching step in the episode.")]
    #[builder(with = |v: String| -> ToolInputResult<_> {
        require_string(v, ADD_PRECEDES, "from_slug")
    })]
    pub from_slug:  String,
    #[schemars(description = "Slug of later teaching step in the episode.")]
    #[builder(with = |v: String| -> ToolInputResult<_> {
        require_string(v, ADD_PRECEDES, "to_slug")
    })]
    pub to_slug:    String,
    #[schemars(description = "Episode identifier; precedes edges must stay within this episode.")]
    #[builder(with = |v: String| -> ToolInputResult<_> {
        require_string(v, ADD_PRECEDES, "episode")
    })]
    pub episode:    String,
    #[schemars(
        description = "Confidence for this discourse ordering; defaults to 1.0 when omitted."
    )]
    #[serde(default = "default_confidence")]
    pub confidence: f32,
    #[serde(default)]
    #[builder(default = false)]
    pub apply:      bool,
}

impl MaybeApply for AddPrecedesArgs {
    fn apply_flag(&self) -> bool {
        self.apply
    }
}

crate::graph_action_tool!(
    add_precedes_meta,
    id: ADD_PRECEDES,
    description: "Add a precedes edge between TeachingSteps in the same episode (acyclic). \
                  Captures authored narrative order, not logical prerequisite.",
    args: AddPrecedesArgs,
    prepare: |raw| super::common::parse_args_with_builder(
        ADD_PRECEDES,
        raw,
        |mut input: AddPrecedesArgs| {
            input.from_slug = require_string(input.from_slug, ADD_PRECEDES, "from_slug")?;
            input.to_slug = require_string(input.to_slug, ADD_PRECEDES, "to_slug")?;
            input.episode = require_string(input.episode, ADD_PRECEDES, "episode")?;
            Ok(input)
        },
    ),
    build: |args: &AddPrecedesArgs| AddPrecedes {
        from:       args.from_slug.clone(),
        to:         args.to_slug.clone(),
        attrs:      PrecedesAttrs {
            episode: args.episode.clone(),
        },
        confidence: args.confidence,
    },
    ok: |args: &AddPrecedesArgs, _| {
        info!(tool = ADD_PRECEDES, from = %args.from_slug, to = %args.to_slug, "graph add precedes");
        command_ok(
            ADD_PRECEDES,
            json!({"from": args.from_slug, "to": args.to_slug, "episode": args.episode}),
        )
    },
    map_err: |e| map_send_err(e, ADD_PRECEDES),
    preflight: Some(Arc::new(
        |args: &AddPrecedesArgs, state: &CallState| -> Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<(), crate::tools::llm::ToolExecutionError>
                    > + Send,
            >,
        > {
            let from = args.from_slug.clone();
            let to = args.to_slug.clone();
            let graph = state.graph.clone();
            Box::pin(async move {
                resolve_typed::<TeachingStep>(&graph, Slug::new(from), ADD_PRECEDES).await?;
                resolve_typed::<TeachingStep>(&graph, Slug::new(to), ADD_PRECEDES).await?;
                Ok(())
            })
        },
    ))
);

#[derive(Debug, Clone, Builder, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AddAnchorsArgs {
    #[schemars(description = "TeachingStep slug anchoring to knowledge/LO/assessment.")]
    #[builder(with = |v: String| -> ToolInputResult<_> {
        require_string(v, ADD_ANCHORS, "from_slug")
    })]
    pub from_slug:  String,
    #[schemars(description = "Target knowledge/LO/assessment slug this step touches.")]
    #[builder(with = |v: String| -> ToolInputResult<_> {
        require_string(v, ADD_ANCHORS, "to_slug")
    })]
    pub to_slug:    String,
    #[schemars(
        description = "Impact of this step on the target: introduce | use | refine | motivate | \
                       target."
    )]
    pub impact:     AnchorImpact,
    #[serde(default = "default_confidence")]
    #[schemars(description = "Confidence for this anchor; defaults to 1.0 if omitted.")]
    pub confidence: f32,
    #[serde(default)]
    #[builder(default = false)]
    pub apply:      bool,
}

impl MaybeApply for AddAnchorsArgs {
    fn apply_flag(&self) -> bool {
        self.apply
    }
}

crate::graph_action_tool!(
    add_anchors_meta,
    id: ADD_ANCHORS,
    description: "Add an anchors edge from a TeachingStep to knowledge/LO/assessment with a \
                  specific impact: introduce/refine (instructional knowledge), target (LO), \
                  use/motivate (any; assessment targets must use). Enforces discourse \
                  impact/target rules.",
    args: AddAnchorsArgs,
    prepare: |raw| super::common::parse_args_with_builder(
        ADD_ANCHORS,
        raw,
        |mut input: AddAnchorsArgs| {
            input.from_slug = require_string(input.from_slug, ADD_ANCHORS, "from_slug")?;
            input.to_slug = require_string(input.to_slug, ADD_ANCHORS, "to_slug")?;
            Ok(input)
        },
    ),
    build: |args: &AddAnchorsArgs| AddAnchors {
        from:       args.from_slug.clone(),
        to:         args.to_slug.clone(),
        attrs:      AnchorsAttrs {
            impact: args.impact,
        },
        confidence: args.confidence,
    },
    ok: |args: &AddAnchorsArgs, _| {
        info!(tool = ADD_ANCHORS, from = %args.from_slug, to = %args.to_slug, "graph add anchors");
        command_ok(
            ADD_ANCHORS,
            json!({"from": args.from_slug, "to": args.to_slug, "impact": args.impact}),
        )
    },
    map_err: |e| map_send_err(e, ADD_ANCHORS),
    preflight: Some(Arc::new(
        |args: &AddAnchorsArgs, state: &CallState| -> Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<(), crate::tools::llm::ToolExecutionError>
                    > + Send,
            >,
        > {
            let from = args.from_slug.clone();
            let to = args.to_slug.clone();
            let graph = state.graph.clone();
            Box::pin(async move {
                resolve_typed::<TeachingStep>(&graph, Slug::new(from), ADD_ANCHORS).await?;
                resolve_typed::<AnyKnowledge>(&graph, Slug::new(to), ADD_ANCHORS).await?;
                Ok(())
            })
        },
    ))
);

// ---------- Rename / remove ----------

const RENAME_NODE: &str = "graph_rename_node";
const REMOVE_NODE: &str = "graph_remove_node";

#[derive(Debug, Clone, Builder, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RenameNodeArgs {
    #[builder(with = |v: String| -> ToolInputResult<_> {
        require_string(v, RENAME_NODE, "old_slug")
    })]
    pub old_slug: String,
    #[builder(with = |v: String| -> ToolInputResult<_> {
        require_string(v, RENAME_NODE, "new_slug")
    })]
    pub new_slug: String,
    #[serde(default)]
    #[builder(default = false)]
    pub apply:    bool,
}

impl MaybeApply for RenameNodeArgs {
    fn apply_flag(&self) -> bool {
        self.apply
    }
}

crate::graph_action_tool!(
    rename_node_meta,
    id: RENAME_NODE,
    description: "Rename a node slug and update assesses.claim if needed.",
    args: RenameNodeArgs,
    prepare: |raw| super::common::parse_args_with_builder(RENAME_NODE, raw, |mut input: RenameNodeArgs| {
        input.old_slug = require_string(input.old_slug, RENAME_NODE, "old_slug")?;
        input.new_slug = require_string(input.new_slug, RENAME_NODE, "new_slug")?;
        Ok(input)
    }),
    build: |args: &RenameNodeArgs| RenameNode {
        old_slug: args.old_slug.clone(),
        new_slug: args.new_slug.clone(),
    },
    ok: |args: &RenameNodeArgs, _| {
        command_ok(RENAME_NODE, json!({"old_slug": args.old_slug, "new_slug": args.new_slug}))
    },
    map_err: |e| map_send_err(e, RENAME_NODE)
);

#[derive(Debug, Clone, Builder, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RemoveNodeArgs {
    #[builder(with = |v: String| -> ToolInputResult<_> {
        require_string(v, REMOVE_NODE, "slug")
    })]
    pub slug:  String,
    #[serde(default)]
    #[builder(default = false)]
    pub apply: bool,
}

impl MaybeApply for RemoveNodeArgs {
    fn apply_flag(&self) -> bool {
        self.apply
    }
}

crate::graph_action_tool!(
    remove_node_meta,
    id: REMOVE_NODE,
    description: "Delete a node and all connected edges.",
    args: RemoveNodeArgs,
    prepare: |raw| super::common::parse_args_with_builder(
        REMOVE_NODE,
        raw,
        |mut input: RemoveNodeArgs| {
            input.slug = require_string(input.slug, REMOVE_NODE, "slug")?;
            Ok(input)
        },
    ),
    build: |args: &RemoveNodeArgs| RemoveNode {
        slug: args.slug.clone(),
    },
    ok: |args: &RemoveNodeArgs, _| {
        info!(tool = REMOVE_NODE, slug = %args.slug, "graph remove node");
        command_ok(REMOVE_NODE, json!({"slug": args.slug}))
    },
    map_err: |e| map_send_err(e, REMOVE_NODE)
);

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
    ]
}
