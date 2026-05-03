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
        slug::Slug as CanonicalSlug,
    },
    schema::types::{
        AssessmentScope, EvidenceLink, IntendedEffect, KnowledgeType, SourceRef, Strength,
        SupportKind,
    },
    tools::llm::{
        CallState, ToolInputError, ToolInputResult, ToolPrototype,
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

fn fill_source_ref_revisions(source_refs: &mut [SourceRef], course_commit: &str) {
    for span in source_refs.iter_mut() {
        if span.revision.is_empty() {
            span.revision = course_commit.to_string();
        }
    }
}

fn reject_mixed_case_knowledge_type(
    raw: &serde_json::Value,
    tool: &'static str,
) -> ToolInputResult<()> {
    if let Some(value) = raw.get("knowledge_type").and_then(|v| v.as_str()) {
        let lower = value.to_ascii_lowercase();
        if value != lower {
            return Err(ToolInputError::InvalidPayload {
                tool,
                message: format!(
                    "knowledge_type must be snake_case (e.g., \"conceptual\"), got \"{}\"",
                    value
                ),
            });
        }
    }
    Ok(())
}

// ---------- Insert / update knowledge ----------

const INSERT_KNOWLEDGE: &str = "graph_insert_knowledge";
const UPDATE_KNOWLEDGE: &str = "graph_update_knowledge";

#[derive(Debug, Clone, Builder, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InsertKnowledgeArgs {
    #[schemars(
        description = "Unique identifier following {Kind}.{name} pattern. Examples: \
                       C.contract_components, P.design_recipe, LO.write_docstring, A.exercise_1. \
                       Kind prefixes: F=factual, C=conceptual, P=procedural, M=metacognitive, \
                       LO=learning_outcome, A=assessment_item"
    )]
    #[builder(with = |value: String| -> ToolInputResult<_> {
        require_string(value, INSERT_KNOWLEDGE, "slug")
    })]
    pub slug: String,
    #[schemars(
        description = "Human-readable title. Keep concise (3-8 words). Example: 'Python Docstring \
                       Format'"
    )]
    #[builder(with = |value: String| -> ToolInputResult<_> {
        require_string(value, INSERT_KNOWLEDGE, "title")
    })]
    pub title: String,
    #[schemars(
        description = "Self-contained description of the knowledge. Must pass the Assessable Atom \
                       Test: 'Can I write ONE exam question targeting ONLY this?' Keep to 1-3 \
                       sentences."
    )]
    #[builder(with = |value: String| -> ToolInputResult<_> {
        require_string(value, INSERT_KNOWLEDGE, "statement")
    })]
    pub statement: String,
    #[schemars(description = "Type of knowledge from Bloom's taxonomy. Use: factual \
                              (terms/definitions), conceptual (principles/models), procedural \
                              (algorithms/methods), metacognitive (self-regulation strategies), \
                              learning_outcome (measurable goals), assessment_item \
                              (exercises/tests)")]
    pub knowledge_type: KnowledgeType,
    #[schemars(
        description = "REQUIRED for learning_outcome only. List of observable, measurable \
                       behaviors. Example: ['identifies preconditions', 'documents raises \
                       clause', 'tests boundary cases']. assesses edges' observation_features \
                       must cover these."
    )]
    #[serde(default)]
    pub rubric_criteria: Vec<String>,
    #[schemars(
        description = "OPTIONAL for assessment_item only. Skills the assessment requires but \
                       doesn't intend to measure. Example: ['prose writing ability']. Used in \
                       construct validity checks."
    )]
    #[serde(default)]
    pub construct_irrelevant_demands: Vec<String>,
    #[schemars(
        description = "REQUIRED. Where in the source text this knowledge appears. Array of \
                       objects with ONLY these fields: {path: 'relative/path.ptx', start_line: \
                       10, end_line: 25}. Do NOT include 'revision' - it is auto-filled."
    )]
    #[serde(default)]
    pub source_refs: Vec<SourceRef>,
    #[schemars(
        description = "Your confidence in this extraction (0.0-1.0). Default 1.0. Use lower \
                       values to flag uncertain extractions for review."
    )]
    #[serde(default)]
    pub confidence: f32,
    #[serde(default)]
    #[schemars(
        description = "Granularity level. mid (default) = one exam question. macro = needs \
                       splitting. micro = too small, fold into parent."
    )]
    pub grain_level: Option<crate::graph::GrainLevel>,
    #[serde(default)]
    #[schemars(
        description = "Cognitive load. high = needs extra scaffolding (examples). medium = \
                       typical. low = straightforward."
    )]
    pub intrinsic_load: Option<crate::graph::IntrinsicLoad>,
    #[serde(default)]
    #[schemars(
        description = "Where this knowledge is introduced. in_course (default) = taught here. \
                       prior = assumed known. external = referenced but not taught."
    )]
    pub introduction_scope: Option<IntroductionScope>,
    #[serde(default)]
    #[schemars(description = "Tags for filtering and organization. REQUIRED: \
                              'source:<chapter_path>' and 'spec:<niche>'. Hints for weavers: \
                              'req:<slug>', 'sup:<slug>', 'ref:<slug>'")]
    pub tags: Vec<String>,
    #[serde(default)]
    #[builder(default = false)]
    #[schemars(
        description = "Set true to actually create the node. Default false returns a preview \
                       showing what would be created. Always preview first on uncertain \
                       extractions."
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

pub(super) fn insert_knowledge_meta() -> ToolPrototype {
    ToolPrototype {
        id:          INSERT_KNOWLEDGE,
        description: "Create a Knowledge, LearningOutcome, or AssessmentItem node. Each node \
                      should represent ONE assessable idea (pass the 'can I write one exam \
                      question for this?' test). Use apply=false first to preview, then \
                      apply=true to create.",
        schema:      crate::tools::llm::schema_for_args::<InsertKnowledgeArgs>(),
        parse:       |raw, state| {
            reject_mixed_case_knowledge_type(&raw, INSERT_KNOWLEDGE)?;
            let mut args: InsertKnowledgeArgs = super::common::parse_args_with_builder(
                INSERT_KNOWLEDGE,
                raw,
                |mut input: InsertKnowledgeArgs| {
                    input.slug = require_string(input.slug, INSERT_KNOWLEDGE, "slug")?;
                    input.title = require_string(input.title, INSERT_KNOWLEDGE, "title")?;
                    input.statement =
                        require_string(input.statement, INSERT_KNOWLEDGE, "statement")?;
                    Ok(input)
                },
            )?;
            fill_source_ref_revisions(&mut args.source_refs, state.course_commit.as_ref());
            let raw_args =
                serde_json::to_value(&args).expect("failed to serialize graph action args");
            super::common::parse_graph_command(
                INSERT_KNOWLEDGE,
                raw_args,
                state,
                build_insert_knowledge,
                |args: &InsertKnowledgeArgs, _| {
                    info!(tool = INSERT_KNOWLEDGE, slug = %args.slug, "graph insert knowledge");
                    command_ok(INSERT_KNOWLEDGE, json!({"slug": args.slug}))
                },
                |e| map_send_err(e, INSERT_KNOWLEDGE),
                None,
                None,
            )
        },
    }
}

pub(super) fn update_knowledge_meta() -> ToolPrototype {
    ToolPrototype {
        id:          UPDATE_KNOWLEDGE,
        description: "Update an existing Knowledge node's metadata. Useful for fixing statements, \
                      updating tags (e.g., removing harvest hint tags like req:*), or changing \
                      introduction_scope. Provide all fields including unchanged ones.",
        schema:      crate::tools::llm::schema_for_args::<InsertKnowledgeArgs>(),
        parse:       |raw, state| {
            reject_mixed_case_knowledge_type(&raw, UPDATE_KNOWLEDGE)?;
            let mut args: InsertKnowledgeArgs = super::common::parse_args_with_builder(
                UPDATE_KNOWLEDGE,
                raw,
                |mut input: InsertKnowledgeArgs| {
                    input.slug = require_string(input.slug, UPDATE_KNOWLEDGE, "slug")?;
                    input.title = require_string(input.title, UPDATE_KNOWLEDGE, "title")?;
                    input.statement =
                        require_string(input.statement, UPDATE_KNOWLEDGE, "statement")?;
                    Ok(input)
                },
            )?;
            fill_source_ref_revisions(&mut args.source_refs, state.course_commit.as_ref());
            let raw_args =
                serde_json::to_value(&args).expect("failed to serialize graph action args");
            super::common::parse_graph_command(
                UPDATE_KNOWLEDGE,
                raw_args,
                state,
                |args: &InsertKnowledgeArgs| UpdateKnowledge {
                    slug:    args.slug.clone(),
                    payload: build_insert_knowledge(args).payload,
                    tags:    args.tags.clone(),
                },
                |args: &InsertKnowledgeArgs, _| {
                    info!(tool = UPDATE_KNOWLEDGE, slug = %args.slug, "graph update knowledge");
                    command_ok(UPDATE_KNOWLEDGE, json!({"slug": args.slug}))
                },
                |e| map_send_err(e, UPDATE_KNOWLEDGE),
                None,
                None,
            )
        },
    }
}

// ---------- Insert / update teaching step ----------

const INSERT_TEACHING: &str = "graph_insert_teaching_step";
const UPDATE_TEACHING: &str = "graph_update_teaching_step";

#[derive(Debug, Clone, Builder, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InsertTeachingArgs {
    #[schemars(
        description = "Unique identifier following TS.{name} pattern. Example: \
                       TS.contract_motivation, TS.docstring_definition"
    )]
    #[builder(with = |value: String| -> ToolInputResult<_> {
        require_string(value, INSERT_TEACHING, "slug")
    })]
    pub slug:        String,
    #[schemars(
        description = "Human-readable title (3-8 words). Example: 'Motivating the Need for \
                       Contracts'"
    )]
    #[builder(with = |value: String| -> ToolInputResult<_> {
        require_string(value, INSERT_TEACHING, "title")
    })]
    pub title:       String,
    #[schemars(
        description = "Brief summary of what this teaching moment does (1-2 sentences). Example: \
                       'Opens with a crashing function to motivate explicit contracts.'"
    )]
    #[builder(with = |value: String| -> ToolInputResult<_> {
        require_string(value, INSERT_TEACHING, "statement")
    })]
    pub statement:   String,
    #[schemars(
        description = "Pedagogical purpose: setup (motivation/framing), idea (introduces new \
                       concept), use (applies known knowledge), consolidate \
                       (summarizes/reinforces)"
    )]
    pub purpose:     TeachingPurpose,
    #[serde(default)]
    #[schemars(
        description = "Pedagogical technique markers. Examples: ['worked-example'], \
                       ['naive-first'], ['analogy'], ['guided-practice'], ['breakdown']"
    )]
    pub method_tags: Vec<String>,
    #[schemars(
        description = "Section/lesson identifier. Used to scope precedes edges (steps in same \
                       episode are ordered together). Example: '02_contracts', 'chapter3_loops'"
    )]
    #[builder(with = |value: String| -> ToolInputResult<_> {
        require_string(value, INSERT_TEACHING, "episode")
    })]
    pub episode:     String,
    #[serde(default)]
    #[schemars(
        description = "REQUIRED. Source location: [{path: 'relative/path.ptx', start_line: 10, \
                       end_line: 25}]. Do NOT include 'revision' - it is auto-filled."
    )]
    pub source_refs: Vec<SourceRef>,
    #[serde(default)]
    #[schemars(
        description = "Tags for filtering. REQUIRED: 'source:<chapter_path>' and \
                       'spec:teaching_steps'. Hints: 'anchors:<slug>:<impact>', 'precedes:<slug>'"
    )]
    pub tags:        Vec<String>,
    #[serde(default)]
    #[schemars(
        description = "Justification if this step has no anchors edges. Every TeachingStep should \
                       anchor to knowledge/LOs; if not, explain why."
    )]
    pub rationale:   Option<String>,
    #[serde(default)]
    #[builder(default = false)]
    #[schemars(description = "Set true to create. Default false = preview. Always preview first.")]
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

pub(super) fn insert_teaching_meta() -> ToolPrototype {
    ToolPrototype {
        id:          INSERT_TEACHING,
        description: "Create a TeachingStep node representing a narrative moment in the textbook \
                      (the Discourse Layer). Each step has a purpose (setup/idea/use/consolidate) \
                      and should be connected via 'precedes' edges to order steps and 'anchors' \
                      edges to link to knowledge.",
        schema:      crate::tools::llm::schema_for_args::<InsertTeachingArgs>(),
        parse:       |raw, state| {
            let mut args: InsertTeachingArgs = super::common::parse_args_with_builder(
                INSERT_TEACHING,
                raw,
                |mut input: InsertTeachingArgs| {
                    input.slug = require_string(input.slug, INSERT_TEACHING, "slug")?;
                    input.title = require_string(input.title, INSERT_TEACHING, "title")?;
                    input.statement =
                        require_string(input.statement, INSERT_TEACHING, "statement")?;
                    input.episode = require_string(input.episode, INSERT_TEACHING, "episode")?;
                    Ok(input)
                },
            )?;
            fill_source_ref_revisions(&mut args.source_refs, state.course_commit.as_ref());
            let raw_args =
                serde_json::to_value(&args).expect("failed to serialize graph action args");
            super::common::parse_graph_command(
                INSERT_TEACHING,
                raw_args,
                state,
                build_insert_teaching,
                |args: &InsertTeachingArgs, _| {
                    info!(tool = INSERT_TEACHING, slug = %args.slug, "graph insert teaching_step");
                    command_ok(INSERT_TEACHING, json!({"slug": args.slug}))
                },
                |e| map_send_err(e, INSERT_TEACHING),
                None,
                None,
            )
        },
    }
}

pub(super) fn update_teaching_meta() -> ToolPrototype {
    ToolPrototype {
        id:          UPDATE_TEACHING,
        description: "Update an existing TeachingStep node's metadata. Useful for fixing \
                      statements, updating tags, or adding rationale. Provide all fields \
                      including unchanged ones.",
        schema:      crate::tools::llm::schema_for_args::<InsertTeachingArgs>(),
        parse:       |raw, state| {
            let mut args: InsertTeachingArgs = super::common::parse_args_with_builder(
                UPDATE_TEACHING,
                raw,
                |mut input: InsertTeachingArgs| {
                    input.slug = require_string(input.slug, UPDATE_TEACHING, "slug")?;
                    input.title = require_string(input.title, UPDATE_TEACHING, "title")?;
                    input.statement =
                        require_string(input.statement, UPDATE_TEACHING, "statement")?;
                    input.episode = require_string(input.episode, UPDATE_TEACHING, "episode")?;
                    Ok(input)
                },
            )?;
            fill_source_ref_revisions(&mut args.source_refs, state.course_commit.as_ref());
            let raw_args =
                serde_json::to_value(&args).expect("failed to serialize graph action args");
            super::common::parse_graph_command(
                UPDATE_TEACHING,
                raw_args,
                state,
                |args: &InsertTeachingArgs| UpdateTeachingStep {
                    slug:    args.slug.clone(),
                    payload: build_insert_teaching(args).payload,
                    tags:    args.tags.clone(),
                },
                |args: &InsertTeachingArgs, _| {
                    info!(tool = UPDATE_TEACHING, slug = %args.slug, "graph update teaching_step");
                    command_ok(UPDATE_TEACHING, json!({"slug": args.slug}))
                },
                |e| map_send_err(e, UPDATE_TEACHING),
                None,
                None,
            )
        },
    }
}

// ---------- Edge tools ----------

const ADD_REQUIRES: &str = "graph_add_requires";
const ADD_SUPPORTS: &str = "graph_add_supports";
const ADD_ASSESSES: &str = "graph_add_assesses";
const ADD_PRECEDES: &str = "graph_add_precedes";
const ADD_ANCHORS: &str = "graph_add_anchors";

#[derive(Debug, Clone, Builder, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AddRequiresArgs {
    #[schemars(
        description = "Source node slug (the prerequisite). Must be a Knowledge node \
                       (factual/conceptual/procedural/metacognitive). Example: 'C.variable_scope'"
    )]
    #[builder(with = |v: String| -> ToolInputResult<_> {
        require_string(v, ADD_REQUIRES, "from_slug")
    })]
    pub from_slug:     String,
    #[schemars(
        description = "Target node slug (depends on prerequisite). Can be Knowledge or \
                       AssessmentItem. Example: 'P.write_function' or 'A.scope_exercise'. \
                       Learning outcomes cannot be targets."
    )]
    #[builder(with = |v: String| -> ToolInputResult<_> {
        require_string(v, ADD_REQUIRES, "to_slug")
    })]
    pub to_slug:       String,
    #[schemars(
        description = "Dependency strength: necessary (cannot proceed without), strong (very \
                       difficult without), helpful (makes learning easier but not required)"
    )]
    pub strength:      Strength,
    #[schemars(description = "REQUIRED. Why does from_slug enable to_slug? Example: \
                              'Understanding variable scope is necessary to correctly identify \
                              which variables a function can access.'")]
    #[builder(with = |v: String| -> ToolInputResult<_> {
        require_string(v, ADD_REQUIRES, "rationale")
    })]
    pub rationale:     String,
    #[schemars(
        description = "REQUIRED: source locations supporting this dependency claim. Provide at \
                       least one entry. Array of {path, start_line, end_line, revision}. If \
                       revision is empty, it will be filled with the course commit."
    )]
    pub evidence_refs: Vec<SourceRef>,
    #[serde(default = "default_confidence")]
    #[schemars(description = "Your confidence in this edge (0.0-1.0). Default 1.0.")]
    pub confidence:    f32,
    #[serde(default)]
    #[builder(default = false)]
    #[schemars(
        description = "Set true to create. Default false = preview. The edge will be REJECTED if \
                       it creates a cycle in the DAG. Run graph_dag_check() after adding to \
                       verify."
    )]
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
    description: "Add a prerequisite (requires) edge from a Knowledge node to another Knowledge or AssessmentItem. Forms the dependency DAG. CRITICAL: The system rejects edges that would create cycles. Always run graph_dag_check() after adding requires edges to verify acyclicity.",
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
    )),
    mutate: Some(|args: &mut AddRequiresArgs, state: &CallState| {
        fill_source_ref_revisions(&mut args.evidence_refs, state.course_commit.as_ref());
        Ok(())
    })
);

#[derive(Debug, Clone, Builder, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AddSupportsArgs {
    #[schemars(
        description = "Source node slug (the scaffold/example). Must be a Knowledge node. \
                       Example: 'P.docstring_typical_example'"
    )]
    #[builder(with = |v: String| -> ToolInputResult<_> {
        require_string(v, ADD_SUPPORTS, "from_slug")
    })]
    pub from_slug:       String,
    #[schemars(
        description = "Target node slug (what the scaffold supports). Can be Knowledge or \
                       LearningOutcome. Example: 'P.python_docstring' or \
                       'LO.write_documented_function'"
    )]
    #[builder(with = |v: String| -> ToolInputResult<_> {
        require_string(v, ADD_SUPPORTS, "to_slug")
    })]
    pub to_slug:         String,
    #[schemars(
        description = "Type of scaffold: worked_example (step-by-step demo), analogy (comparison \
                       to familiar), counterexample (what NOT to do), misconception_fix (corrects \
                       common mistake), strategy_hint (tip for applying), rubric_note (clarifies \
                       LO grading)"
    )]
    pub support_kind:    SupportKind,
    #[schemars(
        description = "Cognitive purpose: reduce_extraneous_load (simplify learning), \
                       increase_germane_load (deepen understanding), motivate (create desire to \
                       learn), contrast (highlight boundaries/differences)"
    )]
    pub intended_effect: IntendedEffect,
    #[schemars(
        description = "REQUIRED case tag: typical (happy path), edge (boundary), error_case \
                       (failure mode). For non-example scaffolds, use typical. Procedural nodes \
                       need both typical AND edge/error_case examples."
    )]
    pub case_tag:        CaseTag,
    #[serde(default)]
    #[schemars(description = "Specific constraints this scaffold covers. Example: \
                              ['negative_input', 'empty_list']. Helps track which edge cases \
                              are addressed.")]
    pub coverage_tags:   Vec<String>,
    #[schemars(
        description = "REQUIRED: source locations for this scaffold. Provide at least one entry. \
                       Array of {path, start_line, end_line, revision}. If revision is empty, it \
                       will be filled with the course commit."
    )]
    pub evidence_refs:   Vec<SourceRef>,
    #[serde(default = "default_confidence")]
    #[schemars(description = "Your confidence in this edge (0.0-1.0). Default 1.0.")]
    pub confidence:      f32,
    #[serde(default)]
    #[builder(default = false)]
    #[schemars(
        description = "Set true to create. Default false = preview. IMPORTANT: Supports must be \
                       fadeable—they cannot be the only path to an assessment. Run \
                       graph_fadeability_view() to check."
    )]
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
    description: "Add a scaffolding (supports) edge connecting an example/analogy/hint to the knowledge it supports. CRITICAL: Supports must stay 'fadeable'—they cannot be the only path to an assessment. If removing all supports would break assessment reachability, the supports edge should be a requires edge instead. Check with graph_fadeability_view().",
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
            case_tag:        Some(args.case_tag),
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
    )),
    mutate: Some(|args: &mut AddSupportsArgs, state: &CallState| {
        fill_source_ref_revisions(&mut args.evidence_refs, state.course_commit.as_ref());
        Ok(())
    })
);

#[derive(Debug, Clone, Builder, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AddAssessesArgs {
    #[schemars(description = "Source node slug. MUST be an AssessmentItem \
                              (knowledge_type=assessment_item). Example: 'A.docstring_exercise'")]
    #[builder(with = |v: String| -> ToolInputResult<_> {
        require_string(v, ADD_ASSESSES, "from_slug")
    })]
    pub from_slug:            String,
    #[schemars(description = "Target node slug. MUST be a LearningOutcome \
                              (knowledge_type=learning_outcome). Example: \
                              'LO.write_documented_function'")]
    #[builder(with = |v: String| -> ToolInputResult<_> {
        require_string(v, ADD_ASSESSES, "to_slug")
    })]
    pub to_slug:              String,
    #[schemars(
        description = "Coverage scope: target (assessment directly measures this LO—every LO \
                       needs at least one target), enabling (assessment measures a prerequisite \
                       or sub-part of the LO)"
    )]
    pub scope:                AssessmentScope,
    #[serde(default)]
    #[schemars(
        description = "CRITICAL: Observable behaviors that can be scored from this assessment. \
                       MUST cover the LO's rubric_criteria. Example: ['test suite passes', \
                       'docstring includes raises clause']. Use graph_get_node() to check the \
                       LO's criteria first."
    )]
    pub observation_features: Vec<String>,
    #[serde(default = "default_confidence")]
    #[schemars(description = "Your confidence in this link (0.0-1.0). Default 1.0.")]
    pub confidence:           f32,
    #[serde(default)]
    #[builder(default = false)]
    #[schemars(
        description = "Set true to create. Default false = preview. After creating, run \
                       graph_lo_alignment_summary(lo_slug=...) to verify coverage."
    )]
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
    description: "Link an AssessmentItem to a LearningOutcome. This is how you prove an LO is measurable. Each LO MUST have at least one assesses edge with scope=target. The observation_features list MUST cover the LO's rubric_criteria. Check coverage with graph_lo_alignment_summary().",
    args: AddAssessesArgs,
    prepare: |raw| super::common::parse_args_with_builder(
        ADD_ASSESSES,
        raw,
        |mut input: AddAssessesArgs| {
            input.from_slug = require_string(input.from_slug, ADD_ASSESSES, "from_slug")?;
            input.to_slug = require_string(input.to_slug, ADD_ASSESSES, "to_slug")?;
            let from = CanonicalSlug::parse(&input.from_slug).map_err(|err| {
                ToolInputError::InvalidPayload {
                    tool:     ADD_ASSESSES,
                    message: format!("invalid from_slug `{}`: {}", input.from_slug, err),
                }
            })?;
            let to = CanonicalSlug::parse(&input.to_slug).map_err(|err| {
                ToolInputError::InvalidPayload {
                    tool:     ADD_ASSESSES,
                    message: format!("invalid to_slug `{}`: {}", input.to_slug, err),
                }
            })?;
            input.from_slug = from.to_string();
            input.to_slug = to.to_string();
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
    )),
    mutate: None
);

#[derive(Debug, Clone, Builder, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AddPrecedesArgs {
    #[schemars(
        description = "Earlier TeachingStep slug. Must be a TeachingStep node. Example: \
                       'TS.motivation'"
    )]
    #[builder(with = |v: String| -> ToolInputResult<_> {
        require_string(v, ADD_PRECEDES, "from_slug")
    })]
    pub from_slug:  String,
    #[schemars(
        description = "Later TeachingStep slug. Must be a TeachingStep node. Example: \
                       'TS.definition'"
    )]
    #[builder(with = |v: String| -> ToolInputResult<_> {
        require_string(v, ADD_PRECEDES, "to_slug")
    })]
    pub to_slug:    String,
    #[schemars(
        description = "Episode/section identifier. MUST match both steps' episode field. Precedes \
                       edges are scoped within episodes. Example: '02_contracts'"
    )]
    #[builder(with = |v: String| -> ToolInputResult<_> {
        require_string(v, ADD_PRECEDES, "episode")
    })]
    pub episode:    String,
    #[serde(default = "default_confidence")]
    #[schemars(description = "Your confidence (0.0-1.0). Default 1.0.")]
    pub confidence: f32,
    #[serde(default)]
    #[builder(default = false)]
    #[schemars(
        description = "Set true to create. Default false = preview. Precedes edges must be \
                       acyclic within each episode."
    )]
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
    description: "Order TeachingSteps within an episode (section/lesson). Creates a sequence representing the authored reading order. Both steps must belong to the same episode. The sequence must be acyclic.",
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
    )),
    mutate: None
);

#[derive(Debug, Clone, Builder, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AddAnchorsArgs {
    #[schemars(description = "TeachingStep slug. Example: 'TS.docstring_definition'")]
    #[builder(with = |v: String| -> ToolInputResult<_> {
        require_string(v, ADD_ANCHORS, "from_slug")
    })]
    pub from_slug:  String,
    #[schemars(
        description = "Target Knowledge, LO, or AssessmentItem. What does this teaching step \
                       interact with? Example: 'C.contract_components', 'LO.write_docstring', \
                       'A.exercise'"
    )]
    #[builder(with = |v: String| -> ToolInputResult<_> {
        require_string(v, ADD_ANCHORS, "to_slug")
    })]
    pub to_slug:    String,
    #[schemars(
        description = "How does this step interact with the target? introduce (first \
                       presentation), use (applies known knowledge), refine (adds \
                       nuance/specialization), motivate (creates desire to learn), target \
                       (articulates LO expectations/rubric)"
    )]
    pub impact:     AnchorImpact,
    #[serde(default = "default_confidence")]
    #[schemars(description = "Your confidence (0.0-1.0). Default 1.0.")]
    pub confidence: f32,
    #[serde(default)]
    #[builder(default = false)]
    #[schemars(
        description = "Set true to create. Default false = preview. Each TeachingStep should have \
                       at least one anchor (or a rationale explaining why not)."
    )]
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
    description: "Connect a TeachingStep to the Knowledge/LO/Assessment it interacts with. The impact describes HOW: introduce (first presentation of concept), use (applies known knowledge), refine (adds nuance), motivate (frames why it matters), target (articulates LO rubric).",
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
    )),
    mutate: None
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
    description: "Rename a node's slug. Automatically updates all edge references (from_slug, to_slug, evidence_link.claim). Use this instead of delete+recreate to preserve edges.",
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
    description: "Delete a node and ALL edges connected to it (both incoming and outgoing). Use with caution—this is destructive. Consider graph_rename_node if you just need to change the slug.",
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
