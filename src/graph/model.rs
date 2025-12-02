use std::path::PathBuf;

use petgraph::{
    Directed,
    stable_graph::{EdgeIndex, NodeIndex, StableGraph},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::schema::types::{
    EvidenceLink, IntendedEffect, KnowledgeType, SourceRef, Strength, SupportKind,
};

/// Runtime configuration for graph persistence/metadata.
#[derive(Clone, Debug)]
pub struct GraphConfig {
    pub course_commit:         String,
    pub autosave_path:         PathBuf,
    pub autosave_secs:         u64,
    pub strict_quality:        bool,
    pub validation_timeout_ms: u64,
    pub skip_dedup_on_insert:  bool,
}

/// Primary graph type alias (stable indices survive deletions).
pub type GraphIx = u32;
pub type NodeId = NodeIndex<GraphIx>;
pub type EdgeId = EdgeIndex<GraphIx>;
pub type CurriculumGraph = StableGraph<NodePayload, EdgePayload, Directed, GraphIx>;

/// Payload attached to every node.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodePayload {
    pub logical_id: Uuid,
    pub slug:       String,
    pub kind:       NodeKind,
    pub tags:       Vec<String>,
}

/// Distinguishes knowledge/assessment/LO nodes from discourse TeachingSteps.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum NodeKind {
    Knowledge(KnowledgeNode),
    TeachingStep(TeachingStepNode),
}

/// Knowledge node payload (covers LOs and assessments via `knowledge_type`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KnowledgeNode {
    pub title: String,
    pub statement: String,
    pub knowledge_type: KnowledgeType,
    pub source_refs: Vec<SourceRef>,
    pub confidence: f32,
    pub rubric_criteria: Vec<String>,
    pub construct_irrelevant_demands: Vec<String>,
    pub grain_level: Option<GrainLevel>,
    pub intrinsic_load: Option<IntrinsicLoad>,
    pub introduction_scope: IntroductionScope,
}

/// Teaching step payload for discourse layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TeachingStepNode {
    pub title:       String,
    pub statement:   String,
    pub purpose:     TeachingPurpose,
    pub method_tags: Vec<String>,
    pub episode:     String,
    pub source_refs: Vec<SourceRef>,
    #[serde(default)]
    pub rationale:   Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TeachingPurpose {
    Setup,
    Idea,
    Use,
    Consolidate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GrainLevel {
    Macro,
    Mid,
    Micro,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum IntrinsicLoad {
    Low,
    Medium,
    High,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum IntroductionScope {
    InCourse,
    Prior,
    External,
}

/// Edge payload plus layer discriminator.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct EdgePayload {
    pub kind:       EdgeKind,
    pub confidence: f32,
    #[serde(default)]
    pub conflicts:  Vec<EdgeConflict>,
}

impl EdgePayload {
    pub fn new(kind: EdgeKind, confidence: f32) -> Self {
        Self {
            kind,
            confidence,
            conflicts: Vec::new(),
        }
    }
}

/// Multiplex edge types.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub enum EdgeKind {
    Requires(RequiresAttrs),
    Supports(SupportsAttrs),
    Assesses(AssessesAttrs),
    Precedes(PrecedesAttrs),
    Anchors(AnchorsAttrs),
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct RequiresAttrs {
    pub strength:      Strength,
    pub rationale:     String,
    pub evidence_refs: Vec<SourceRef>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct EdgeConflict {
    pub kind:       EdgeKind,
    pub confidence: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SupportsAttrs {
    pub support_kind:    SupportKind,
    pub intended_effect: IntendedEffect,
    pub case_tag:        Option<CaseTag>,
    pub coverage_tags:   Vec<String>,
    pub evidence_refs:   Vec<SourceRef>,
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum CaseTag {
    Typical,
    Edge,
    ErrorCase,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AssessesAttrs {
    pub evidence_link: EvidenceLink,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct PrecedesAttrs {
    pub episode: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AnchorsAttrs {
    pub impact: AnchorImpact,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AnchorImpact {
    Introduce,
    Use,
    Refine,
    Motivate,
    Target,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InvariantCode {
    StatementEmpty,
    ProvenanceRevision,
    RequiresDag,
    Fadeability,
    LoTargetAssessment,
    LoReachability,
    RubricCoverage,
    RubricUnusedObservationFeatures,
    RubricDrift,
    ExampleMinimums,
    ProceduralPractice,
    PurityIntendedMissing,
    PurityExtraneous,
    BorrowAhead,
    DiscourseOrphan,
    TeachingStepAnchorOrRationale,
    IntroduceAnchor,
    GrainOverbundled,
    GrainFragment,
    IntrinsicLoadMismatch,
    IntrinsicLoadSupport,
}

impl InvariantCode {
    pub fn as_str(self) -> &'static str {
        match self {
            InvariantCode::StatementEmpty => "statement_empty",
            InvariantCode::ProvenanceRevision => "provenance_revision",
            InvariantCode::RequiresDag => "requires_dag",
            InvariantCode::Fadeability => "fadeability",
            InvariantCode::LoTargetAssessment => "lo_target_assessment",
            InvariantCode::LoReachability => "lo_reachability",
            InvariantCode::RubricCoverage => "rubric_coverage",
            InvariantCode::RubricUnusedObservationFeatures => "rubric_unused_observation_features",
            InvariantCode::RubricDrift => "rubric_drift",
            InvariantCode::ExampleMinimums => "example_minimums",
            InvariantCode::ProceduralPractice => "procedural_practice",
            InvariantCode::PurityIntendedMissing => "purity_intended_missing",
            InvariantCode::PurityExtraneous => "purity_extraneous",
            InvariantCode::BorrowAhead => "borrow_ahead",
            InvariantCode::DiscourseOrphan => "discourse_orphan",
            InvariantCode::TeachingStepAnchorOrRationale => "teaching_step_anchor_or_rationale",
            InvariantCode::IntroduceAnchor => "introduce_anchor",
            InvariantCode::GrainOverbundled => "grain_overbundled",
            InvariantCode::GrainFragment => "grain_fragment",
            InvariantCode::IntrinsicLoadMismatch => "intrinsic_load_mismatch",
            InvariantCode::IntrinsicLoadSupport => "intrinsic_load_support",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InvariantViolation {
    pub code:    InvariantCode,
    pub message: String,
}

#[derive(thiserror::Error, Debug)]
pub enum GraphOperationalError {
    #[error("graph invariant validation timed out after {timeout_ms} ms")]
    InvariantTimeout { timeout_ms: u64 },
    #[error("graph invariant validation task failed: {message}")]
    InvariantTaskFailed { message: String },
}

/// Errors returned by GraphService.
#[derive(thiserror::Error, Debug)]
pub enum GraphError {
    #[error("slug `{0}` not found")]
    MissingSlug(String),
    #[error("invalid edge endpoints for {edge}: from={from:?}, to={to:?}")]
    InvalidEndpoints {
        edge: &'static str,
        from: Option<NodeKindPreview>,
        to:   Option<NodeKindPreview>,
    },
    #[error("edge would create a cycle in requires layer: {cycle_slugs:?}")]
    RequiresCycle { cycle_slugs: Vec<String> },
    #[error("validator error: {0}")]
    Schema(String),
    #[error(transparent)]
    Operational(#[from] GraphOperationalError),
    #[error("graph invariant violation(s): {violations:?}")]
    InvariantViolation { violations: Vec<InvariantViolation> },
}

/// Lightweight snapshot of a node's kind used for error messages.
#[derive(Clone, Debug)]
pub enum NodeKindPreview {
    Knowledge(KnowledgeType),
    TeachingStep,
}

impl NodeKindPreview {
    pub fn from_node(node: &NodeKind) -> Self {
        match node {
            NodeKind::Knowledge(k) => NodeKindPreview::Knowledge(k.knowledge_type),
            NodeKind::TeachingStep(_) => NodeKindPreview::TeachingStep,
        }
    }
}

/// Helper to validate source refs and surface warnings consistently.
pub(crate) fn validate_source_refs(spans: &[SourceRef]) -> Result<(), GraphError> {
    for span in spans {
        crate::schema::validate::validate_source_ref(span)
            .map_err(|e| GraphError::Schema(e.to_string()))?;
    }
    Ok(())
}
