use std::{fmt, str::FromStr};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Primary knowledge categories from the white paper ontology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Hash)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeType {
    #[serde(alias = "Factual")]
    Factual,
    #[serde(alias = "Conceptual")]
    Conceptual,
    #[serde(alias = "Procedural")]
    Procedural,
    #[serde(alias = "Metacognitive")]
    Metacognitive,
    #[serde(alias = "LearningOutcome")]
    LearningOutcome,
    #[serde(alias = "AssessmentItem")]
    AssessmentItem,
}

impl KnowledgeType {
    pub fn is_learning_outcome(self) -> bool {
        matches!(self, KnowledgeType::LearningOutcome)
    }

    pub fn is_assessment_item(self) -> bool {
        matches!(self, KnowledgeType::AssessmentItem)
    }

    pub fn is_instructional_knowledge(self) -> bool {
        matches!(
            self,
            KnowledgeType::Factual
                | KnowledgeType::Conceptual
                | KnowledgeType::Procedural
                | KnowledgeType::Metacognitive
        )
    }
}

impl fmt::Display for KnowledgeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            KnowledgeType::Factual => "factual",
            KnowledgeType::Conceptual => "conceptual",
            KnowledgeType::Procedural => "procedural",
            KnowledgeType::Metacognitive => "metacognitive",
            KnowledgeType::LearningOutcome => "learning_outcome",
            KnowledgeType::AssessmentItem => "assessment_item",
        })
    }
}

impl FromStr for KnowledgeType {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "factual" => Ok(KnowledgeType::Factual),
            "conceptual" => Ok(KnowledgeType::Conceptual),
            "procedural" => Ok(KnowledgeType::Procedural),
            "metacognitive" => Ok(KnowledgeType::Metacognitive),
            "learning_outcome" => Ok(KnowledgeType::LearningOutcome),
            "assessment_item" => Ok(KnowledgeType::AssessmentItem),
            _ => Err("unknown knowledge_type"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Strength {
    #[serde(alias = "Necessary")]
    Necessary,
    #[serde(alias = "Strong")]
    Strong,
    #[serde(alias = "Helpful")]
    Helpful,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SupportKind {
    #[serde(alias = "WorkedExample", alias = "Worked_Example")]
    WorkedExample,
    #[serde(alias = "Analogy")]
    Analogy,
    #[serde(alias = "Counterexample", alias = "CounterExample")]
    Counterexample,
    #[serde(alias = "MisconceptionFix", alias = "Misconception_Fix")]
    MisconceptionFix,
    #[serde(alias = "StrategyHint", alias = "Strategy_Hint")]
    StrategyHint,
    #[serde(alias = "RubricNote", alias = "Rubric_Note")]
    RubricNote,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum IntendedEffect {
    #[serde(alias = "ReduceExtraneousLoad", alias = "Reduce_Extraneous_Load")]
    ReduceExtraneousLoad,
    #[serde(alias = "IncreaseGermaneLoad", alias = "Increase_Germane_Load")]
    IncreaseGermaneLoad,
    #[serde(alias = "Motivate")]
    Motivate,
    #[serde(alias = "Contrast")]
    Contrast,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AssessmentScope {
    #[serde(alias = "Target")]
    Target,
    #[serde(alias = "Enabling")]
    Enabling,
}

/// Reference to a source span inside the PreTeXt repository (path + line range
/// + revision).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SourceRef {
    #[schemars(
        description = "Workspace-relative path to the source file (e.g., source/Unit/1.ptx)."
    )]
    pub path:       String,
    #[schemars(description = "Starting line (1-based, inclusive) for the cited span.")]
    pub start_line: u32,
    #[schemars(description = "Ending line (1-based, inclusive) for the cited span.")]
    pub end_line:   u32,
    #[serde(default)]
    #[schemars(
        description = "DO NOT PROVIDE - auto-filled from course_commit. Leave empty or omit."
    )]
    pub revision:   String,
}

impl SourceRef {
    pub fn new(
        path: impl Into<String>,
        start_line: u32,
        end_line: u32,
        revision: impl Into<String>,
    ) -> Self {
        Self {
            path: path.into(),
            start_line,
            end_line,
            revision: revision.into(),
        }
    }
}

/// Evidence Centered Design payload stored on `assesses` edges.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct EvidenceLink {
    pub claim:                String,
    #[serde(default)]
    pub observation_features: Vec<String>,
    pub scope:                AssessmentScope,
}
