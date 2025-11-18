use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};

/// Primary knowledge categories from the white paper ontology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeType {
    Factual,
    Conceptual,
    Procedural,
    Metacognitive,
    LearningOutcome,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Strength {
    Necessary,
    Strong,
    Helpful,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SupportKind {
    WorkedExample,
    Analogy,
    Counterexample,
    MisconceptionFix,
    StrategyHint,
    RubricNote,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntendedEffect {
    ReduceExtraneousLoad,
    IncreaseGermaneLoad,
    Motivate,
    Contrast,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssessmentScope {
    Target,
    Enabling,
}

/// Reference to a source span inside the PreTeXt repository (path + line range
/// + revision).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRef {
    pub path:       String,
    pub start_line: u32,
    pub end_line:   u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub revision:   Option<String>,
}

impl SourceRef {
    pub fn new(
        path: impl Into<String>,
        start_line: u32,
        end_line: u32,
        revision: Option<String>,
    ) -> Self {
        Self {
            path: path.into(),
            start_line,
            end_line,
            revision,
        }
    }
}

/// Evidence Centered Design payload stored on `assesses` edges.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceLink {
    pub claim:                String,
    #[serde(default)]
    pub observation_features: Vec<String>,
    pub scope:                AssessmentScope,
}
