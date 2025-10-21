use petgraph::graph::NodeIndex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const MAX_NODE_LEVEL: u8 = 3;
pub const ALLOWED_TAGS: &[&str] = &[
    "design_recipe",
    "contract",
    "purpose",
    "tests",
    "stub",
    "implementation",
    "refactor",
];

/// Type of node in the learning graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, JsonSchema)]
pub enum NodeKind {
    Concept,
    LearningOutcome,
}

/// Content granularity for nodes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, JsonSchema)]
pub enum Granularity {
    Sentence,
}

/// Fully validated node stored in the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: Uuid,
    pub kind: NodeKind,
    pub granularity: Granularity,
    pub level: u8,
    pub text: String,
    pub tags: Option<Vec<String>>,
}

/// Relation type between two nodes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, JsonSchema)]
pub enum Relation {
    PrerequisiteFor,
    Supports,
}

/// Fully validated edge stored in the graph.
#[derive(Debug, Clone)]
pub struct Edge {
    pub from: NodeIndex,
    pub to: NodeIndex,
    pub relation: Relation,
    pub rationale: String,
}

/// Proposed node emitted by a generator (LLM or fallback).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct NodeProposal {
    pub kind: NodeKind,
    pub granularity: Granularity,
    pub level: u8,
    pub text: String,
    pub tags: Option<Vec<String>>,
}

/// Proposed edge emitted by a generator (LLM or fallback).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EdgeProposal {
    pub relation: Relation,
    pub from_id: Uuid,
    pub to_id: Uuid,
    pub rationale: String,
}

/// Decision result returned by the GraphAdder for node and edge proposals.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Decision {
    pub accepted: bool,
    pub reason: Option<String>,
    pub assigned_id: Option<Uuid>,
}

impl Decision {
    /// Construct an accepted decision with an optional assigned UUID.
    pub fn accepted(assigned_id: Option<Uuid>) -> Self {
        Self {
            accepted: true,
            reason: None,
            assigned_id,
        }
    }

    /// Construct a rejected decision with the provided reason.
    pub fn rejected(reason: impl Into<String>) -> Self {
        Self {
            accepted: false,
            reason: Some(reason.into()),
            assigned_id: None,
        }
    }
}

/// Normalizes text for deduplication: trim, collapse whitespace, lowercase.
pub fn normalize_text(text: &str) -> String {
    text.split_whitespace()
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Trim leading/trailing whitespace and collapse interior whitespace.
pub fn clean_text(text: &str) -> String {
    text.split_whitespace()
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::normalize_text;

    #[test]
    fn normalize_text_lowercases_and_collapses_whitespace() {
        let input = "  Hello   World ";
        let normalized = normalize_text(input);
        assert_eq!(normalized, "hello world");
    }
}
