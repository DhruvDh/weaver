use kameo::{
    Actor,
    message::{Context, Message},
};
use tracing::warn;

use crate::{
    llm::{LlmClient, LlmError},
    model::{ALLOWED_TAGS, Granularity, MAX_NODE_LEVEL, NodeKind, NodeProposal},
};

/// Configuration for generating node proposals.
#[derive(Debug, Clone)]
pub struct NodeGeneratorConfig {
    pub use_llm: bool,
    pub default_concepts: usize,
    pub default_learning_outcomes: usize,
}

impl Default for NodeGeneratorConfig {
    fn default() -> Self {
        Self {
            use_llm: false,
            default_concepts: 25,
            default_learning_outcomes: 5,
        }
    }
}

/// Request a batch of node proposals.
pub struct GenerateNodes {
    pub concepts: usize,
    pub learning_outcomes: usize,
}

/// Actor responsible for producing node proposals via LLM or deterministic fallback.
#[derive(Debug, Actor)]
pub struct NodeGenerator {
    config: NodeGeneratorConfig,
    llm: Option<LlmClient>,
}

impl NodeGenerator {
    pub fn new(config: NodeGeneratorConfig) -> Self {
        let llm = match LlmClient::new(config.use_llm) {
            Ok(client) => Some(client),
            Err(LlmError::Disabled) => None,
            Err(err) => {
                warn!(error = %err, "node_generator.llm_unavailable");
                None
            }
        };
        Self { config, llm }
    }

    fn fallback_nodes(concepts: usize, learning_outcomes: usize) -> Vec<NodeProposal> {
        const CONCEPT_SUBJECTS: [&str; 6] = [
            "Students",
            "Learners",
            "Developers",
            "Programmers",
            "Teams",
            "Analysts",
        ];
        const CONCEPT_VERBS: [&str; 6] =
            ["map", "trace", "refine", "compare", "document", "simulate"];
        const CONCEPT_OBJECTS: [&str; 6] = [
            "data flow through recursive helpers",
            "contract edge cases in failing tests",
            "design trade-offs across module boundaries",
            "error handling strategies in Rust services",
            "state transitions across program phases",
            "naming conventions for public APIs",
        ];
        const CONCEPT_PURPOSES: [&str; 6] = [
            "surface boundary cases early",
            "share intent with collaborators",
            "select appropriate abstractions",
            "sustain maintainable codebases",
            "reason about runtime performance",
            "guide incremental refactoring",
        ];

        const LO_VERBS: [&str; 6] = [
            "trace",
            "justify",
            "refactor",
            "explain",
            "coordinate",
            "debug",
        ];
        const LO_OBJECTS: [&str; 6] = [
            "recursion invariants across modules",
            "test suites that capture tricky failures",
            "design recipe steps for new problems",
            "interface contracts alongside implementations",
            "learning pathways that sequence key ideas",
            "stateful workflows using structured logs",
        ];
        const LO_CONTEXTS: [&str; 6] = [
            "with evidence from runnable examples",
            "while articulating trade-offs to peers",
            "without relying on external hints",
            "under varying time constraints",
            "so teammates can adopt the approach confidently",
            "while respecting performance budgets",
        ];
        const MAX_TAGS: usize = 3;

        let mut proposals = Vec::with_capacity(concepts + learning_outcomes);

        for i in 0..concepts {
            let subject = CONCEPT_SUBJECTS[i % CONCEPT_SUBJECTS.len()];
            let verb = CONCEPT_VERBS[(i / CONCEPT_SUBJECTS.len()) % CONCEPT_VERBS.len()];
            let object = CONCEPT_OBJECTS
                [(i / (CONCEPT_SUBJECTS.len() * CONCEPT_VERBS.len())) % CONCEPT_OBJECTS.len()];
            let purpose = CONCEPT_PURPOSES[(i
                / (CONCEPT_SUBJECTS.len() * CONCEPT_VERBS.len() * CONCEPT_OBJECTS.len()))
                % CONCEPT_PURPOSES.len()];
            let sentence = format!("{subject} {verb} {object} to {purpose}.");
            let level = (i % ((MAX_NODE_LEVEL as usize) + 1)) as u8;
            let tags = Self::fallback_tags(i, 1, MAX_TAGS);
            proposals.push(NodeProposal {
                kind: NodeKind::Concept,
                granularity: Granularity::Sentence,
                level,
                text: sentence,
                tags,
            });
        }

        for i in 0..learning_outcomes {
            let verb = LO_VERBS[i % LO_VERBS.len()];
            let object = LO_OBJECTS[(i / LO_VERBS.len()) % LO_OBJECTS.len()];
            let context =
                LO_CONTEXTS[(i / (LO_VERBS.len() * LO_OBJECTS.len())) % LO_CONTEXTS.len()];
            let sentence = format!("I can {verb} {object} {context}.");
            let mut level = MAX_NODE_LEVEL.saturating_sub(1) + (i as u8 % 2);
            if level > MAX_NODE_LEVEL {
                level = MAX_NODE_LEVEL;
            }
            let tags = Self::fallback_tags(concepts + i, 2, MAX_TAGS);
            proposals.push(NodeProposal {
                kind: NodeKind::LearningOutcome,
                granularity: Granularity::Sentence,
                level,
                text: sentence,
                tags,
            });
        }

        proposals
    }

    fn fallback_tags(seed: usize, desired: usize, max_tags: usize) -> Option<Vec<String>> {
        if desired == 0 {
            return None;
        }
        let mut tags = Vec::new();
        for offset in 0..desired {
            if tags.len() >= max_tags {
                break;
            }
            let tag = ALLOWED_TAGS[(seed + offset) % ALLOWED_TAGS.len()].to_string();
            if !tags.contains(&tag) {
                tags.push(tag);
            }
        }
        if tags.is_empty() { None } else { Some(tags) }
    }
}

impl Message<GenerateNodes> for NodeGenerator {
    type Reply = Vec<NodeProposal>;

    fn handle(
        &mut self,
        msg: GenerateNodes,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> impl std::future::Future<Output = Self::Reply> + Send {
        let llm = self.llm.clone();
        let concepts = if msg.concepts == 0 {
            self.config.default_concepts
        } else {
            msg.concepts
        };
        let learning_outcomes = if msg.learning_outcomes == 0 {
            self.config.default_learning_outcomes
        } else {
            msg.learning_outcomes
        };

        async move {
            if let Some(client) = llm {
                match client.generate_nodes(concepts, learning_outcomes).await {
                    Ok(nodes) if !nodes.is_empty() => return nodes,
                    Ok(_) => {
                        warn!("node_generator.llm_returned_empty_batch");
                    }
                    Err(err) => {
                        warn!(error = %err, "node_generator.llm_failed");
                    }
                }
            }

            NodeGenerator::fallback_nodes(concepts, learning_outcomes)
        }
    }
}
