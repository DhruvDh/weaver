use std::collections::HashSet;

use kameo::{
    Actor,
    message::{Context, Message},
};
use tracing::warn;
use uuid::Uuid;

use crate::{
    llm::{LlmClient, LlmError},
    model::{EdgeProposal, NodeKind, Relation},
};

/// Configuration for generating edge proposals.
#[derive(Debug, Clone)]
pub struct EdgeGeneratorConfig {
    pub use_llm: bool,
    pub default_target_edges: usize,
}

impl Default for EdgeGeneratorConfig {
    fn default() -> Self {
        Self {
            use_llm: false,
            default_target_edges: 40,
        }
    }
}

/// Request a batch of edge proposals.
pub struct GenerateEdges {
    pub inventory: Vec<(Uuid, NodeKind, u8, String, Option<Vec<String>>)>,
    pub target_edges: usize,
}

/// Actor responsible for producing edge proposals.
#[derive(Debug, Actor)]
pub struct EdgeGenerator {
    config: EdgeGeneratorConfig,
    llm: Option<LlmClient>,
}

impl EdgeGenerator {
    pub fn new(config: EdgeGeneratorConfig) -> Self {
        let llm = match LlmClient::new(config.use_llm) {
            Ok(client) => Some(client),
            Err(LlmError::Disabled) => None,
            Err(err) => {
                warn!(error = %err, "edge_generator.llm_unavailable");
                None
            }
        };
        Self { config, llm }
    }

    fn fallback_edges(
        inventory: &[(Uuid, NodeKind, u8, String, Option<Vec<String>>)],
        target_edges: usize,
    ) -> Vec<EdgeProposal> {
        let mut edges = Vec::new();
        let mut seen = HashSet::new();

        let mut concepts: Vec<_> = inventory
            .iter()
            .filter(|(_, kind, _, _, _)| matches!(kind, NodeKind::Concept))
            .cloned()
            .collect();
        let mut learning_outcomes: Vec<_> = inventory
            .iter()
            .filter(|(_, kind, _, _, _)| matches!(kind, NodeKind::LearningOutcome))
            .cloned()
            .collect();

        concepts.sort_by(|a, b| a.3.to_lowercase().cmp(&b.3.to_lowercase()));
        learning_outcomes.sort_by(|a, b| a.3.to_lowercase().cmp(&b.3.to_lowercase()));

        if !concepts.is_empty() {
            for (lo_index, (lo_id, _, _, lo_text, _)) in learning_outcomes.iter().enumerate() {
                let supports_needed = 3;
                for offset in 0..supports_needed {
                    let concept = &concepts[(lo_index + offset * 2) % concepts.len()];
                    let key = (concept.0, *lo_id, Relation::Supports);
                    if seen.insert(key) {
                        edges.push(EdgeProposal {
                            relation: Relation::Supports,
                            from_id: concept.0,
                            to_id: *lo_id,
                            rationale: format!(
                                "{} underpins {}",
                                truncate_sentence(&concept.3),
                                truncate_sentence(lo_text)
                            ),
                        });
                    }
                    if edges.len() >= target_edges {
                        return edges;
                    }
                }
            }
        }

        for window in concepts.windows(2) {
            if let [from, to] = window {
                let key = (from.0, to.0, Relation::PrerequisiteFor);
                if seen.insert(key) {
                    edges.push(EdgeProposal {
                        relation: Relation::PrerequisiteFor,
                        from_id: from.0,
                        to_id: to.0,
                        rationale: format!(
                            "{} prepares learners for {}",
                            truncate_sentence(&from.3),
                            truncate_sentence(&to.3)
                        ),
                    });
                }
            }
            if edges.len() >= target_edges {
                return edges;
            }
        }

        edges
    }
}

impl Message<GenerateEdges> for EdgeGenerator {
    type Reply = Vec<EdgeProposal>;

    fn handle(
        &mut self,
        msg: GenerateEdges,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> impl std::future::Future<Output = Self::Reply> + Send {
        let llm = self.llm.clone();
        let fallback_target = if msg.target_edges == 0 {
            self.config.default_target_edges
        } else {
            msg.target_edges
        };

        async move {
            if let Some(client) = llm {
                match client.generate_edges(&msg.inventory, fallback_target).await {
                    Ok(edges) if !edges.is_empty() => return edges,
                    Ok(_) => warn!("edge_generator.llm_returned_empty_batch"),
                    Err(err) => warn!(error = %err, "edge_generator.llm_failed"),
                }
            }

            EdgeGenerator::fallback_edges(&msg.inventory, fallback_target)
        }
    }
}

pub(crate) fn truncate_sentence(sentence: &str) -> String {
    const LIMIT: usize = 80;
    let cleaned = sentence.trim();
    let mut chars = cleaned.chars();
    let mut truncated = String::with_capacity(LIMIT + 3);

    for _ in 0..LIMIT {
        match chars.next() {
            Some(ch) => truncated.push(ch),
            None => return truncated,
        }
    }

    if chars.next().is_some() {
        truncated.push_str("...");
    }

    truncated
}
