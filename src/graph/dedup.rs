use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
};

use strsim::jaro_winkler;

use crate::{
    constants::{DEDUP_INSERT_SIMHASH_DISTANCE, DEDUP_TITLE_SIMILARITY},
    graph::{CurriculumGraph, KnowledgeNode, NodeId, NodeKind},
    schema::types::KnowledgeType,
};

/// Result of a duplicate detection probe for a new or updated node.
#[derive(Debug, Clone)]
pub enum DuplicateCheck {
    Exact { existing: NodeId },
    HighSimilarity { candidates: Vec<(NodeId, f64)> },
    SimilarTitle { existing: NodeId, similarity: f64 },
    Unique,
}

/// Maintains lightweight fingerprints to spot duplicates quickly during
/// insertion/update.
#[derive(Default)]
pub struct NodeDeduplicator {
    content_hashes: HashMap<(KnowledgeType, u64), NodeId>,
    fingerprints:   HashMap<NodeId, u64>,
}

impl NodeDeduplicator {
    pub fn new() -> Self {
        Self {
            content_hashes: HashMap::new(),
            fingerprints:   HashMap::new(),
        }
    }

    pub fn rebuild(&mut self, graph: &CurriculumGraph) {
        self.content_hashes.clear();
        self.fingerprints.clear();
        for node in graph.node_indices() {
            if let NodeKind::Knowledge(k) = &graph[node].kind {
                self.record(node, k);
            }
        }
    }

    pub fn record(&mut self, id: NodeId, node: &KnowledgeNode) {
        let hash = normalized_statement_hash(&node.statement);
        self.content_hashes.insert((node.knowledge_type, hash), id);
        let fp = simhash(&node.statement);
        self.fingerprints.insert(id, fp);
    }

    pub fn remove(&mut self, id: NodeId) {
        self.fingerprints.remove(&id);
        self.content_hashes.retain(|_, v| *v != id);
    }

    pub fn check(
        &self,
        title: &str,
        statement: &str,
        knowledge_type: KnowledgeType,
        graph: &CurriculumGraph,
        exclude: Option<NodeId>,
    ) -> DuplicateCheck {
        let hash = normalized_statement_hash(statement);
        if let Some(existing) = self.content_hashes.get(&(knowledge_type, hash))
            && Some(*existing) != exclude
        {
            return DuplicateCheck::Exact {
                existing: *existing,
            };
        }

        let fp = simhash(statement);
        let mut similar: Vec<(NodeId, f64)> = self
            .fingerprints
            .iter()
            .filter_map(|(id, existing_fp)| {
                if Some(id) == exclude.as_ref() {
                    return None;
                }
                if let NodeKind::Knowledge(k) = &graph[*id].kind
                    && k.knowledge_type == knowledge_type
                {
                    let dist = hamming_distance(fp, *existing_fp);
                    if dist <= DEDUP_INSERT_SIMHASH_DISTANCE {
                        let similarity = 1.0 - (dist as f64 / 64.0);
                        return Some((*id, similarity));
                    }
                }
                None
            })
            .collect();
        similar.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if !similar.is_empty() {
            return DuplicateCheck::HighSimilarity {
                candidates: similar,
            };
        }

        let normalized_title = title.trim().to_ascii_lowercase();
        let mut best: Option<(NodeId, f64)> = None;
        for node in graph.node_indices() {
            if Some(node) == exclude {
                continue;
            }
            let NodeKind::Knowledge(k) = &graph[node].kind else {
                continue;
            };
            if k.knowledge_type != knowledge_type {
                continue;
            }
            let existing_title = k.title.to_ascii_lowercase();
            let score = jaro_winkler(&normalized_title, &existing_title);
            if score > DEDUP_TITLE_SIMILARITY {
                let candidate = (node, score);
                if let Some(prev) = best {
                    if score > prev.1 {
                        best = Some(candidate);
                    }
                } else {
                    best = Some(candidate);
                }
            } else {
                continue;
            }
        }
        if let Some((existing, similarity)) = best {
            return DuplicateCheck::SimilarTitle {
                existing,
                similarity,
            };
        }

        DuplicateCheck::Unique
    }

    pub fn fingerprints(&self) -> &HashMap<NodeId, u64> {
        &self.fingerprints
    }
}

pub(crate) fn normalized_statement_hash(statement: &str) -> u64 {
    let normalized = statement
        .split_whitespace()
        .map(|part| part.to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(" ");
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    normalized.hash(&mut hasher);
    hasher.finish()
}

pub(crate) fn simhash(statement: &str) -> u64 {
    let mut bits = [0i64; 64];
    for token in statement.split_whitespace() {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        token.to_ascii_lowercase().hash(&mut hasher);
        let hash = hasher.finish();
        for (i, bit) in bits.iter_mut().enumerate() {
            if (hash >> i) & 1 == 1 {
                *bit += 1;
            } else {
                *bit -= 1;
            }
        }
    }
    let mut fp = 0u64;
    for (i, bit) in bits.iter().enumerate() {
        if *bit >= 0 {
            fp |= 1 << i;
        }
    }
    fp
}

pub(crate) fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use super::*;

    #[test]
    fn detects_exact_duplicate() {
        let mut graph = CurriculumGraph::default();
        let node = KnowledgeNode {
            title: "Loop basics".into(),
            statement: "Loops repeat work".into(),
            knowledge_type: KnowledgeType::Conceptual,
            source_refs: vec![],
            confidence: 1.0,
            rubric_criteria: vec![],
            construct_irrelevant_demands: vec![],
            grain_level: None,
            intrinsic_load: None,
            introduction_scope: crate::graph::IntroductionScope::InCourse,
        };
        let id = graph.add_node(crate::graph::NodePayload {
            logical_id: Uuid::new_v4(),
            slug:       "c.loop_basics".into(),
            kind:       crate::graph::NodeKind::Knowledge(node.clone()),
            tags:       vec![],
        });
        let mut dedup = NodeDeduplicator::new();
        dedup.record(id, &node);
        let check = dedup.check(
            "Loop basics",
            "Loops repeat work",
            KnowledgeType::Conceptual,
            &graph,
            None,
        );
        assert!(matches!(check, DuplicateCheck::Exact { existing } if existing == id));
    }
}
