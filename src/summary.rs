use std::collections::HashMap;

use serde::Serialize;
use uuid::Uuid;

use crate::{
    graph::GraphStore,
    model::{NodeKind, Relation},
};

/// Aggregate statistics about the current graph state.
#[derive(Debug, Clone, Serialize, kameo::Reply)]
pub struct Summary {
    pub total_nodes:           usize,
    pub concepts:              usize,
    pub learning_outcomes:     usize,
    pub total_edges:           usize,
    pub prerequisite_edges:    usize,
    pub supports_edges:        usize,
    pub prerequisite_dag_ok:   bool,
    pub top_learning_outcomes: Vec<TopLearningOutcome>,
}

/// Lightweight view of a learning outcome ranked by inbound supports.
#[derive(Debug, Clone, Serialize)]
pub struct TopLearningOutcome {
    pub id:       Uuid,
    pub text:     String,
    pub supports: usize,
}

impl Summary {
    /// Create an empty summary placeholder.
    pub fn empty() -> Self {
        Self {
            total_nodes:           0,
            concepts:              0,
            learning_outcomes:     0,
            total_edges:           0,
            prerequisite_edges:    0,
            supports_edges:        0,
            prerequisite_dag_ok:   true,
            top_learning_outcomes: Vec::new(),
        }
    }

    /// Build a summary snapshot from the current graph store.
    pub fn from_store(store: &GraphStore) -> Self {
        let mut summary = Summary::empty();

        for index in store.node_indices() {
            if let Some(node) = store.node(index) {
                summary.total_nodes += 1;
                match node.kind {
                    NodeKind::Concept => summary.concepts += 1,
                    NodeKind::LearningOutcome => summary.learning_outcomes += 1,
                }
            }
        }

        summary.total_edges = store.edge_indices().count();
        summary.prerequisite_edges = store.prerequisite_edges();
        summary.supports_edges = store.supports_edges();
        summary.prerequisite_dag_ok = store.is_prerequisite_dag();

        let mut support_counts: HashMap<Uuid, usize> = HashMap::new();
        for edge_index in store.edge_indices() {
            if let Some(edge) = store.edge_weight(edge_index)
                && matches!(edge.relation, Relation::Supports)
                && let Some(node) = store.node(edge.to)
            {
                *support_counts.entry(node.id).or_default() += 1;
            }
        }

        let mut learning_outcomes = Vec::new();
        for index in store.node_indices() {
            if let Some(node) = store.node(index)
                && matches!(node.kind, NodeKind::LearningOutcome)
            {
                let supports = support_counts.get(&node.id).copied().unwrap_or(0);
                learning_outcomes.push(TopLearningOutcome {
                    id: node.id,
                    text: node.text.clone(),
                    supports,
                });
            }
        }

        learning_outcomes.sort_by(|a, b| {
            b.supports
                .cmp(&a.supports)
                .then_with(|| a.text.cmp(&b.text))
        });
        learning_outcomes.truncate(5);

        summary.top_learning_outcomes = learning_outcomes;
        summary
    }
}
