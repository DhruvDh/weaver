use std::{collections::HashSet, future::ready};

use kameo::{
    Actor,
    message::{Context, Message},
};
use tokio::sync::mpsc::UnboundedSender;
use tracing::{info, warn};
use uuid::Uuid;

use crate::{
    graph::GraphStore,
    model::{
        ALLOWED_TAGS, Decision, Edge, EdgeProposal, Granularity, MAX_NODE_LEVEL, Node, NodeKind,
        NodeProposal, Relation, clean_text, normalize_text,
    },
    summary::{Summary, TopLearningOutcome},
    viz::Event,
};

const MAX_TAGS_PER_NODE: usize = 3;

/// Message requesting that new nodes be considered.
pub struct AddNodes(pub Vec<NodeProposal>);

/// Message requesting that new edges be considered.
pub struct AddEdges(pub Vec<EdgeProposal>);

/// Message requesting current inventory of accepted nodes.
#[derive(Default)]
pub struct Inventory;

/// Message requesting summary statistics.
#[derive(Default)]
pub struct Summarize;

/// Message requesting DOT export of the graph.
pub struct ExportDot;

/// Primary mutator actor that validates and applies graph updates.
#[derive(Debug, Actor)]
pub struct GraphAdder {
    store: GraphStore,
    event_sender: Option<UnboundedSender<Event>>,
}

impl GraphAdder {
    pub fn with_event_sender(
        store: GraphStore,
        event_sender: Option<UnboundedSender<Event>>,
    ) -> Self {
        Self {
            store,
            event_sender,
        }
    }

    fn handle_add_nodes(&mut self, proposals: Vec<NodeProposal>) -> Vec<Decision> {
        let mut decisions = Vec::with_capacity(proposals.len());
        let mut batch_seen = HashSet::new();

        for proposal in proposals {
            let decision = self.validate_and_add_node(proposal, &mut batch_seen);
            decisions.push(decision);
        }

        decisions
    }

    fn validate_and_add_node(
        &mut self,
        proposal: NodeProposal,
        batch_seen: &mut HashSet<String>,
    ) -> Decision {
        let NodeProposal {
            kind,
            granularity,
            level,
            text,
            tags,
        } = proposal;

        if granularity != Granularity::Sentence {
            let reason = "granularity must be sentence";
            warn!(reason = reason, "node.rejected");
            self.emit_event(Event::NodeRejected {
                text,
                reason: reason.to_string(),
            });
            return Decision::rejected(reason);
        }

        if level > MAX_NODE_LEVEL {
            let reason = "level must be between 0 and 3";
            warn!(reason = reason, "node.rejected");
            self.emit_event(Event::NodeRejected {
                text,
                reason: reason.to_string(),
            });
            return Decision::rejected(reason);
        }

        let cleaned_text = clean_text(&text);
        if cleaned_text.is_empty() {
            let reason = "node text is empty after trimming";
            warn!(reason = reason, "node.rejected");
            self.emit_event(Event::NodeRejected {
                text,
                reason: reason.to_string(),
            });
            return Decision::rejected(reason);
        }

        if !Self::is_single_sentence(&cleaned_text) {
            let reason = "node text must be a single sentence";
            warn!(reason = reason, "node.rejected");
            self.emit_event(Event::NodeRejected {
                text: cleaned_text,
                reason: reason.to_string(),
            });
            return Decision::rejected(reason);
        }

        if matches!(kind, NodeKind::LearningOutcome) {
            let lowered = cleaned_text.to_lowercase();
            if !(lowered.starts_with("i can ") || lowered.starts_with("students can ")) {
                let reason = "learning outcomes must start with 'I can' or 'Students can'";
                warn!(reason = reason, "node.rejected");
                self.emit_event(Event::NodeRejected {
                    text: cleaned_text,
                    reason: reason.to_string(),
                });
                return Decision::rejected(reason);
            }
        }

        if !batch_seen.insert(normalize_text(&cleaned_text)) {
            let reason = "duplicate node within batch";
            warn!(reason = reason, "node.rejected");
            self.emit_event(Event::NodeRejected {
                text: cleaned_text,
                reason: reason.to_string(),
            });
            return Decision::rejected(reason);
        }

        if self.store.find_by_text(&cleaned_text).is_some() {
            let reason = "duplicate node already present";
            warn!(reason = reason, "node.rejected");
            self.emit_event(Event::NodeRejected {
                text: cleaned_text,
                reason: reason.to_string(),
            });
            return Decision::rejected(reason);
        }

        let tags = Self::sanitize_tags(tags);

        let node = Node {
            id: Uuid::new_v4(),
            kind: kind.clone(),
            granularity,
            level,
            text: cleaned_text.clone(),
            tags: tags.clone(),
        };

        let node_id = node.id;
        self.store.add_node(node);

        info!(node_id = %node_id, kind = ?kind, level, "node.accepted");
        self.emit_event(Event::NodeAccepted {
            id: node_id,
            kind,
            level,
            tags,
            text: cleaned_text,
        });

        Decision::accepted(Some(node_id))
    }

    fn handle_add_edges(&mut self, proposals: Vec<EdgeProposal>) -> Vec<Decision> {
        let mut decisions = Vec::with_capacity(proposals.len());
        let mut batch_seen: HashSet<(Uuid, Uuid, Relation)> = HashSet::new();

        for proposal in proposals {
            let decision = self.validate_and_add_edge(proposal, &mut batch_seen);
            decisions.push(decision);
        }

        decisions
    }

    fn validate_and_add_edge(
        &mut self,
        proposal: EdgeProposal,
        batch_seen: &mut HashSet<(Uuid, Uuid, Relation)>,
    ) -> Decision {
        let EdgeProposal {
            relation,
            from_id,
            to_id,
            rationale,
        } = proposal;

        let key = (from_id, to_id, relation.clone());
        if !batch_seen.insert(key) {
            let reason = "duplicate edge within batch";
            warn!(reason = reason, relation = ?relation, "edge.rejected");
            self.emit_event(Event::EdgeRejected {
                relation,
                reason: reason.to_string(),
            });
            return Decision::rejected(reason);
        }

        let Some(from_index) = self.store.find_by_id(&from_id) else {
            let reason = format!("unknown from_id {}", from_id);
            warn!(relation = ?relation, from_id = %from_id, reason = %reason, "edge.rejected");
            self.emit_event(Event::EdgeRejected {
                relation,
                reason: reason.clone(),
            });
            return Decision::rejected(reason);
        };

        let Some(to_index) = self.store.find_by_id(&to_id) else {
            let reason = format!("unknown to_id {}", to_id);
            warn!(relation = ?relation, to_id = %to_id, reason = %reason, "edge.rejected");
            self.emit_event(Event::EdgeRejected {
                relation,
                reason: reason.clone(),
            });
            return Decision::rejected(reason);
        };

        if from_index == to_index {
            let reason = "self-loops are not allowed";
            warn!(relation = ?relation, reason = reason, "edge.rejected");
            self.emit_event(Event::EdgeRejected {
                relation,
                reason: reason.to_string(),
            });
            return Decision::rejected(reason);
        }

        if let Some(existing) = self.store.graph().find_edge(from_index, to_index) {
            if let Some(weight) = self.store.graph().edge_weight(existing) {
                if weight.relation == relation {
                    let reason = "edge already exists";
                    warn!(relation = ?relation, reason = reason, "edge.rejected");
                    self.emit_event(Event::EdgeRejected {
                        relation,
                        reason: reason.to_string(),
                    });
                    return Decision::rejected(reason);
                }
            }
        }

        let Some(from_node) = self.store.node(from_index).cloned() else {
            let reason = "from node missing from store";
            warn!(relation = ?relation, reason = reason, "edge.rejected");
            self.emit_event(Event::EdgeRejected {
                relation,
                reason: reason.to_string(),
            });
            return Decision::rejected(reason);
        };

        let Some(to_node) = self.store.node(to_index).cloned() else {
            let reason = "to node missing from store";
            warn!(relation = ?relation, reason = reason, "edge.rejected");
            self.emit_event(Event::EdgeRejected {
                relation,
                reason: reason.to_string(),
            });
            return Decision::rejected(reason);
        };

        if matches!(relation, Relation::PrerequisiteFor)
            && self.store.would_cycle_prereq(from_index, to_index)
        {
            let reason = "edge would introduce a prerequisite cycle";
            warn!(relation = ?relation, reason = reason, "edge.rejected");
            self.emit_event(Event::EdgeRejected {
                relation,
                reason: reason.to_string(),
            });
            return Decision::rejected(reason);
        }

        if matches!(relation, Relation::PrerequisiteFor)
            && !matches!(from_node.kind, NodeKind::Concept)
        {
            let reason = "prerequisite edges must originate from a concept";
            warn!(relation = ?relation, reason = reason, "edge.rejected");
            self.emit_event(Event::EdgeRejected {
                relation,
                reason: reason.to_string(),
            });
            return Decision::rejected(reason);
        }

        if matches!(relation, Relation::PrerequisiteFor)
            && !matches!(to_node.kind, NodeKind::Concept | NodeKind::LearningOutcome)
        {
            let reason = "prerequisite edges must target a concept or learning outcome";
            warn!(relation = ?relation, reason = reason, "edge.rejected");
            self.emit_event(Event::EdgeRejected {
                relation,
                reason: reason.to_string(),
            });
            return Decision::rejected(reason);
        }

        if matches!(relation, Relation::PrerequisiteFor) && from_node.level > to_node.level {
            let reason = "prerequisite edges must not decrease level";
            warn!(relation = ?relation, reason = reason, "edge.rejected");
            self.emit_event(Event::EdgeRejected {
                relation,
                reason: reason.to_string(),
            });
            return Decision::rejected(reason);
        }

        let rationale = rationale.trim();
        if rationale.is_empty() {
            let reason = "edge rationale missing";
            warn!(relation = ?relation, reason = reason, "edge.rejected");
            self.emit_event(Event::EdgeRejected {
                relation,
                reason: reason.to_string(),
            });
            return Decision::rejected(reason);
        }

        let rationale = rationale.to_string();

        let edge = Edge {
            from: from_index,
            to: to_index,
            relation: relation.clone(),
            rationale: rationale.clone(),
        };

        self.store.add_edge(edge);

        info!(relation = ?relation, from = %from_id, to = %to_id, "edge.accepted");

        self.emit_event(Event::EdgeAccepted {
            relation,
            from: from_id,
            to: to_id,
            rationale,
        });

        Decision::accepted(None)
    }

    fn summarize(&self) -> Summary {
        Summary::from_store(&self.store)
    }

    fn sanitize_tags(tags: Option<Vec<String>>) -> Option<Vec<String>> {
        let mut result = Vec::new();
        let Some(tags) = tags else { return None };

        for tag in tags {
            if result.len() == MAX_TAGS_PER_NODE {
                break;
            }
            let normalized = tag.trim().to_lowercase();
            if normalized.is_empty() {
                continue;
            }
            if !ALLOWED_TAGS.contains(&normalized.as_str()) {
                continue;
            }
            if !result
                .iter()
                .any(|existing: &String| existing == &normalized)
            {
                result.push(normalized);
            }
        }

        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }

    fn is_single_sentence(text: &str) -> bool {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return false;
        }

        let mut ender_count = 0;
        for ch in trimmed.chars() {
            if matches!(ch, '.' | '!' | '?') {
                ender_count += 1;
            }
        }

        matches!(trimmed.chars().last(), Some('.') | Some('!') | Some('?')) && ender_count == 1
    }

    fn emit_event(&self, event: Event) {
        if let Some(sender) = &self.event_sender {
            let _ = sender.send(event);
        }
    }
}

impl Message<AddNodes> for GraphAdder {
    type Reply = Vec<Decision>;

    fn handle(
        &mut self,
        msg: AddNodes,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> impl std::future::Future<Output = Self::Reply> + Send {
        let decisions = self.handle_add_nodes(msg.0);
        ready(decisions)
    }
}

impl Message<AddEdges> for GraphAdder {
    type Reply = Vec<Decision>;

    fn handle(
        &mut self,
        msg: AddEdges,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> impl std::future::Future<Output = Self::Reply> + Send {
        let decisions = self.handle_add_edges(msg.0);
        ready(decisions)
    }
}

impl Message<Inventory> for GraphAdder {
    type Reply = Vec<(Uuid, NodeKind, u8, String, Option<Vec<String>>)>;

    fn handle(
        &mut self,
        _msg: Inventory,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> impl std::future::Future<Output = Self::Reply> + Send {
        ready(self.store.inventory())
    }
}

impl Message<Summarize> for GraphAdder {
    type Reply = Summary;

    fn handle(
        &mut self,
        _msg: Summarize,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> impl std::future::Future<Output = Self::Reply> + Send {
        let summary = self.summarize();

        for TopLearningOutcome { id, text, supports } in &summary.top_learning_outcomes {
            self.emit_event(Event::SummaryLine {
                message: format!("LO {id} ({supports} supports): {}", text),
            });
        }

        ready(summary)
    }
}

impl Message<ExportDot> for GraphAdder {
    type Reply = String;

    fn handle(
        &mut self,
        _msg: ExportDot,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> impl std::future::Future<Output = Self::Reply> + Send {
        ready(self.store.export_dot())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Granularity, NodeKind};
    use uuid::Uuid;

    fn sample_concept(text: &str) -> NodeProposal {
        NodeProposal {
            kind: NodeKind::Concept,
            granularity: Granularity::Sentence,
            level: 0,
            text: text.to_string(),
            tags: Some(vec!["tests".to_string()]),
        }
    }

    #[test]
    fn test_add_node_validation() {
        let mut adder = GraphAdder::with_event_sender(GraphStore::new(), None);

        let proposals = vec![
            sample_concept("Graphs model dependencies with directed edges."),
            sample_concept("Graphs model dependencies with directed edges."),
            NodeProposal {
                kind: NodeKind::LearningOutcome,
                granularity: Granularity::Sentence,
                level: 2,
                text: "Understand recursion across modules.".to_string(),
                tags: Some(vec!["purpose".to_string()]),
            },
            NodeProposal {
                kind: NodeKind::LearningOutcome,
                granularity: Granularity::Sentence,
                level: 2,
                text: "I can trace prerequisite chains in a learning network.".to_string(),
                tags: Some(vec!["implementation".to_string()]),
            },
        ];

        let decisions = adder.handle_add_nodes(proposals);

        assert!(decisions[0].accepted, "first node should be accepted");
        assert!(!decisions[1].accepted, "duplicate node should be rejected");
        assert!(
            !decisions[2].accepted,
            "learning outcome without prefix should be rejected"
        );
        assert!(
            decisions[3].accepted,
            "valid learning outcome should be accepted"
        );
    }

    #[test]
    fn test_add_edge_cycle_rejected() {
        let mut adder = GraphAdder::with_event_sender(GraphStore::new(), None);

        let decisions = adder.handle_add_nodes(vec![
            sample_concept("Concept A establishes foundational syntax."),
            sample_concept("Concept B introduces control flow variations."),
            sample_concept("Concept C covers data encapsulation."),
        ]);

        let node_ids: Vec<Uuid> = decisions.iter().filter_map(|d| d.assigned_id).collect();
        assert_eq!(node_ids.len(), 3, "expected three accepted nodes");

        let edges = vec![
            EdgeProposal {
                relation: Relation::PrerequisiteFor,
                from_id: node_ids[0],
                to_id: node_ids[1],
                rationale: "Concept A informs Concept B.".to_string(),
            },
            EdgeProposal {
                relation: Relation::PrerequisiteFor,
                from_id: node_ids[1],
                to_id: node_ids[2],
                rationale: "Concept B prepares learners for Concept C.".to_string(),
            },
            EdgeProposal {
                relation: Relation::PrerequisiteFor,
                from_id: node_ids[2],
                to_id: node_ids[0],
                rationale: "Concept C loops back to Concept A.".to_string(),
            },
        ];

        let decisions = adder.handle_add_edges(edges);
        assert!(decisions[0].accepted);
        assert!(decisions[1].accepted);
        assert!(
            !decisions[2].accepted,
            "cycle-forming edge must be rejected"
        );
    }

    #[test]
    fn test_export_dot_contains_labels() {
        let mut adder = GraphAdder::with_event_sender(GraphStore::new(), None);

        let decisions = adder.handle_add_nodes(vec![
            sample_concept("Concept A explores closures."),
            sample_concept("Concept B applies closures to iterators."),
        ]);

        let ids: Vec<Uuid> = decisions.iter().filter_map(|d| d.assigned_id).collect();

        let edges = vec![EdgeProposal {
            relation: Relation::Supports,
            from_id: ids[0],
            to_id: ids[1],
            rationale: "Closures provide reusable iterator adapters.".to_string(),
        }];

        let edge_result = adder.handle_add_edges(edges);
        assert!(edge_result[0].accepted);

        let dot = adder.store.export_dot();
        assert!(dot.contains("Concept A explores closures"));
        assert!(dot.contains("supports"));
    }
}
