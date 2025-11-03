use std::collections::{HashMap, VecDeque};

use rerun::{
    GraphEdges, GraphNodes, RecordingStream, RecordingStreamBuilder, archetypes::TextLog,
    components::TextLogLevel,
};
use tokio::sync::mpsc::UnboundedReceiver;
use uuid::Uuid;

use crate::{
    edge_synth::truncate_sentence,
    events::DomainEvent,
    model::{NodeKind, Relation},
};

#[derive(Debug, Clone)]
struct NodeCache {
    #[allow(dead_code)]
    kind:  NodeKind,
    #[allow(dead_code)]
    level: u8,
    #[allow(dead_code)]
    text:  String,
    order: usize,
}

/// Rerun-backed visualizer actor state.
#[derive(Debug)]
pub struct Viz {
    stream:     Option<RecordingStream>,
    nodes:      HashMap<Uuid, NodeCache>,
    edges:      Vec<(Uuid, Uuid, Relation, String)>,
    next_order: usize,
}

impl Viz {
    pub fn new(enabled: bool) -> Self {
        let stream = if enabled {
            RecordingStreamBuilder::new("weaver-mvp").spawn().ok()
        } else {
            None
        };

        Self {
            stream,
            nodes: HashMap::new(),
            edges: Vec::new(),
            next_order: 0,
        }
    }

    pub async fn run(mut self, mut rx: UnboundedReceiver<DomainEvent>) {
        while let Some(event) = rx.recv().await {
            self.handle_event(event);
        }
    }

    fn handle_event(&mut self, event: DomainEvent) {
        match event {
            DomainEvent::NodeAccepted {
                id,
                kind,
                level,
                tags,
                text,
            } => self.handle_node_accepted(id, kind, level, tags, text),
            DomainEvent::NodeRejected { proposal, reason } => {
                let text = truncate_sentence(&proposal.text);
                self.log_text(
                    "graph/nodes_rejected",
                    TextLogLevel::WARN,
                    format!("REJECT node {}: {reason}", text),
                );
            }
            DomainEvent::EdgeAccepted {
                relation,
                from,
                to,
                rationale,
            } => self.handle_edge_accepted(relation, from, to, rationale),
            DomainEvent::EdgeRejected { proposal, reason } => {
                self.log_text(
                    "graph/edges_rejected",
                    TextLogLevel::WARN,
                    format!("REJECT edge {:?}: {reason}", proposal.relation),
                );
            }
            DomainEvent::SummaryLine { message } => {
                self.log_text("graph/summary", TextLogLevel::INFO, message);
            }
        }
    }

    fn handle_node_accepted(
        &mut self,
        id: Uuid,
        kind: NodeKind,
        level: u8,
        tags: Option<Vec<String>>,
        text: String,
    ) {
        let order = match self.nodes.get(&id) {
            Some(cache) => cache.order,
            None => {
                let current = self.next_order;
                self.next_order += 1;
                current
            }
        };

        self.nodes.insert(
            id,
            NodeCache {
                kind: kind.clone(),
                level,
                text: text.clone(),
                order,
            },
        );

        let label = truncate_sentence(&text);
        let tag_suffix = tags
            .clone()
            .filter(|t| !t.is_empty())
            .map(|t| format!(" tags=[{}]", t.join(",")))
            .unwrap_or_default();

        self.log_text(
            "graph/events",
            TextLogLevel::INFO,
            format!("ACCEPT node {:?} lvl {} {}: {}{}", kind, level, id, label, tag_suffix),
        );

        self.log_nodes();
    }

    fn handle_edge_accepted(
        &mut self,
        relation: Relation,
        from: Uuid,
        to: Uuid,
        rationale: String,
    ) {
        self.edges
            .push((from, to, relation.clone(), rationale.clone()));

        self.log_text(
            "graph/events",
            TextLogLevel::INFO,
            format!(
                "ACCEPT edge {:?} {} -> {}: {}",
                relation,
                from,
                to,
                truncate_sentence(&rationale)
            ),
        );

        self.log_edges();
    }

    fn log_text(&self, entity: &str, level: impl Into<TextLogLevel>, message: String) {
        if let Some(stream) = &self.stream {
            let _ = stream.log(entity, &TextLog::new(message).with_level(level));
        }
    }

    fn log_nodes(&self) {
        let Some(stream) = &self.stream else { return };
        if self.nodes.is_empty() {
            return;
        }

        let ranks = compute_prerequisite_ranks(
            self.nodes.keys().copied(),
            self.edges.iter().filter_map(|(from, to, relation, _)| {
                if matches!(relation, Relation::PrerequisiteFor) {
                    Some((*from, *to))
                } else {
                    None
                }
            }),
        );

        let mut entries: Vec<_> = self.nodes.iter().collect();
        entries.sort_by(|(id_a, cache_a), (id_b, cache_b)| {
            let rank_a = ranks.get(id_a).copied().unwrap_or(0);
            let rank_b = ranks.get(id_b).copied().unwrap_or(0);
            rank_a
                .cmp(&rank_b)
                .then_with(|| cache_a.order.cmp(&cache_b.order))
                .then_with(|| cache_a.level.cmp(&cache_b.level))
                .then_with(|| id_a.as_bytes().cmp(id_b.as_bytes()))
        });

        let node_ids: Vec<_> = entries.iter().map(|(id, _)| id.to_string()).collect();
        let labels: Vec<_> = entries
            .iter()
            .map(|(_, cache)| truncate_sentence(&cache.text))
            .collect();
        let graph_nodes = GraphNodes::new(node_ids).with_labels(labels);
        let _ = stream.log("graph/nodes", &graph_nodes);
    }

    fn log_edges(&self) {
        let Some(stream) = &self.stream else { return };
        if self.edges.is_empty() {
            return;
        }

        let mut directed: Vec<(String, String)> = Vec::new();
        let mut supports: Vec<(String, String)> = Vec::new();

        for (from, to, relation, _) in &self.edges {
            let edge = (from.to_string(), to.to_string());
            match relation {
                Relation::PrerequisiteFor => directed.push(edge),
                Relation::Supports => supports.push(edge),
            }
        }

        if !directed.is_empty() {
            directed.sort();
            directed.dedup();
            let _ =
                stream.log("graph/edges", &GraphEdges::new(directed.clone()).with_directed_edges());
        }

        if !supports.is_empty() {
            supports.sort();
            supports.dedup();
            let _ = stream
                .log("graph/edges_supports", &GraphEdges::new(supports).with_undirected_edges());
        }
    }
}

pub(crate) fn compute_prerequisite_ranks(
    node_ids: impl IntoIterator<Item = Uuid>,
    prereq_edges: impl IntoIterator<Item = (Uuid, Uuid)>,
) -> HashMap<Uuid, usize> {
    let mut indegree: HashMap<Uuid, usize> = HashMap::new();
    let mut predecessors: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
    let mut successors: HashMap<Uuid, Vec<Uuid>> = HashMap::new();

    for id in node_ids {
        indegree.entry(id).or_insert(0);
        predecessors.entry(id).or_insert_with(Vec::new);
        successors.entry(id).or_insert_with(Vec::new);
    }

    for (from, to) in prereq_edges {
        indegree.entry(from).or_insert(0);
        indegree.entry(to).or_insert(0);
        predecessors.entry(from).or_insert_with(Vec::new);
        predecessors.entry(to).or_insert_with(Vec::new);
        successors.entry(from).or_insert_with(Vec::new).push(to);
        predecessors.entry(to).or_insert_with(Vec::new).push(from);
        if let Some(entry) = indegree.get_mut(&to) {
            *entry += 1;
        }
    }

    let mut initial: Vec<Uuid> = indegree
        .iter()
        .filter_map(|(node, &deg)| if deg == 0 { Some(*node) } else { None })
        .collect();
    initial.sort_unstable();
    let mut queue: VecDeque<Uuid> = initial.into();

    let mut ranks: HashMap<Uuid, usize> = HashMap::new();

    while let Some(node) = queue.pop_front() {
        let rank = predecessors
            .get(&node)
            .map(|parents| {
                parents
                    .iter()
                    .filter_map(|parent| ranks.get(parent).map(|parent_rank| parent_rank + 1))
                    .max()
            })
            .flatten()
            .unwrap_or(0);
        ranks.insert(node, rank);

        if let Some(children) = successors.get(&node) {
            let mut newly_zero = Vec::new();
            for child in children {
                if let Some(entry) = indegree.get_mut(child) {
                    if *entry > 0 {
                        *entry -= 1;
                        if *entry == 0 {
                            newly_zero.push(*child);
                        }
                    }
                }
            }
            if !newly_zero.is_empty() {
                newly_zero.sort_unstable();
                for child in newly_zero {
                    queue.push_back(child);
                }
            }
        }
    }

    for node in indegree.keys() {
        ranks.entry(*node).or_insert(0);
    }

    ranks
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use super::*;

    #[test]
    fn compute_prerequisite_ranks_orders_chain() {
        let a = Uuid::from_u128(1);
        let b = Uuid::from_u128(2);
        let c = Uuid::from_u128(3);

        let ranks = compute_prerequisite_ranks(vec![a, b, c], vec![(a, b), (b, c)]);
        assert_eq!(ranks.get(&a), Some(&0));
        assert_eq!(ranks.get(&b), Some(&1));
        assert_eq!(ranks.get(&c), Some(&2));
    }

    #[test]
    fn compute_prerequisite_ranks_defaults_to_zero_without_edges() {
        let a = Uuid::from_u128(10);
        let b = Uuid::from_u128(20);

        let ranks = compute_prerequisite_ranks(vec![a, b], std::iter::empty());
        assert_eq!(ranks.get(&a), Some(&0));
        assert_eq!(ranks.get(&b), Some(&0));
        assert_eq!(ranks.len(), 2);
    }
}
