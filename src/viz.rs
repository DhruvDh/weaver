use std::collections::HashMap;

use rerun::{
    GraphEdges, GraphNodes, RecordingStream, RecordingStreamBuilder, archetypes::TextLog,
    components::TextLogLevel,
};
use tokio::sync::mpsc::UnboundedReceiver;
use uuid::Uuid;

use crate::{
    edge_synth::truncate_sentence,
    model::{NodeKind, Relation},
};

/// Events emitted by the GraphAdder for visualization purposes.
#[derive(Debug, Clone)]
pub enum Event {
    NodeAccepted {
        id: Uuid,
        kind: NodeKind,
        level: u8,
        tags: Option<Vec<String>>,
        text: String,
    },
    NodeRejected {
        text: String,
        reason: String,
    },
    EdgeAccepted {
        relation: Relation,
        from: Uuid,
        to: Uuid,
        rationale: String,
    },
    EdgeRejected {
        relation: Relation,
        reason: String,
    },
    SummaryLine {
        message: String,
    },
}

#[derive(Debug, Clone)]
struct NodeCache {
    #[allow(dead_code)]
    kind: NodeKind,
    #[allow(dead_code)]
    level: u8,
    #[allow(dead_code)]
    text: String,
    order: usize,
}

/// Rerun-backed visualizer actor state.
#[derive(Debug)]
pub struct Viz {
    stream: Option<RecordingStream>,
    nodes: HashMap<Uuid, NodeCache>,
    edges: Vec<(Uuid, Uuid, Relation, String)>,
    level_counts: HashMap<u8, usize>,
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
            level_counts: HashMap::new(),
        }
    }

    pub async fn run(mut self, mut rx: UnboundedReceiver<Event>) {
        while let Some(event) = rx.recv().await {
            self.handle_event(event);
        }
    }

    fn handle_event(&mut self, event: Event) {
        match event {
            Event::NodeAccepted {
                id,
                kind,
                level,
                tags,
                text,
            } => self.handle_node_accepted(id, kind, level, tags, text),
            Event::NodeRejected { text, reason } => self.log_text(
                "graph/nodes_rejected",
                TextLogLevel::WARN,
                format!("REJECT node {}: {reason}", truncate_sentence(&text)),
            ),
            Event::EdgeAccepted {
                relation,
                from,
                to,
                rationale,
            } => self.handle_edge_accepted(relation, from, to, rationale),
            Event::EdgeRejected { relation, reason } => self.log_text(
                "graph/edges_rejected",
                TextLogLevel::WARN,
                format!("REJECT edge {:?}: {reason}", relation),
            ),
            Event::SummaryLine { message } => {
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
                let entry = self.level_counts.entry(level).or_insert(0);
                let current = *entry;
                *entry += 1;
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
            format!(
                "ACCEPT node {:?} lvl {} {}: {}{}",
                kind, level, id, label, tag_suffix
            ),
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

        let mut entries: Vec<_> = self.nodes.iter().collect();
        entries.sort_by(|(id_a, cache_a), (id_b, cache_b)| {
            cache_a
                .level
                .cmp(&cache_b.level)
                .then_with(|| cache_a.order.cmp(&cache_b.order))
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
            let _ = stream.log(
                "graph/edges",
                &GraphEdges::new(directed.clone()).with_directed_edges(),
            );
        }

        if !supports.is_empty() {
            supports.sort();
            supports.dedup();
            let _ = stream.log(
                "graph/edges_supports",
                &GraphEdges::new(supports).with_undirected_edges(),
            );
        }
    }
}
