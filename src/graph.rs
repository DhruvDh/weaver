use std::collections::{HashMap, HashSet};

use petgraph::{
    Directed, Graph, algo,
    graph::{EdgeIndex, NodeIndex},
    visit::EdgeRef,
};
use uuid::Uuid;

use crate::model::{Edge, Node, NodeKind, Relation, clean_text, normalize_text};

/// Wrapper around the petgraph store with convenient indexes.
#[derive(Debug)]
pub struct GraphStore {
    graph: Graph<Node, Edge, Directed>,
    text_index: HashMap<String, NodeIndex>,
    id_index: HashMap<Uuid, NodeIndex>,
}

impl GraphStore {
    pub fn new() -> Self {
        Self {
            graph: Graph::default(),
            text_index: HashMap::new(),
            id_index: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: Node) -> NodeIndex {
        let norm = normalize_text(&node.text);
        let id = node.id;
        let index = self.graph.add_node(node);
        self.text_index.insert(norm, index);
        self.id_index.insert(id, index);
        index
    }

    pub fn add_edge(&mut self, edge: Edge) -> EdgeIndex {
        self.graph.add_edge(edge.from, edge.to, edge)
    }

    pub fn find_by_text(&self, text: &str) -> Option<NodeIndex> {
        let norm = normalize_text(text);
        self.text_index.get(&norm).copied()
    }

    pub fn find_by_id(&self, id: &Uuid) -> Option<NodeIndex> {
        self.id_index.get(id).copied()
    }

    pub fn node(&self, index: NodeIndex) -> Option<&Node> {
        self.graph.node_weight(index)
    }

    pub fn inventory(&self) -> Vec<(Uuid, NodeKind, u8, String, Option<Vec<String>>)> {
        self.graph
            .node_indices()
            .filter_map(|index| {
                self.graph.node_weight(index).map(|node| {
                    (
                        node.id,
                        node.kind.clone(),
                        node.level,
                        node.text.clone(),
                        node.tags.clone(),
                    )
                })
            })
            .collect()
    }

    pub fn would_cycle_prereq(&self, from: NodeIndex, to: NodeIndex) -> bool {
        if from == to {
            return true;
        }

        let mut stack = vec![to];
        let mut visited = HashSet::new();
        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            if current == from {
                return true;
            }
            for edge in self.graph.edges(current) {
                if matches!(edge.weight().relation, Relation::PrerequisiteFor) {
                    stack.push(edge.target());
                }
            }
        }

        false
    }

    pub fn export_dot(&self) -> String {
        let mut output = String::from("digraph weaver {\n");

        for index in self.graph.node_indices() {
            if let Some(node) = self.graph.node_weight(index) {
                let label = clean_text(&node.text).replace('"', "\\\"");
                output.push_str(&format!("  \"{}\" [label=\"{}\"];\n", node.id, label));
            }
        }

        for edge in self.graph.edge_references() {
            let weight = edge.weight();
            let relation = match weight.relation {
                Relation::PrerequisiteFor => "prerequisite_for",
                Relation::Supports => "supports",
            };
            let mut label = relation.to_string();
            if !weight.rationale.trim().is_empty() {
                let sanitized = clean_text(&weight.rationale).replace('"', "\\\"");
                label = format!("{}: {}", relation, sanitized);
            }
            if let (Some(from_node), Some(to_node)) = (
                self.graph.node_weight(weight.from),
                self.graph.node_weight(weight.to),
            ) {
                output.push_str(&format!(
                    "  \"{}\" -> \"{}\" [label=\"{}\"];\n",
                    from_node.id, to_node.id, label
                ));
            }
        }

        output.push_str("}\n");
        output
    }

    pub fn graph(&self) -> &Graph<Node, Edge, Directed> {
        &self.graph
    }

    pub fn node_indices(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        self.graph.node_indices()
    }

    pub fn edge_indices(&self) -> impl Iterator<Item = EdgeIndex> + '_ {
        self.graph.edge_indices()
    }

    pub fn edge_weight(&self, index: EdgeIndex) -> Option<&Edge> {
        self.graph.edge_weight(index)
    }

    pub fn prerequisite_edges(&self) -> usize {
        self.graph
            .edge_references()
            .filter(|edge_ref| matches!(edge_ref.weight().relation, Relation::PrerequisiteFor))
            .count()
    }

    pub fn supports_edges(&self) -> usize {
        self.graph
            .edge_references()
            .filter(|edge_ref| matches!(edge_ref.weight().relation, Relation::Supports))
            .count()
    }

    pub fn is_prerequisite_dag(&self) -> bool {
        let mut check_graph = Graph::<(), (), Directed>::new();
        let mut mapping = HashMap::new();

        for node_index in self.graph.node_indices() {
            let mapped = check_graph.add_node(());
            mapping.insert(node_index, mapped);
        }

        for edge in self
            .graph
            .edge_references()
            .filter(|edge| matches!(edge.weight().relation, Relation::PrerequisiteFor))
        {
            if let (Some(&from), Some(&to)) =
                (mapping.get(&edge.source()), mapping.get(&edge.target()))
            {
                check_graph.add_edge(from, to, ());
            }
        }

        !algo::is_cyclic_directed(&check_graph)
    }
}
