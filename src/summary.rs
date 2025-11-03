use std::collections::{BTreeMap, HashMap, HashSet};

use petgraph::visit::EdgeRef;
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
    pub degree_metrics:        DegreeMetrics,
    pub centrality_metrics:    CentralityMetrics,
    pub clustering:            ClusteringMetrics,
    pub assortativity:         AssortativityMetrics,
    pub community_analysis:    CommunityAnalysis,
}

/// Lightweight view of a learning outcome ranked by inbound supports.
#[derive(Debug, Clone, Serialize)]
pub struct TopLearningOutcome {
    pub id:       Uuid,
    pub text:     String,
    pub supports: usize,
}

/// Histogram and top-degree summaries.
#[derive(Debug, Clone, Serialize, Default)]
pub struct DegreeMetrics {
    pub in_histogram:         Vec<(usize, usize)>,
    pub out_histogram:        Vec<(usize, usize)>,
    pub undirected_histogram: Vec<(usize, usize)>,
    pub top_in_degree:        Vec<NodeDegreeEntry>,
    pub top_out_degree:       Vec<NodeDegreeEntry>,
    pub top_undirected:       Vec<NodeDegreeEntry>,
}

/// Individual node degree entry for reporting.
#[derive(Debug, Clone, Serialize)]
pub struct NodeDegreeEntry {
    pub id:     Uuid,
    pub text:   String,
    pub degree: usize,
}

/// Centrality metrics (currently Katz centrality).
#[derive(Debug, Clone, Serialize, Default)]
pub struct CentralityMetrics {
    pub top_katz: Vec<CentralityEntry>,
}

/// Katz centrality entry.
#[derive(Debug, Clone, Serialize)]
pub struct CentralityEntry {
    pub id:    Uuid,
    pub text:  String,
    pub score: f64,
}

/// Clustering and transitivity statistics.
#[derive(Debug, Clone, Serialize, Default)]
pub struct ClusteringMetrics {
    pub average_local: f64,
    pub transitivity:  f64,
}

/// Assortativity by various node attributes.
#[derive(Debug, Clone, Serialize, Default)]
pub struct AssortativityMetrics {
    pub by_level: Option<f64>,
    pub by_kind:  Option<f64>,
    pub by_tag:   Option<f64>,
}

/// Community detection summary derived from spectral modularity.
#[derive(Debug, Clone, Serialize, Default)]
pub struct CommunityAnalysis {
    pub modularity:  f64,
    pub communities: Vec<CommunitySummary>,
}

/// Single community summary.
#[derive(Debug, Clone, Serialize)]
pub struct CommunitySummary {
    pub id:           usize,
    pub size:         usize,
    pub sample_nodes: Vec<CommunityMember>,
}

/// Representative node included in a community summary.
#[derive(Debug, Clone, Serialize)]
pub struct CommunityMember {
    pub id:   Uuid,
    pub text: String,
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
            degree_metrics:        DegreeMetrics::default(),
            centrality_metrics:    CentralityMetrics::default(),
            clustering:            ClusteringMetrics::default(),
            assortativity:         AssortativityMetrics::default(),
            community_analysis:    CommunityAnalysis::default(),
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

        if summary.total_nodes > 0 {
            let analytics = compute_analytics(store);
            summary.degree_metrics = analytics.degree_metrics;
            summary.centrality_metrics = analytics.centrality_metrics;
            summary.clustering = analytics.clustering;
            summary.assortativity = analytics.assortativity;
            summary.community_analysis = analytics.community_analysis;
        }

        summary
    }
}

struct AnalyticsBundle {
    degree_metrics:     DegreeMetrics,
    centrality_metrics: CentralityMetrics,
    clustering:         ClusteringMetrics,
    assortativity:      AssortativityMetrics,
    community_analysis: CommunityAnalysis,
}

fn compute_analytics(store: &GraphStore) -> AnalyticsBundle {
    let mut index_map = HashMap::new();
    let mut nodes = Vec::new();
    for (position, node_index) in store.node_indices().enumerate() {
        index_map.insert(node_index, position);
        nodes.push(node_index);
    }

    let n = nodes.len();
    if n == 0 {
        return AnalyticsBundle {
            degree_metrics:     DegreeMetrics::default(),
            centrality_metrics: CentralityMetrics::default(),
            clustering:         ClusteringMetrics::default(),
            assortativity:      AssortativityMetrics::default(),
            community_analysis: CommunityAnalysis::default(),
        };
    }

    let mut in_degrees = vec![0usize; n];
    let mut out_degrees = vec![0usize; n];
    let mut incoming: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut undirected_sets: Vec<HashSet<usize>> = vec![HashSet::new(); n];

    for edge in store.graph().edge_references() {
        let from = index_map[&edge.source()];
        let to = index_map[&edge.target()];
        out_degrees[from] += 1;
        in_degrees[to] += 1;
        incoming[to].push(from);
        if undirected_sets[from].insert(to) {
            undirected_sets[to].insert(from);
        }
    }

    let undirected_degrees: Vec<usize> = undirected_sets
        .iter()
        .map(|neighbors| neighbors.len())
        .collect();
    let undirected_edges: Vec<(usize, usize)> = undirected_sets
        .iter()
        .enumerate()
        .flat_map(|(i, neighbors)| {
            neighbors
                .iter()
                .filter(move |&&j| i < j)
                .map(move |&j| (i, j))
        })
        .collect();

    let degree_metrics = DegreeMetrics {
        in_histogram:         degree_histogram(&in_degrees),
        out_histogram:        degree_histogram(&out_degrees),
        undirected_histogram: degree_histogram(&undirected_degrees),
        top_in_degree:        top_degree_entries(store, &nodes, &in_degrees, 5),
        top_out_degree:       top_degree_entries(store, &nodes, &out_degrees, 5),
        top_undirected:       top_degree_entries(store, &nodes, &undirected_degrees, 5),
    };

    let katz_scores = compute_katz_centrality(&incoming);
    let centrality_metrics = CentralityMetrics {
        top_katz: top_centrality_entries(store, &nodes, &katz_scores, 5),
    };

    let clustering = compute_clustering_metrics(&undirected_sets);

    let mut level_categories = vec![0usize; n];
    let mut kind_categories = vec![0usize; n];
    let mut tag_categories = vec![0usize; n];
    let mut tag_lookup: HashMap<String, usize> = HashMap::new();
    let mut next_tag_id = 0usize;

    for (i, node_index) in nodes.iter().enumerate() {
        if let Some(node) = store.node(*node_index) {
            level_categories[i] = node.level as usize;
            kind_categories[i] = match node.kind {
                NodeKind::Concept => 0,
                NodeKind::LearningOutcome => 1,
            };
            let tag_key = node
                .tags
                .as_ref()
                .and_then(|tags| tags.iter().find(|tag| !tag.trim().is_empty()))
                .map(|tag| tag.trim().to_string())
                .unwrap_or_else(|| "none".to_string());
            let entry = tag_lookup.entry(tag_key).or_insert_with(|| {
                let id = next_tag_id;
                next_tag_id += 1;
                id
            });
            tag_categories[i] = *entry;
        }
    }

    let assortativity = AssortativityMetrics {
        by_level: compute_assortativity(&level_categories, &undirected_edges),
        by_kind:  compute_assortativity(&kind_categories, &undirected_edges),
        by_tag:   compute_assortativity(&tag_categories, &undirected_edges),
    };

    let community_analysis =
        compute_community_analysis(store, &nodes, &undirected_sets, &undirected_edges);

    AnalyticsBundle {
        degree_metrics,
        centrality_metrics,
        clustering,
        assortativity,
        community_analysis,
    }
}

fn degree_histogram(degrees: &[usize]) -> Vec<(usize, usize)> {
    let mut histogram = BTreeMap::new();
    for &value in degrees {
        *histogram.entry(value).or_insert(0usize) += 1;
    }
    histogram.into_iter().collect()
}

fn top_degree_entries(
    store: &GraphStore,
    nodes: &[petgraph::graph::NodeIndex],
    degrees: &[usize],
    limit: usize,
) -> Vec<NodeDegreeEntry> {
    let mut entries: Vec<NodeDegreeEntry> = nodes
        .iter()
        .enumerate()
        .filter_map(|(i, node_index)| store.node(*node_index).map(|node| (i, node)))
        .map(|(i, node)| NodeDegreeEntry {
            id:     node.id,
            text:   node.text.clone(),
            degree: degrees[i],
        })
        .collect();

    entries.sort_by(|a, b| {
        b.degree
            .cmp(&a.degree)
            .then_with(|| a.text.cmp(&b.text))
            .then_with(|| a.id.as_bytes().cmp(b.id.as_bytes()))
    });
    if entries.len() > limit {
        entries.truncate(limit);
    }
    entries
}

fn top_centrality_entries(
    store: &GraphStore,
    nodes: &[petgraph::graph::NodeIndex],
    scores: &[f64],
    limit: usize,
) -> Vec<CentralityEntry> {
    let mut entries: Vec<CentralityEntry> = nodes
        .iter()
        .enumerate()
        .filter_map(|(i, node_index)| store.node(*node_index).map(|node| (i, node)))
        .map(|(i, node)| CentralityEntry {
            id:    node.id,
            text:  node.text.clone(),
            score: scores.get(i).copied().unwrap_or(0.0),
        })
        .collect();

    entries.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.text.cmp(&b.text))
            .then_with(|| a.id.as_bytes().cmp(b.id.as_bytes()))
    });

    if entries.len() > limit {
        entries.truncate(limit);
    }

    entries
}

fn compute_katz_centrality(incoming: &[Vec<usize>]) -> Vec<f64> {
    let n = incoming.len();
    if n == 0 {
        return Vec::new();
    }

    let alpha = 0.05;
    let max_iterations = 100;
    let tolerance = 1e-6;

    let mut scores = vec![1.0; n];
    normalize_vector(&mut scores);
    let mut next = vec![0.0; n];

    for _ in 0..max_iterations {
        for (i, predecessors) in incoming.iter().enumerate() {
            let sum: f64 = predecessors.iter().map(|&j| scores[j]).sum();
            next[i] = 1.0 + alpha * sum;
        }

        normalize_vector(&mut next);
        let max_delta = scores
            .iter()
            .zip(&next)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        scores.copy_from_slice(&next);
        if max_delta < tolerance {
            break;
        }
    }

    normalize_sum(&mut scores);
    scores
}

fn normalize_vector(vector: &mut [f64]) {
    let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm > 0.0 {
        for value in vector {
            *value /= norm;
        }
    }
}

fn normalize_sum(vector: &mut [f64]) {
    let total = vector.iter().sum::<f64>();
    if total > 0.0 {
        for value in vector {
            *value /= total;
        }
    }
}

fn compute_clustering_metrics(adj: &[HashSet<usize>]) -> ClusteringMetrics {
    let mut total_local = 0.0;
    let mut contributing_nodes = 0usize;
    let mut closed_triplets = 0usize;
    let mut total_triplets = 0usize;

    for (_idx, neighbors) in adj.iter().enumerate() {
        let degree = neighbors.len();
        if degree < 2 {
            continue;
        }

        let mut neighbor_list: Vec<usize> = neighbors.iter().copied().collect();
        neighbor_list.sort_unstable();

        let mut closed = 0usize;
        for i in 0..neighbor_list.len() {
            for j in (i + 1)..neighbor_list.len() {
                if adj[neighbor_list[i]].contains(&neighbor_list[j]) {
                    closed += 1;
                }
            }
        }

        let possible = degree * (degree - 1) / 2;
        if possible > 0 {
            total_local += closed as f64 / possible as f64;
            contributing_nodes += 1;
        }

        closed_triplets += closed;
        total_triplets += possible;
    }

    ClusteringMetrics {
        average_local: if contributing_nodes > 0 {
            total_local / contributing_nodes as f64
        } else {
            0.0
        },
        transitivity:  if total_triplets > 0 {
            closed_triplets as f64 / total_triplets as f64
        } else {
            0.0
        },
    }
}

fn compute_assortativity(categories: &[usize], edges: &[(usize, usize)]) -> Option<f64> {
    if edges.is_empty() || categories.is_empty() {
        return None;
    }

    let mut unique_categories = categories.to_vec();
    unique_categories.sort_unstable();
    unique_categories.dedup();

    if unique_categories.len() < 2 {
        return None;
    }

    let mut index_map = HashMap::new();
    for (idx, category) in unique_categories.iter().enumerate() {
        index_map.insert(*category, idx);
    }

    let size = unique_categories.len();
    let mut mixing = vec![vec![0.0_f64; size]; size];
    let m = edges.len() as f64;

    for &(u, v) in edges {
        let a = index_map[&categories[u]];
        let b = index_map[&categories[v]];
        mixing[a][b] += 1.0;
        mixing[b][a] += 1.0;
    }

    for row in &mut mixing {
        for value in row {
            *value /= 2.0 * m;
        }
    }

    let mut row_sums = vec![0.0_f64; size];
    for i in 0..size {
        row_sums[i] = mixing[i].iter().copied().sum();
    }

    let trace: f64 = (0..size).map(|i| mixing[i][i]).sum();
    let squared_row_sum: f64 = row_sums.iter().map(|value| value * value).sum();
    let denominator = 1.0 - squared_row_sum;

    if denominator.abs() < 1e-9 {
        None
    } else {
        Some((trace - squared_row_sum) / denominator)
    }
}

fn compute_community_analysis(
    store: &GraphStore,
    nodes: &[petgraph::graph::NodeIndex],
    adjacency_sets: &[HashSet<usize>],
    edges: &[(usize, usize)],
) -> CommunityAnalysis {
    let mut analysis = CommunityAnalysis::default();
    let n = nodes.len();
    let total_edges = edges.len();

    if n == 0 {
        return analysis;
    }

    if total_edges == 0 {
        analysis.communities.push(CommunitySummary {
            id:           0,
            size:         n,
            sample_nodes: sample_nodes(store, nodes, (0..n).collect(), 3),
        });
        return analysis;
    }

    let degrees: Vec<usize> = adjacency_sets
        .iter()
        .map(|neighbors| neighbors.len())
        .collect();
    let adjacency: Vec<Vec<usize>> = adjacency_sets
        .iter()
        .map(|neighbors| {
            let mut list: Vec<usize> = neighbors.iter().copied().collect();
            list.sort_unstable();
            list
        })
        .collect();

    if let Some(partition) = spectral_partition(&adjacency, &degrees, total_edges) {
        let modularity =
            compute_modularity(&partition, adjacency_sets, &degrees, total_edges as f64);
        analysis.modularity = modularity;

        let groups = [(-1, 0usize), (1, 1usize)];
        for (sign, label) in groups {
            let members: Vec<usize> = partition
                .iter()
                .enumerate()
                .filter_map(|(idx, value)| if *value == sign { Some(idx) } else { None })
                .collect();

            if members.is_empty() {
                continue;
            }

            analysis.communities.push(CommunitySummary {
                id:           label,
                size:         members.len(),
                sample_nodes: sample_nodes(store, nodes, members, 3),
            });
        }

        if analysis.communities.is_empty() {
            analysis.communities.push(CommunitySummary {
                id:           0,
                size:         n,
                sample_nodes: sample_nodes(store, nodes, (0..n).collect(), 3),
            });
        }
    } else {
        analysis.communities.push(CommunitySummary {
            id:           0,
            size:         n,
            sample_nodes: sample_nodes(store, nodes, (0..n).collect(), 3),
        });
    }

    analysis
}

fn spectral_partition(
    adjacency: &[Vec<usize>],
    degrees: &[usize],
    total_edges: usize,
) -> Option<Vec<i8>> {
    let n = adjacency.len();
    if n < 2 || total_edges == 0 {
        return None;
    }

    let mut vector = vec![1.0_f64; n];
    let mean = vector.iter().sum::<f64>() / n as f64;
    for value in &mut vector {
        *value -= mean;
    }
    normalize_vector(&mut vector);
    let mut next = vec![0.0_f64; n];

    for _ in 0..100 {
        modularity_multiply(&mut next, &vector, adjacency, degrees, total_edges);
        let norm = next.iter().map(|value| value * value).sum::<f64>().sqrt();
        if norm < 1e-9 {
            return None;
        }
        for value in &mut next {
            *value /= norm;
        }
        let delta = vector
            .iter()
            .zip(&next)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        vector.copy_from_slice(&next);
        if delta < 1e-6 {
            break;
        }
    }

    modularity_multiply(&mut next, &vector, adjacency, degrees, total_edges);
    let eigenvalue = vector.iter().zip(&next).map(|(a, b)| a * b).sum::<f64>();

    if eigenvalue <= 1e-6 {
        return None;
    }

    let mut partition = vec![0_i8; n];
    let mut has_positive = false;
    let mut has_negative = false;

    for (i, value) in vector.iter().enumerate() {
        if *value >= 0.0 {
            partition[i] = 1;
            has_positive = true;
        } else {
            partition[i] = -1;
            has_negative = true;
        }
    }

    if has_positive && has_negative {
        Some(partition)
    } else {
        None
    }
}

fn modularity_multiply(
    output: &mut [f64],
    vector: &[f64],
    adjacency: &[Vec<usize>],
    degrees: &[usize],
    total_edges: usize,
) {
    let m2 = 2.0 * total_edges as f64;
    let degree_dot_vector: f64 = degrees
        .iter()
        .zip(vector.iter())
        .map(|(degree, value)| *degree as f64 * value)
        .sum();

    for (i, neighbors) in adjacency.iter().enumerate() {
        let sum_neighbors: f64 = neighbors.iter().map(|&j| vector[j]).sum();
        output[i] = sum_neighbors - degrees[i] as f64 * degree_dot_vector / m2;
    }

    let mean = output.iter().sum::<f64>() / output.len() as f64;
    for value in output {
        *value -= mean;
    }
}

fn compute_modularity(
    partition: &[i8],
    adjacency_sets: &[HashSet<usize>],
    degrees: &[usize],
    total_edges: f64,
) -> f64 {
    if total_edges <= 0.0 {
        return 0.0;
    }

    let m2 = 2.0 * total_edges;
    let n = adjacency_sets.len();
    let mut sum = 0.0_f64;

    for i in 0..n {
        for j in 0..n {
            if partition[i] == partition[j] {
                let adjacency = if adjacency_sets[i].contains(&j) {
                    1.0
                } else {
                    0.0
                };
                sum += adjacency - (degrees[i] as f64 * degrees[j] as f64) / m2;
            }
        }
    }

    sum / m2
}

fn sample_nodes(
    store: &GraphStore,
    nodes: &[petgraph::graph::NodeIndex],
    mut members: Vec<usize>,
    limit: usize,
) -> Vec<CommunityMember> {
    members.sort_unstable();
    let mut samples = Vec::new();
    for index in members.into_iter().take(limit) {
        if let Some(node) = store.node(nodes[index]) {
            samples.push(CommunityMember {
                id:   node.id,
                text: node.text.clone(),
            });
        }
    }
    samples
}
