use std::collections::{HashMap, HashSet, hash_map::RandomState};

use petgraph::{
    algo::{all_simple_paths, has_path_connecting, toposort},
    graph::Graph,
    visit::{EdgeFiltered, EdgeRef},
};

use super::{CurriculumGraph, EdgeKind, EdgePayload, GraphIx, NodeId};

/// Filtered view over only `requires` edges.
pub fn requires_view<'a>(
    g: &'a CurriculumGraph,
) -> EdgeFiltered<
    &'a CurriculumGraph,
    impl Fn(petgraph::stable_graph::EdgeReference<'a, EdgePayload, GraphIx>) -> bool + 'a,
> {
    EdgeFiltered::from_fn(g, |e| matches!(e.weight().kind, EdgeKind::Requires(_)))
}

/// Filtered view over `precedes` edges limited to a single episode.
pub fn precedes_view<'a>(
    g: &'a CurriculumGraph,
    episode: &'a str,
) -> EdgeFiltered<
    &'a CurriculumGraph,
    impl Fn(petgraph::stable_graph::EdgeReference<'a, EdgePayload, GraphIx>) -> bool + 'a,
> {
    EdgeFiltered::from_fn(
        g,
        move |e| matches!(e.weight().kind, EdgeKind::Precedes(ref p) if p.episode == episode),
    )
}

/// Filtered view over requires or supports edges (used for fadeability/paths).
pub fn requires_or_supports_view<'a>(
    g: &'a CurriculumGraph,
) -> EdgeFiltered<
    &'a CurriculumGraph,
    impl Fn(petgraph::stable_graph::EdgeReference<'a, EdgePayload, GraphIx>) -> bool + 'a,
> {
    EdgeFiltered::from_fn(g, |e| {
        matches!(e.weight().kind, EdgeKind::Requires(_) | EdgeKind::Supports(_))
    })
}

/// Fast path-existence check along `requires` edges.
pub fn requires_path_exists(g: &CurriculumGraph, from: NodeId, to: NodeId) -> bool {
    has_path_connecting(&requires_view(g), from, to, None)
}

/// Return one requires path (if any) between two nodes using petgraph's
/// simple-path iterator.
pub fn requires_one_path(g: &CurriculumGraph, from: NodeId, to: NodeId) -> Option<Vec<NodeId>> {
    all_simple_paths::<Vec<_>, _, RandomState>(&requires_view(g), from, to, 0, None).next()
}

/// Path-existence check over `precedes` edges scoped to an episode.
pub fn precedes_path_exists(g: &CurriculumGraph, episode: &str, from: NodeId, to: NodeId) -> bool {
    has_path_connecting(&precedes_view(g, episode), from, to, None)
}

/// Return the set of requires edges that are redundant under transitive
/// reduction. Only defined when the requires layer is a DAG.
pub fn requires_transitive_reduction(g: &CurriculumGraph) -> Option<HashSet<(NodeId, NodeId)>> {
    // Build a temporary Graph (implements NodeCompactIndexable) mirroring the
    // requires layer.
    let mut g2: Graph<(), (), petgraph::Directed> =
        Graph::with_capacity(g.node_count(), g.edge_count());
    let mut map: HashMap<NodeId, petgraph::graph::NodeIndex> = HashMap::new();
    for n in g.node_indices() {
        let idx = g2.add_node(());
        map.insert(n, idx);
    }
    for e in g.edge_indices() {
        if let EdgeKind::Requires(_) = g[e].kind
            && let Some((u, v)) = g.edge_endpoints(e)
        {
            g2.add_edge(map[&u], map[&v], ());
        }
    }
    let topo = toposort(&g2, None).ok()?;
    let (adj, revmap) = petgraph::algo::tred::dag_to_toposorted_adjacency_list(&g2, &topo);
    let (tred, _tclos) = petgraph::algo::tred::dag_transitive_reduction_closure::<
        (),
        petgraph::graph::DefaultIx,
    >(&adj);

    let mut redundant = HashSet::new();
    for e in g2.edge_references() {
        let keep = tred
            .find_edge(revmap[e.source().index()], revmap[e.target().index()])
            .is_some();
        if !keep {
            // map back to original NodeId
            let orig_u = map
                .iter()
                .find(|(_, v)| **v == e.source())
                .map(|(k, _)| *k)
                .unwrap();
            let orig_v = map
                .iter()
                .find(|(_, v)| **v == e.target())
                .map(|(k, _)| *k)
                .unwrap();
            redundant.insert((orig_u, orig_v));
        }
    }
    Some(redundant)
}
