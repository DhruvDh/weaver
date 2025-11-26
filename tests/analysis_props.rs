use proptest::prelude::*;
use weaver::{
    graph::{GraphService, KnowledgeNode, NodeId},
    schema::types::{KnowledgeType, SourceRef},
};

fn mk_kn(title: &str, kt: KnowledgeType) -> KnowledgeNode {
    KnowledgeNode {
        title: title.to_string(),
        statement: title.to_string(),
        knowledge_type: kt,
        source_refs: vec![SourceRef {
            path:       "dummy".into(),
            start_line: 1,
            end_line:   2,
            revision:   "deadbeef".into(),
        }],
        confidence: 1.0,
        rubric_criteria: vec![],
        construct_irrelevant_demands: vec![],
        grain_level: None,
        intrinsic_load: None,
        introduction_scope: weaver::graph::IntroductionScope::InCourse,
    }
}

fn add_nodes(service: &mut GraphService, n: usize) -> Vec<NodeId> {
    (0..n)
        .map(|i| {
            service
                .add_knowledge_node(
                    format!("n{i}"),
                    mk_kn(&format!("n{i}"), KnowledgeType::Conceptual),
                    vec![],
                )
                .unwrap()
        })
        .collect()
}

proptest! {
    // Adding edges along a topological order keeps requires acyclic; a back edge creates a cycle.
    #[test]
    fn requires_dag_guard_prevents_cycle(size in 2_usize..15) {
        let mut svc = GraphService::new();
        let nodes = add_nodes(&mut svc, size);

        // chain edges n0->n1->...->n{k}
        for win in nodes.windows(2) {
            svc.add_edge::<weaver::graph::RequiresSpec>(win[0], win[1],
                weaver::graph::RequiresAttrs { strength: weaver::schema::types::Strength::Necessary, rationale: "r".into(), evidence_refs: vec![SourceRef { path: "dummy".into(), start_line: 1, end_line: 2, revision: "deadbeef".into() }] },
                1.0).unwrap();
        }

        // current graph must be DAG
        assert!(weaver::analysis::requires_is_dag(svc.graph()));

        // adding a back edge to an ancestor should be rejected
        let last = nodes[nodes.len()-1];
        let first = nodes[0];
        let err = svc.add_edge::<weaver::graph::RequiresSpec>(last, first,
            weaver::graph::RequiresAttrs { strength: weaver::schema::types::Strength::Necessary, rationale: "r".into(), evidence_refs: vec![SourceRef { path: "dummy".into(), start_line: 1, end_line: 2, revision: "deadbeef".into() }] },
            1.0);
        assert!(err.is_err());
    }

    // First principles are exactly nodes with zero incoming requires.
    #[test]
    fn first_principles_zero_indegree(
        size in 2_usize..10,
        extra_edges in prop::collection::vec((0_usize..20, 0_usize..20), 0..15)
    ) {
        let mut svc = GraphService::new();
        let nodes = add_nodes(&mut svc, size);

        for (a,b) in extra_edges {
            if a < size && b < size && a != b {
                let _ = svc.add_edge::<weaver::graph::RequiresSpec>(nodes[a], nodes[b],
                    weaver::graph::RequiresAttrs { strength: weaver::schema::types::Strength::Necessary, rationale: "r".into(), evidence_refs: vec![SourceRef { path: "dummy".into(), start_line: 1, end_line: 2, revision: "deadbeef".into() }] },
                    1.0);
            }
        }

        let fps = weaver::analysis::first_principles(svc.graph());
        for &fp in &fps {
            let indeg = svc.graph().edges_directed(fp, petgraph::Direction::Incoming)
                .filter(|e| matches!(e.weight().kind, weaver::graph::EdgeKind::Requires(_)))
                .count();
            assert_eq!(indeg, 0);
        }
    }
}
