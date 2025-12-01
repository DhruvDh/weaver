use std::sync::Arc;

use weaver::{
    graph::{GraphService, KnowledgeNode, NodeId},
    schema::types::{AssessmentScope, EvidenceLink, KnowledgeType, SourceRef},
    tools::llm::graph_tools::{
        analysis::{
            CachedGapBundle, CachedLoBundle, cache_gap_bundle, cache_lo_bundle, decode_cached,
        },
        analysis_cache::{AnalysisCache, AnalysisCacheValue},
    },
};

fn mk_source_ref() -> SourceRef {
    SourceRef {
        path:       "dummy".into(),
        start_line: 1,
        end_line:   2,
        revision:   "deadbeef".into(),
    }
}

fn mk_lo_node() -> KnowledgeNode {
    KnowledgeNode {
        title: "lo".into(),
        statement: "lo".into(),
        knowledge_type: KnowledgeType::LearningOutcome,
        source_refs: vec![mk_source_ref()],
        confidence: 1.0,
        rubric_criteria: vec!["crit1".into()],
        construct_irrelevant_demands: vec![],
        grain_level: None,
        intrinsic_load: None,
        introduction_scope: weaver::graph::IntroductionScope::InCourse,
    }
}

fn mk_assessment_node() -> KnowledgeNode {
    KnowledgeNode {
        title: "a1".into(),
        statement: "a1".into(),
        knowledge_type: KnowledgeType::AssessmentItem,
        source_refs: vec![mk_source_ref()],
        confidence: 1.0,
        rubric_criteria: vec![],
        construct_irrelevant_demands: vec![],
        grain_level: None,
        intrinsic_load: None,
        introduction_scope: weaver::graph::IntroductionScope::InCourse,
    }
}

fn build_minimal_graph() -> (Arc<weaver::graph::CurriculumGraph>, u64, NodeId) {
    let mut svc = GraphService::new();
    let lo = svc
        .add_knowledge_node("lo".into(), mk_lo_node(), vec![])
        .unwrap();
    let assess = svc
        .add_knowledge_node("a1".into(), mk_assessment_node(), vec![])
        .unwrap();
    svc.add_edge::<weaver::graph::AssessesSpec>(
        assess,
        lo,
        weaver::graph::AssessesAttrs {
            evidence_link: EvidenceLink {
                claim:                "lo".into(),
                observation_features: vec!["crit1".into()],
                scope:                AssessmentScope::Target,
            },
        },
        1.0,
    )
    .unwrap();

    let version = svc.graph_version();
    (svc.shared_graph(), version, lo)
}

#[test]
fn lo_bundle_cache_reused_across_calls() {
    let cache = AnalysisCache::new();
    let (graph, version, lo) = build_minimal_graph();

    let first: Arc<AnalysisCacheValue> = cache_lo_bundle(&cache, version, &graph, lo, "lo");
    let second: Arc<AnalysisCacheValue> = cache_lo_bundle(&cache, version, &graph, lo, "lo");

    assert!(Arc::ptr_eq(&first, &second));

    let bundle: CachedLoBundle = decode_cached(&first.payload).unwrap();
    assert_eq!(bundle.assessments.len(), 1);
    assert_eq!(bundle.coverage.covered.len(), 1);
}

#[test]
fn gap_bundle_cache_reused_across_calls() {
    let cache = AnalysisCache::new();
    let (graph, version, _) = build_minimal_graph();

    let first: Arc<AnalysisCacheValue> = cache_gap_bundle(&cache, version, &graph);
    let second: Arc<AnalysisCacheValue> = cache_gap_bundle(&cache, version, &graph);

    assert!(Arc::ptr_eq(&first, &second));

    let bundle: CachedGapBundle = decode_cached(&first.payload).unwrap();
    if let Some(first_gap) = bundle.example_gaps.first() {
        assert!(!first_gap.slug.is_empty());
    }
}
