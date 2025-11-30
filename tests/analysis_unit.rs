use weaver::{
    analysis,
    graph::{self, GraphService, KnowledgeNode, NodeId, manager::GraphManagerState, traversal},
    schema::types::{IntendedEffect, KnowledgeType, SourceRef, SupportKind},
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
        introduction_scope: graph::IntroductionScope::InCourse,
    }
}

fn add_support(
    svc: &mut GraphService,
    from: NodeId,
    to: NodeId,
    kind: SupportKind,
    intended: IntendedEffect,
    case_tag: Option<graph::CaseTag>,
    coverage_tags: Vec<String>,
) {
    svc.add_edge::<graph::SupportsSpec>(
        from,
        to,
        graph::SupportsAttrs {
            support_kind: kind,
            intended_effect: intended,
            case_tag,
            coverage_tags,
            evidence_refs: vec![SourceRef {
                path:       "dummy".into(),
                start_line: 1,
                end_line:   2,
                revision:   "deadbeef".into(),
            }],
        },
        1.0,
    )
    .unwrap();
}

#[test]
fn example_gaps_deduplicates_messages() {
    let mut svc = GraphService::new();
    // target procedural, high intrinsic load
    let mut proc = mk_kn("proc", KnowledgeType::Procedural);
    proc.intrinsic_load = Some(graph::IntrinsicLoad::High);
    let proc_id = svc.add_knowledge_node("proc".into(), proc, vec![]).unwrap();

    // support node
    let support_id = svc
        .add_knowledge_node("support".into(), mk_kn("support", KnowledgeType::Conceptual), vec![])
        .unwrap();

    // Only one typical worked example, no edge/error case, no coverage tags
    add_support(
        &mut svc,
        support_id,
        proc_id,
        SupportKind::WorkedExample,
        IntendedEffect::ReduceExtraneousLoad,
        Some(graph::CaseTag::Typical),
        vec![],
    );

    let gaps = analysis::example_gaps(svc.graph());
    let proc_gap = gaps
        .into_iter()
        .find(|g| svc.graph()[g.node].slug == "proc")
        .expect("procedural gap present");
    let desc = proc_gap.description;
    assert!(desc.contains("worked examples"));
    assert!(desc.contains("high intrinsic_load"));
}

#[test]
fn borrow_ahead_flags_use_without_intro() {
    let mut svc = GraphService::new();
    // knowledge node to be used
    let k_id = svc
        .add_knowledge_node("k".into(), mk_kn("k", KnowledgeType::Conceptual), vec![])
        .unwrap();

    // teaching step that uses k before any introduce
    let ts_use = svc
        .add_teaching_step(
            "ts_use".into(),
            graph::TeachingStepNode {
                title:       "use".into(),
                statement:   "use it".into(),
                purpose:     graph::TeachingPurpose::Use,
                method_tags: vec![],
                episode:     "ep1".into(),
                source_refs: vec![SourceRef {
                    path:       "dummy".into(),
                    start_line: 1,
                    end_line:   2,
                    revision:   "deadbeef".into(),
                }],
            },
            vec![],
        )
        .unwrap();

    svc.add_edge::<graph::AnchorsSpec>(
        ts_use,
        k_id,
        graph::AnchorsAttrs {
            impact: graph::AnchorImpact::Use,
        },
        1.0,
    )
    .unwrap();

    let results = analysis::borrow_ahead(svc.graph(), "ep1");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].severity, analysis::BorrowSeverity::NoIntro);
}

#[test]
fn supports_guard_rejects_prereq_load() {
    let mut svc = GraphService::new();
    // first principle
    let fp = svc
        .add_knowledge_node("fp".into(), mk_kn("fp", KnowledgeType::Conceptual), vec![])
        .unwrap();
    // assessment item reachable only via support (would carry prerequisite load)
    let assess = svc
        .add_knowledge_node("a".into(), mk_kn("a", KnowledgeType::AssessmentItem), vec![])
        .unwrap();
    // support edge fp -> assessment (allowed combo but should be rejected as
    // non-fadeable)
    let err = svc.add_edge::<graph::SupportsSpec>(
        fp,
        assess,
        graph::SupportsAttrs {
            support_kind:    SupportKind::RubricNote,
            intended_effect: IntendedEffect::Motivate,
            case_tag:        None,
            coverage_tags:   vec![],
            evidence_refs:   vec![SourceRef {
                path:       "dummy".into(),
                start_line: 1,
                end_line:   2,
                revision:   "deadbeef".into(),
            }],
        },
        1.0,
    );
    assert!(err.is_err(), "support edge should be rejected as non-fadeable");
    assert!(svc.graph().find_edge(fp, assess).is_none());
}

#[test]
fn example_gaps_parallel_reports_missing_supports() {
    let mut svc = GraphService::new();
    // high intrinsic load procedural with no supports
    let mut proc = mk_kn("proc", KnowledgeType::Procedural);
    proc.intrinsic_load = Some(graph::IntrinsicLoad::High);
    svc.add_knowledge_node("proc".into(), proc, vec![]).unwrap();

    let gaps = analysis::example_gaps(svc.graph());
    assert_eq!(gaps.len(), 1);
    assert!(gaps[0].description.contains("worked examples"));
}

#[test]
fn practice_gaps_parallel_flags_missing_assessment() {
    let mut svc = GraphService::new();
    let _proc = svc
        .add_knowledge_node("proc".into(), mk_kn("proc", KnowledgeType::Procedural), vec![])
        .unwrap();
    // unrelated assessment without linkage
    let assess = svc
        .add_knowledge_node("a".into(), mk_kn("a", KnowledgeType::AssessmentItem), vec![])
        .unwrap();
    // assessment targets some LO so it would be valid if reachable
    let lo = svc
        .add_knowledge_node("lo".into(), mk_kn("lo", KnowledgeType::LearningOutcome), vec![])
        .unwrap();
    svc.add_edge::<graph::AssessesSpec>(
        assess,
        lo,
        graph::AssessesAttrs {
            evidence_link: weaver::schema::types::EvidenceLink {
                scope:                weaver::schema::types::AssessmentScope::Target,
                claim:                "lo".into(),
                observation_features: vec!["feat".into()],
            },
        },
        1.0,
    )
    .unwrap();

    let gaps = analysis::procedural_practice_gaps(svc.graph());
    assert_eq!(gaps.len(), 1);
    assert_eq!(svc.graph()[gaps[0].node].slug, "proc");
}

#[test]
fn requires_transitive_reduction_identifies_redundant_edge() {
    // Build a small DAG: a -> b, b -> c, a -> c (redundant)
    let mut svc = GraphService::new();
    let a = svc
        .add_knowledge_node("a".into(), mk_kn("a", KnowledgeType::Conceptual), vec![])
        .unwrap();
    let b = svc
        .add_knowledge_node("b".into(), mk_kn("b", KnowledgeType::Conceptual), vec![])
        .unwrap();
    let c = svc
        .add_knowledge_node("c".into(), mk_kn("c", KnowledgeType::Conceptual), vec![])
        .unwrap();

    svc.add_edge::<graph::RequiresSpec>(
        a,
        b,
        graph::RequiresAttrs {
            strength:      weaver::schema::types::Strength::Necessary,
            rationale:     "r".into(),
            evidence_refs: vec![SourceRef {
                path:       "dummy".into(),
                start_line: 1,
                end_line:   2,
                revision:   "deadbeef".into(),
            }],
        },
        1.0,
    )
    .unwrap();
    svc.add_edge::<graph::RequiresSpec>(
        b,
        c,
        graph::RequiresAttrs {
            strength:      weaver::schema::types::Strength::Necessary,
            rationale:     "r".into(),
            evidence_refs: vec![SourceRef {
                path:       "dummy".into(),
                start_line: 1,
                end_line:   2,
                revision:   "deadbeef".into(),
            }],
        },
        1.0,
    )
    .unwrap();
    svc.add_edge::<graph::RequiresSpec>(
        a,
        c,
        graph::RequiresAttrs {
            strength:      weaver::schema::types::Strength::Necessary,
            rationale:     "r".into(),
            evidence_refs: vec![SourceRef {
                path:       "dummy".into(),
                start_line: 1,
                end_line:   2,
                revision:   "deadbeef".into(),
            }],
        },
        1.0,
    )
    .unwrap();

    let redundants = traversal::requires_transitive_reduction(svc.graph()).unwrap();
    assert!(redundants.contains(&(a, c)));
    // reachability still holds
    assert!(traversal::requires_path_exists(svc.graph(), a, c));
}

#[test]
fn graph_manager_state_round_trip_preserves_version() {
    let mut svc = GraphService::new();
    let _ = svc
        .add_knowledge_node("k".into(), mk_kn("k", KnowledgeType::Conceptual), vec![])
        .unwrap();
    let version = svc.graph_version();

    let state = GraphManagerState::new(svc.snapshot_graph(), "deadbeef".into(), false, version);

    let data = postcard::to_stdvec(&state).expect("serialize state");
    let decoded: GraphManagerState = postcard::from_bytes(&data).expect("deserialize state");

    let restored = GraphService::from_parts(
        decoded.graph.clone(),
        decoded.strict_quality,
        decoded.graph_version,
    )
    .expect("restored graph should validate");

    assert_eq!(restored.graph_version(), version);
    restored.validate_global_invariants().unwrap();
}
