use weaver::{
    analysis,
    graph::{self, GraphService, KnowledgeNode, NodeId},
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
