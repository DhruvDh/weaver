use std::time::Instant;

use weaver::{
    analysis,
    graph::{
        self, AnchorImpact, AnchorsAttrs, EdgeKind, EdgePayload, GraphService, KnowledgeNode,
        NodeId, PrecedesAttrs, TeachingPurpose, TeachingStepNode, manager::GraphManagerState,
        traversal,
    },
    schema::types::{
        AssessmentScope, EvidenceLink, IntendedEffect, KnowledgeType, SourceRef, Strength,
        SupportKind,
    },
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

fn canonical_slug(title: &str, kt: KnowledgeType) -> String {
    weaver::graph::slug::Slug::generate(kt, title).to_string()
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

mod supports_examples {
    use super::*;

    #[test]
    fn example_gaps_deduplicates_messages() {
        let mut svc = GraphService::new();
        let mut proc = mk_kn("proc", KnowledgeType::Procedural);
        proc.intrinsic_load = Some(graph::IntrinsicLoad::High);
        let proc_id = svc.add_knowledge_node("proc".into(), proc, vec![]).unwrap();
        let support_id = svc
            .add_knowledge_node(
                "support".into(),
                mk_kn("support", KnowledgeType::Conceptual),
                vec![],
            )
            .unwrap();
        add_support(
            &mut svc,
            support_id,
            proc_id,
            SupportKind::WorkedExample,
            IntendedEffect::ReduceExtraneousLoad,
            Some(graph::CaseTag::Typical),
            vec!["trace".into()],
        );
        let gaps = analysis::example_gaps(svc.graph());
        let expected_slug = canonical_slug("proc", KnowledgeType::Procedural);
        let proc_gap = gaps
            .into_iter()
            .find(|g| svc.graph()[g.node].slug == expected_slug)
            .expect("procedural gap present");
        assert!(proc_gap.description.contains("worked examples"));
    }

    #[test]
    fn example_gaps_parallel_reports_missing_supports() {
        let mut svc = GraphService::new();
        let mut proc = mk_kn("proc", KnowledgeType::Procedural);
        proc.intrinsic_load = Some(graph::IntrinsicLoad::High);
        svc.add_knowledge_node("proc".into(), proc, vec![]).unwrap();
        let gaps = analysis::example_gaps(svc.graph());
        assert_eq!(gaps.len(), 1);
        assert!(gaps[0].description.contains("worked examples"));
    }

    #[test]
    fn supports_guard_rejects_prereq_load() {
        let mut svc = GraphService::new();
        let fp = svc
            .add_knowledge_node("fp".into(), mk_kn("fp", KnowledgeType::Conceptual), vec![])
            .unwrap();
        let assess = svc
            .add_knowledge_node("a".into(), mk_kn("a", KnowledgeType::AssessmentItem), vec![])
            .unwrap();
        let err = svc.add_edge::<graph::SupportsSpec>(
            fp,
            assess,
            graph::SupportsAttrs {
                support_kind:    SupportKind::RubricNote,
                intended_effect: IntendedEffect::Motivate,
                case_tag:        Some(graph::CaseTag::Typical),
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
        assert!(err.is_err());
        assert!(svc.graph().find_edge(fp, assess).is_none());
    }

    #[test]
    fn supports_require_case_tag_and_coverage_tags() {
        let mut svc = GraphService::new();
        let from = svc
            .add_knowledge_node("a".into(), mk_kn("a", KnowledgeType::Conceptual), vec![])
            .unwrap();
        let mut high = mk_kn("b", KnowledgeType::Conceptual);
        high.intrinsic_load = Some(graph::IntrinsicLoad::High);
        let to = svc.add_knowledge_node("b".into(), high, vec![]).unwrap();
        let err = svc
            .add_edge::<graph::SupportsSpec>(
                from,
                to,
                graph::SupportsAttrs {
                    support_kind:    SupportKind::WorkedExample,
                    intended_effect: IntendedEffect::ReduceExtraneousLoad,
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
            )
            .expect_err("missing case_tag + coverage_tags should be rejected");
        assert!(matches!(err, graph::GraphError::Schema(_)));
    }
}

mod practice_alignment {
    use super::*;

    #[test]
    fn practice_gaps_parallel_flags_missing_assessment() {
        let mut svc = GraphService::new();
        let _proc = svc
            .add_knowledge_node("proc".into(), mk_kn("proc", KnowledgeType::Procedural), vec![])
            .unwrap();
        let assess = svc
            .add_knowledge_node("a".into(), mk_kn("a", KnowledgeType::AssessmentItem), vec![])
            .unwrap();
        let mut lo_node = mk_kn("lo", KnowledgeType::LearningOutcome);
        lo_node.introduction_scope = graph::IntroductionScope::Prior;
        let lo = svc
            .add_knowledge_node("lo".into(), lo_node, vec![])
            .unwrap();
        let lo_slug = svc.graph()[lo].slug.clone();
        svc.add_edge::<graph::AssessesSpec>(
            assess,
            lo,
            graph::AssessesAttrs {
                evidence_link: weaver::schema::types::EvidenceLink {
                    scope:                weaver::schema::types::AssessmentScope::Target,
                    claim:                lo_slug.clone(),
                    observation_features: vec!["feat".into()],
                },
            },
            1.0,
        )
        .unwrap();

        let gaps = analysis::procedural_practice_gaps(svc.graph());
        assert_eq!(gaps.len(), 1);
        assert_eq!(
            svc.graph()[gaps[0].node].slug,
            canonical_slug("proc", KnowledgeType::Procedural)
        );
    }

    #[test]
    fn strict_mode_flags_stranded_lo() {
        let mut svc = GraphService::new();
        svc.add_knowledge_node("lo".into(), mk_kn("lo", KnowledgeType::LearningOutcome), vec![])
            .unwrap();
        let err = svc
            .set_strict_quality(true)
            .expect_err("strict mode should reject unreachable LO");
        if let graph::GraphError::InvariantViolation { violations } = err {
            assert!(
                violations
                    .iter()
                    .any(|v| v.code == graph::InvariantCode::LoTargetAssessment),
                "violations missing stranded LO code: {violations:?}"
            );
        } else {
            panic!("expected invariant violation, got {err:?}");
        }
    }

    #[test]
    fn strict_mode_flags_practice_gap() {
        let mut svc = GraphService::new();
        svc.add_knowledge_node("proc".into(), mk_kn("proc", KnowledgeType::Procedural), vec![])
            .unwrap();
        let err = svc
            .set_strict_quality(true)
            .expect_err("strict mode should reject missing practice");
        if let graph::GraphError::InvariantViolation { violations } = err {
            assert!(
                violations
                    .iter()
                    .any(|v| v.code == graph::InvariantCode::ProceduralPractice),
                "violations missing practice gap code: {violations:?}"
            );
        } else {
            panic!("expected invariant violation, got {err:?}");
        }
    }

    #[test]
    fn strict_mode_flags_coverage_gap() {
        let mut svc = GraphService::new();
        let concept = svc
            .add_knowledge_node("c".into(), mk_kn("c", KnowledgeType::Conceptual), vec![])
            .unwrap();
        let support = svc
            .add_knowledge_node("s".into(), mk_kn("s", KnowledgeType::Conceptual), vec![])
            .unwrap();
        svc.add_edge::<graph::SupportsSpec>(
            support,
            concept,
            graph::SupportsAttrs {
                support_kind:    SupportKind::Analogy,
                intended_effect: IntendedEffect::IncreaseGermaneLoad,
                case_tag:        Some(graph::CaseTag::Typical),
                coverage_tags:   vec![],
                evidence_refs:   vec![SourceRef {
                    path:       "dummy".into(),
                    start_line: 1,
                    end_line:   2,
                    revision:   "deadbeef".into(),
                }],
            },
            1.0,
        )
        .unwrap();
        let assess = svc
            .add_knowledge_node("a".into(), mk_kn("a", KnowledgeType::AssessmentItem), vec![])
            .unwrap();
        let mut lo_node = mk_kn("lo", KnowledgeType::LearningOutcome);
        lo_node.rubric_criteria = vec!["criterion_b".into()];
        let lo = svc
            .add_knowledge_node("lo".into(), lo_node, vec![])
            .unwrap();
        let lo_slug = svc.graph()[lo].slug.clone();
        svc.add_edge::<graph::RequiresSpec>(
            concept,
            assess,
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
        svc.add_edge::<graph::AssessesSpec>(
            assess,
            lo,
            graph::AssessesAttrs {
                evidence_link: weaver::schema::types::EvidenceLink {
                    scope:                weaver::schema::types::AssessmentScope::Target,
                    claim:                lo_slug.clone(),
                    observation_features: vec!["criterion_a".into()],
                },
            },
            1.0,
        )
        .unwrap();
        let err = svc
            .set_strict_quality(true)
            .expect_err("strict mode should reject missing rubric coverage");
        if let graph::GraphError::InvariantViolation { violations } = err {
            assert!(
                violations
                    .iter()
                    .any(|v| v.code == graph::InvariantCode::RubricCoverage),
                "expected rubric coverage code, got {violations:?}"
            );
        } else {
            panic!("expected invariant violation, got {err:?}");
        }
    }

    #[test]
    fn rubric_drift_recomputes_coverage() {
        let mut svc = GraphService::new();
        let concept = svc
            .add_knowledge_node("c".into(), mk_kn("c", KnowledgeType::Conceptual), vec![])
            .unwrap();
        let assess = svc
            .add_knowledge_node("a".into(), mk_kn("a", KnowledgeType::AssessmentItem), vec![])
            .unwrap();
        let mut lo_node = mk_kn("lo", KnowledgeType::LearningOutcome);
        lo_node.rubric_criteria = vec!["a".into()];
        let lo = svc
            .add_knowledge_node("lo".into(), lo_node, vec![])
            .unwrap();
        let lo_slug = svc.graph()[lo].slug.clone();
        svc.add_edge::<graph::RequiresSpec>(
            concept,
            assess,
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
        svc.add_edge::<graph::AssessesSpec>(
            assess,
            lo,
            graph::AssessesAttrs {
                evidence_link: weaver::schema::types::EvidenceLink {
                    scope:                weaver::schema::types::AssessmentScope::Target,
                    claim:                lo_slug.clone(),
                    observation_features: vec!["a".into()],
                },
            },
            1.0,
        )
        .unwrap();
        svc.validate_global_invariants().unwrap();
        let mut updated_lo = mk_kn("lo", KnowledgeType::LearningOutcome);
        updated_lo.rubric_criteria = vec!["a".into(), "b".into()];
        svc.update_knowledge_node("lo", updated_lo, vec![]).unwrap();
        let err = svc
            .set_strict_quality(true)
            .expect_err("rubric drift with missing coverage should fail validation");
        if let graph::GraphError::InvariantViolation { violations } = err {
            assert!(
                violations
                    .iter()
                    .any(|v| v.code == graph::InvariantCode::RubricCoverage),
                "expected rubric coverage code after drift, got {violations:?}"
            );
        } else {
            panic!("expected invariant violation, got {err:?}");
        }
    }
}

mod purity {
    use super::*;

    #[test]
    fn extraneous_prerequisite_blocked() {
        let mut svc = GraphService::new();
        let intended = svc
            .add_knowledge_node(
                "k_intended".into(),
                mk_kn("k_intended", KnowledgeType::Conceptual),
                vec![],
            )
            .unwrap();
        let extraneous = svc
            .add_knowledge_node(
                "k_extra".into(),
                mk_kn("k_extra", KnowledgeType::Conceptual),
                vec![],
            )
            .unwrap();
        let assess = svc
            .add_knowledge_node("a".into(), mk_kn("a", KnowledgeType::AssessmentItem), vec![])
            .unwrap();
        let lo = svc
            .add_knowledge_node("lo".into(), mk_kn("lo", KnowledgeType::LearningOutcome), vec![])
            .unwrap();
        let lo_slug = svc.graph()[lo].slug.clone();
        let ts = svc
            .add_teaching_step(
                "ts".into(),
                graph::TeachingStepNode {
                    title:       "target".into(),
                    statement:   "targets LO".into(),
                    purpose:     graph::TeachingPurpose::Idea,
                    method_tags: vec![],
                    episode:     "ep".into(),
                    source_refs: vec![SourceRef {
                        path:       "dummy".into(),
                        start_line: 1,
                        end_line:   2,
                        revision:   "deadbeef".into(),
                    }],
                    rationale:   None,
                },
                vec![],
            )
            .unwrap();
        svc.add_edge::<graph::AnchorsSpec>(
            ts,
            lo,
            graph::AnchorsAttrs {
                impact: graph::AnchorImpact::Target,
            },
            1.0,
        )
        .unwrap();
        svc.add_edge::<graph::AnchorsSpec>(
            ts,
            intended,
            graph::AnchorsAttrs {
                impact: graph::AnchorImpact::Introduce,
            },
            1.0,
        )
        .unwrap();
        svc.add_edge::<graph::RequiresSpec>(
            intended,
            assess,
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
            extraneous,
            assess,
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
        let err = svc
            .add_edge::<graph::AssessesSpec>(
                assess,
                lo,
                graph::AssessesAttrs {
                    evidence_link: weaver::schema::types::EvidenceLink {
                        scope:                weaver::schema::types::AssessmentScope::Target,
                        claim:                lo_slug.clone(),
                        observation_features: vec!["feat".into()],
                    },
                },
                1.0,
            )
            .expect_err("purity violation should block assesses edge");
        if let graph::GraphError::InvariantViolation { violations } = err {
            assert!(
                violations
                    .iter()
                    .any(|v| v.code == graph::InvariantCode::PurityExtraneous),
                "expected purity_extraneous code, got {violations:?}"
            );
        } else {
            panic!("expected invariant violation, got {err:?}");
        }
    }
}

mod discourse {
    use super::*;

    #[test]
    fn borrow_ahead_flags_use_without_intro() {
        let mut svc = GraphService::new();
        let k_id = svc
            .add_knowledge_node("k".into(), mk_kn("k", KnowledgeType::Conceptual), vec![])
            .unwrap();
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
                    rationale:   None,
                },
                vec![],
            )
            .unwrap();
        let err = svc
            .add_edge::<graph::AnchorsSpec>(
                ts_use,
                k_id,
                graph::AnchorsAttrs {
                    impact: graph::AnchorImpact::Use,
                },
                1.0,
            )
            .expect_err("borrow-ahead should block use before introduce");
        if let graph::GraphError::InvariantViolation { violations } = err {
            assert!(
                violations
                    .iter()
                    .any(|v| v.code == graph::InvariantCode::BorrowAhead),
            );
        } else {
            panic!("expected invariant violation, got {err:?}");
        }
    }

    #[test]
    fn orphan_step_rejected_under_strict() {
        let mut svc = GraphService::new();
        svc.add_teaching_step(
            "ts".into(),
            graph::TeachingStepNode {
                title:       "orphan".into(),
                statement:   "lonely step".into(),
                purpose:     graph::TeachingPurpose::Setup,
                method_tags: vec![],
                episode:     "ep".into(),
                source_refs: vec![SourceRef {
                    path:       "dummy".into(),
                    start_line: 1,
                    end_line:   2,
                    revision:   "deadbeef".into(),
                }],
                rationale:   None,
            },
            vec![],
        )
        .unwrap();
        let err = svc
            .set_strict_quality(true)
            .expect_err("strict mode should reject discourse orphans");
        if let graph::GraphError::InvariantViolation { violations } = err {
            assert!(
                violations
                    .iter()
                    .any(|v| v.code == graph::InvariantCode::DiscourseOrphan),
                "expected discourse_orphan code"
            );
        } else {
            panic!("expected invariant violation, got {err:?}");
        }
    }

    #[test]
    fn borrow_ahead_marks_cross_episode_use() {
        let mut svc = GraphService::new();
        let k = svc
            .add_knowledge_node("k".into(), mk_kn("k", KnowledgeType::Conceptual), vec![])
            .unwrap();
        let ts_intro = svc
            .add_teaching_step(
                "ts_intro".into(),
                graph::TeachingStepNode {
                    title:       "intro".into(),
                    statement:   "introduce".into(),
                    purpose:     graph::TeachingPurpose::Setup,
                    method_tags: vec![],
                    episode:     "ep0".into(),
                    source_refs: vec![SourceRef {
                        path:       "dummy".into(),
                        start_line: 1,
                        end_line:   2,
                        revision:   "deadbeef".into(),
                    }],
                    rationale:   None,
                },
                vec![],
            )
            .unwrap();
        let ts_use = svc
            .add_teaching_step(
                "ts_use".into(),
                graph::TeachingStepNode {
                    title:       "use".into(),
                    statement:   "use".into(),
                    purpose:     graph::TeachingPurpose::Use,
                    method_tags: vec![],
                    episode:     "ep1".into(),
                    source_refs: vec![SourceRef {
                        path:       "dummy".into(),
                        start_line: 1,
                        end_line:   2,
                        revision:   "deadbeef".into(),
                    }],
                    rationale:   None,
                },
                vec![],
            )
            .unwrap();
        svc.add_edge::<graph::AnchorsSpec>(
            ts_intro,
            k,
            graph::AnchorsAttrs {
                impact: graph::AnchorImpact::Introduce,
            },
            1.0,
        )
        .unwrap();
        svc.add_edge::<graph::AnchorsSpec>(
            ts_use,
            k,
            graph::AnchorsAttrs {
                impact: graph::AnchorImpact::Use,
            },
            1.0,
        )
        .expect_err("cross-episode borrow-ahead should be flagged");
        let err = svc
            .set_strict_quality(true)
            .expect_err("should still flag borrow-ahead");
        if let graph::GraphError::InvariantViolation { violations } = err {
            assert!(
                violations
                    .iter()
                    .any(|v| v.code == graph::InvariantCode::TeachingStepAnchorOrRationale),
                "expected anchor/rationale violation when use step remains unanchored"
            );
            assert!(
                !violations
                    .iter()
                    .any(|v| v.code == graph::InvariantCode::BorrowAhead),
                "unanchored use without anchors should avoid double-reporting borrow_ahead"
            );
        }
    }

    #[test]
    fn borrow_ahead_suppresses_prior_scope() {
        let mut svc = GraphService::new();
        let mut prior_kn = mk_kn("k", KnowledgeType::Conceptual);
        prior_kn.introduction_scope = graph::IntroductionScope::Prior;
        let k = svc
            .add_knowledge_node("k".into(), prior_kn, vec![])
            .unwrap();
        let ts_use = svc
            .add_teaching_step(
                "ts_use".into(),
                graph::TeachingStepNode {
                    title:       "use".into(),
                    statement:   "use".into(),
                    purpose:     graph::TeachingPurpose::Use,
                    method_tags: vec![],
                    episode:     "ep1".into(),
                    source_refs: vec![SourceRef {
                        path:       "dummy".into(),
                        start_line: 1,
                        end_line:   2,
                        revision:   "deadbeef".into(),
                    }],
                    rationale:   None,
                },
                vec![],
            )
            .unwrap();
        svc.add_edge::<graph::AnchorsSpec>(
            ts_use,
            k,
            graph::AnchorsAttrs {
                impact: graph::AnchorImpact::Use,
            },
            1.0,
        )
        .unwrap();

        let results = analysis::borrow_ahead(svc.graph(), "ep1");
        assert!(results.is_empty());
    }

    #[test]
    fn unanchored_use_without_rationale_reports_once() {
        let mut svc = GraphService::new();
        svc.add_teaching_step(
            "ts_use".into(),
            graph::TeachingStepNode {
                title:       "use".into(),
                statement:   "use it".into(),
                purpose:     graph::TeachingPurpose::Use,
                method_tags: vec![],
                episode:     "ep".into(),
                source_refs: vec![SourceRef {
                    path:       "dummy".into(),
                    start_line: 1,
                    end_line:   2,
                    revision:   "deadbeef".into(),
                }],
                rationale:   None,
            },
            vec![],
        )
        .unwrap();
        let err = svc
            .set_strict_quality(true)
            .expect_err("missing anchor/rationale should fail in strict mode");
        if let graph::GraphError::InvariantViolation { violations } = err {
            assert!(
                violations
                    .iter()
                    .any(|v| v.code == graph::InvariantCode::TeachingStepAnchorOrRationale),
                "expected anchor/rationale warning promoted to error"
            );
            assert!(
                !violations
                    .iter()
                    .any(|v| v.code == graph::InvariantCode::BorrowAhead),
                "borrow_ahead should not double-report unanchored use without rationale"
            );
        } else {
            panic!("expected invariant violation");
        }
    }

    #[test]
    fn unanchored_use_with_rationale_warns_about_borrow_ahead() {
        let mut svc = GraphService::new();
        svc.add_teaching_step(
            "ts_use".into(),
            graph::TeachingStepNode {
                title:       "use".into(),
                statement:   "use it".into(),
                purpose:     graph::TeachingPurpose::Use,
                method_tags: vec![],
                episode:     "ep".into(),
                source_refs: vec![SourceRef {
                    path:       "dummy".into(),
                    start_line: 1,
                    end_line:   2,
                    revision:   "deadbeef".into(),
                }],
                rationale:   Some("we intend to reuse prior knowledge".into()),
            },
            vec![],
        )
        .unwrap();
        let err = svc
            .set_strict_quality(true)
            .expect_err("borrow-ahead check should fire for unanchored use with rationale");
        if let graph::GraphError::InvariantViolation { violations } = err {
            assert!(
                violations
                    .iter()
                    .any(|v| v.code == graph::InvariantCode::BorrowAhead),
                "expected borrow_ahead code"
            );
            assert!(
                !violations
                    .iter()
                    .any(|v| v.code == graph::InvariantCode::TeachingStepAnchorOrRationale),
                "anchor/rationale warning should not double-report when rationale present"
            );
        } else {
            panic!("expected invariant violation");
        }
    }
}

mod hygiene {
    use super::*;

    #[test]
    fn source_ref_revision_mismatch_is_error() {
        let mut svc = GraphService::new();
        svc.set_expected_revision(Some("expected".into()));
        let res = svc.add_knowledge_node("k".into(), mk_kn("k", KnowledgeType::Conceptual), vec![]);
        assert!(res.is_err());
    }

    #[test]
    fn empty_statement_rejected() {
        let mut svc = GraphService::new();
        let mut node = mk_kn("k", KnowledgeType::Conceptual);
        node.statement = "".into();
        let err = svc
            .add_knowledge_node("k".into(), node, vec![])
            .expect_err("empty statement should fail");
        assert!(matches!(err, graph::GraphError::Schema(_)));
    }
}

mod persistence_topology {
    use super::*;

    #[test]
    fn graph_manager_state_round_trip_preserves_version() {
        let mut svc = GraphService::new();
        let _ = svc
            .add_knowledge_node("k".into(), mk_kn("k", KnowledgeType::Conceptual), vec![])
            .unwrap();
        let version = svc.graph_version();
        let state = GraphManagerState::new(
            svc.snapshot_graph_owned(),
            "deadbeef".into(),
            false,
            version,
            2_000,
            false,
        );
        let data = postcard::to_stdvec(&state).expect("serialize state");
        let decoded: GraphManagerState = postcard::from_bytes(&data).expect("deserialize state");
        let restored = GraphService::from_parts(
            decoded.graph.clone(),
            decoded.strict_quality,
            decoded.graph_version,
            None,
            decoded.skip_dedup_on_insert,
        )
        .expect("restored graph should validate");
        assert_eq!(restored.graph_version(), version);
        restored.validate_global_invariants().unwrap();
    }

    #[test]
    fn requires_transitive_reduction_identifies_redundant_edge() {
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
        for (u, v) in &[(a, b), (b, c), (a, c)] {
            svc.add_edge::<graph::RequiresSpec>(
                *u,
                *v,
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
        }
        let redundants = traversal::requires_transitive_reduction(svc.graph()).unwrap();
        assert!(redundants.contains(&(a, c)));
        assert!(traversal::requires_path_exists(svc.graph(), a, c));
    }

    #[test]
    fn borrow_ahead_handles_long_episode() {
        let mut g = graph::CurriculumGraph::default();
        let src = SourceRef {
            path:       "dummy".into(),
            start_line: 1,
            end_line:   2,
            revision:   "deadbeef".into(),
        };
        let target = g.add_node(graph::NodePayload {
            logical_id: uuid::Uuid::new_v4(),
            slug:       "k".into(),
            kind:       graph::NodeKind::Knowledge(KnowledgeNode {
                title: "k".into(),
                statement: "k".into(),
                knowledge_type: KnowledgeType::Conceptual,
                source_refs: vec![src.clone()],
                confidence: 1.0,
                rubric_criteria: vec![],
                construct_irrelevant_demands: vec![],
                grain_level: None,
                intrinsic_load: None,
                introduction_scope: graph::IntroductionScope::InCourse,
            }),
            tags:       vec![],
        });

        let episode = "episode-1";
        let mut steps = Vec::new();
        for i in 0..200usize {
            let node = graph::NodePayload {
                logical_id: uuid::Uuid::new_v4(),
                slug:       format!("ts{i}"),
                kind:       graph::NodeKind::TeachingStep(TeachingStepNode {
                    title:       format!("Step {i}"),
                    statement:   format!("step {i}"),
                    purpose:     TeachingPurpose::Use,
                    method_tags: vec![],
                    episode:     episode.into(),
                    source_refs: vec![src.clone()],
                    rationale:   None,
                }),
                tags:       vec![],
            };
            steps.push(g.add_node(node));
        }

        for window in steps.windows(2) {
            if let [from, to] = *window {
                g.add_edge(
                    from,
                    to,
                    EdgePayload {
                        kind:       EdgeKind::Precedes(PrecedesAttrs {
                            episode: episode.into(),
                        }),
                        confidence: 1.0,
                    },
                );
            }
        }

        let intro_idx = 120usize;
        g.add_edge(
            steps[intro_idx],
            target,
            EdgePayload {
                kind:       EdgeKind::Anchors(AnchorsAttrs {
                    impact: AnchorImpact::Introduce,
                }),
                confidence: 1.0,
            },
        );

        for (idx, step) in steps.iter().enumerate() {
            if idx == intro_idx {
                continue;
            }
            g.add_edge(
                *step,
                target,
                EdgePayload {
                    kind:       EdgeKind::Anchors(AnchorsAttrs {
                        impact: AnchorImpact::Use,
                    }),
                    confidence: 1.0,
                },
            );
        }

        let start = std::time::Instant::now();
        let results = analysis::borrow_ahead(&g, episode);
        let elapsed = start.elapsed();

        assert!(
            elapsed < std::time::Duration::from_millis(500),
            "borrow_ahead should remain near-linear even for long episodes (elapsed: {:?})",
            elapsed
        );
        assert_eq!(results.len(), intro_idx, "uses before introduction should be flagged");
        assert!(
            results
                .iter()
                .all(|b| matches!(b.severity, analysis::BorrowSeverity::InEpisode))
        );
    }

    #[test]
    fn invariant_validation_latency_is_bounded_under_mutation_load() {
        let mut svc = GraphService::new();
        svc.set_skip_dedup_on_insert(true);

        let evidence = SourceRef {
            path:       "dummy".into(),
            start_line: 1,
            end_line:   2,
            revision:   "deadbeef".into(),
        };

        let mut first_principles = Vec::new();
        for i in 0..8 {
            first_principles.push(
                svc.add_knowledge_node(
                    format!("fp-{i}"),
                    KnowledgeNode {
                        introduction_scope: graph::IntroductionScope::Prior,
                        ..mk_kn(&format!("fp-{i}"), KnowledgeType::Factual)
                    },
                    vec![],
                )
                .expect("first principle insertion should succeed"),
            );
        }

        let mut assessments = Vec::new();
        for i in 0..48 {
            let lo_slug = format!("lo-{i}");
            let crit = format!("crit-{i}");
            let mut lo = mk_kn(&lo_slug, KnowledgeType::LearningOutcome);
            lo.rubric_criteria = vec![crit.clone()];
            let lo_id = svc
                .add_knowledge_node(lo_slug.clone(), lo, vec![])
                .expect("add lo");
            let lo_slug_canonical = svc.graph()[lo_id].slug.clone();

            let assess_slug = format!("assess-{i}");
            let assess = mk_kn(&assess_slug, KnowledgeType::AssessmentItem);
            let assess_id = svc
                .add_knowledge_node(assess_slug, assess, vec![])
                .expect("add assessment");

            svc.add_edge::<graph::AssessesSpec>(
                assess_id,
                lo_id,
                graph::AssessesAttrs {
                    evidence_link: EvidenceLink {
                        claim:                lo_slug_canonical,
                        observation_features: vec![crit],
                        scope:                AssessmentScope::Target,
                    },
                },
                1.0,
            )
            .expect("assesses edge");

            assessments.push(assess_id);
        }

        let requires_attr = |label: String| graph::RequiresAttrs {
            strength:      Strength::Necessary,
            rationale:     label,
            evidence_refs: vec![evidence.clone()],
        };

        let start = Instant::now();
        for (i, assess) in assessments.iter().enumerate() {
            let fp = first_principles[i % first_principles.len()];
            svc.add_edge::<graph::RequiresSpec>(
                fp,
                *assess,
                requires_attr(format!("fp->{i}")),
                1.0,
            )
            .expect("requires edge");
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        assert!(
            elapsed_ms < 2500.0,
            "batched invariant checks should stay bounded (elapsed_ms={elapsed_ms})"
        );
    }
}

mod granularity {
    use super::*;

    fn requires_attrs() -> graph::RequiresAttrs {
        graph::RequiresAttrs {
            strength:      Strength::Necessary,
            rationale:     "prereq".into(),
            evidence_refs: vec![SourceRef {
                path:       "dummy".into(),
                start_line: 1,
                end_line:   2,
                revision:   "deadbeef".into(),
            }],
        }
    }

    #[test]
    fn overbundled_nodes_flagged_in_strict_mode() {
        let mut svc = GraphService::new();
        let mut target = mk_kn("bundle", KnowledgeType::Conceptual);
        target.statement = "First sentence. Second sentence. Third sentence.".into();
        let target_id = svc
            .add_knowledge_node("bundle".into(), target, vec![])
            .unwrap();
        for idx in 0..4 {
            let prereq = svc
                .add_knowledge_node(
                    format!("p{idx}"),
                    mk_kn(&format!("p{idx}"), KnowledgeType::Conceptual),
                    vec![],
                )
                .unwrap();
            svc.add_edge::<graph::RequiresSpec>(prereq, target_id, requires_attrs(), 1.0)
                .unwrap();
        }

        let err = svc
            .set_strict_quality(true)
            .expect_err("should fail granularity audit");
        let violations = match err {
            graph::GraphError::InvariantViolation { violations } => violations,
            other => panic!("unexpected error {other:?}"),
        };
        assert!(
            violations
                .iter()
                .any(|v| matches!(v.code, graph::InvariantCode::GrainOverbundled)),
            "expected GrainOverbundled violation"
        );
    }

    #[test]
    fn high_intrinsic_load_requires_supports() {
        let mut svc = GraphService::new();
        let mut target = mk_kn("dense", KnowledgeType::Procedural);
        target.intrinsic_load = Some(graph::IntrinsicLoad::High);
        target.statement =
            "This is a dense procedural node that should not be treated as a fragment.".into();
        let target_id = svc
            .add_knowledge_node("dense".into(), target, vec![])
            .unwrap();
        let prereq = svc
            .add_knowledge_node("p0".into(), mk_kn("p0", KnowledgeType::Conceptual), vec![])
            .unwrap();
        svc.add_edge::<graph::RequiresSpec>(prereq, target_id, requires_attrs(), 1.0)
            .unwrap();

        let err = svc
            .set_strict_quality(true)
            .expect_err("high intrinsic load should require supports");
        let violations = match err {
            graph::GraphError::InvariantViolation { violations } => violations,
            other => panic!("unexpected error {other:?}"),
        };
        assert!(
            violations
                .iter()
                .any(|v| matches!(v.code, graph::InvariantCode::IntrinsicLoadSupport)),
            "expected IntrinsicLoadSupport violation"
        );
    }

    #[test]
    fn fragmented_nodes_detected() {
        let mut svc = GraphService::new();
        let mut target = mk_kn("frag", KnowledgeType::Conceptual);
        target.statement = "too small".into();
        svc.add_knowledge_node("frag".into(), target, vec![])
            .unwrap();

        let err = svc
            .set_strict_quality(true)
            .expect_err("fragmented node should fail strict validation");
        let violations = match err {
            graph::GraphError::InvariantViolation { violations } => violations,
            other => panic!("unexpected error {other:?}"),
        };
        assert!(
            violations
                .iter()
                .any(|v| matches!(v.code, graph::InvariantCode::GrainFragment)),
            "expected GrainFragment violation"
        );
    }
}
