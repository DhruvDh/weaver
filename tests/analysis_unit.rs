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
        let proc_gap = gaps
            .into_iter()
            .find(|g| svc.graph()[g.node].slug == "proc")
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
                    .any(|v| v.contains("learning_outcome") && v.contains("assesses")),
                "violations missing stranded LO message: {violations:?}"
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
                    .any(|v| v.contains("lacks reachable assessment")),
                "violations missing practice gap: {violations:?}"
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
                    claim:                "lo".into(),
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
            assert!(violations.iter().any(|v| v.contains("missing coverage")));
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
                    claim:                "lo".into(),
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
            assert!(violations.iter().any(|v| v.contains("missing coverage")));
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
                        claim:                "lo".into(),
                        observation_features: vec!["feat".into()],
                    },
                },
                1.0,
            )
            .expect_err("purity violation should block assesses edge");
        if let graph::GraphError::InvariantViolation { violations } = err {
            assert!(violations.iter().any(|v| v.contains("purity violation")));
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
                    .any(|v| v.contains("borrow-ahead") && v.contains("NoIntro"))
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
            assert!(violations.iter().any(|v| v.contains("orphaned")));
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
        let state = GraphManagerState::new(svc.snapshot_graph(), "deadbeef".into(), false, version);
        let data = postcard::to_stdvec(&state).expect("serialize state");
        let decoded: GraphManagerState = postcard::from_bytes(&data).expect("deserialize state");
        let restored = GraphService::from_parts(
            decoded.graph.clone(),
            decoded.strict_quality,
            decoded.graph_version,
            None,
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
}
