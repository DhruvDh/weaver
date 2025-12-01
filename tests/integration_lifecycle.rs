use weaver::{
    constants::MAX_DELEGATED_TASKS,
    graph::{
        GraphError, GraphService, IntroductionScope, KnowledgeNode,
        specs::{AssessesSpec, RequiresSpec},
    },
    schema::types::{AssessmentScope, EvidenceLink, KnowledgeType, SourceRef, Strength},
    tools::llm::DelegateTasksArgs,
};

fn source_ref() -> SourceRef {
    SourceRef {
        path:       "dummy".into(),
        start_line: 1,
        end_line:   1,
        revision:   "deadbeef".into(),
    }
}

fn mk_kn(title: &str, kt: KnowledgeType) -> KnowledgeNode {
    KnowledgeNode {
        title: title.to_string(),
        statement: title.to_string(),
        knowledge_type: kt,
        source_refs: vec![source_ref()],
        confidence: 1.0,
        rubric_criteria: vec!["feature".into()],
        construct_irrelevant_demands: Vec::new(),
        grain_level: None,
        intrinsic_load: None,
        introduction_scope: IntroductionScope::InCourse,
    }
}

fn requires() -> weaver::graph::RequiresAttrs {
    weaver::graph::RequiresAttrs {
        strength:      Strength::Necessary,
        rationale:     "prerequisite".into(),
        evidence_refs: vec![source_ref()],
    }
}

fn assesses(claim: &str) -> weaver::graph::AssessesAttrs {
    weaver::graph::AssessesAttrs {
        evidence_link: EvidenceLink {
            claim:                claim.to_string(),
            observation_features: vec!["feature".into()],
            scope:                AssessmentScope::Target,
        },
    }
}

#[test]
fn graph_lifecycle_validates_and_versions() -> Result<(), GraphError> {
    let mut svc = GraphService::new();
    let concept_slug =
        weaver::graph::slug::Slug::generate(KnowledgeType::Conceptual, "concept_alpha");
    let lo_slug =
        weaver::graph::slug::Slug::generate(KnowledgeType::LearningOutcome, "outcome_alpha");
    let assessment_slug =
        weaver::graph::slug::Slug::generate(KnowledgeType::AssessmentItem, "assessment_alpha");
    let concept = svc.add_knowledge_node(
        concept_slug.to_string(),
        mk_kn("Concept Alpha", KnowledgeType::Conceptual),
        vec![],
    )?;
    let lo = svc.add_knowledge_node(
        lo_slug.to_string(),
        mk_kn("Outcome Alpha", KnowledgeType::LearningOutcome),
        vec![],
    )?;
    let assessment = svc.add_knowledge_node(
        assessment_slug.to_string(),
        mk_kn("Assessment Alpha", KnowledgeType::AssessmentItem),
        vec![],
    )?;

    svc.add_edge::<RequiresSpec>(concept, assessment, requires(), 1.0)?;
    svc.add_edge::<AssessesSpec>(assessment, lo, assesses(lo_slug.as_str()), 1.0)?;

    svc.validate_global_invariants()?;
    assert!(svc.graph_version() > 0);
    Ok(())
}

#[test]
fn requires_cycles_are_rejected() {
    let mut svc = GraphService::new();
    let a = svc
        .add_knowledge_node("a".into(), mk_kn("A", KnowledgeType::Conceptual), vec![])
        .unwrap();
    let b = svc
        .add_knowledge_node("b".into(), mk_kn("B", KnowledgeType::Conceptual), vec![])
        .unwrap();

    svc.add_edge::<RequiresSpec>(a, b, requires(), 1.0).unwrap();
    let err = svc
        .add_edge::<RequiresSpec>(b, a, requires(), 1.0)
        .unwrap_err();
    match err {
        GraphError::RequiresCycle { .. } => {}
        other => panic!("expected requires cycle error, got {other:?}"),
    }
}

#[test]
fn delegate_tasks_limits_batch_size() {
    let tasks = vec!["t".to_string(); MAX_DELEGATED_TASKS + 1];
    let result = DelegateTasksArgs::builder()
        .tasks(tasks)
        .map(|builder| builder.build());
    assert!(result.is_err(), "builder should enforce max delegated tasks");
}
