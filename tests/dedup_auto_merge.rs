use kameo::actor::Spawn;
use weaver::{
    agents::deduplication::{DeduplicationAgent, RunDeduplication},
    graph::{
        IntroductionScope, KnowledgeNode, RequiresAttrs, RequiresSpec,
        commands::{GetGraphVersion, ResolveSlug},
        manager::{GraphManager, GraphManagerState},
        service::GraphService,
    },
    schema::types::{KnowledgeType, SourceRef, Strength},
};

fn sample_node(title: &str, statement: &str, revision: &str) -> KnowledgeNode {
    KnowledgeNode {
        title: title.into(),
        statement: statement.into(),
        knowledge_type: KnowledgeType::Conceptual,
        source_refs: vec![SourceRef {
            path:       "dummy".into(),
            start_line: 1,
            end_line:   1,
            revision:   revision.to_string(),
        }],
        confidence: 1.0,
        rubric_criteria: vec![],
        construct_irrelevant_demands: vec![],
        grain_level: None,
        intrinsic_load: None,
        introduction_scope: IntroductionScope::InCourse,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn auto_merge_rolls_back_on_invariant_failure() -> anyhow::Result<()> {
    let revision = "deadbeef";
    let requires = RequiresAttrs {
        strength:      Strength::Necessary,
        rationale:     "prereq chain".into(),
        evidence_refs: vec![SourceRef {
            path:       "dummy".into(),
            start_line: 1,
            end_line:   1,
            revision:   revision.into(),
        }],
    };

    let mut svc = GraphService::new();
    svc.add_knowledge_node(
        "c.alpha".into(),
        sample_node("Alpha", "Loops repeat work", revision),
        vec![],
    )?;
    svc.add_knowledge_node(
        "c.beta".into(),
        sample_node("Beta", "Loop repeats work", revision),
        vec![],
    )?;
    svc.add_knowledge_node(
        "c.gamma".into(),
        sample_node("Gamma", "Helper node", revision),
        vec![],
    )?;

    let alpha = svc.node_by_slug("c.alpha")?;
    let beta = svc.node_by_slug("c.beta")?;
    let gamma = svc.node_by_slug("c.gamma")?;

    svc.add_edge::<RequiresSpec>(beta, gamma, requires.clone(), 1.0)?;
    svc.add_edge::<RequiresSpec>(gamma, alpha, requires, 1.0)?;

    let state = GraphManagerState::new(
        svc.snapshot_graph_owned(),
        revision.into(),
        false,
        svc.graph_version(),
        2_000,
        false,
    );
    let graph_actor = GraphManager::spawn(state);
    let dedup_actor = DeduplicationAgent::spawn(DeduplicationAgent::new(graph_actor.clone(), None));

    let before_version: u64 = graph_actor.ask(GetGraphVersion).await?;
    let report = dedup_actor
        .ask(RunDeduplication {
            auto_merge_threshold: 0.9,
            dry_run:              false,
        })
        .await?;

    assert_eq!(report.auto_merged.len(), 0, "merge should fail validation");
    assert_eq!(report.auto_merged_clusters, 0);
    assert_eq!(report.skipped_clusters, 1);
    assert_eq!(report.skipped_due_to_budget, 0);
    assert_eq!(report.pending_review.len(), 1, "cluster should remain for review");

    let after_version: u64 = graph_actor.ask(GetGraphVersion).await?;
    assert_eq!(before_version, after_version, "failed merge should roll back graph version");

    let duplicate_alive = graph_actor
        .ask(ResolveSlug {
            slug: "c.beta".into(),
        })
        .await
        .is_ok();
    assert!(duplicate_alive, "duplicate node should still exist after rollback");

    let _ = dedup_actor.stop_gracefully().await;
    let _ = graph_actor.stop_gracefully().await;
    Ok(())
}
