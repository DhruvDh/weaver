use std::{fs, path::PathBuf};

use kameo::{actor::Spawn, error::SendError, prelude::*};
use kameo_persistence::PersistentActor;
use url::Url;
use uuid::Uuid;
use weaver::{
    graph::{
        GraphService, KnowledgeNode,
        commands::{InsertKnowledge, ResolveSlug},
        manager::{GraphManager, GraphManagerState, PersistSnapshot},
    },
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

struct ValidationDelayGuard {
    prev_ms: u64,
}

impl ValidationDelayGuard {
    fn set(delay_ms: u64) -> Self {
        let prev_ms = weaver::graph::service::test_support::set_test_validation_delay_ms(delay_ms);
        Self { prev_ms }
    }
}

impl Drop for ValidationDelayGuard {
    fn drop(&mut self) {
        let _ = weaver::graph::service::test_support::set_test_validation_delay_ms(self.prev_ms);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn graph_manager_persists_and_restores_state() -> anyhow::Result<()> {
    let state_dir: PathBuf =
        std::env::temp_dir().join(format!("weaver-graph-state-{}", Uuid::new_v4()));
    fs::create_dir_all(&state_dir)?;
    let state_url =
        Url::from_directory_path(&state_dir).map_err(|_| anyhow::anyhow!("invalid state url"))?;

    let mut svc = GraphService::new();
    svc.add_knowledge_node("k1".into(), mk_kn("k1", KnowledgeType::Conceptual), vec![])?;
    let base_version = svc.graph_version();

    let state = GraphManagerState::new(
        svc.snapshot_graph_owned(),
        String::new(),
        false,
        base_version,
        2_000,
        false,
    );

    let actor = GraphManager::spawn_persistent(state_url.clone(), state).await?;

    actor
        .ask(InsertKnowledge {
            slug:    "k2".into(),
            payload: mk_kn("k2", KnowledgeType::Procedural),
            tags:    vec![],
        })
        .await?;

    actor.ask(PersistSnapshot).await?;
    actor.stop_gracefully().await.expect("stop graph manager");
    actor.wait_for_shutdown().await;
    drop(actor);

    let restored = GraphManager::respawn_persistent(state_url.clone()).await?;
    restored
        .ask(ResolveSlug { slug: "k1".into() })
        .await
        .expect("k1 restored");
    restored
        .ask(ResolveSlug { slug: "k2".into() })
        .await
        .expect("k2 restored");

    let snapshot_bytes = fs::read(state_dir.join("index.bin"))?;
    let snapshot: GraphManagerState = postcard::from_bytes(&snapshot_bytes)?;
    assert_eq!(snapshot.graph_version, base_version + 1);
    assert!(snapshot.course_commit.is_empty());

    restored
        .stop_gracefully()
        .await
        .expect("stop restored graph manager");
    restored.wait_for_shutdown().await;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn apply_runtime_config_honors_validation_timeout() -> anyhow::Result<()> {
    let _delay_guard = ValidationDelayGuard::set(50);

    let evidence = SourceRef {
        path:       "dummy".into(),
        start_line: 1,
        end_line:   1,
        revision:   "deadbeef".into(),
    };

    let mut svc = GraphService::new();
    svc.add_knowledge_node("k1".into(), mk_kn("k1", KnowledgeType::Conceptual), vec![])?;
    svc.add_knowledge_node("k2".into(), mk_kn("k2", KnowledgeType::Procedural), vec![])?;
    let from = svc.node_by_slug("k1")?;
    let to = svc.node_by_slug("k2")?;
    let attrs = weaver::graph::RequiresAttrs {
        strength:      weaver::schema::types::Strength::Necessary,
        rationale:     "seq".into(),
        evidence_refs: vec![evidence.clone()],
    };
    svc.add_edge::<weaver::graph::RequiresSpec>(from, to, attrs, 1.0)?;

    let _delay_guard = ValidationDelayGuard::set(50);

    let state = GraphManagerState::new(
        svc.snapshot_graph_owned(),
        String::new(),
        false,
        svc.graph_version(),
        2_000,
        false,
    );

    let actor: ActorRef<GraphManager> = GraphManager::spawn(state);

    match actor
        .ask(weaver::graph::manager::ApplyRuntimeConfig {
            course_commit:         String::new(),
            strict_quality:        false,
            validation_timeout_ms: 1,
            source_root:           None,
        })
        .await
    {
        Ok(()) => panic!("expected validation timeout, got success"),
        Err(SendError::HandlerError(err)) => {
            let graph_err = err
                .downcast_ref::<weaver::graph::GraphError>()
                .expect("graph error");
            match graph_err {
                weaver::graph::GraphError::Operational(
                    weaver::graph::GraphOperationalError::InvariantTimeout { timeout_ms },
                ) => assert_eq!(*timeout_ms, 1),
                other => panic!("unexpected error {other:?}"),
            }
        }
        Err(send_err) => panic!("unexpected send error: {send_err:?}"),
    }

    actor
        .stop_gracefully()
        .await
        .expect("stop graph manager actor");
    actor.wait_for_shutdown().await;

    Ok(())
}
