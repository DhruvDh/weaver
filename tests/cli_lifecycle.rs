use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Once},
};

use futures::FutureExt;
use kameo_persistence::PersistentActor;
use tracing_subscriber::EnvFilter;
use url::Url;
use uuid::Uuid;
use weaver::{
    app::{Cli, GatewayMode, RuntimeOptions, run_app},
    graph::{
        CurriculumGraph, EdgeKind, EdgePayload, IntroductionScope, KnowledgeNode, NodeKind,
        commands::InsertKnowledge,
        manager::{ApplyRuntimeConfig, GraphManager, GraphManagerState},
        model::{NodePayload, RequiresAttrs},
        persist,
    },
    schema::types::{KnowledgeType, SourceRef, Strength},
};

static LOG_INIT: Once = Once::new();

fn init_test_logging() {
    // Silence extremely chatty invariant warnings in test runs while preserving
    // other warning/error output.
    LOG_INIT.call_once(|| {
        let filter =
            EnvFilter::new("warn,weaver.graph.invariants=error,weaver.graph.validation=error");
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .try_init();
    });
}

fn temp_root(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("weaver-cli-{label}-{}", Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("create temp root");
    dir
}

fn make_node(commit: &str) -> KnowledgeNode {
    KnowledgeNode {
        title: "k1".into(),
        statement: "k1".into(),
        knowledge_type: KnowledgeType::Conceptual,
        source_refs: vec![SourceRef {
            path:       "dummy".into(),
            start_line: 1,
            end_line:   1,
            revision:   commit.to_string(),
        }],
        confidence: 1.0,
        rubric_criteria: vec![],
        construct_irrelevant_demands: vec![],
        grain_level: None,
        intrinsic_load: None,
        introduction_scope: IntroductionScope::InCourse,
    }
}

fn base_cli(root: &Path, snapshot: &Path, commit: &str, autosave_secs: u64) -> Cli {
    Cli {
        rerun_file:                  Some(root.join("noop.rrd")),
        workspace:                   root.join("workspace"),
        graph_snapshot_path:         snapshot.to_path_buf(),
        graph_autosave_secs:         autosave_secs,
        graph_course_commit:         Some(commit.to_string()),
        graph_strict_quality:        false,
        graph_prune_requires_s:      None,
        skip_demo:                   true,
        graph_validation_timeout_ms: 2_000,
        dedup_interval_secs:         0,
        dedup_auto_merge_threshold:  0.95,
        skip_dedup_on_insert:        false,
        interactive:                 true,
        interactive_writable:        false,
        analyst:                     false,
        harvest_timeout_hours:       None,
        weave_timeout_hours:         None,
        max_concurrent_chapters:     4,
        chapters_pattern:            None,
        chapters_dir:                None,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn cli_persists_snapshot_and_hook_mutation() -> anyhow::Result<()> {
    init_test_logging();

    let root = temp_root("persist");
    fs::create_dir_all(root.join("workspace"))?;
    let snapshot = root.join("graph_snapshot.json");
    let commit = "deadbeef";
    let cli = base_cli(&root, &snapshot, commit, 2);

    let runtime = RuntimeOptions {
        gateway_mode: GatewayMode::Stub,
        on_started: Some(Arc::new({
            let commit = commit.to_string();
            move |handles| {
                let node = make_node(&commit);
                async move {
                    handles
                        .graph
                        .ask(InsertKnowledge {
                            slug:    "k1".into(),
                            payload: node,
                            tags:    vec![],
                        })
                        .await?;
                    Ok(())
                }
                .boxed()
            }
        })),
        ..Default::default()
    };

    run_app(cli, runtime).await?;

    assert!(snapshot.exists(), "snapshot should be written after run");
    let loaded = persist::load_graph(&snapshot).await?;
    let expected_slug = weaver::graph::slug::Slug::generate(KnowledgeType::Conceptual, "k1");
    let has_k1 = loaded
        .graph
        .node_indices()
        .any(|n| loaded.graph[n].slug == expected_slug.as_str());
    assert!(has_k1, "graph snapshot should include inserted node");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn autosave_interval_zero_is_rejected() {
    init_test_logging();

    let root = temp_root("autosave-zero");
    fs::create_dir_all(root.join("workspace")).unwrap();
    let snapshot = root.join("graph_snapshot.json");
    let cli = base_cli(&root, &snapshot, "badc0de", 0);
    let runtime = RuntimeOptions {
        gateway_mode: GatewayMode::Stub,
        ..Default::default()
    };

    let result = run_app(cli, runtime).await;
    assert!(result.is_err(), "zero autosave interval should fail fast");
}

#[tokio::test(flavor = "multi_thread")]
async fn skip_demo_allows_missing_openai_model() -> anyhow::Result<()> {
    init_test_logging();

    // Safe in test process: we only need to clear this for the next call and
    // no other threads rely on it.
    unsafe {
        std::env::remove_var("OPENAI_MODEL");
    }
    let root = temp_root("skip-demo");
    fs::create_dir_all(root.join("workspace"))?;
    let snapshot = root.join("graph_snapshot.json");
    let cli = base_cli(&root, &snapshot, "feedbabe", 5);
    let runtime = RuntimeOptions {
        gateway_mode: GatewayMode::Stub,
        ..Default::default()
    };

    run_app(cli, runtime).await?;

    // Without skip_demo the same config should fail due to OPENAI_MODEL missing.
    let mut no_skip_cli = base_cli(&root, &snapshot, "feedbabe", 5);
    no_skip_cli.skip_demo = false;
    let runtime = RuntimeOptions {
        gateway_mode: GatewayMode::Stub,
        ..Default::default()
    };
    let result = run_app(no_skip_cli, runtime).await;
    assert!(
        result.is_err(),
        "expected FileReader init to fail when skip_demo is false and OPENAI_MODEL is absent"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn apply_runtime_config_respects_timeout() -> anyhow::Result<()> {
    init_test_logging();

    const NODES: usize = 2_000;
    let state_dir: PathBuf =
        std::env::temp_dir().join(format!("weaver-apply-config-{}", Uuid::new_v4()));
    fs::create_dir_all(&state_dir)?;
    let state_url =
        Url::from_directory_path(&state_dir).map_err(|_| anyhow::anyhow!("invalid state url"))?;

    let evidence = SourceRef {
        path:       "dummy".into(),
        start_line: 1,
        end_line:   1,
        revision:   "deadbeef".into(),
    };

    let mut graph = CurriculumGraph::default();
    let mut ids = Vec::with_capacity(NODES);
    for i in 0..NODES {
        let node = KnowledgeNode {
            title: format!("k{i}"),
            statement: "stmt".into(),
            knowledge_type: KnowledgeType::Conceptual,
            source_refs: vec![evidence.clone()],
            confidence: 1.0,
            rubric_criteria: vec![],
            construct_irrelevant_demands: vec![],
            grain_level: None,
            intrinsic_load: None,
            introduction_scope: IntroductionScope::InCourse,
        };
        let payload = NodePayload {
            logical_id: uuid::Uuid::new_v4(),
            slug:       format!("k{i}"),
            kind:       NodeKind::Knowledge(node),
            tags:       vec![],
        };
        ids.push(graph.add_node(payload));
    }

    let requires = RequiresAttrs {
        strength:      Strength::Necessary,
        rationale:     "chain".into(),
        evidence_refs: vec![evidence.clone()],
    };
    for i in 0..(NODES - 1) {
        graph.add_edge(
            ids[i],
            ids[i + 1],
            EdgePayload::new(EdgeKind::Requires(requires.clone()), 1.0),
        );
    }

    let state = GraphManagerState::new(graph, String::new(), false, 0, 2_000, false);
    let actor = GraphManager::spawn_persistent(state_url, state).await?;

    let result = actor
        .ask(ApplyRuntimeConfig {
            course_commit:         String::new(),
            strict_quality:        false,
            validation_timeout_ms: 1,
        })
        .await;

    let err = match result {
        Err(kameo::error::SendError::HandlerError(err)) => err,
        other => panic!("expected handler timeout error, got {other:?}"),
    };
    let graph_err = err
        .downcast_ref::<weaver::graph::GraphError>()
        .expect("error should be GraphError");
    match graph_err {
        weaver::graph::GraphError::Operational(
            weaver::graph::GraphOperationalError::InvariantTimeout { timeout_ms },
        ) => assert_eq!(*timeout_ms, 1, "timeout should propagate configured limit"),
        other => panic!("expected invariant timeout, got {other:?}"),
    }

    actor.stop_gracefully().await.expect("stop graph manager");
    actor.wait_for_shutdown().await;
    Ok(())
}
