use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use futures::FutureExt;
use uuid::Uuid;
use weaver::{
    app::{Cli, GatewayMode, RerunMode, RuntimeOptions, run_app},
    graph::{IntroductionScope, commands::InsertKnowledge, model::KnowledgeNode, persist},
    schema::types::{KnowledgeType, SourceRef},
};

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
        rerun_mode:                  RerunMode::None,
        rerun_file:                  root.join("noop.rrd"),
        workspace:                   root.join("workspace"),
        graph_snapshot_path:         snapshot.to_path_buf(),
        graph_autosave_secs:         autosave_secs,
        graph_course_commit:         Some(commit.to_string()),
        graph_strict_quality:        false,
        graph_prune_requires_s:      None,
        skip_demo:                   true,
        graph_validation_timeout_ms: 2_000,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn cli_persists_snapshot_and_hook_mutation() -> anyhow::Result<()> {
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
    let has_k1 = loaded
        .graph
        .node_indices()
        .any(|n| loaded.graph[n].slug == "k1");
    assert!(has_k1, "graph snapshot should include inserted node");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn autosave_interval_zero_is_rejected() {
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
