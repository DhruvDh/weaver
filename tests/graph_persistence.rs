use std::{fs, path::PathBuf};

use kameo_persistence::PersistentActor;
use url::Url;
use uuid::Uuid;
use weaver::{
    graph::{
        IntroductionScope,
        manager::{GraphManager, GraphManagerState},
        model::{CurriculumGraph, KnowledgeNode, NodeKind, NodePayload},
        service::GraphService,
    },
    schema::types::{KnowledgeType, SourceRef},
};

#[tokio::test(flavor = "multi_thread")]
async fn corrupt_snapshot_fails_start_and_quarantines() -> anyhow::Result<()> {
    let state_dir: PathBuf =
        std::env::temp_dir().join(format!("weaver-quarantine-{}", Uuid::new_v4()));
    fs::create_dir_all(&state_dir)?;
    let persistence_key =
        Url::from_directory_path(&state_dir).expect("valid temp dir url for persistence");

    // Build a graph with provenance that mismatches course_commit to force
    // validation failure.
    let mut graph = CurriculumGraph::default();
    let bad_source = SourceRef {
        path:       "dummy".into(),
        start_line: 1,
        end_line:   1,
        revision:   "badbeef".into(),
    };
    let node = NodePayload {
        logical_id: Uuid::new_v4(),
        slug:       "k1".into(),
        kind:       NodeKind::Knowledge(KnowledgeNode {
            title: "k1".into(),
            statement: "k1".into(),
            knowledge_type: KnowledgeType::Conceptual,
            source_refs: vec![bad_source.clone()],
            confidence: 1.0,
            rubric_criteria: vec![],
            construct_irrelevant_demands: vec![],
            grain_level: None,
            intrinsic_load: None,
            introduction_scope: IntroductionScope::InCourse,
        }),
        tags:       vec![],
    };
    graph.add_node(node);

    assert!(
        GraphService::from_parts(graph.clone(), false, 7, Some("deadbeef".to_string()), false)
            .is_err(),
        "invalid snapshot should fail validation"
    );

    let state = GraphManagerState::new(graph.clone(), "deadbeef".into(), false, 7, 2_000, false);
    let snapshot_bytes = postcard::to_stdvec(&state)?;
    fs::write(state_dir.join("index.bin"), snapshot_bytes)?;

    let actor = GraphManager::respawn_persistent(persistence_key).await?;
    let startup = actor.wait_for_startup_result().await;
    assert!(startup.is_err(), "startup should fail for corrupt persisted snapshot");

    // Persisted bytes should be intact (no silent zeroing).
    let saved: GraphManagerState = postcard::from_bytes(&fs::read(state_dir.join("index.bin"))?)?;
    assert_eq!(saved.course_commit, "deadbeef");
    assert_eq!(saved.graph.node_count(), graph.node_count());

    let quarantines: Vec<_> = fs::read_dir(&state_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with("quarantine-")
        })
        .collect();
    assert!(!quarantines.is_empty(), "expected quarantined snapshot alongside index.bin");
    let quarantine_body = fs::read_to_string(quarantines[0].path())?;
    assert!(quarantine_body.contains("deadbeef"));
    assert!(quarantine_body.contains("revision"));

    Ok(())
}
