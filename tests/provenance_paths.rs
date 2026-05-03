use std::{fs, path::PathBuf};

use uuid::Uuid;
use weaver::{
    graph::{
        CurriculumGraph, GraphError, GraphService, InvariantCode, NodeKind, NodePayload,
        model::KnowledgeNode, persist::GraphSnapshot,
    },
    schema::types::{KnowledgeType, SourceRef},
};

fn temp_workspace(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("weaver-{name}-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).expect("create temp workspace");
    root
}

fn graph_with_source(path: impl Into<String>, end_line: u32) -> CurriculumGraph {
    let mut graph = CurriculumGraph::default();
    graph.add_node(NodePayload {
        logical_id: Uuid::new_v4(),
        slug:       "k1".into(),
        kind:       NodeKind::Knowledge(KnowledgeNode {
            title: "k1".into(),
            statement: "k1".into(),
            knowledge_type: KnowledgeType::Conceptual,
            source_refs: vec![SourceRef {
                path: path.into(),
                start_line: 1,
                end_line,
                revision: "deadbeef".into(),
            }],
            confidence: 1.0,
            rubric_criteria: vec![],
            construct_irrelevant_demands: vec![],
            grain_level: None,
            intrinsic_load: None,
            introduction_scope: weaver::graph::IntroductionScope::InCourse,
        }),
        tags:       vec![],
    });
    graph
}

fn build_with_source_root(
    root: &PathBuf,
    path: impl Into<String>,
    end_line: u32,
) -> Result<GraphService, GraphError> {
    GraphService::from_parts_with_source_root(
        graph_with_source(path, end_line),
        false,
        0,
        Some("deadbeef".into()),
        false,
        Some(root.clone()),
    )
}

fn assert_provenance_path_error(err: GraphError) {
    match err {
        GraphError::InvariantViolation { violations } => assert!(
            violations
                .iter()
                .any(|v| v.code == InvariantCode::ProvenancePath),
            "expected provenance_path violation, got {violations:?}"
        ),
        other => panic!("expected invariant violation, got {other:?}"),
    }
}

fn unwrap_err(result: Result<GraphService, GraphError>) -> GraphError {
    match result {
        Ok(_) => panic!("expected provenance path error"),
        Err(err) => err,
    }
}

#[test]
fn valid_relative_source_ref_path_passes_with_source_root() {
    let root = temp_workspace("provenance-valid");
    fs::write(root.join("source.ptx"), "one\ntwo\n").expect("write source");

    build_with_source_root(&root, "source.ptx", 2).expect("valid provenance");
}

#[test]
fn absolute_source_ref_path_fails_with_source_root() {
    let root = temp_workspace("provenance-absolute");
    let source = root.join("source.ptx");
    fs::write(&source, "one\n").expect("write source");

    let err = unwrap_err(build_with_source_root(&root, source.display().to_string(), 1));

    assert_provenance_path_error(err);
}

#[test]
fn parent_traversal_source_ref_path_fails_with_source_root() {
    let root = temp_workspace("provenance-parent-root");
    let outside = temp_workspace("provenance-parent-outside");
    fs::write(outside.join("source.ptx"), "one\n").expect("write source");

    let err = unwrap_err(build_with_source_root(&root, "../source.ptx", 1));

    assert_provenance_path_error(err);
}

#[test]
fn missing_source_ref_file_fails_with_source_root() {
    let root = temp_workspace("provenance-missing");

    let err = unwrap_err(build_with_source_root(&root, "missing.ptx", 1));

    assert_provenance_path_error(err);
}

#[test]
fn source_ref_end_line_past_eof_fails_with_source_root() {
    let root = temp_workspace("provenance-eof");
    fs::write(root.join("source.ptx"), "one\n").expect("write source");

    let err = unwrap_err(build_with_source_root(&root, "source.ptx", 2));

    assert_provenance_path_error(err);
}

#[cfg(unix)]
#[test]
fn symlink_escape_source_ref_path_fails_with_source_root() {
    let root = temp_workspace("provenance-symlink-root");
    let outside = temp_workspace("provenance-symlink-outside");
    let outside_file = outside.join("source.ptx");
    fs::write(&outside_file, "one\n").expect("write source");
    std::os::unix::fs::symlink(&outside_file, root.join("linked.ptx")).expect("create symlink");

    let err = unwrap_err(build_with_source_root(&root, "linked.ptx", 1));

    assert_provenance_path_error(err);
}

#[test]
fn checked_in_graph_snapshot_passes_source_root_validation() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let snapshot_path = manifest_dir.join("graph_snapshot.json");
    let source_root = manifest_dir.join("uncc_cs2-pretext-project");
    let data = fs::read_to_string(&snapshot_path).expect("read checked-in graph snapshot");
    let snapshot: GraphSnapshot = serde_json::from_str(&data).expect("deserialize graph snapshot");

    GraphService::from_parts_with_source_root(
        snapshot.graph,
        false,
        snapshot.graph_version,
        Some(snapshot.course_commit),
        false,
        Some(source_root),
    )
    .expect("checked-in snapshot should pass provenance path validation");
}
