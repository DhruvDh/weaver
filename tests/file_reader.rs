use std::path::PathBuf;

use weaver::{
    app::{Cli, interactive_session_mode},
    file_reader::{AgentMode, make_conversation_id, tool_identifiers_for_mode},
};

#[test]
fn conversation_ids_use_uuid_v4_and_remain_unique() {
    let actor_name = "FileReader/Root";

    let first = make_conversation_id(actor_name);
    let second = make_conversation_id(actor_name);

    assert!(first.starts_with(actor_name));
    assert!(second.starts_with(actor_name));

    let first_uuid = first.split('#').nth(1).expect("missing uuid segment");
    let second_uuid = second.split('#').nth(1).expect("missing uuid segment");

    let first_parsed = uuid::Uuid::parse_str(first_uuid).expect("invalid uuid format");
    let second_parsed = uuid::Uuid::parse_str(second_uuid).expect("invalid uuid format");

    assert_eq!(first_parsed.get_version_num(), 4);
    assert_eq!(second_parsed.get_version_num(), 4);
    assert_ne!(first_parsed, second_parsed);
}

fn sample_cli() -> Cli {
    Cli {
        rerun_file:                  None,
        graph_snapshot_path:         PathBuf::from("graph_snapshot.json"),
        graph_autosave_secs:         300,
        graph_course_commit:         None,
        graph_strict_quality:        false,
        graph_prune_requires_s:      None,
        skip_demo:                   true,
        graph_validation_timeout_ms: 2_000,
        dedup_interval_secs:         3_600,
        dedup_auto_merge_threshold:  0.95,
        skip_dedup_on_insert:        true,
        interactive:                 true,
        interactive_writable:        false,
        analyst:                     false,
        harvest_timeout_hours:       None,
        weave_timeout_hours:         None,
        max_concurrent_chapters:     4,
        chapters_pattern:            None,
        chapters_dir:                None,
        workspace:                   PathBuf::from("."),
    }
}

#[test]
fn analyst_mode_is_default_for_interactive() {
    let cli = sample_cli();
    let mode = interactive_session_mode(&cli);
    assert_eq!(mode, Some(AgentMode::Analyst));
}

#[test]
fn interactive_writable_opt_in_selects_file_reader() {
    let mut cli = sample_cli();
    cli.interactive_writable = true;

    let mode = interactive_session_mode(&cli);
    assert_eq!(mode, Some(AgentMode::Interactive));
}

#[test]
fn analyst_tool_list_is_read_only() {
    let mut tools = tool_identifiers_for_mode(AgentMode::Analyst).expect("analyst tool list");
    tools.sort();

    let mut expected = vec![
        "graph_assessment_gaps",
        "graph_borrow_ahead",
        "graph_course_commit",
        "graph_dag_check",
        "graph_discourse_orphans",
        "graph_example_gaps_view",
        "graph_extraneous",
        "graph_fadeability_view",
        "graph_first_principles",
        "graph_first_principles_summary",
        "graph_gap_summary",
        "graph_get_node",
        "graph_keystone",
        "graph_list_nodes_by_kind",
        "graph_list_nodes_by_tag",
        "graph_list_tags",
        "graph_lo_alignment_summary",
        "graph_lo_anchors_view",
        "graph_lo_assessments_view",
        "graph_lo_coverage",
        "graph_lo_missing_criteria_view",
        "graph_lo_reachability",
        "graph_neighbors",
        "graph_practice_gaps_view",
        "graph_redundant_requires",
        "graph_search_nodes",
    ];
    expected.sort();

    assert_eq!(tools, expected);
}
