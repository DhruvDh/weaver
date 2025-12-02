// graph_tools module split
pub mod algorithms;
pub mod analysis;
pub mod analysis_cache;
pub mod cache_admin;
pub mod commands;
pub mod common;
pub mod edge_conflicts;
pub mod inspection;
pub mod persist;
pub mod redundant_requires;

use std::collections::HashSet;

use crate::tools::llm::ToolPrototype;

crate::tool_ids! {
        pub(crate) CURATED_TOOL_IDS {
            GRAPH_INSERT_KNOWLEDGE = "graph_insert_knowledge";
            GRAPH_UPDATE_KNOWLEDGE = "graph_update_knowledge";
            GRAPH_INSERT_TEACHING_STEP = "graph_insert_teaching_step";
            GRAPH_UPDATE_TEACHING_STEP = "graph_update_teaching_step";
            GRAPH_ADD_REQUIRES = "graph_add_requires";
            GRAPH_ADD_SUPPORTS = "graph_add_supports";
            GRAPH_ADD_ASSESSES = "graph_add_assesses";
            GRAPH_ADD_PRECEDES = "graph_add_precedes";
            GRAPH_ADD_ANCHORS = "graph_add_anchors";
            GRAPH_RENAME_NODE = "graph_rename_node";
            GRAPH_REMOVE_NODE = "graph_remove_node";
            GRAPH_SAVE_NOW = "graph_save_now";
            GRAPH_LOAD_SNAPSHOT = "graph_load_snapshot";
            GRAPH_NEIGHBORS = "graph_neighbors";
            GRAPH_GET_NODE = "graph_get_node";
            GRAPH_EDGE_CONFLICTS = "graph_edge_conflicts";
            GRAPH_RESOLVE_EDGE_CONFLICT = "graph_resolve_edge_conflict";
            GRAPH_FIRST_PRINCIPLES = "graph_first_principles";
            GRAPH_FIRST_PRINCIPLES_SUMMARY = "graph_first_principles_summary";
            GRAPH_DAG_CHECK = "graph_dag_check";
            GRAPH_LO_REACHABILITY = "graph_lo_reachability";
            GRAPH_LO_COVERAGE = "graph_lo_coverage";
            GRAPH_LO_ALIGNMENT_SUMMARY = "graph_lo_alignment_summary";
            GRAPH_LO_ASSESSMENTS_VIEW = "graph_lo_assessments_view";
            GRAPH_LO_MISSING_CRITERIA_VIEW = "graph_lo_missing_criteria_view";
            GRAPH_LO_ANCHORS_VIEW = "graph_lo_anchors_view";
            GRAPH_GAP_SUMMARY = "graph_gap_summary";
            GRAPH_EXAMPLE_GAPS_VIEW = "graph_example_gaps_view";
            GRAPH_FADEABILITY_VIEW = "graph_fadeability_view";
            GRAPH_PRACTICE_GAPS_VIEW = "graph_practice_gaps_view";
            GRAPH_KEYSTONE = "graph_keystone";
            GRAPH_EXTRANEOUS = "graph_extraneous";
            GRAPH_REDUNDANT_REQUIRES = "graph_redundant_requires";
            GRAPH_ASSESSMENT_GAPS = "graph_assessment_gaps";
            GRAPH_BORROW_AHEAD = "graph_borrow_ahead";
            GRAPH_DISCOURSE_ORPHANS = "graph_discourse_orphans";
            GRAPH_ANALYSIS_CACHE_CLEAR = "graph_analysis_cache_clear";
    }
}

pub fn graph_tool_prototypes() -> Vec<ToolPrototype> {
    let mut v = Vec::new();
    v.extend(commands::graph_tool_prototypes());
    v.extend(inspection::tool_prototypes());
    v.extend(analysis::tool_prototypes());
    v.extend(persist::tool_prototypes());
    v.extend(algorithms::tool_prototypes());
    v.extend(cache_admin::tool_prototypes());
    v.extend(redundant_requires::tool_prototypes());
    v.extend(edge_conflicts::tool_prototypes());
    let mut allowed: HashSet<&str> = CURATED_TOOL_IDS.iter().copied().collect();
    if std::env::var("WEAVER_DEBUG_GRAPH_ALGORITHMS")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        allowed.extend(algorithms::ALGORITHM_TOOL_IDS.iter().copied());
    }
    v.retain(|meta| allowed.contains(meta.id));
    v
}
