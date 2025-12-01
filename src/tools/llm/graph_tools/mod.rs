// graph_tools module split
pub mod algorithms;
pub mod analysis;
pub mod analysis_cache;
pub mod cache_admin;
pub mod commands;
pub mod common;
pub mod inspection;
pub mod persist;
pub mod redundant_requires;

use std::collections::HashSet;

use crate::tools::llm::ToolPrototype;

const CURATED_TOOL_IDS: &[&str] = &[
    // commands
    "graph_insert_knowledge",
    "graph_update_knowledge",
    "graph_insert_teaching_step",
    "graph_update_teaching_step",
    "graph_add_requires",
    "graph_add_supports",
    "graph_add_assesses",
    "graph_add_precedes",
    "graph_add_anchors",
    "graph_rename_node",
    "graph_remove_node",
    // persistence
    "graph_save_now",
    "graph_load_snapshot",
    // inspection
    "graph_neighbors",
    "graph_get_node",
    // analyses
    "graph_dag_check",
    "graph_lo_alignment_summary",
    "graph_gap_summary",
    "graph_keystone",
    "graph_redundant_requires",
    "graph_assessment_gaps",
    "graph_borrow_ahead",
    "graph_discourse_orphans",
    "graph_analysis_cache_clear",
];

pub fn graph_tool_prototypes() -> Vec<ToolPrototype> {
    let mut v = Vec::new();
    v.extend(commands::graph_tool_prototypes());
    v.extend(inspection::tool_prototypes());
    v.extend(analysis::tool_prototypes());
    v.extend(persist::tool_prototypes());
    v.extend(algorithms::tool_prototypes());
    v.extend(cache_admin::tool_prototypes());
    v.extend(redundant_requires::tool_prototypes());
    let allowed: HashSet<&str> = CURATED_TOOL_IDS.iter().copied().collect();
    v.retain(|meta| allowed.contains(meta.id));
    v
}
