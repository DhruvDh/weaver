// graph_tools module split
pub mod algorithms;
pub mod analysis;
pub mod commands;
pub mod common;
pub mod inspection;
pub mod persist;
pub mod redundant_requires;

use crate::tools::llm::ToolPrototype;

pub fn graph_tool_prototypes() -> Vec<ToolPrototype> {
    let mut v = Vec::new();
    v.extend(commands::graph_tool_prototypes());
    v.extend(inspection::tool_prototypes());
    v.extend(analysis::tool_prototypes());
    v.extend(persist::tool_prototypes());
    v.extend(algorithms::tool_prototypes());
    v.extend(redundant_requires::tool_prototypes());
    v
}
