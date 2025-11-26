// graph_tools module split
pub mod analysis;
pub mod commands;
pub mod common;
pub mod inspection;
pub mod persist;

use crate::tools::llm::ToolPrototype;

pub fn graph_tool_prototypes() -> Vec<ToolPrototype> {
    let mut v = Vec::new();
    v.extend(commands::graph_tool_prototypes());
    v.extend(inspection::tool_prototypes());
    v.extend(analysis::tool_prototypes());
    v.extend(persist::tool_prototypes());
    v
}
