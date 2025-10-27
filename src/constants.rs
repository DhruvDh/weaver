/// Default temperature applied to LLM requests.
pub const DEFAULT_TEMPERATURE: f32 = 0.7;

/// Default nucleus sampling value applied to LLM requests.
pub const DEFAULT_TOP_P: f32 = 1.0;

/// Relative path to the UNCC CS2 PreTeXt project within the workspace.
pub const PRETEXT_SUBDIR: &str = "uncc_cs2-pretext-project";

/// Maximum number of tool-calling loops before giving up on an assistant
/// response.
pub const MAX_TOOL_ITERATIONS: usize = 60;

/// Maximum depth of recursive delegation for FileReader agents.
pub const DEFAULT_MAX_SUBDELEGATIONS: usize = 2;
