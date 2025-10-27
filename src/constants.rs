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

/// Default number of delegate subtasks to run in parallel when using the
/// `subtasks` array form.
pub const DEFAULT_PARALLEL_DELEGATIONS: usize = 4;

/// Hard ceiling on delegate subtask parallelism to prevent runaway fan-out.
pub const MAX_PARALLEL_DELEGATIONS: usize = 8;

/// Timeout applied to each chat-completion request (seconds).
pub const REQUEST_TIMEOUT_SECS: u64 = 300;

/// Base delay (milliseconds) used for exponential backoff after a failed
/// request.
pub const RETRY_BASE_DELAY_MS: u64 = 250;

/// Maximum jitter (milliseconds) added to each backoff sleep.
pub const RETRY_MAX_JITTER_MS: u64 = 250;

/// Maximum backoff exponent; backoff is min(iteration, RETRY_MAX_EXP).
pub const RETRY_MAX_EXP: u32 = 6;
