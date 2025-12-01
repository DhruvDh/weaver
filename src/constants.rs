/// Default temperature applied to LLM requests.
// TODO: Change to 0.7 before final run.
pub const DEFAULT_TEMPERATURE: f32 = 1.0;

/// Default nucleus sampling value applied to LLM requests.
pub const DEFAULT_TOP_P: f32 = 1.0;

/// Relative path to the UNCC CS2 PreTeXt project within the workspace.
pub const PRETEXT_SUBDIR: &str = "uncc_cs2-pretext-project";

/// Maximum number of tool-calling loops before giving up on an assistant
/// response.
pub const MAX_TOOL_ITERATIONS: usize = 60;

/// Maximum depth of recursive delegation for FileReader agents.
pub const DEFAULT_MAX_SUBDELEGATIONS: usize = 6;

/// Hard ceiling on delegate task parallelism to prevent runaway fan-out.
pub const MAX_PARALLEL_DELEGATIONS: usize = 8;

/// Maximum number of in-flight OpenAI requests handled by the gateway.
pub const LLM_MAX_CONCURRENT_REQUESTS: usize = 256;

/// Maximum number of retry attempts for a single OpenAI request.
pub const LLM_MAX_RETRIES: usize = 5;

/// Timeout applied to each chat-completion request (seconds).
pub const REQUEST_TIMEOUT_SECS: u64 = 300;

/// Base delay (milliseconds) used for exponential backoff after a failed
/// request.
pub const RETRY_BASE_DELAY_MS: u64 = 250;

/// Maximum backoff exponent; backoff is min(iteration, RETRY_MAX_EXP).
pub const RETRY_MAX_EXP: u32 = 6;

/// Hard cap for exponential backoff waits (milliseconds).
pub const RETRY_MAX_BACKOFF_MS: u64 = 60_000;

/// Maximum number of cached analysis entries retained at once.
pub const ANALYSIS_CACHE_MAX_ENTRIES: usize = 256;

/// Retention window for cached analyses (seconds).
pub const ANALYSIS_CACHE_TTL_SECS: u64 = 1_800;

/// Maximum historical versions kept per analysis kind.
pub const ANALYSIS_CACHE_VERSIONS_PER_KIND: usize = 2;

/// Default validation timeout applied to invariant checks (milliseconds).
pub const GRAPH_VALIDATION_TIMEOUT_MS: u64 = 2_000;

/// Maximum user-supplied path length accepted by LLM tools.
pub const MAX_TOOL_PATH_LEN: usize = 4_096;

/// Maximum number of delegated tasks accepted per call.
pub const MAX_DELEGATED_TASKS: usize = 24;

/// Maximum length of a delegated task description.
pub const MAX_DELEGATED_TASK_LEN: usize = 512;

/// Maximum regex pattern length accepted by search tools.
pub const MAX_SEARCH_PATTERN_LEN: usize = 256;
