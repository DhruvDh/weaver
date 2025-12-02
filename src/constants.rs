/// Default temperature applied to LLM requests. Some models only accept the
/// default of 1.0, so keep this at the neutral setting.
/// TODO: Change to 0.6 when running with gpt-oss-120b
pub const DEFAULT_TEMPERATURE: f32 = 0.6;

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

/// --- Graph visualization ---
pub mod viz {
    /// Inflate collision circles to keep the layout sparse in force layouts.
    pub const NODE_RADIUS_SCALE: f32 = 6.0;
}

/// --- Search tools ---
pub mod search {
    /// Directories skipped by filesystem search to avoid noise and slowness.
    pub const DEFAULT_BLOCKED_DIRS: &[&str] = &[".git", "target", "node_modules", "vendor"];
}

/// --- LLM tool configuration and metadata ---
pub mod tools {
    pub mod search_text {
        pub const IDENTIFIER: &str = "search_text";
        pub const DESCRIPTION: &str =
            "Search for text patterns using regex (ripgrep-style). Use to find specific content \
             before reading files, like '<section' to find section boundaries or 'def\\s+\\w+' to \
             find function definitions. Returns file paths and line numbers of matches. Use \
             fetch_body=false to preview match count, then true to get content.";
        pub const PREVIEW_MATCH_CAP: usize = 200;
        pub const PREVIEW_BYTE_CAP: u64 = 64 * 1024;
        pub const BODY_MATCH_CAP: usize = 2_000;
        pub const BODY_BYTE_CAP: u64 = 1_000_000;
    }

    pub mod read_file_range {
        pub const IDENTIFIER: &str = "read_file_range";
        pub const DESCRIPTION: &str = "Read a specific line range from a file. Use for targeted \
                                       reads instead of reading entire files. Lines are 1-based \
                                       and inclusive. First call with fetch_body=false to preview \
                                       size, then fetch_body=true to get content. Prefer this \
                                       over read_file_full for large files.";
    }
}

/// --- UI ---
pub mod ui {
    /// Maximum number of log lines shown in UI summaries.
    pub const MAX_LOG_LINES: usize = 3;
}

/// Maximum number of in-flight OpenAI requests handled by the gateway.
pub const LLM_MAX_CONCURRENT_REQUESTS: usize = 128;

/// Maximum number of retry attempts for a single OpenAI request.
pub const LLM_MAX_RETRIES: usize = 5;

/// Maximum number of models tracked in GatewayMetrics (estimators, context).
pub const GATEWAY_METRICS_MAX_MODELS: usize = 128;

/// Maximum number of conversations tracked in GatewayMetrics.
pub const GATEWAY_METRICS_MAX_CONVERSATIONS: usize = 512;

/// Maximum nodes permitted in the graph to prevent runaway growth.
pub const MAX_GRAPH_NODES: usize = 50_000;

/// Maximum edges permitted in the graph to prevent runaway growth.
pub const MAX_GRAPH_EDGES: usize = 200_000;

/// Maximum milliseconds allowed for filesystem search tasks to run.
pub const SEARCH_TASK_TIMEOUT_MS: u64 = 5_000;

/// Sentence threshold for over-bundling detection.
pub const GRAIN_OVERBUNDLED_SENTENCE_THRESHOLD: usize = 2;

/// Requires in-degree threshold for over-bundling/intrinsic load warnings.
pub const GRAIN_OVERBUNDLED_REQUIRES_THRESHOLD: usize = 4;

/// Token threshold for fragment detection.
pub const GRAIN_FRAGMENT_TOKEN_THRESHOLD: usize = 15;

/// Requires in-degree threshold for intrinsic load mismatch.
pub const INTRINSIC_COMPLEXITY_REQUIRES_THRESHOLD: usize = 4;

/// Minimum supports required for high intrinsic load nodes.
pub const HIGH_INTRINSIC_MIN_SUPPORTS: usize = 2;

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

/// Maximum regex pattern length accepted by search tools.
pub const MAX_SEARCH_PATTERN_LEN: usize = 256;

/// Maximum normalized slug name length (excluding kind prefix).
pub const MAX_SLUG_NAME_LEN: usize = 64;

/// Simhash Hamming distance tolerated for insert-time duplicate detection.
pub const DEDUP_INSERT_SIMHASH_DISTANCE: u32 = 3;

/// Minimum Jaro-Winkler similarity used for background clustering.
pub const DEDUP_CLUSTER_SIMILARITY: f64 = 0.92;

/// Hard cap on auto-merges performed in a single dedup run.
pub const DEDUP_MAX_AUTO_MERGES_PER_RUN: usize = 25;

/// Title similarity threshold for duplicate warnings.
pub const DEDUP_TITLE_SIMILARITY: f64 = 0.85;
