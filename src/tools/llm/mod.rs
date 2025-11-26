use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result, bail};
use async_openai::types::{
    ChatCompletionTool, ChatCompletionToolArgs, ChatCompletionToolType, FunctionObjectArgs,
};
use async_trait::async_trait;
use kameo::prelude::ActorRef;
use once_cell::sync::Lazy;
use schemars::{JsonSchema, schema_for};
use serde_json::{self, Value, json};
use thiserror::Error;
use tracing::warn;

use self::{
    delegate_tasks::delegate_tasks_meta, list_directory::list_directory_meta,
    read_file_full::read_file_full_meta, read_file_range::read_file_range_meta,
    search_text::search_text_meta,
};
use crate::{
    graph::manager::GraphManager,
    llm_gateway::{GatewayMetrics, LLMGateway},
};

mod delegate_tasks;
mod graph_tools;
mod list_directory;
mod read_file_full;
mod read_file_range;
mod search_text;

const MASKED_PATH: &str = "<path-unavailable>";

#[derive(Debug)]
pub enum ToolExecutionError {
    Input(ToolInputError),
    Internal(anyhow::Error),
}

impl ToolExecutionError {
    pub fn user(err: ToolInputError) -> Self {
        Self::Input(err)
    }

    pub fn system(err: anyhow::Error) -> Self {
        Self::Internal(err)
    }
}

impl From<anyhow::Error> for ToolExecutionError {
    fn from(err: anyhow::Error) -> Self {
        ToolExecutionError::Internal(err)
    }
}
impl From<ToolInputError> for ToolExecutionError {
    fn from(err: ToolInputError) -> Self {
        ToolExecutionError::Input(err)
    }
}

pub struct ToolOutput {
    pub payload:   Value,
    pub byte_hint: Option<u64>,
}

impl ToolOutput {
    pub fn new(payload: Value) -> Self {
        Self {
            payload,
            byte_hint: None,
        }
    }

    pub fn with_byte_hint(payload: Value, byte_hint: u64) -> Self {
        Self {
            payload,
            byte_hint: Some(byte_hint),
        }
    }
}

pub type ToolExecutionResult = Result<ToolOutput, ToolExecutionError>;

#[async_trait]
pub trait ToolInstance: Send {
    async fn execute(&self) -> ToolExecutionResult;
}

pub type ToolParser = fn(Value, &CallState) -> ToolInputResult<Box<dyn ToolInstance>>;

pub struct ToolPrototype {
    pub id:          &'static str,
    pub description: &'static str,
    pub schema:      Value,
    pub parse:       ToolParser,
}

/// Mode selected for a tool payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolPayloadMode {
    Preview,
    Body,
}

impl ToolPayloadMode {
    pub fn from_fetch_flag(fetch_body: bool) -> Self {
        match fetch_body {
            true => ToolPayloadMode::Body,
            false => ToolPayloadMode::Preview,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ToolPayloadMode::Preview => "preview",
            ToolPayloadMode::Body => "body",
        }
    }
}

pub fn estimate_tokens_from_characters(characters: usize) -> u64 {
    let chars = characters as u64;
    // Heuristic: ~4 characters per token for English text; ensure at least one
    // token for non-empty inputs.
    if chars == 0 {
        0
    } else {
        chars.div_ceil(4).max(1)
    }
}
#[derive(Debug, Clone, Copy)]
pub struct PayloadEstimates {
    pub safe_tokens: Option<u64>,
}

pub fn prepare_payload_estimates(
    metrics: &GatewayMetrics,
    model: &str,
    bytes: u64,
) -> PayloadEstimates {
    let heuristic = if bytes == 0 {
        None
    } else {
        Some(estimate_tokens_from_characters(bytes as usize))
    };
    let approx = metrics.estimate_tokens(model, bytes).or(heuristic);
    let safe = apply_safety_margin(approx);
    PayloadEstimates { safe_tokens: safe }
}

pub fn apply_safety_margin(tokens: Option<u64>) -> Option<u64> {
    tokens.map(|value| value + value.saturating_div(5) + 1)
}

pub fn payload_size_bytes(value: &Value) -> u64 {
    serde_json::to_vec(value)
        .map(|buffer| buffer.len() as u64)
        .unwrap_or(0)
}

pub fn build_cost_preview(
    tool: &'static str,
    bytes_total: u64,
    approx_tokens: Option<u64>,
    hints: Vec<String>,
) -> Value {
    json!({
        "type": "preview",
        "mode": "preview",
        "tool": tool,
        "cost": {
            "bytes_total": bytes_total,
            "approx_tokens": approx_tokens,
            "preview_tokens": Value::Null,
            "remaining_tokens": Value::Null,
            "remaining_ratio": Value::Null,
        },
        "hints": hints,
    })
}

pub fn apply_preview_cost(
    payload: &mut Value,
    metrics: &GatewayMetrics,
    model: &str,
    conversation: &str,
    preview_tokens: u64,
    future_tokens: Option<u64>,
) {
    let context_limit = metrics.context_limit(model) as u64;
    let used_tokens = metrics
        .latest_prompt_tokens_for_conversation(conversation)
        .unwrap_or(0);
    let mut remaining = context_limit.saturating_sub(used_tokens);
    remaining = remaining.saturating_sub(preview_tokens);
    if let Some(cost) = future_tokens {
        remaining = remaining.saturating_sub(cost);
    }
    let ratio = if context_limit == 0 {
        0.0
    } else {
        (remaining as f64 / context_limit as f64).clamp(0.0, 1.0)
    };
    if let Some(cost) = payload.get_mut("cost").and_then(|c| c.as_object_mut()) {
        cost.insert("preview_tokens".to_string(), json!(preview_tokens));
        cost.insert("remaining_tokens".to_string(), json!(remaining));
        cost.insert("remaining_ratio".to_string(), json!(ratio));
    }
}

static TOOL_PROTOTYPES: Lazy<Vec<ToolPrototype>> = Lazy::new(|| {
    let mut metas = vec![
        list_directory_meta(),
        read_file_full_meta(),
        read_file_range_meta(),
        search_text_meta(),
        delegate_tasks_meta(),
    ];

    metas.extend(graph_tools::graph_tool_prototypes());

    let mut seen = HashMap::new();
    for meta in &metas {
        if let Some(existing) = seen.insert(meta.id, meta.description) {
            panic!(
                "duplicate tool identifier `{}` detected (existing description: `{}`, new \
                 description: `{}`)",
                meta.id, existing, meta.description,
            );
        }
    }

    metas
});

static TOOL_PROTOTYPE_INDEX: Lazy<HashMap<&'static str, &'static ToolPrototype>> =
    Lazy::new(|| TOOL_PROTOTYPES.iter().map(|meta| (meta.id, meta)).collect());

#[derive(Clone)]
pub struct CallState {
    pub depth:              usize,
    pub max_subdelegations: usize,
    pub workspace_root:     Arc<PathBuf>,
    pub gateway:            ActorRef<LLMGateway>,
    pub model:              Arc<String>,
    pub metrics:            Arc<GatewayMetrics>,
    pub graph:              ActorRef<GraphManager>,
    pub actor_name:         Arc<String>,
    pub conversation_id:    Arc<String>,
}

pub const fn default_false() -> bool {
    false
}

pub fn all_tools() -> &'static [ToolPrototype] {
    &TOOL_PROTOTYPES
}

pub fn lookup_tool(id: &str) -> Option<&'static ToolPrototype> {
    TOOL_PROTOTYPE_INDEX.get(id).copied()
}

pub fn tool_specs(ids: &[&str]) -> Result<Vec<ChatCompletionTool>> {
    let metas: Vec<&ToolPrototype> = if ids.is_empty() {
        TOOL_PROTOTYPES.iter().collect()
    } else {
        let mut selected = Vec::with_capacity(ids.len());
        for id in ids {
            match lookup_tool(id) {
                Some(meta) => selected.push(meta),
                None => warn!(tool = *id, "tool identifier not registered; skipping"),
            }
        }
        selected
    };

    if metas.is_empty() {
        bail!("no registered tools available for schema generation");
    }

    metas
        .into_iter()
        .map(|meta| {
            ChatCompletionToolArgs::default()
                .r#type(ChatCompletionToolType::Function)
                .function(
                    FunctionObjectArgs::default()
                        .name(meta.id)
                        .description(meta.description)
                        .parameters(meta.schema.clone())
                        .build()
                        .context("failed to build tool schema")?,
                )
                .build()
                .context("failed to build tool specification")
        })
        .collect()
}

pub fn resolve_workspace_path(
    root: &Path,
    relative: impl AsRef<Path>,
    tool: &'static str,
) -> ToolInputResult<PathBuf> {
    let rel = relative.as_ref();
    let candidate = if rel.is_absolute() {
        rel.to_path_buf()
    } else {
        root.join(rel)
    };
    let canonical = candidate
        .canonicalize()
        .map_err(|err| ToolInputError::InvalidPath {
            tool,
            path: candidate.display().to_string(),
            message: err.to_string(),
        })?;
    if !canonical.starts_with(root) {
        return Err(ToolInputError::InvalidPath {
            tool,
            path: canonical.display().to_string(),
            message: format!("escapes workspace root {}", root.display()),
        });
    }
    Ok(canonical)
}

pub fn render_relative_path(root: &Path, path: &Path) -> String {
    pathdiff::diff_paths(path, root)
        .and_then(|p| p.to_str().map(|s| s.to_string()))
        .unwrap_or_else(|| MASKED_PATH.to_string())
}

#[derive(Debug, Error, Clone)]
pub enum ToolInputError {
    #[error("{tool} requires field `{field}`")]
    MissingField {
        tool:  &'static str,
        field: &'static str,
    },
    #[error("{tool} field `{field}` must be non-empty")]
    EmptyField {
        tool:  &'static str,
        field: &'static str,
    },
    #[error("{tool} field `{field}` must be >= {min}")]
    BelowMinimum {
        tool:  &'static str,
        field: &'static str,
        min:   usize,
    },
    #[error("{tool} field `{upper_field}` ({upper}) must be >= `{lower_field}` ({lower})")]
    InvalidRange {
        tool:        &'static str,
        lower_field: &'static str,
        lower:       usize,
        upper_field: &'static str,
        upper:       usize,
    },
    #[error("{tool} requires at least one {item}")]
    EmptyCollection {
        tool: &'static str,
        item: &'static str,
    },
    #[error("unsupported tool call: {identifier}")]
    UnsupportedTool { identifier: String },
    #[error("{tool} payload is invalid: {message}")]
    InvalidPayload {
        tool:    &'static str,
        message: String,
    },
    #[error("{tool} depth {depth} exceeds limit {limit}")]
    DepthExceeded {
        tool:  &'static str,
        depth: usize,
        limit: usize,
    },
    #[error("{tool} path `{path}` is invalid: {message}")]
    InvalidPath {
        tool:    &'static str,
        path:    String,
        message: String,
    },
}

pub type ToolInputResult<T> = std::result::Result<T, ToolInputError>;

pub(crate) fn trim_optional(input: Option<String>) -> Option<String> {
    input.and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

pub(crate) fn require_string(
    value: String,
    tool: &'static str,
    field: &'static str,
) -> ToolInputResult<String> {
    require_non_empty(value, tool, field)
}

pub(crate) fn require_usize_min(
    value: usize,
    min: usize,
    tool: &'static str,
    field: &'static str,
) -> ToolInputResult<usize> {
    ensure_min(value, min, tool, field)
}

pub(crate) fn ensure_ordering(
    start: usize,
    end: usize,
    tool: &'static str,
    start_field: &'static str,
    end_field: &'static str,
) -> ToolInputResult<()> {
    ensure_range(start, end, tool, start_field, end_field)
}

fn require_non_empty(
    value: String,
    tool: &'static str,
    field: &'static str,
) -> ToolInputResult<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(ToolInputError::EmptyField { tool, field });
    }
    Ok(trimmed.to_string())
}

fn ensure_min(
    value: usize,
    min: usize,
    tool: &'static str,
    field: &'static str,
) -> ToolInputResult<usize> {
    if value < min {
        return Err(ToolInputError::BelowMinimum { tool, field, min });
    }
    Ok(value)
}

fn ensure_range(
    start: usize,
    end: usize,
    tool: &'static str,
    start_field: &'static str,
    end_field: &'static str,
) -> ToolInputResult<()> {
    if end < start {
        return Err(ToolInputError::InvalidRange {
            tool,
            lower_field: start_field,
            lower: start,
            upper_field: end_field,
            upper: end,
        });
    }
    Ok(())
}

fn schema_value_for<T: JsonSchema>() -> Value {
    let schema = schema_for!(T);
    serde_json::to_value(&schema).expect("serialize tool schema")
}

pub fn schema_for_args<T: JsonSchema>() -> Value {
    schema_value_for::<T>()
}

pub fn unsupported_tool(identifier: &str) -> ToolInputError {
    ToolInputError::UnsupportedTool {
        identifier: identifier.to_string(),
    }
}

pub fn invalid_payload(tool: &'static str, err: impl std::fmt::Display) -> ToolInputError {
    ToolInputError::InvalidPayload {
        tool,
        message: err.to_string(),
    }
}

pub fn depth_exceeded(tool: &'static str, depth: usize, limit: usize) -> ToolInputError {
    ToolInputError::DepthExceeded { tool, depth, limit }
}
