use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result, anyhow, bail};
use async_openai::types::{
    ChatCompletionTool, ChatCompletionToolArgs, ChatCompletionToolType, FunctionObjectArgs,
};
use async_trait::async_trait;
use kameo::prelude::ActorRef;
use once_cell::sync::Lazy;
use schemars::{JsonSchema, schema_for};
use serde_json::{self, Value};
use thiserror::Error;

use self::{
    delegate_tasks::delegate_tasks_meta, list_directory::list_directory_meta,
    read_file_full::read_file_full_meta, read_file_range::read_file_range_meta,
    search_text::search_text_meta,
};
use crate::llm_gateway::LLMGateway;

mod delegate_tasks;
mod list_directory;
mod read_file_full;
mod read_file_range;
mod search_text;

const MASKED_PATH: &str = "<path-unavailable>";

#[async_trait]
pub trait Tool: Send + Sync {
    fn id(&self) -> &'static str;
    async fn execute(&self) -> Result<Value>;
}

pub type ToolParser = fn(Value, &CallState) -> ToolInputResult<Box<dyn Tool>>;

pub struct ToolMeta {
    pub id:          &'static str,
    pub description: &'static str,
    pub schema:      Value,
    pub parse:       ToolParser,
}

static TOOL_METADATA: Lazy<Vec<ToolMeta>> = Lazy::new(|| {
    vec![
        list_directory_meta(),
        read_file_full_meta(),
        read_file_range_meta(),
        search_text_meta(),
        delegate_tasks_meta(),
    ]
});

static TOOL_INDEX: Lazy<HashMap<&'static str, &'static ToolMeta>> =
    Lazy::new(|| TOOL_METADATA.iter().map(|meta| (meta.id, meta)).collect());

#[derive(Clone)]
pub struct CallState {
    pub depth:              usize,
    pub max_subdelegations: usize,
    pub workspace_root:     Arc<PathBuf>,
    pub gateway:            ActorRef<LLMGateway>,
    pub model:              Arc<String>,
}

pub fn all_tools() -> &'static [ToolMeta] {
    &TOOL_METADATA
}

pub fn lookup_tool(id: &str) -> Option<&'static ToolMeta> {
    TOOL_INDEX.get(id).copied()
}

pub fn tool_specs(ids: &[&str]) -> Result<Vec<ChatCompletionTool>> {
    let metas: Vec<&ToolMeta> = if ids.is_empty() {
        TOOL_METADATA.iter().collect()
    } else {
        let mut selected = Vec::with_capacity(ids.len());
        for id in ids {
            let meta = lookup_tool(id).ok_or_else(|| anyhow!(unsupported_tool(id)))?;
            selected.push(meta);
        }
        selected
    };

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

pub fn resolve_workspace_path(root: &Path, relative: impl AsRef<Path>) -> Result<PathBuf> {
    let rel = relative.as_ref();
    let candidate = if rel.is_absolute() {
        rel.to_path_buf()
    } else {
        root.join(rel)
    };
    let canonical = candidate
        .canonicalize()
        .with_context(|| format!("failed to canonicalize resolved path {}", candidate.display()))?;
    if !canonical.starts_with(root) {
        bail!("path {} escapes workspace root {}", canonical.display(), root.display());
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
