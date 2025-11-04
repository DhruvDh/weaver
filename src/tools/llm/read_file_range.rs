use std::{path::PathBuf, sync::Arc};

use anyhow::{Context, Result};
use async_trait::async_trait;
use bon::Builder;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::info;

use super::{
    CallState, Tool, ToolInputError, ToolInputResult, ToolMeta, ensure_ordering,
    render_relative_path, resolve_workspace_path, schema_for_args,
};
use crate::tools::filesystem;

const IDENTIFIER: &str = "read_file_range";
const DESCRIPTION: &str = "Read a specific inclusive line range from a UTF-8 text file.";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadFileRangeArgs {
    #[schemars(length(min = 1), description = "File path relative to workspace root.")]
    #[builder(with = |value: String| -> ToolInputResult<_> {
        super::require_string(value, IDENTIFIER, "path")
    })]
    pub path:       String,
    #[schemars(range(min = 1), description = "1-based inclusive start line (>= 1).")]
    #[builder(with = |value: usize| -> ToolInputResult<_> {
        super::require_usize_min(value, 1, IDENTIFIER, "start_line")
    })]
    pub start_line: usize,
    #[schemars(range(min = 1), description = "1-based inclusive end line (>= 1).")]
    #[builder(with = |value: usize| -> ToolInputResult<_> {
        super::require_usize_min(value, 1, IDENTIFIER, "end_line")
    })]
    pub end_line:   usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadFileRangePayload {
    path:       String,
    start_line: usize,
    end_line:   usize,
}

pub(super) fn read_file_range_meta() -> ToolMeta {
    ToolMeta {
        id:          IDENTIFIER,
        description: DESCRIPTION,
        schema:      schema_for_args::<ReadFileRangeArgs>(),
        parse:       parse_read_file_range,
    }
}

fn parse_read_file_range(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn Tool>> {
    let payload: ReadFileRangePayload =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    IDENTIFIER,
            message: err.to_string(),
        })?;

    let args = ReadFileRangeArgs::builder()
        .path(payload.path)?
        .start_line(payload.start_line)?
        .end_line(payload.end_line)?
        .build();

    ensure_ordering(args.start_line, args.end_line, IDENTIFIER, "start_line", "end_line")?;

    Ok(Box::new(ReadFileRangeTool {
        args,
        depth: state.depth,
        workspace_root: Arc::clone(&state.workspace_root),
    }))
}

struct ReadFileRangeTool {
    args:           ReadFileRangeArgs,
    depth:          usize,
    workspace_root: Arc<PathBuf>,
}

#[async_trait]
impl Tool for ReadFileRangeTool {
    fn id(&self) -> &'static str {
        IDENTIFIER
    }

    async fn execute(&self) -> Result<Value> {
        let resolved = resolve_workspace_path(self.workspace_root.as_ref(), &self.args.path)?;
        let range =
            filesystem::read_file_range(&resolved, self.args.start_line, self.args.end_line)
                .await
                .with_context(|| {
                    format!(
                        "read_file_range failed for {} ({}-{})",
                        resolved.display(),
                        self.args.start_line,
                        self.args.end_line
                    )
                })?;

        let line_count = range.end_line.saturating_sub(range.start_line) + 1;
        info!(
            "tool_call read_file_range depth={} path={} start_line={} end_line={} line_count={}",
            self.depth,
            resolved.display(),
            range.start_line,
            range.end_line,
            line_count
        );

        Ok(json!({
            "path": render_relative_path(self.workspace_root.as_ref(), &range.path),
            "start_line": range.start_line,
            "end_line": range.end_line,
            "content": range.text,
        }))
    }
}
