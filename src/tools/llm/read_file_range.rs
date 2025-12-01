use anyhow::Context;
use async_trait::async_trait;
use bon::Builder;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use tracing::info;

use super::{
    CallState, ToolExecutionError, ToolInputError, ToolInputResult, ToolInstance, ToolOutput,
    ToolPayloadMode,
    common::{ToolRunPayload, ToolRunner},
    ensure_ordering, render_relative_path, resolve_workspace_path,
};
use crate::{constants::MAX_TOOL_PATH_LEN, tools::filesystem};

const IDENTIFIER: &str = "read_file_range";
const DESCRIPTION: &str = "Read a specific inclusive line range from a UTF-8 text file.";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadFileRangeArgs {
    #[schemars(
        length(min = 1, max = MAX_TOOL_PATH_LEN),
        description = "File path relative to workspace root."
    )]
    #[builder(with = |value: String| -> ToolInputResult<_> {
        let path = super::require_string(value, IDENTIFIER, "path")?;
        super::ensure_max_len(&path, MAX_TOOL_PATH_LEN, IDENTIFIER, "path")?;
        Ok(path)
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
    #[serde(default)]
    #[schemars(
        description = "When true, return the requested range immediately; otherwise respond with \
                       a preview.",
        default = "crate::tools::llm::default_false"
    )]
    #[builder(default = false)]
    pub fetch_body: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadFileRangePayload {
    path:       String,
    start_line: usize,
    end_line:   usize,
    #[serde(default)]
    fetch_body: bool,
}

crate::basic_tool!(
    read_file_range_meta,
    id: IDENTIFIER,
    description: DESCRIPTION,
    args: ReadFileRangeArgs,
    prepare: |raw, _state: &CallState| {
        let payload: ReadFileRangePayload =
            serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
                tool:    IDENTIFIER,
                message: err.to_string(),
            })?;

        let args = ReadFileRangeArgs::builder()
            .path(payload.path)?
            .start_line(payload.start_line)?
            .end_line(payload.end_line)?
            .fetch_body(payload.fetch_body)
            .build();

        ensure_ordering(args.start_line, args.end_line, IDENTIFIER, "start_line", "end_line")?;

        Ok(args)
    },
    runner: |args: ReadFileRangeArgs, state: &CallState| ReadFileRangeTool {
        args,
        state: state.clone(),
    }
);

struct ReadFileRangeTool {
    args:  ReadFileRangeArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for ReadFileRangeTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let resolved = resolve_workspace_path(
            self.state.workspace_root.as_ref(),
            &self.args.path,
            IDENTIFIER,
        )?;
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
        let range_bytes = range.text.len() as u64;
        let mode = ToolPayloadMode::from_fetch_flag(self.args.fetch_body);
        let path_hint = render_relative_path(self.state.workspace_root.as_ref(), &resolved);
        let body_path = render_relative_path(self.state.workspace_root.as_ref(), &range.path);
        let start_line = range.start_line;
        let end_line = range.end_line;
        let range_text = range.text.clone();

        let runner = ToolRunner::new(IDENTIFIER, &self.state)
            .with_mode(mode)
            .hints(vec![
                format!("Path: {path_hint}"),
                format!("Span covers {} lines ({}-{}).", line_count, start_line, end_line),
                "Re-run read_file_range with fetch_body=true to retrieve this span.".to_string(),
                "Narrow the start/end lines to stay within budget.".to_string(),
            ]);

        info!(
            mode = ?mode,
            depth = self.state.depth,
            range_bytes,
            start_line,
            end_line,
            "tool_call read_file_range",
        );

        let preview_path = path_hint.clone();

        runner
            .run(move |_mode| async move {
                Ok(ToolRunPayload {
                    body:          json!({
                        "type": "data",
                        "mode": "body",
                        "path": body_path,
                        "start_line": start_line,
                        "end_line": end_line,
                        "content": range_text,
                        "line_count": line_count,
                        "bytes": range_bytes,
                    }),
                    approx_bytes:  Some(range_bytes),
                    preview:       Some(json!({
                        "type": "data",
                        "tool": IDENTIFIER,
                        "mode": "preview",
                        "path": preview_path,
                        "start_line": start_line,
                        "end_line": end_line,
                        "bytes": range_bytes,
                        "line_count": line_count,
                    })),
                    preview_hints: Vec::new(),
                    page:          None,
                })
            })
            .await
    }
}
