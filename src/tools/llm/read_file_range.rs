use std::{path::PathBuf, sync::Arc};

use anyhow::Context;
use async_trait::async_trait;
use bon::Builder;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::info;

use super::{
    CallState, ToolExecutionError, ToolInputError, ToolInputResult, ToolInstance, ToolOutput,
    ToolPayloadMode, ToolPrototype, apply_preview_cost, build_cost_preview, ensure_ordering,
    estimate_tokens_from_characters, payload_size_bytes, prepare_payload_estimates,
    render_relative_path, resolve_workspace_path, schema_for_args,
};
use crate::{llm_gateway::GatewayMetrics, tools::filesystem};

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

pub(super) fn read_file_range_meta() -> ToolPrototype {
    ToolPrototype {
        id:          IDENTIFIER,
        description: DESCRIPTION,
        schema:      schema_for_args::<ReadFileRangeArgs>(),
        parse:       parse_read_file_range,
    }
}

fn parse_read_file_range(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
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

    Ok(Box::new(ReadFileRangeTool {
        args,
        depth: state.depth,
        workspace_root: Arc::clone(&state.workspace_root),
        metrics: Arc::clone(&state.metrics),
        model: Arc::clone(&state.model),
        conversation_id: Arc::clone(&state.conversation_id),
    }))
}

struct ReadFileRangeTool {
    args:            ReadFileRangeArgs,
    depth:           usize,
    workspace_root:  Arc<PathBuf>,
    metrics:         Arc<GatewayMetrics>,
    model:           Arc<String>,
    conversation_id: Arc<String>,
}

#[async_trait]
impl ToolInstance for ReadFileRangeTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let resolved =
            resolve_workspace_path(self.workspace_root.as_ref(), &self.args.path, IDENTIFIER)?;
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
        let range_bytes = range.text.as_bytes().len() as u64;
        let token_estimates =
            prepare_payload_estimates(&self.metrics, self.model.as_str(), range_bytes);
        let safe_tokens = token_estimates.safe_tokens;
        let mode = ToolPayloadMode::from_fetch_flag(self.args.fetch_body);

        match mode {
            ToolPayloadMode::Preview => {
                let hints = vec![
                    format!(
                        "Span covers {} lines ({}-{}).",
                        line_count, range.start_line, range.end_line
                    ),
                    "Re-run read_file_range with fetch_body=true to retrieve this span."
                        .to_string(),
                    "Narrow the start/end lines to stay within budget.".to_string(),
                ];
                let mut value = build_cost_preview(IDENTIFIER, range_bytes, safe_tokens, hints);
                let payload_bytes = payload_size_bytes(&value);
                let preview_tokens = estimate_tokens_from_characters(payload_bytes as usize);
                apply_preview_cost(
                    &mut value,
                    &self.metrics,
                    self.model.as_str(),
                    self.conversation_id.as_str(),
                    preview_tokens,
                    safe_tokens,
                );
                info!(
                    mode = %mode.as_str(),
                    depth = self.depth,
                    preview_bytes = payload_bytes,
                    range_bytes,
                    start_line = range.start_line,
                    end_line = range.end_line,
                    "tool_call read_file_range preview",
                );
                Ok(ToolOutput::with_byte_hint(value, payload_bytes))
            }
            ToolPayloadMode::Body => {
                let value = json!({
                    "type": "data",
                    "mode": "body",
                    "path": render_relative_path(self.workspace_root.as_ref(), &range.path),
                    "start_line": range.start_line,
                    "end_line": range.end_line,
                    "content": range.text,
                    "line_count": line_count,
                    "bytes": range_bytes,
                    "approx_tokens": safe_tokens,
                });
                let payload_bytes = payload_size_bytes(&value);
                info!(
                    mode = %mode.as_str(),
                    depth = self.depth,
                    payload_bytes,
                    range_bytes,
                    start_line = range.start_line,
                    end_line = range.end_line,
                    "tool_call read_file_range body",
                );
                Ok(ToolOutput::with_byte_hint(value, payload_bytes))
            }
        }
    }
}
