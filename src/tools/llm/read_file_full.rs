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
    ToolPayloadMode, ToolPrototype, apply_preview_cost, build_cost_preview,
    estimate_tokens_from_characters, payload_size_bytes, prepare_payload_estimates,
    render_relative_path, resolve_workspace_path, schema_for_args,
};
use crate::{llm_gateway::GatewayMetrics, tools::filesystem};

const IDENTIFIER: &str = "read_file_full";
const DESCRIPTION: &str = "Read the full contents of a UTF-8 text file.";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadFileFullArgs {
    #[schemars(length(min = 1), description = "File path relative to workspace root.")]
    #[builder(with = |value: String| -> ToolInputResult<_> {
        super::require_string(value, IDENTIFIER, "path")
    })]
    pub path:       String,
    #[serde(default)]
    #[schemars(
        description = "When true, return the file contents instead of a preview header.",
        default = "crate::tools::llm::default_false"
    )]
    #[builder(default = false)]
    pub fetch_body: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadFileFullPayload {
    path:       String,
    #[serde(default)]
    fetch_body: bool,
}

pub(super) fn read_file_full_meta() -> ToolPrototype {
    ToolPrototype {
        id:          IDENTIFIER,
        description: DESCRIPTION,
        schema:      schema_for_args::<ReadFileFullArgs>(),
        parse:       parse_read_file_full,
    }
}

fn parse_read_file_full(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let payload: ReadFileFullPayload =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    IDENTIFIER,
            message: err.to_string(),
        })?;

    let args = ReadFileFullArgs::builder()
        .path(payload.path)?
        .fetch_body(payload.fetch_body)
        .build();

    Ok(Box::new(ReadFileFullTool {
        args,
        depth: state.depth,
        workspace_root: Arc::clone(&state.workspace_root),
        metrics: Arc::clone(&state.metrics),
        model: Arc::clone(&state.model),
        conversation_id: Arc::clone(&state.conversation_id),
    }))
}

struct ReadFileFullTool {
    args:            ReadFileFullArgs,
    depth:           usize,
    workspace_root:  Arc<PathBuf>,
    metrics:         Arc<GatewayMetrics>,
    model:           Arc<String>,
    conversation_id: Arc<String>,
}

#[async_trait]
impl ToolInstance for ReadFileFullTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let resolved =
            resolve_workspace_path(self.workspace_root.as_ref(), &self.args.path, IDENTIFIER)?;
        let content = filesystem::read_file_full(&resolved)
            .await
            .with_context(|| format!("read_file_full failed for {}", resolved.display()))?;

        let file_bytes = content.len() as u64;
        let line_count = content.lines().count() as u64;
        let token_estimates =
            prepare_payload_estimates(&self.metrics, self.model.as_str(), file_bytes);
        let safe_tokens = token_estimates.safe_tokens;
        let mode = ToolPayloadMode::from_fetch_flag(self.args.fetch_body);

        match mode {
            ToolPayloadMode::Preview => {
                let hints = vec![
                    format!(
                        "Path: {}",
                        render_relative_path(self.workspace_root.as_ref(), &resolved)
                    ),
                    format!("Target file has {} lines.", line_count),
                    "Re-run read_file_full with fetch_body=true if you need the entire file."
                        .to_string(),
                    "Call read_file_range to focus on a smaller portion.".to_string(),
                ];
                let mut value = build_cost_preview(IDENTIFIER, file_bytes, safe_tokens, hints);
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
                    file_bytes,
                    line_count,
                    "tool_call read_file_full preview",
                );
                Ok(ToolOutput::with_byte_hint(value, payload_bytes))
            }
            ToolPayloadMode::Body => {
                let value = json!({
                    "type": "data",
                    "mode": "body",
                    "path": render_relative_path(self.workspace_root.as_ref(), &resolved),
                    "content": content,
                    "bytes": file_bytes,
                    "approx_tokens": safe_tokens,
                });
                let payload_bytes = payload_size_bytes(&value);
                info!(
                    mode = %mode.as_str(),
                    depth = self.depth,
                    payload_bytes,
                    file_bytes,
                    "tool_call read_file_full body",
                );
                Ok(ToolOutput::with_byte_hint(value, payload_bytes))
            }
        }
    }
}
