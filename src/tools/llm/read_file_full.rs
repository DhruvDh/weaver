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
    render_relative_path, resolve_workspace_path,
};
use crate::tools::filesystem;

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

crate::basic_tool!(
    read_file_full_meta,
    id: IDENTIFIER,
    description: DESCRIPTION,
    args: ReadFileFullArgs,
    prepare: |raw, _state: &CallState| {
        let payload: ReadFileFullPayload =
            serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
                tool:    IDENTIFIER,
                message: err.to_string(),
            })?;

        let args = ReadFileFullArgs::builder()
            .path(payload.path)?
            .fetch_body(payload.fetch_body)
            .build();
        Ok(args)
    },
    runner: |args: ReadFileFullArgs, state: &CallState| ReadFileFullTool {
        args,
        state: state.clone(),
    }
);

struct ReadFileFullTool {
    args:  ReadFileFullArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for ReadFileFullTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let resolved = resolve_workspace_path(
            self.state.workspace_root.as_ref(),
            &self.args.path,
            IDENTIFIER,
        )?;
        let content = filesystem::read_file_full(&resolved)
            .await
            .with_context(|| format!("read_file_full failed for {}", resolved.display()))?;

        let file_bytes = content.len() as u64;
        let line_count = content.lines().count() as u64;
        let mode = ToolPayloadMode::from_fetch_flag(self.args.fetch_body);
        let path_hint = render_relative_path(self.state.workspace_root.as_ref(), &resolved);
        let runner = ToolRunner::new(IDENTIFIER, &self.state)
            .with_mode(mode)
            .hints(vec![
                format!("Path: {path_hint}"),
                format!("Target file has {} lines.", line_count),
                "Re-run read_file_full with fetch_body=true if you need the entire file."
                    .to_string(),
                "Call read_file_range to focus on a smaller portion.".to_string(),
            ]);

        info!(
            mode = ?mode,
            depth = self.state.depth,
            file_bytes,
            line_count,
            "tool_call read_file_full",
        );

        runner
            .run(move |_mode| async move {
                Ok(ToolRunPayload {
                    body:          json!({
                        "type": "data",
                        "mode": "body",
                        "path": path_hint,
                        "content": content,
                        "bytes": file_bytes,
                    }),
                    approx_bytes:  Some(file_bytes),
                    preview:       Some(json!({
                        "type": "data",
                        "tool": IDENTIFIER,
                        "mode": "preview",
                        "path": path_hint,
                        "bytes": file_bytes,
                        "line_count": line_count,
                    })),
                    preview_hints: Vec::new(),
                    page:          None,
                })
            })
            .await
    }
}
