use std::{path::PathBuf, sync::Arc};

use anyhow::Context;
use async_trait::async_trait;
use bon::Builder;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::info;

use super::{
    CallState, ToolExecutionError, ToolInputError, ToolInputResult, ToolInstance, ToolPrototype,
    render_relative_path, resolve_workspace_path, schema_for_args,
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
    pub path: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadFileFullPayload {
    path: String,
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

    let args = ReadFileFullArgs::builder().path(payload.path)?.build();

    Ok(Box::new(ReadFileFullTool {
        args,
        depth: state.depth,
        workspace_root: Arc::clone(&state.workspace_root),
    }))
}

struct ReadFileFullTool {
    args:           ReadFileFullArgs,
    depth:          usize,
    workspace_root: Arc<PathBuf>,
}

#[async_trait]
impl ToolInstance for ReadFileFullTool {
    async fn execute(&self) -> Result<Value, ToolExecutionError> {
        let resolved =
            resolve_workspace_path(self.workspace_root.as_ref(), &self.args.path, IDENTIFIER)?;
        let content = filesystem::read_file_full(&resolved)
            .await
            .with_context(|| format!("read_file_full failed for {}", resolved.display()))?;

        info!(
            "tool_call read_file_full depth={} path={} bytes={}",
            self.depth,
            resolved.display(),
            content.len()
        );

        Ok(json!({
            "path": render_relative_path(self.workspace_root.as_ref(), &resolved),
            "content": content,
        }))
    }
}
