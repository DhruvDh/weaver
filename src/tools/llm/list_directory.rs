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
    render_relative_path, resolve_workspace_path, trim_optional,
};
use crate::{constants::MAX_TOOL_PATH_LEN, tools::filesystem};

const IDENTIFIER: &str = "list_directory";
const DESCRIPTION: &str = "List the entries of a directory relative to the workspace root.";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ListDirectoryArgs {
    #[serde(default)]
    #[schemars(
        length(min = 1, max = MAX_TOOL_PATH_LEN),
        description = "Directory path relative to workspace root. Defaults to \".\"."
    )]
    #[builder(with = |value: String| -> ToolInputResult<_> {
        let path = super::require_string(value, IDENTIFIER, "path")?;
        super::ensure_max_len(&path, MAX_TOOL_PATH_LEN, IDENTIFIER, "path")?;
        Ok(path)
    })]
    pub path: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ListDirectoryPayload {
    #[serde(default)]
    path: Option<String>,
}

crate::basic_tool!(
    list_directory_meta,
    id: IDENTIFIER,
    description: DESCRIPTION,
    args: ListDirectoryArgs,
    prepare: |raw, _state: &CallState| {
        let payload: ListDirectoryPayload =
            serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
                tool:    IDENTIFIER,
                message: err.to_string(),
            })?;

        let args = match trim_optional(payload.path) {
            Some(path) => ListDirectoryArgs::builder().path(path)?.build(),
            None => ListDirectoryArgs::builder().build(),
        };
        Ok(args)
    },
    runner: |args: ListDirectoryArgs, state: &CallState| ListDirectoryTool {
        args,
        state: state.clone(),
    }
);

struct ListDirectoryTool {
    args:  ListDirectoryArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for ListDirectoryTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let relative = self.args.path.as_deref().unwrap_or(".");
        let resolved =
            resolve_workspace_path(self.state.workspace_root.as_ref(), relative, IDENTIFIER)?;
        let entries = filesystem::list_dir(&resolved)
            .await
            .with_context(|| format!("list_directory failed for {}", resolved.display()))?;
        info!(
            "tool_call list_directory depth={} path={} entry_count={}",
            self.state.depth,
            resolved.display(),
            entries.len()
        );
        let rendered = entries
            .into_iter()
            .map(|entry| {
                json!({
                    "name": entry.name,
                    "path": render_relative_path(self.state.workspace_root.as_ref(), &entry.path),
                    "kind": entry.kind.as_str(),
                    "size": entry.size,
                })
            })
            .collect::<Vec<_>>();
        let entries_payload = rendered;
        ToolRunner::new(IDENTIFIER, &self.state)
            .with_mode(ToolPayloadMode::Body)
            .run(move |_| async move {
                Ok(ToolRunPayload::new(json!({
                    "type": "data",
                    "mode": "body",
                    "entries": entries_payload
                })))
            })
            .await
    }
}
