use std::{path::PathBuf, sync::Arc};

use anyhow::{Context, Result};
use async_trait::async_trait;
use bon::Builder;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::info;

use super::{
    CallState, Tool, ToolInputError, ToolInputResult, ToolMeta, render_relative_path,
    resolve_workspace_path, schema_for_args, trim_optional,
};
use crate::tools::filesystem;

const IDENTIFIER: &str = "list_directory";
const DESCRIPTION: &str = "List the entries of a directory relative to the workspace root.";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ListDirectoryArgs {
    #[serde(default)]
    #[schemars(description = "Directory path relative to workspace root. Defaults to \".\".")]
    #[builder(with = |value: String| -> ToolInputResult<_> {
        super::require_string(value, IDENTIFIER, "path")
    })]
    pub path: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ListDirectoryPayload {
    #[serde(default)]
    path: Option<String>,
}

pub(super) fn list_directory_meta() -> ToolMeta {
    ToolMeta {
        id:          IDENTIFIER,
        description: DESCRIPTION,
        schema:      schema_for_args::<ListDirectoryArgs>(),
        parse:       parse_list_directory,
    }
}

fn parse_list_directory(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn Tool>> {
    let payload: ListDirectoryPayload =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    IDENTIFIER,
            message: err.to_string(),
        })?;

    let args = match trim_optional(payload.path) {
        Some(path) => ListDirectoryArgs::builder().path(path)?.build(),
        None => ListDirectoryArgs::builder().build(),
    };

    Ok(Box::new(ListDirectoryTool {
        args,
        depth: state.depth,
        workspace_root: Arc::clone(&state.workspace_root),
    }))
}

struct ListDirectoryTool {
    args:           ListDirectoryArgs,
    depth:          usize,
    workspace_root: Arc<PathBuf>,
}

#[async_trait]
impl Tool for ListDirectoryTool {
    fn id(&self) -> &'static str {
        IDENTIFIER
    }

    async fn execute(&self) -> Result<Value> {
        let relative = self.args.path.as_deref().unwrap_or(".");
        let resolved = resolve_workspace_path(self.workspace_root.as_ref(), relative)?;
        let entries = filesystem::list_dir(&resolved)
            .await
            .with_context(|| format!("list_directory failed for {}", resolved.display()))?;
        info!(
            "tool_call list_directory depth={} path={} entry_count={}",
            self.depth,
            resolved.display(),
            entries.len()
        );
        let rendered = entries
            .into_iter()
            .map(|entry| {
                json!({
                    "name": entry.name,
                    "path": render_relative_path(self.workspace_root.as_ref(), &entry.path),
                    "kind": entry.kind.as_str(),
                    "size": entry.size,
                })
            })
            .collect::<Vec<_>>();
        Ok(json!({ "entries": rendered }))
    }
}
