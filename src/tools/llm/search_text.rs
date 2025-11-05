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
    render_relative_path, resolve_workspace_path, schema_for_args, trim_optional,
};
use crate::tools::search;

const IDENTIFIER: &str = "search_text";
const DESCRIPTION: &str = "Run a regex search (ripgrep-style) within the workspace.";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SearchTextArgs {
    #[schemars(length(min = 1), description = "Rust-style regular expression.")]
    #[builder(with = |value: String| -> ToolInputResult<_> {
        super::require_string(value, IDENTIFIER, "pattern")
    })]
    pub pattern: String,
    #[serde(default)]
    #[schemars(description = "Optional directory to scope the search. Defaults to root.")]
    #[builder(with = |value: String| -> ToolInputResult<_> {
        super::require_string(value, IDENTIFIER, "path")
    })]
    pub path:    Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchTextPayload {
    pattern: String,
    #[serde(default)]
    path:    Option<String>,
}

pub(super) fn search_text_meta() -> ToolPrototype {
    ToolPrototype {
        id:          IDENTIFIER,
        description: DESCRIPTION,
        schema:      schema_for_args::<SearchTextArgs>(),
        parse:       parse_search_text,
    }
}

fn parse_search_text(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    let payload: SearchTextPayload =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    IDENTIFIER,
            message: err.to_string(),
        })?;

    let args = match trim_optional(payload.path) {
        Some(path) => SearchTextArgs::builder()
            .pattern(payload.pattern)?
            .path(path)?
            .build(),
        None => SearchTextArgs::builder().pattern(payload.pattern)?.build(),
    };

    Ok(Box::new(SearchTextTool {
        args,
        depth: state.depth,
        workspace_root: Arc::clone(&state.workspace_root),
    }))
}

struct SearchTextTool {
    args:           SearchTextArgs,
    depth:          usize,
    workspace_root: Arc<PathBuf>,
}

#[async_trait]
impl ToolInstance for SearchTextTool {
    async fn execute(&self) -> Result<Value, ToolExecutionError> {
        let scope = match self.args.path.as_deref() {
            Some(relative) => {
                resolve_workspace_path(self.workspace_root.as_ref(), relative, IDENTIFIER)?
            }
            None => (*self.workspace_root).clone(),
        };

        let matches = search::search_recursive(&scope, &self.args.pattern)
            .await
            .with_context(|| {
                format!(
                    "search_text failed for pattern `{}` in {}",
                    self.args.pattern,
                    scope.display()
                )
            })?;

        info!(
            "tool_call search_text depth={} scope={} pattern={} match_count={}",
            self.depth,
            scope.display(),
            self.args.pattern,
            matches.len()
        );

        let rendered = matches
            .into_iter()
            .map(|m| {
                json!({
                    "path": render_relative_path(self.workspace_root.as_ref(), &m.path),
                    "line_number": m.line_number,
                    "line": m.context,
                })
            })
            .collect::<Vec<_>>();

        Ok(json!({ "matches": rendered }))
    }
}
