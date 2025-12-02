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

const IDENTIFIER: &str = "locate_snippet";
const DESCRIPTION: &str = "Locate a snippet inside a file and return its line range.";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LocateSnippetArgs {
    #[schemars(description = "File path relative to workspace root.")]
    #[builder(with = |value: String| -> ToolInputResult<_> {
        let path = super::require_string(value, IDENTIFIER, "path")?;
        Ok(path)
    })]
    pub path:           String,
    #[schemars(description = "Plaintext snippet to search for.")]
    #[builder(with = |value: String| -> ToolInputResult<_> {
        let snippet = super::require_string(value, IDENTIFIER, "snippet")?;
        Ok(snippet)
    })]
    pub snippet:        String,
    #[serde(default)]
    #[schemars(description = "If false (default), search is case-insensitive.")]
    pub case_sensitive: bool,
    #[serde(default)]
    #[schemars(description = "1-based occurrence index to return (default 1 = first).")]
    pub occurrence:     Option<usize>,
    #[serde(default)]
    #[schemars(
        description = "When true, return the matched content immediately; otherwise return a \
                       preview."
    )]
    #[builder(default = false)]
    pub fetch_body:     bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LocateSnippetPayload {
    path:           String,
    snippet:        String,
    #[serde(default)]
    case_sensitive: bool,
    #[serde(default)]
    occurrence:     Option<usize>,
    #[serde(default)]
    fetch_body:     bool,
}

crate::basic_tool!(
    locate_snippet_meta,
    id: IDENTIFIER,
    description: DESCRIPTION,
    args: LocateSnippetArgs,
    prepare: |raw, _state: &CallState| {
        let payload: LocateSnippetPayload =
            serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
                tool:    IDENTIFIER,
                message: err.to_string(),
            })?;

        let args = LocateSnippetArgs::builder()
            .path(payload.path)?
            .snippet(payload.snippet)?
            .case_sensitive(payload.case_sensitive)
            .occurrence(payload.occurrence.unwrap_or(1))
            .fetch_body(payload.fetch_body)
            .build();
        Ok(args)
    },
    runner: |args: LocateSnippetArgs, state: &CallState| LocateSnippetTool {
        args,
        state: state.clone(),
    }
);

struct LocateSnippetTool {
    args:  LocateSnippetArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for LocateSnippetTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let resolved = resolve_workspace_path(
            self.state.workspace_root.as_ref(),
            &self.args.path,
            IDENTIFIER,
        )?;
        let content = filesystem::read_file_full(&resolved)
            .await
            .with_context(|| format!("locate_snippet failed to read {}", resolved.display()))?;

        let needle = if self.args.case_sensitive {
            self.args.snippet.clone()
        } else {
            self.args.snippet.to_ascii_lowercase()
        };
        let haystack = if self.args.case_sensitive {
            content.clone()
        } else {
            content.to_ascii_lowercase()
        };

        let occurrence = self.args.occurrence.unwrap_or(1).max(1);
        let mut search_start: usize = 0;
        let mut found: Option<usize> = None;

        for _ in 0..occurrence {
            match haystack[search_start..].find(&needle) {
                Some(pos) => {
                    let abs = search_start + pos;
                    found = Some(abs);
                    search_start = abs + needle.len();
                }
                None => {
                    found = None;
                    break;
                }
            }
        }

        let start_byte = found.ok_or_else(|| {
            ToolExecutionError::Input(ToolInputError::InvalidPayload {
                tool:    IDENTIFIER,
                message: format!(
                    "snippet occurrence {} not found in {}",
                    occurrence,
                    render_relative_path(self.state.workspace_root.as_ref(), &resolved)
                ),
            })
        })?;
        let end_byte = start_byte
            + self
                .args
                .snippet
                .len()
                .min(content.len().saturating_sub(start_byte));

        let pre = &content[..start_byte];
        let match_text = content[start_byte..end_byte].to_string();
        let start_line = pre.lines().count().saturating_add(1);
        let end_line = start_line + match_text.lines().count().saturating_sub(1);
        let case_sensitive = self.args.case_sensitive;
        let requested_snippet = self.args.snippet.clone();

        let mode = ToolPayloadMode::from_fetch_flag(self.args.fetch_body);
        let path_hint = render_relative_path(self.state.workspace_root.as_ref(), &resolved);
        let runner = ToolRunner::new(IDENTIFIER, &self.state)
            .with_mode(mode)
            .hints(vec![
                format!("Path: {path_hint}"),
                format!("Lines: {start_line}-{end_line}"),
                "Use these line numbers in source_refs.".to_string(),
            ]);

        info!(
            mode = ?mode,
            depth = self.state.depth,
            start_line,
            end_line,
            start_byte,
            end_byte,
            occurrence,
            case_sensitive,
            "tool_call locate_snippet",
        );

        runner
            .run(move |_mode| async move {
                Ok(ToolRunPayload {
                    body:          json!({
                        "type": "data",
                        "mode": "body",
                        "path": path_hint,
                        "snippet": match_text,
                        "requested_snippet": requested_snippet,
                        "start_line": start_line,
                        "end_line": end_line,
                        "start_byte": start_byte,
                        "end_byte": end_byte,
                        "occurrence": occurrence,
                        "case_sensitive": case_sensitive,
                    }),
                    approx_bytes:  None,
                    preview:       Some(json!({
                        "type": "data",
                        "tool": IDENTIFIER,
                        "mode": "preview",
                        "path": path_hint,
                        "start_line": start_line,
                        "end_line": end_line,
                        "occurrence": occurrence,
                        "case_sensitive": case_sensitive,
                    })),
                    preview_hints: Vec::new(),
                    page:          None,
                })
            })
            .await
    }
}
