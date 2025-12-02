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
use crate::{
    constants::{
        MAX_SEARCH_PATTERN_LEN, MAX_TOOL_PATH_LEN,
        tools::search_text::{
            BODY_BYTE_CAP, BODY_MATCH_CAP, DESCRIPTION, IDENTIFIER, PREVIEW_BYTE_CAP,
            PREVIEW_MATCH_CAP,
        },
    },
    tools::search::{self, SearchOptions},
};

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SearchTextArgs {
    #[schemars(
        length(min = 1, max = MAX_SEARCH_PATTERN_LEN),
        description = "Regex pattern (Rust/ripgrep style). Examples: '<section' (XML tags), \
                       'def\\s+\\w+' (function defs), 'TODO|FIXME' (comments). Case-sensitive by \
                       default; use (?i) prefix for case-insensitive."
    )]
    #[builder(with = |value: String| -> ToolInputResult<_> {
        let pattern = super::require_string(value, IDENTIFIER, "pattern")?;
        super::ensure_max_len(&pattern, MAX_SEARCH_PATTERN_LEN, IDENTIFIER, "pattern")?;
        Ok(pattern)
    })]
    pub pattern:    String,
    #[serde(default)]
    #[schemars(
        length(min = 1, max = MAX_TOOL_PATH_LEN),
        description = "Directory or file to search within (relative to workspace). Omit to search \
                       entire workspace. Example: '02_contracts' to search only that chapter."
    )]
    #[builder(with = |value: String| -> ToolInputResult<_> {
        let path = super::require_string(value, IDENTIFIER, "path")?;
        super::ensure_max_len(&path, MAX_TOOL_PATH_LEN, IDENTIFIER, "path")?;
        Ok(path)
    })]
    pub path:       Option<String>,
    #[serde(default)]
    #[schemars(
        description = "Include normally-excluded directories in search. Rarely needed. Examples: \
                       ['vendor'] to search vendored code, ['.git'] to search git metadata."
    )]
    #[builder(default)]
    pub allow:      Vec<String>,
    #[serde(default)]
    #[schemars(
        description = "Set true to return match content. Default false = returns match count and \
                       locations only. Use false first to check result size, then true to fetch.",
        default = "crate::tools::llm::default_false"
    )]
    #[builder(default = false)]
    pub fetch_body: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchTextPayload {
    pattern:    String,
    #[serde(default)]
    path:       Option<String>,
    #[serde(default)]
    allow:      Vec<String>,
    #[serde(default)]
    fetch_body: bool,
}

crate::basic_tool!(
    search_text_meta,
    id: IDENTIFIER,
    description: DESCRIPTION,
    args: SearchTextArgs,
    prepare: |raw, _state: &CallState| {
        let payload: SearchTextPayload =
            serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
                tool:    IDENTIFIER,
                message: err.to_string(),
            })?;

        for entry in &payload.allow {
            super::ensure_max_len(entry, MAX_TOOL_PATH_LEN, IDENTIFIER, "allow")?;
        }

        let args = match trim_optional(payload.path) {
            Some(path) => SearchTextArgs::builder()
                .pattern(payload.pattern)?
                .path(path)?
                .allow(payload.allow)
                .fetch_body(payload.fetch_body)
                .build(),
            None => SearchTextArgs::builder()
                .pattern(payload.pattern)?
                .allow(payload.allow)
                .fetch_body(payload.fetch_body)
                .build(),
        };

        Ok(args)
    },
    runner: |args: SearchTextArgs, state: &CallState| SearchTextTool {
        args,
        state: state.clone(),
    }
);

struct SearchTextTool {
    args:  SearchTextArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for SearchTextTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let requested_path = self.args.path.clone();
        let scope = match requested_path.as_deref() {
            Some(relative) => {
                resolve_workspace_path(self.state.workspace_root.as_ref(), relative, IDENTIFIER)?
            }
            None => (*self.state.workspace_root).clone(),
        };

        let mode = ToolPayloadMode::from_fetch_flag(self.args.fetch_body);
        let (max_matches, max_bytes) = match mode {
            ToolPayloadMode::Preview => (PREVIEW_MATCH_CAP, PREVIEW_BYTE_CAP),
            ToolPayloadMode::Body => (BODY_MATCH_CAP, BODY_BYTE_CAP),
        };

        let result = search::search_recursive(
            &scope,
            &self.args.pattern,
            SearchOptions {
                allow:       &self.args.allow,
                max_matches: Some(max_matches),
                max_bytes:   Some(max_bytes),
                stop_early:  true,
            },
        )
        .await
        .with_context(|| {
            format!("search_text failed for pattern `{}` in {}", self.args.pattern, scope.display())
        })?;

        let match_count = result.total_matches;
        let approx_bytes = result.total_bytes;
        let truncated = result.truncated;
        let scope_rendered = render_relative_path(self.state.workspace_root.as_ref(), &scope);
        let pattern = self.args.pattern.clone();
        let allow = self.args.allow.clone();

        let mut hints = vec![
            format!("{match_count} matches across roughly {approx_bytes} bytes of context."),
            if truncated {
                format!(
                    "Preview truncated at {} matches or {} bytes; set fetch_body=true to stream \
                     more or narrow the pattern.",
                    max_matches, max_bytes
                )
            } else {
                "Set fetch_body=true to retrieve all matches.".to_string()
            },
            format!("Pattern: `{pattern}`"),
            format!("Scope: {scope_rendered}"),
            "Refine the regex or narrow the path to reduce match volume.".to_string(),
        ];
        if !allow.is_empty() {
            hints.push(format!("Allowing directories: {}", allow.join(", ")));
        }

        let matches = result
            .matches
            .into_iter()
            .map(|m| {
                json!({
                    "path": render_relative_path(self.state.workspace_root.as_ref(), &m.path),
                    "line_number": m.line_number,
                    "line": m.context,
                })
            })
            .collect::<Vec<_>>();

        info!(
            mode = ?mode,
            depth = self.state.depth,
            approx_bytes,
            scope = %scope.display(),
            pattern = %pattern,
            match_count,
            truncated,
            "tool_call search_text",
        );

        ToolRunner::new(IDENTIFIER, &self.state)
            .with_mode(mode)
            .hints(hints)
            .run(|_| async move {
                Ok(ToolRunPayload {
                    body:          json!({
                        "type": "data",
                        "mode": "body",
                        "pattern": pattern,
                        "scope": scope_rendered,
                        "match_count": match_count,
                        "truncated": truncated,
                        "caps": { "max_matches": max_matches, "max_bytes": max_bytes },
                        "matches": matches,
                        "bytes": approx_bytes,
                    }),
                    approx_bytes:  Some(approx_bytes),
                    preview:       Some(json!({
                        "type": "data",
                        "tool": IDENTIFIER,
                        "mode": "preview",
                        "pattern": pattern,
                        "scope": scope_rendered,
                        "match_count": match_count,
                        "truncated": truncated,
                        "approx_bytes": approx_bytes,
                    })),
                    preview_hints: Vec::new(),
                    page:          None,
                })
            })
            .await
    }
}
