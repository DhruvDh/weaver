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
    render_relative_path, resolve_workspace_path, schema_for_args, trim_optional,
};
use crate::{
    llm_gateway::GatewayMetrics,
    tools::search::{self, SearchOptions},
};

const IDENTIFIER: &str = "search_text";
const DESCRIPTION: &str = "Run a regex search (ripgrep-style) within the workspace.";
const PREVIEW_MATCH_CAP: usize = 200;
const PREVIEW_BYTE_CAP: u64 = 64 * 1024;
const BODY_MATCH_CAP: usize = 2_000;
const BODY_BYTE_CAP: u64 = 1_000_000;

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SearchTextArgs {
    #[schemars(length(min = 1), description = "Rust-style regular expression.")]
    #[builder(with = |value: String| -> ToolInputResult<_> {
        super::require_string(value, IDENTIFIER, "pattern")
    })]
    pub pattern:    String,
    #[serde(default)]
    #[schemars(description = "Optional directory to scope the search. Defaults to root.")]
    #[builder(with = |value: String| -> ToolInputResult<_> {
        super::require_string(value, IDENTIFIER, "path")
    })]
    pub path:       Option<String>,
    #[serde(default)]
    #[schemars(
        description = "Directory names to include even if normally excluded (target, .git, \
                       node_modules, vendor)."
    )]
    #[builder(default)]
    pub allow:      Vec<String>,
    #[serde(default)]
    #[schemars(
        description = "When true, return all search matches immediately; otherwise return a \
                       preview header.",
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
            .allow(payload.allow)
            .fetch_body(payload.fetch_body)
            .build(),
        None => SearchTextArgs::builder()
            .pattern(payload.pattern)?
            .allow(payload.allow)
            .fetch_body(payload.fetch_body)
            .build(),
    };

    Ok(Box::new(SearchTextTool {
        args,
        depth: state.depth,
        workspace_root: Arc::clone(&state.workspace_root),
        metrics: Arc::clone(&state.metrics),
        model: Arc::clone(&state.model),
        conversation_id: Arc::clone(&state.conversation_id),
    }))
}

struct SearchTextTool {
    args:            SearchTextArgs,
    depth:           usize,
    workspace_root:  Arc<PathBuf>,
    metrics:         Arc<GatewayMetrics>,
    model:           Arc<String>,
    conversation_id: Arc<String>,
}

#[async_trait]
impl ToolInstance for SearchTextTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let requested_path = self.args.path.clone();
        let scope = match requested_path.as_deref() {
            Some(relative) => {
                resolve_workspace_path(self.workspace_root.as_ref(), relative, IDENTIFIER)?
            }
            None => (*self.workspace_root).clone(),
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
        let token_estimates =
            prepare_payload_estimates(&self.metrics, self.model.as_str(), approx_bytes);
        let safe_tokens = token_estimates.safe_tokens;
        let truncated = result.truncated;

        match mode {
            ToolPayloadMode::Preview => {
                let mut hints = vec![
                    format!(
                        "{match_count} matches across roughly {approx_bytes} bytes of context."
                    ),
                    if truncated {
                        format!(
                            "Preview truncated at {} matches or {} bytes; set fetch_body=true to \
                             stream more or narrow the pattern.",
                            max_matches, max_bytes
                        )
                    } else {
                        "Set fetch_body=true to retrieve all matches.".to_string()
                    },
                    format!("Pattern: `{}`", self.args.pattern),
                    format!(
                        "Scope: {}",
                        render_relative_path(self.workspace_root.as_ref(), &scope)
                    ),
                    "Refine the regex or narrow the path to reduce match volume.".to_string(),
                ];
                if !self.args.allow.is_empty() {
                    hints.push(format!("Allowing directories: {}", self.args.allow.join(", ")));
                }
                let mut value = build_cost_preview(IDENTIFIER, approx_bytes, safe_tokens, hints);
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
                    approx_bytes,
                    scope = %scope.display(),
                    pattern = %self.args.pattern,
                    match_count,
                    truncated,
                    "tool_call search_text preview",
                );
                Ok(ToolOutput::with_byte_hint(value, payload_bytes))
            }
            ToolPayloadMode::Body => {
                let rendered = result
                    .matches
                    .into_iter()
                    .map(|m| {
                        json!({
                            "path": render_relative_path(self.workspace_root.as_ref(), &m.path),
                            "line_number": m.line_number,
                            "line": m.context,
                        })
                    })
                    .collect::<Vec<_>>();

                let payload = json!({
                    "type": "data",
                    "mode": "body",
                    "pattern": &self.args.pattern,
                    "scope": render_relative_path(self.workspace_root.as_ref(), &scope),
                    "match_count": match_count,
                    "truncated": truncated,
                    "caps": { "max_matches": max_matches, "max_bytes": max_bytes },
                    "matches": rendered,
                    "bytes": approx_bytes,
                    "approx_tokens": safe_tokens,
                });
                let payload_bytes = payload_size_bytes(&payload);
                info!(
                    mode = %mode.as_str(),
                    depth = self.depth,
                    payload_bytes,
                    approx_bytes,
                    scope = %scope.display(),
                    pattern = %self.args.pattern,
                    match_count,
                    truncated,
                    "tool_call search_text body",
                );
                Ok(ToolOutput::with_byte_hint(payload, payload_bytes))
            }
        }
    }
}
