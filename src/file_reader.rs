use std::{
    env,
    path::{Path, PathBuf},
};

use anyhow::{Context as _, Result, anyhow, bail};
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::{
        ChatCompletionMessageToolCall, ChatCompletionRequestAssistantMessageArgs,
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestToolMessageArgs, ChatCompletionRequestUserMessageArgs,
        ChatCompletionTool, ChatCompletionToolArgs, ChatCompletionToolType,
        CreateChatCompletionRequestArgs, FunctionObjectArgs,
    },
};
use kameo::prelude::*;
use serde_json::{Value, json};
use tracing::{debug, info, warn};

use crate::{
    constants::{DEFAULT_TEMPERATURE, DEFAULT_TOP_P, MAX_TOOL_ITERATIONS, PRETEXT_SUBDIR},
    tools::{filesystem, search},
};

const SYSTEM_PROMPT_TEMPLATE: &str = r#"You are Weaver's file-reading assistant assigned to explore the UNCC CS2 PreTeXt project.
Always stay within the UNCC CS2 PreTeXt workspace and rely on the provided tools to inspect files.
The primary course content lives under `./uncc_cs2-pretext-project/`; call `list_directory` whenever you need to confirm the current structure.
Never assume content from file names alone—use `read_file_full` or `read_file_range` to inspect source material before describing or citing it.
Only answer after gathering the necessary context via tool calls, and reference the specific files you actually examined."#;

#[derive(Clone, Copy, Debug)]
pub enum ToolName {
    ListDirectory,
    ReadFileFull,
    ReadFileRange,
    SearchText,
}

impl ToolName {
    pub const fn identifier(self) -> &'static str {
        match self {
            ToolName::ListDirectory => "list_directory",
            ToolName::ReadFileFull => "read_file_full",
            ToolName::ReadFileRange => "read_file_range",
            ToolName::SearchText => "search_text",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            ToolName::ListDirectory => {
                "List the entries of a directory relative to the workspace root."
            }
            ToolName::ReadFileFull => "Read the full contents of a UTF-8 text file.",
            ToolName::ReadFileRange => {
                "Read a specific inclusive line range from a UTF-8 text file."
            }
            ToolName::SearchText => "Run a regex search (ripgrep-style) within the workspace.",
        }
    }

    fn json_schema(self) -> Value {
        match self {
            ToolName::ListDirectory => json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to workspace root. Defaults to \".\""
                    }
                }
            }),
            ToolName::ReadFileFull => json!({
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to workspace root."
                    }
                }
            }),
            ToolName::ReadFileRange => json!({
                "type": "object",
                "required": ["path", "start_line", "end_line"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to workspace root."
                    },
                    "start_line": {
                        "type": "integer",
                        "minimum": 1
                    },
                    "end_line": {
                        "type": "integer",
                        "minimum": 1
                    }
                }
            }),
            ToolName::SearchText => json!({
                "type": "object",
                "required": ["pattern"],
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Rust-style regular expression."
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional directory to scope the search. Defaults to root."
                    }
                }
            }),
        }
    }

    fn from_identifier(name: &str) -> Option<Self> {
        match name {
            "list_directory" => Some(ToolName::ListDirectory),
            "read_file_full" => Some(ToolName::ReadFileFull),
            "read_file_range" => Some(ToolName::ReadFileRange),
            "search_text" => Some(ToolName::SearchText),
            _ => None,
        }
    }
}

const TOOL_NAMES: [ToolName; 4] = [
    ToolName::ListDirectory,
    ToolName::ReadFileFull,
    ToolName::ReadFileRange,
    ToolName::SearchText,
];

/// Actor that exposes local filesystem utilities to LLM collaborators.
#[derive(Actor)]
pub struct FileReader {
    client: Client<OpenAIConfig>,
    model:  String,
    root:   PathBuf,
}

impl FileReader {
    /// Build a new [`FileReader`] using `OPENAI_MODEL`/`OPENAI_API_BASE`.
    pub fn from_env(root: impl AsRef<Path>) -> Result<Self> {
        let model = env::var("OPENAI_MODEL")
            .map_err(|_| anyhow!("OPENAI_MODEL environment variable must be set"))?;

        let mut config = OpenAIConfig::default();
        if let Ok(url) = env::var("OPENAI_API_BASE") {
            config = config.with_api_base(url);
        }

        let root = root
            .as_ref()
            .canonicalize()
            .with_context(|| format!("failed to canonicalize root {}", root.as_ref().display()))?;

        Ok(Self {
            client: Client::with_config(config),
            model,
            root,
        })
    }

    fn system_prompt(&self) -> String {
        format!(
            "{SYSTEM_PROMPT_TEMPLATE}\n\nWorkspace root: {root}\nPreTeXt project path: \
             {root}/{subdir}",
            root = self.root.display(),
            subdir = PRETEXT_SUBDIR
        )
    }

    pub fn workspace_root(&self) -> &Path {
        &self.root
    }

    pub fn tool_names() -> &'static [ToolName] {
        &TOOL_NAMES
    }

    fn tool_specs(&self) -> Vec<ChatCompletionTool> {
        TOOL_NAMES
            .iter()
            .map(|tool| {
                ChatCompletionToolArgs::default()
                    .r#type(ChatCompletionToolType::Function)
                    .function(
                        FunctionObjectArgs::default()
                            .name(tool.identifier())
                            .description(tool.description())
                            .parameters(tool.json_schema())
                            .build()
                            .expect("tool schema"),
                    )
                    .build()
                    .expect("tool")
            })
            .collect()
    }

    fn resolve_path(&self, relative: impl AsRef<Path>) -> Result<PathBuf> {
        let rel = relative.as_ref();
        let candidate = if rel.is_absolute() {
            rel.to_path_buf()
        } else {
            self.root.join(rel)
        };
        let canonical = candidate.canonicalize().with_context(|| {
            format!("failed to canonicalize resolved path {}", candidate.display())
        })?;
        if !canonical.starts_with(&self.root) {
            bail!("path {} escapes workspace root {}", canonical.display(), self.root.display());
        }
        Ok(canonical)
    }

    async fn execute_tool(&self, call: &ChatCompletionMessageToolCall) -> Result<Value> {
        let args: Value = serde_json::from_str(&call.function.arguments)
            .with_context(|| format!("invalid JSON arguments for {}", call.function.name))?;

        let tool = ToolName::from_identifier(call.function.name.as_str())
            .ok_or_else(|| anyhow!("unsupported tool call: {}", call.function.name))?;

        match tool {
            ToolName::ListDirectory => {
                let path_str = args.get("path").and_then(Value::as_str).unwrap_or(".");
                let path = self.resolve_path(path_str)?;
                let entries = filesystem::list_dir(&path)
                    .await
                    .with_context(|| format!("list_directory failed for {}", path.display()))?;
                info!(
                    "tool_call list_directory path={} entry_count={}",
                    path.display(),
                    entries.len()
                );
                let rendered = entries
                    .into_iter()
                    .map(|entry| {
                        json!({
                            "name": entry.name,
                            "path": pathdiff::diff_paths(&entry.path, &self.root)
                                .and_then(|p| p.to_str().map(|s| s.to_string()))
                                .unwrap_or_else(|| entry.path.display().to_string()),
                            "kind": entry.kind.as_str(),
                            "size": entry.size,
                        })
                    })
                    .collect::<Vec<_>>();
                Ok(json!({ "entries": rendered }))
            }
            ToolName::ReadFileFull => {
                let path_str = args
                    .get("path")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("read_file_full requires `path`"))?;
                let path = self.resolve_path(path_str)?;
                let content = filesystem::read_file_full(&path)
                    .await
                    .with_context(|| format!("read_file_full failed for {}", path.display()))?;
                info!(
                    "tool_call read_file_full path={} bytes={}",
                    path.display(),
                    content.as_bytes().len()
                );
                Ok(json!({
                    "path": pathdiff::diff_paths(&path, &self.root)
                        .and_then(|p| p.to_str().map(|s| s.to_string()))
                        .unwrap_or_else(|| path.display().to_string()),
                    "content": content,
                }))
            }
            ToolName::ReadFileRange => {
                let path_str = args
                    .get("path")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("read_file_range requires `path`"))?;
                let start_line = args
                    .get("start_line")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| anyhow!("read_file_range requires `start_line`"))?
                    as usize;
                let end_line = args
                    .get("end_line")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| anyhow!("read_file_range requires `end_line`"))?
                    as usize;
                let path = self.resolve_path(path_str)?;
                let range = filesystem::read_file_range(&path, start_line, end_line)
                    .await
                    .with_context(|| {
                        format!(
                            "read_file_range failed for {} ({}-{})",
                            path.display(),
                            start_line,
                            end_line
                        )
                    })?;
                info!(
                    "tool_call read_file_range path={} start_line={} end_line={} line_count={}",
                    range.path.display(),
                    range.start_line,
                    range.end_line,
                    range.end_line.saturating_sub(range.start_line) + 1
                );
                Ok(json!({
                    "path": pathdiff::diff_paths(&range.path, &self.root)
                        .and_then(|p| p.to_str().map(|s| s.to_string()))
                        .unwrap_or_else(|| range.path.display().to_string()),
                    "start_line": range.start_line,
                    "end_line": range.end_line,
                    "content": range.text,
                }))
            }
            ToolName::SearchText => {
                let pattern = args
                    .get("pattern")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("search_text requires `pattern`"))?;
                let path = args
                    .get("path")
                    .and_then(Value::as_str)
                    .map(|p| self.resolve_path(p))
                    .transpose()?
                    .unwrap_or_else(|| self.root.clone());

                let matches = search::search_recursive(&path, pattern)
                    .await
                    .with_context(|| {
                        format!(
                            "search_text failed for pattern `{}` in {}",
                            pattern,
                            path.display()
                        )
                    })?;
                info!(
                    "tool_call search_text scope={} pattern={} match_count={}",
                    path.display(),
                    pattern,
                    matches.len()
                );
                let rendered = matches
                    .into_iter()
                    .map(|m| {
                        json!({
                            "path": pathdiff::diff_paths(m.path, &self.root)
                                .and_then(|p| p.to_str().map(|s| s.to_string()))
                                .unwrap_or_else(|| "".to_string()),
                            "line_number": m.line_number,
                            "line": m.context,
                        })
                    })
                    .collect::<Vec<_>>();
                Ok(json!({ "matches": rendered }))
            }
        }
    }
}

/// Primary message for querying the file reader via LLM tools.
pub struct FileReaderQuery {
    pub prompt: String,
}

impl Message<FileReaderQuery> for FileReader {
    type Reply = Result<String>;

    async fn handle(
        &mut self,
        FileReaderQuery { prompt }: FileReaderQuery,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let mut messages: Vec<ChatCompletionRequestMessage> = vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content(self.system_prompt())
                .build()?
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content(prompt)
                .build()?
                .into(),
        ];

        'retry: for iteration in 0..MAX_TOOL_ITERATIONS {
            debug!(iteration, "Starting LLM tool iteration");
            let request = CreateChatCompletionRequestArgs::default()
                .model(self.model.clone())
                .messages(messages.clone())
                .temperature(DEFAULT_TEMPERATURE)
                .top_p(DEFAULT_TOP_P)
                .tools(self.tool_specs())
                .build()?;

            let response = match self.client.chat().create(request).await {
                Ok(resp) => resp,
                Err(err) => {
                    warn!(iteration, error = %err, "LLM request failed; retrying");
                    continue 'retry;
                }
            };
            let mut choices = response.choices.into_iter();
            let message = choices
                .next()
                .ok_or_else(|| anyhow!("chat completion returned no choices"))?
                .message;

            debug!(
                iteration,
                role = ?message.role,
                has_content = message.content.as_ref().map(|c| !c.trim().is_empty()),
                tool_call_count = message.tool_calls.as_ref().map(|c| c.len()),
                content = ?message.content,
                refusal = ?message.refusal,
                "Assistant message received"
            );

            match message.tool_calls {
                Some(tool_calls) if !tool_calls.is_empty() => {
                    debug!(
                        iteration,
                        tool_call_count = tool_calls.len(),
                        "Assistant requested tool calls"
                    );
                    let assistant_msg = ChatCompletionRequestAssistantMessageArgs::default()
                        .tool_calls(tool_calls.clone())
                        .build()?
                        .into();
                    messages.push(assistant_msg);

                    for tool_call in tool_calls {
                        debug!(
                            iteration,
                            tool = tool_call.function.name.as_str(),
                            "Executing assistant-requested tool"
                        );
                        let tool_name = tool_call.function.name.clone();
                        let tool_msg = match self.execute_tool(&tool_call).await {
                            Ok(result) => ChatCompletionRequestToolMessageArgs::default()
                                .tool_call_id(tool_call.id.clone())
                                .content(result.to_string())
                                .build()?
                                .into(),
                            Err(err) => {
                                warn!(
                                    "tool_error tool={} iteration={} error={}",
                                    tool_name, iteration, err
                                );
                                let args_json: Value =
                                    serde_json::from_str(&tool_call.function.arguments)
                                        .unwrap_or_else(|_| {
                                            Value::String(tool_call.function.arguments.clone())
                                        });
                                let error_payload = json!({
                                    "type": "tool_error",
                                    "tool": tool_name,
                                    "arguments": args_json,
                                    "message": err.to_string(),
                                });
                                ChatCompletionRequestToolMessageArgs::default()
                                    .tool_call_id(tool_call.id.clone())
                                    .content(error_payload.to_string())
                                    .build()?
                                    .into()
                            }
                        };
                        messages.push(tool_msg);
                    }
                    continue;
                }
                Some(empty_calls) => {
                    debug!(
                        iteration,
                        tool_call_count = empty_calls.len(),
                        "Assistant returned empty tool call list; treating as no tool calls"
                    );
                }
                None => {}
            }

            if let Some(content) = message.content {
                let trimmed = content.trim();
                if trimmed.is_empty() {
                    debug!(iteration, "Assistant content was empty; continuing");
                } else {
                    debug!(iteration, "Assistant returned final content");
                    return Ok(content);
                }
            }

            debug!(iteration, "Assistant response had no tool calls and no content; continuing");
        }

        bail!(
            "LLM tool loop did not terminate with a message after {} iterations",
            MAX_TOOL_ITERATIONS
        );
    }
}
