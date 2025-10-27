use std::{
    env,
    future::Future,
    path::{Path, PathBuf},
    pin::Pin,
    time::{Duration, Instant},
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
use tokio::{
    task::JoinSet,
    time::{sleep, timeout},
};
use tracing::{debug, info, warn};

use crate::{
    constants::{
        DEFAULT_MAX_SUBDELEGATIONS, DEFAULT_PARALLEL_DELEGATIONS, DEFAULT_TEMPERATURE,
        DEFAULT_TOP_P, MAX_PARALLEL_DELEGATIONS, MAX_TOOL_ITERATIONS, PRETEXT_SUBDIR,
        REQUEST_TIMEOUT_SECS, RETRY_BASE_DELAY_MS, RETRY_MAX_EXP, RETRY_MAX_JITTER_MS,
    },
    tools::{filesystem, search},
};

const SYSTEM_PROMPT_TEMPLATE: &str = r#"You are Weaver's file-reading assistant assigned to explore the UNCC CS2 PreTeXt project.
Always stay within the UNCC CS2 PreTeXt workspace and rely on the provided tools to inspect files.
The primary course content lives under `./uncc_cs2-pretext-project/`; call `list_directory` whenever you need to confirm the current structure.
Never assume content from file names alone—use `read_file_full` or `read_file_range` to inspect source material before describing or citing it.
When tasks can be partitioned, prefer launching delegate subtasks in parallel. The `delegate_subtask` tool accepts either a single `subtask` string or a `subtasks` array, and it is recommended to batch independent subtasks so they run concurrently. The runtime executes up to 4 subtasks concurrently.
Only answer after gathering the necessary context via tool calls, and reference the specific files you actually examined."#;

#[derive(Clone, Copy, Debug)]
pub enum ToolName {
    ListDirectory,
    ReadFileFull,
    ReadFileRange,
    SearchText,
    DelegateSubtask,
}

impl ToolName {
    pub const fn identifier(self) -> &'static str {
        match self {
            ToolName::ListDirectory => "list_directory",
            ToolName::ReadFileFull => "read_file_full",
            ToolName::ReadFileRange => "read_file_range",
            ToolName::SearchText => "search_text",
            ToolName::DelegateSubtask => "delegate_subtask",
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
            ToolName::DelegateSubtask => {
                "Delegate one or more subtasks to child FileReader agents. This is preferred for \
                 parallelizable work."
            }
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
            ToolName::DelegateSubtask => json!({
                "type": "object",
                "properties": {
                    "subtask": {
                        "type": "string",
                        "description": "Instruction for a delegated FileReader agent."
                    },
                    "subtasks": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Precise description with motivation of subtasks to delegate in parallel."
                    }
                },
                "anyOf": [
                    {"required": ["subtask"]},
                    {"required": ["subtasks"]}
                ]
            }),
        }
    }

    fn from_identifier(name: &str) -> Option<Self> {
        match name {
            "list_directory" => Some(ToolName::ListDirectory),
            "read_file_full" => Some(ToolName::ReadFileFull),
            "read_file_range" => Some(ToolName::ReadFileRange),
            "search_text" => Some(ToolName::SearchText),
            "delegate_subtask" => Some(ToolName::DelegateSubtask),
            _ => None,
        }
    }
}

const TOOL_NAMES: [ToolName; 5] = [
    ToolName::ListDirectory,
    ToolName::ReadFileFull,
    ToolName::ReadFileRange,
    ToolName::SearchText,
    ToolName::DelegateSubtask,
];

/// Actor that exposes local filesystem utilities to LLM collaborators.
#[derive(Actor)]
pub struct FileReader {
    client:             Client<OpenAIConfig>,
    model:              String,
    root:               PathBuf,
    depth:              usize,
    max_subdelegations: usize,
}

impl FileReader {
    /// Build a new [`FileReader`] using `OPENAI_MODEL`/`OPENAI_API_BASE`.
    pub fn from_env(root: impl AsRef<Path>) -> Result<Self> {
        Self::from_env_with_limit(root, DEFAULT_MAX_SUBDELEGATIONS)
    }

    /// Build a new [`FileReader`] with a custom delegation limit.
    pub fn from_env_with_limit(root: impl AsRef<Path>, max_subdelegations: usize) -> Result<Self> {
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

        let client = Client::with_config(config);
        Ok(Self::new(client, model, root, 0, max_subdelegations))
    }

    fn new(
        client: Client<OpenAIConfig>,
        model: String,
        root: PathBuf,
        depth: usize,
        max_subdelegations: usize,
    ) -> Self {
        Self {
            client,
            model,
            root,
            depth,
            max_subdelegations,
        }
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

    fn spawn_child_reader(&self) -> FileReader {
        FileReader::new(
            self.client.clone(),
            self.model.clone(),
            self.root.clone(),
            self.depth + 1,
            self.max_subdelegations,
        )
    }

    fn spawn_delegate_task(
        &self,
        join_set: &mut JoinSet<(usize, String, Result<String>)>,
        idx: usize,
        subtask: String,
    ) -> Result<()> {
        let child = self.spawn_child_reader();
        let system_prompt = child.system_prompt();
        let system_msg = ChatCompletionRequestSystemMessageArgs::default()
            .content(system_prompt)
            .build()?;
        let user_msg = ChatCompletionRequestUserMessageArgs::default()
            .content(subtask.clone())
            .build()?;
        let messages = vec![system_msg.into(), user_msg.into()];

        join_set.spawn(async move {
            let outcome = child.run_conversation(messages).await;
            (idx, subtask, outcome)
        });
        Ok(())
    }

    async fn run_delegate_batch(
        &self,
        subtasks: Vec<String>,
        max_concurrency: usize,
    ) -> Result<Value> {
        let total = subtasks.len();
        if total == 0 {
            bail!("delegate_subtask requires at least one subtask");
        }

        let limit = max_concurrency
            .clamp(1, MAX_PARALLEL_DELEGATIONS)
            .min(total);

        let mut join_set: JoinSet<(usize, String, Result<String>)> = JoinSet::new();
        let mut pending = subtasks.into_iter().enumerate();
        let mut active = 0usize;

        for _ in 0..limit {
            if let Some((idx, subtask)) = pending.next() {
                self.spawn_delegate_task(&mut join_set, idx, subtask)?;
                active += 1;
            }
        }

        let mut results: Vec<Option<Value>> = vec![None; total];

        while active > 0 {
            if let Some(res) = join_set.join_next().await {
                active -= 1;
                match res {
                    Ok((idx, subtask, outcome)) => {
                        let entry = match outcome {
                            Ok(content) => json!({
                                "subtask": subtask,
                                "status": "ok",
                                "content": content,
                            }),
                            Err(err) => json!({
                                "subtask": subtask,
                                "status": "error",
                                "error": err.to_string(),
                            }),
                        };
                        results[idx] = Some(entry);
                    }
                    Err(join_err) => {
                        bail!("delegate subtask panicked: {join_err}");
                    }
                }
            }

            if let Some((idx, subtask)) = pending.next() {
                self.spawn_delegate_task(&mut join_set, idx, subtask)?;
                active += 1;
            }
        }

        if results.iter().any(|entry| entry.is_none()) {
            bail!("missing delegate results after execution");
        }

        let collected: Vec<Value> = results
            .into_iter()
            .map(|entry| entry.expect("guarded above"))
            .collect();

        Ok(json!({
            "type": "delegation_batch_result",
            "depth": self.depth + 1,
            "requested": collected.len(),
            "max_concurrency": limit,
            "results": collected,
        }))
    }

    async fn sleep_backoff(&self, iteration: usize) {
        let exp = (iteration as u32).min(RETRY_MAX_EXP);
        let multiplier = match 1u64.checked_shl(exp) {
            Some(m) => m,
            None => u64::MAX,
        };
        let base_delay = RETRY_BASE_DELAY_MS.saturating_mul(multiplier).min(60_000);
        let jitter = (iteration as u64 * 137) % (RETRY_MAX_JITTER_MS + 1);
        let delay_ms = base_delay + jitter;
        sleep(Duration::from_millis(delay_ms)).await;
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
                    "tool_call list_directory depth={} path={} entry_count={}",
                    self.depth,
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
                    "tool_call read_file_full depth={} path={} bytes={}",
                    self.depth,
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
                    "tool_call read_file_range depth={} path={} start_line={} end_line={} \
                     line_count={}",
                    self.depth,
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
                    "tool_call search_text depth={} scope={} pattern={} match_count={}",
                    self.depth,
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
            ToolName::DelegateSubtask => {
                if self.depth >= self.max_subdelegations {
                    bail!(
                        "delegate_subtask limit reached (depth {} >= {})",
                        self.depth,
                        self.max_subdelegations
                    );
                }
                let mut subtasks = Vec::new();

                if let Some(subtask) = args.get("subtask").and_then(Value::as_str) {
                    let trimmed = subtask.trim();
                    if !trimmed.is_empty() {
                        subtasks.push(trimmed.to_string());
                    }
                }

                if let Some(value) = args.get("subtasks") {
                    let arr = value
                        .as_array()
                        .ok_or_else(|| anyhow!("`subtasks` must be an array of strings"))?;
                    for entry in arr {
                        let text = entry
                            .as_str()
                            .ok_or_else(|| anyhow!("`subtasks` must contain only strings"))?;
                        let trimmed = text.trim();
                        if !trimmed.is_empty() {
                            subtasks.push(trimmed.to_string());
                        }
                    }
                }

                if subtasks.is_empty() {
                    if let Some(prompt) = args.get("prompt").and_then(Value::as_str) {
                        let trimmed = prompt.trim();
                        if !trimmed.is_empty() {
                            subtasks.push(trimmed.to_string());
                        }
                    }
                    if let Some(value) = args.get("prompts") {
                        let arr = value
                            .as_array()
                            .ok_or_else(|| anyhow!("`prompts` must be an array of strings"))?;
                        for entry in arr {
                            let text = entry
                                .as_str()
                                .ok_or_else(|| anyhow!("`prompts` must contain only strings"))?;
                            let trimmed = text.trim();
                            if !trimmed.is_empty() {
                                subtasks.push(trimmed.to_string());
                            }
                        }
                    }
                }

                if subtasks.is_empty() {
                    bail!("delegate_subtask requires `subtask` or a non-empty `subtasks` array");
                }

                info!(
                    "tool_call delegate_subtask depth={} subtasks={} max_concurrency={}",
                    self.depth + 1,
                    subtasks.len(),
                    DEFAULT_PARALLEL_DELEGATIONS
                );

                let result = self
                    .run_delegate_batch(subtasks, DEFAULT_PARALLEL_DELEGATIONS)
                    .await?;
                Ok(result)
            }
        }
    }

    fn run_conversation(
        &self,
        mut messages: Vec<ChatCompletionRequestMessage>,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + '_>> {
        Box::pin(async move {
            let timeout_duration = Duration::from_secs(REQUEST_TIMEOUT_SECS);
            'retry: for iteration in 0..MAX_TOOL_ITERATIONS {
                debug!(iteration, "Starting LLM tool iteration");
                let request = CreateChatCompletionRequestArgs::default()
                    .model(self.model.clone())
                    .messages(messages.clone())
                    .temperature(DEFAULT_TEMPERATURE)
                    .top_p(DEFAULT_TOP_P)
                    .tools(self.tool_specs())
                    .build()?;
                info!(
                    "llm_request_start depth={} iteration={} messages={}",
                    self.depth,
                    iteration,
                    messages.len()
                );
                let start = Instant::now();
                let response_res =
                    timeout(timeout_duration, self.client.chat().create(request)).await;
                let response = match response_res {
                    Ok(Ok(resp)) => {
                        info!(
                            "llm_request_ok depth={} iteration={} elapsed_ms={}",
                            self.depth,
                            iteration,
                            start.elapsed().as_millis()
                        );
                        resp
                    }
                    Ok(Err(err)) => {
                        warn!(
                            "llm_request_error depth={} iteration={} error={}",
                            self.depth, iteration, err
                        );
                        self.sleep_backoff(iteration).await;
                        continue 'retry;
                    }
                    Err(_) => {
                        warn!(
                            "llm_request_timeout depth={} iteration={} timeout_secs={}",
                            self.depth,
                            iteration,
                            timeout_duration.as_secs()
                        );
                        self.sleep_backoff(iteration).await;
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

                debug!(
                    iteration,
                    "Assistant response had no tool calls and no content; continuing"
                );
            }

            bail!(
                "LLM tool loop did not terminate with a message after {} iterations",
                MAX_TOOL_ITERATIONS
            );
        })
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
        let messages: Vec<ChatCompletionRequestMessage> = vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content(self.system_prompt())
                .build()?
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content(prompt)
                .build()?
                .into(),
        ];
        self.run_conversation(messages).await
    }
}
