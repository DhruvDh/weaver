use std::{
    env,
    path::{Path, PathBuf},
};

use anyhow::{Context as _, Result, anyhow, bail};
use async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestUserMessageArgs,
};
use async_trait::async_trait;
use kameo::{prelude::*, reply::DelegatedReply};
use serde_json::{Value, json};
use tokio::task::JoinSet;

use crate::{
    constants::{
        DEFAULT_MAX_SUBDELEGATIONS, DEFAULT_PARALLEL_DELEGATIONS, DEFAULT_TEMPERATURE,
        DEFAULT_TOP_P, MAX_PARALLEL_DELEGATIONS, MAX_TOOL_ITERATIONS, PRETEXT_SUBDIR,
    },
    llm_gateway::{ChatCompletionRequest, LLMGateway},
    tools::{
        filesystem,
        llm::{self, ToolInvocation, ToolName},
        search,
    },
};

const SYSTEM_PROMPT_TEMPLATE: &str = r#"You are Weaver's file-reading assistant assigned to explore the UNCC CS2 PreTeXt project.
Always stay within the UNCC CS2 PreTeXt workspace and rely on the provided tools to inspect files.
The primary course content lives under `./uncc_cs2-pretext-project/`; call `list_directory` whenever you need to confirm the current structure.
Never assume content from file names alone—use `read_file_full` or `read_file_range` to inspect source material before describing or citing it.
When tasks can be partitioned, prefer launching delegate subtasks in parallel. The `delegate_subtask` tool accepts a `subtasks` array; wrap a single instruction in an array when needed. The runtime executes up to 4 subtasks concurrently.
Only answer after gathering the necessary context via tool calls, and reference the specific files you actually examined."#;

const MASKED_PATH: &str = "<path-unavailable>";

/// Message used by the LLM gateway to execute a tool invocation within the
/// file reader context.
pub struct ExecuteTool {
    pub invocation: llm::ToolInvocation,
}

/// Actor that exposes local filesystem utilities to LLM collaborators.
#[derive(Actor)]
pub struct FileReader {
    gateway:            ActorRef<LLMGateway>,
    model:              String,
    root:               PathBuf,
    depth:              usize,
    max_subdelegations: usize,
}

impl FileReader {
    /// Build a new [`FileReader`] using `OPENAI_MODEL`/`OPENAI_API_BASE`.
    pub fn from_env(root: impl AsRef<Path>, gateway: ActorRef<LLMGateway>) -> Result<Self> {
        Self::from_env_with_limit(root, gateway, DEFAULT_MAX_SUBDELEGATIONS)
    }

    /// Build a new [`FileReader`] with a custom delegation limit.
    pub fn from_env_with_limit(
        root: impl AsRef<Path>,
        gateway: ActorRef<LLMGateway>,
        max_subdelegations: usize,
    ) -> Result<Self> {
        let model = env::var("OPENAI_MODEL")
            .map_err(|_| anyhow!("OPENAI_MODEL environment variable must be set"))?;

        let root = root
            .as_ref()
            .canonicalize()
            .with_context(|| format!("failed to canonicalize root {}", root.as_ref().display()))?;

        Ok(Self::new(gateway, model, root, 0, max_subdelegations))
    }

    fn new(
        gateway: ActorRef<LLMGateway>,
        model: String,
        root: PathBuf,
        depth: usize,
        max_subdelegations: usize,
    ) -> Self {
        Self {
            gateway,
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
        llm::all_tools()
    }

    fn spawn_child_reader(&self) -> FileReader {
        FileReader::new(
            self.gateway.clone(),
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
        let actor = FileReader::spawn(child);
        join_set.spawn(async move {
            let outcome = match actor
                .ask(FileReaderQuery {
                    prompt: subtask.clone(),
                })
                .await
            {
                Ok(content) => Ok(content),
                Err(err) => Err(anyhow!(err)),
            };
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

    fn resolve_workspace_path(&self, relative: impl AsRef<Path>) -> Result<PathBuf> {
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

    fn render_workspace_relative_path(&self, path: &Path) -> String {
        pathdiff::diff_paths(path, &self.root)
            .and_then(|p| p.to_str().map(|s| s.to_string()))
            .unwrap_or_else(|| MASKED_PATH.to_string())
    }

    async fn execute_invocation(&self, invocation: ToolInvocation) -> Result<Value> {
        invocation.into_action().execute(self).await
    }
}

#[async_trait]
impl llm::ToolHost for FileReader {
    fn depth(&self) -> usize {
        self.depth
    }

    fn max_subdelegations(&self) -> usize {
        self.max_subdelegations
    }

    fn workspace_root(&self) -> &Path {
        &self.root
    }

    fn resolve_path(&self, relative: &str) -> Result<PathBuf> {
        self.resolve_workspace_path(relative)
    }

    fn render_relative_path(&self, path: &Path) -> String {
        self.render_workspace_relative_path(path)
    }

    async fn list_directory(&self, path: &Path) -> Result<Vec<filesystem::DirEntryInfo>> {
        filesystem::list_dir(path)
            .await
            .with_context(|| format!("list_directory failed for {}", path.display()))
    }

    async fn read_file_full(&self, path: &Path) -> Result<String> {
        filesystem::read_file_full(path)
            .await
            .with_context(|| format!("read_file_full failed for {}", path.display()))
    }

    async fn read_file_range(
        &self,
        path: &Path,
        start_line: usize,
        end_line: usize,
    ) -> Result<filesystem::FileRange> {
        filesystem::read_file_range(path, start_line, end_line)
            .await
            .with_context(|| {
                format!(
                    "read_file_range failed for {} ({}-{})",
                    path.display(),
                    start_line,
                    end_line
                )
            })
    }

    async fn search_recursive(
        &self,
        scope: &Path,
        pattern: &str,
    ) -> Result<Vec<search::SearchMatch>> {
        search::search_recursive(scope, pattern)
            .await
            .with_context(|| {
                format!("search_text failed for pattern `{}` in {}", pattern, scope.display())
            })
    }

    async fn delegate_subtasks(&self, subtasks: Vec<String>) -> Result<Value> {
        self.run_delegate_batch(subtasks, DEFAULT_PARALLEL_DELEGATIONS)
            .await
    }
}

/// Primary message for querying the file reader via LLM tools.
pub struct FileReaderQuery {
    pub prompt: String,
}

impl Message<FileReaderQuery> for FileReader {
    type Reply = DelegatedReply<Result<String>>;

    async fn handle(
        &mut self,
        FileReaderQuery { prompt }: FileReaderQuery,
        ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let tool_host = ctx.actor_ref().clone();
        let gateway = self.gateway.clone();
        let system_prompt = self.system_prompt();
        let model = self.model.clone();
        let tool_names = Self::tool_names().to_vec();

        ctx.spawn(async move {
            let system_msg: ChatCompletionRequestMessage =
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(system_prompt)
                    .build()
                    .map_err(|err| anyhow!(err))?
                    .into();
            let user_msg: ChatCompletionRequestMessage =
                ChatCompletionRequestUserMessageArgs::default()
                    .content(prompt)
                    .build()
                    .map_err(|err| anyhow!(err))?
                    .into();
            let request = ChatCompletionRequest {
                model,
                messages: vec![system_msg, user_msg],
                temperature: DEFAULT_TEMPERATURE,
                top_p: DEFAULT_TOP_P,
                tool_names,
                max_iterations: MAX_TOOL_ITERATIONS,
                tool_host,
            };

            let response = gateway.ask(request).await.map_err(|err| anyhow!(err))?;
            Ok(response)
        })
    }
}

impl Message<ExecuteTool> for FileReader {
    type Reply = Result<Value>;

    async fn handle(
        &mut self,
        ExecuteTool { invocation }: ExecuteTool,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        self.execute_invocation(invocation).await
    }
}
