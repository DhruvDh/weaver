use std::{
    env,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context as _, Result, anyhow, bail};
use async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestUserMessageArgs,
};
use futures::{StreamExt, stream};
use kameo::{prelude::*, reply::DelegatedReply};
use serde_json::{Value, json};

use crate::{
    constants::{
        DEFAULT_MAX_SUBDELEGATIONS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, MAX_PARALLEL_DELEGATIONS,
        MAX_TOOL_ITERATIONS, PRETEXT_SUBDIR,
    },
    llm_gateway::{ChatCompletionRequest, LLMGateway},
    tools::llm::{self, CallState},
};

const SYSTEM_PROMPT_TEMPLATE: &str = r#"You are Weaver's file-reading assistant assigned to explore the UNCC CS2 PreTeXt project.
Always stay within the UNCC CS2 PreTeXt workspace and rely on the provided tools to inspect files.
The primary course content lives under `./uncc_cs2-pretext-project/`; call `list_directory` whenever you need to confirm the current structure.
Never assume content from file names alone—use `read_file_full` or `read_file_range` to inspect source material before describing or citing it.
When tasks can be partitioned, prefer launching delegated tasks in parallel. The `delegate_tasks` tool accepts a `tasks` array; wrap a single instruction in an array when needed. The runtime executes up to 8 tasks concurrently.
Only answer after gathering the necessary context via tool calls, and reference the specific files you actually examined."#;

/// Message used by the LLM gateway to execute a tool invocation within the
/// file reader context.
pub struct ExecuteTool {
    pub identifier: String,
    pub arguments:  Value,
}

/// Actor that exposes local filesystem utilities to LLM collaborators.
#[derive(Actor)]
pub struct FileReader {
    gateway:            ActorRef<LLMGateway>,
    model:              Arc<String>,
    root:               Arc<PathBuf>,
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
        let model = Arc::new(
            env::var("OPENAI_MODEL")
                .map_err(|_| anyhow!("OPENAI_MODEL environment variable must be set"))?,
        );

        let root =
            Arc::new(root.as_ref().canonicalize().with_context(|| {
                format!("failed to canonicalize root {}", root.as_ref().display())
            })?);

        Ok(Self::new(gateway, model, root, 0, max_subdelegations))
    }

    fn new(
        gateway: ActorRef<LLMGateway>,
        model: Arc<String>,
        root: Arc<PathBuf>,
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
        self.root.as_ref()
    }

    pub fn tool_identifiers() -> Vec<&'static str> {
        llm::all_tools().iter().map(|meta| meta.id).collect()
    }
}

pub(crate) async fn run_delegate_batch_with_state(
    gateway: ActorRef<LLMGateway>,
    model: Arc<String>,
    workspace_root: Arc<PathBuf>,
    depth: usize,
    max_subdelegations: usize,
    tasks: Vec<String>,
) -> Result<Value> {
    let total = tasks.len();
    if total == 0 {
        bail!("delegate_tasks requires at least one task");
    }

    let limit = MAX_PARALLEL_DELEGATIONS.min(total);

    let results = stream::iter(tasks.into_iter())
        .map(|task| {
            let gateway = gateway.clone();
            let model = Arc::clone(&model);
            let root = Arc::clone(&workspace_root);
            async move {
                let child =
                    FileReader::new(gateway.clone(), model, root, depth + 1, max_subdelegations);
                let actor = FileReader::spawn(child);
                let prompt = task.clone();
                match actor.ask(FileReaderQuery { prompt }).await {
                    Ok(content) => {
                        json!({
                            "task": task,
                            "status": "ok",
                            "content": content,
                        })
                    }
                    Err(err) => {
                        json!({
                            "task": task,
                            "status": "error",
                            "error": err.to_string(),
                        })
                    }
                }
            }
        })
        .buffered(limit)
        .collect::<Vec<Value>>()
        .await;

    Ok(json!({
        "type": "delegation_batch_result",
        "depth": depth + 1,
        "requested": total,
        "max_concurrency": limit,
        "results": results,
    }))
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
        let tool_ids = Self::tool_identifiers();

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
                tool_ids,
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
        ExecuteTool {
            identifier,
            arguments,
        }: ExecuteTool,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let meta = llm::lookup_tool(&identifier)
            .ok_or_else(|| anyhow!(llm::unsupported_tool(&identifier)))?;

        let state = CallState {
            depth:              self.depth,
            max_subdelegations: self.max_subdelegations,
            workspace_root:     Arc::clone(&self.root),
            gateway:            self.gateway.clone(),
            model:              Arc::clone(&self.model),
        };

        let tool = (meta.parse)(arguments, &state).map_err(|err| anyhow!(err))?;
        tool.execute().await
    }
}
