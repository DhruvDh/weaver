use std::{
    env,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use anyhow::{Context as _, Result, anyhow};
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
        MAX_TOOL_ITERATIONS,
    },
    llm_gateway::{ChatCompletionRequest, GatewayMetrics, LLMGateway},
    tools::llm::{self, CallState, ToolExecutionError, ToolOutput},
};

const SYSTEM_PROMPT_TEMPLATE: &str = r#"You are Weaver's file-reading assistant assigned to explore the UNCC CS2 PreTeXt project.
Always stay within the provided workspace root and rely on the available tools to inspect files.
Call `list_directory` whenever you need to confirm the current structure instead of inferring it from file names.
Never assume content from file names alone—use `read_file_full` or `read_file_range` to inspect source material before describing or citing it.
When tasks can be partitioned, prefer launching delegated tasks in parallel. The `delegate_tasks` tool accepts a `tasks` array; wrap a single instruction in an array when needed. The runtime executes up to 8 tasks concurrently.
Large-result tools return a preview header first. Examine the reported size and token estimates, refine your arguments (e.g., smaller line ranges or narrower regex scopes), or delegate a summarisation task before opting into full payloads. Only set `fetch_body` to true when you are confident the resulting content fits within the conversation budget.
Only answer after gathering the necessary context via tool calls, and reference the specific files you actually examined."#;

/// Message used by the LLM gateway to execute a tool invocation within the
/// file reader context.
pub struct ExecuteTool {
    pub identifier: String,
    pub arguments:  Value,
}

/// Actor that exposes local filesystem utilities to LLM collaborators.
static NEXT_FILE_READER_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Actor)]
pub struct FileReader {
    gateway:            ActorRef<LLMGateway>,
    model:              Arc<String>,
    root:               Arc<PathBuf>,
    metrics:            Arc<GatewayMetrics>,
    graph:              ActorRef<crate::graph::manager::GraphManager>,
    depth:              usize,
    max_subdelegations: usize,
    actor_name:         Arc<String>,
    conversation_id:    Arc<String>,
}

impl FileReader {
    /// Build a new [`FileReader`] using `OPENAI_MODEL`/`OPENAI_API_BASE`.
    pub fn from_env(
        root: impl AsRef<Path>,
        gateway: ActorRef<LLMGateway>,
        metrics: Arc<GatewayMetrics>,
        graph: ActorRef<crate::graph::manager::GraphManager>,
    ) -> Result<Self> {
        Self::from_env_with_limit(root, gateway, metrics, graph, DEFAULT_MAX_SUBDELEGATIONS)
    }

    /// Build a new [`FileReader`] with a custom delegation limit.
    pub fn from_env_with_limit(
        root: impl AsRef<Path>,
        gateway: ActorRef<LLMGateway>,
        metrics: Arc<GatewayMetrics>,
        graph: ActorRef<crate::graph::manager::GraphManager>,
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

        Ok(Self::new(gateway, model, root, metrics, graph, 0, max_subdelegations))
    }

    fn new(
        gateway: ActorRef<LLMGateway>,
        model: Arc<String>,
        root: Arc<PathBuf>,
        metrics: Arc<GatewayMetrics>,
        graph: ActorRef<crate::graph::manager::GraphManager>,
        depth: usize,
        max_subdelegations: usize,
    ) -> Self {
        let actor_name = if depth == 0 {
            "FileReader/Root".to_string()
        } else {
            format!("FileReader/Delegate{}", depth)
        };
        let id = NEXT_FILE_READER_ID.fetch_add(1, Ordering::Relaxed);
        let conversation_id = format!("{}#{}", actor_name, id);
        Self {
            gateway,
            model,
            root,
            metrics,
            graph,
            depth,
            max_subdelegations,
            actor_name: Arc::new(actor_name),
            conversation_id: Arc::new(conversation_id),
        }
    }

    fn system_prompt(&self) -> String {
        format!("{SYSTEM_PROMPT_TEMPLATE}\n\nWorkspace root: {root}", root = self.root.display())
    }

    pub fn workspace_root(&self) -> &Path {
        self.root.as_ref()
    }

    pub fn tool_identifiers() -> Vec<&'static str> {
        llm::all_tools().iter().map(|meta| meta.id).collect()
    }
}

#[allow(clippy::too_many_arguments)]
#[derive(Clone)]
pub(crate) struct DelegateBatchCtx {
    pub gateway:            ActorRef<LLMGateway>,
    pub model:              Arc<String>,
    pub workspace_root:     Arc<PathBuf>,
    pub metrics:            Arc<GatewayMetrics>,
    pub graph:              ActorRef<crate::graph::manager::GraphManager>,
    pub depth:              usize,
    pub max_subdelegations: usize,
}

pub(crate) async fn run_delegate_batch_with_state(
    ctx: DelegateBatchCtx,
    tasks: Vec<String>,
) -> Result<Value> {
    if tasks.is_empty() {
        return Ok(json!({
            "type": "delegation_batch_result",
            "depth": ctx.depth + 1,
            "requested": 0,
            "max_concurrency": 0,
            "results": [],
        }));
    }

    let total = tasks.len();
    let limit = MAX_PARALLEL_DELEGATIONS.min(total);

    let results = stream::iter(tasks.into_iter())
        .map(|task| {
            let gateway = ctx.gateway.clone();
            let model = Arc::clone(&ctx.model);
            let root = Arc::clone(&ctx.workspace_root);
            let metrics = Arc::clone(&ctx.metrics);
            let graph = ctx.graph.clone();
            let depth = ctx.depth;
            let max_subdelegations = ctx.max_subdelegations;
            async move {
                let child = FileReader::new(
                    gateway.clone(),
                    model,
                    root,
                    metrics,
                    graph,
                    depth + 1,
                    max_subdelegations,
                );
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
        "depth": ctx.depth + 1,
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
        let actor_name = (*self.actor_name).clone();
        let conversation_id = (*self.conversation_id).clone();

        ctx.spawn(async move {
            let system_msg: ChatCompletionRequestMessage =
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(system_prompt)
                    .build()?
                    .into();
            let user_msg: ChatCompletionRequestMessage =
                ChatCompletionRequestUserMessageArgs::default()
                    .content(prompt)
                    .build()?
                    .into();
            let request = ChatCompletionRequest {
                model,
                messages: vec![system_msg, user_msg],
                temperature: DEFAULT_TEMPERATURE,
                top_p: DEFAULT_TOP_P,
                tool_ids,
                max_iterations: MAX_TOOL_ITERATIONS,
                tool_host,
                actor_name,
                conversation_id,
            };

            let reply = gateway.ask(request).await?;
            Ok(reply)
        })
    }
}

impl Message<ExecuteTool> for FileReader {
    type Reply = Result<ToolOutput, ToolExecutionError>;

    async fn handle(
        &mut self,
        ExecuteTool {
            identifier,
            arguments,
        }: ExecuteTool,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let meta = match llm::lookup_tool(&identifier) {
            Some(meta) => meta,
            None => return Err(llm::unsupported_tool(&identifier).into()),
        };

        let state = CallState {
            depth:              self.depth,
            max_subdelegations: self.max_subdelegations,
            workspace_root:     Arc::clone(&self.root),
            gateway:            self.gateway.clone(),
            model:              Arc::clone(&self.model),
            metrics:            Arc::clone(&self.metrics),
            graph:              self.graph.clone(),
            actor_name:         Arc::clone(&self.actor_name),
            conversation_id:    Arc::clone(&self.conversation_id),
        };

        let tool = (meta.parse)(arguments, &state).map_err(ToolExecutionError::from)?;

        tool.execute().await
    }
}
