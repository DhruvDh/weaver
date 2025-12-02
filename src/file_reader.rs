use std::{
    env, fmt,
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context as _, Result, anyhow};
use async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestUserMessageArgs,
};
use futures::{StreamExt, stream};
use kameo::{prelude::*, reply::DelegatedReply};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    agents::deduplication::DeduplicationAgent,
    constants::{
        DEFAULT_MAX_SUBDELEGATIONS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, MAX_PARALLEL_DELEGATIONS,
        MAX_TOOL_ITERATIONS,
    },
    llm_gateway::{ChatCompletionRequest, GatewayMetrics, LLMGateway},
    tools::llm::{self, CallState, ToolExecutionError, ToolOutput, analysis_cache::AnalysisCache},
};

/// Operational mode for FileReader actors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AgentMode {
    /// Phase 1: Extract nodes only (no edge creation).
    Harvester,
    /// Phase 2: Connect nodes only (no node creation).
    Weaver,
    /// Traditional mode: unrestricted tool access (default for interactive
    /// use).
    #[default]
    Interactive,
}

/// Optional specialization for harvester actors.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarvesterFocus {
    Factual,
    Conceptual,
    Procedural,
    Metacognitive,
    All,
}

/// Optional specialization for weaver actors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeaverFocus {
    Requires,
    Supports,
    Assesses,
    All,
}

#[derive(Clone, Debug)]
pub enum ToolHost {
    Interactive(ActorRef<Reader<InteractiveSpec>>),
    Harvester(ActorRef<Reader<HarvesterSpec>>),
    Weaver(ActorRef<Reader<WeaverSpec>>),
}

impl ToolHost {
    pub fn mode(&self) -> AgentMode {
        match self {
            ToolHost::Interactive(_) => AgentMode::Interactive,
            ToolHost::Harvester(_) => AgentMode::Harvester,
            ToolHost::Weaver(_) => AgentMode::Weaver,
        }
    }

    pub fn tool_ids(&self) -> Result<Vec<&'static str>> {
        tool_identifiers_for_mode(self.mode())
    }

    pub async fn ask_execute(
        &self,
        msg: ExecuteTool,
    ) -> Result<ToolOutput, kameo::error::SendError<ExecuteTool, ToolExecutionError>> {
        match self {
            ToolHost::Interactive(actor) => actor.ask(msg).await,
            ToolHost::Harvester(actor) => actor.ask(msg).await,
            ToolHost::Weaver(actor) => actor.ask(msg).await,
        }
    }
}

pub trait ModeSpec: Send + Sync + 'static {
    const MODE: AgentMode;
    const SUFFIX: &'static str;
    fn wrap_tool_host(actor: ActorRef<Reader<Self>>) -> ToolHost
    where
        Self: Sized;
}

pub struct HarvesterSpec;
pub struct WeaverSpec;
pub struct InteractiveSpec;

impl ModeSpec for HarvesterSpec {
    const MODE: AgentMode = AgentMode::Harvester;
    const SUFFIX: &'static str = "/Harvester";

    fn wrap_tool_host(actor: ActorRef<Reader<Self>>) -> ToolHost {
        ToolHost::Harvester(actor)
    }
}

impl ModeSpec for WeaverSpec {
    const MODE: AgentMode = AgentMode::Weaver;
    const SUFFIX: &'static str = "/Weaver";

    fn wrap_tool_host(actor: ActorRef<Reader<Self>>) -> ToolHost {
        ToolHost::Weaver(actor)
    }
}

impl ModeSpec for InteractiveSpec {
    const MODE: AgentMode = AgentMode::Interactive;
    const SUFFIX: &'static str = "";

    fn wrap_tool_host(actor: ActorRef<Reader<Self>>) -> ToolHost {
        ToolHost::Interactive(actor)
    }
}

impl fmt::Display for HarvesterFocus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            HarvesterFocus::Factual => "factual",
            HarvesterFocus::Conceptual => "conceptual",
            HarvesterFocus::Procedural => "procedural",
            HarvesterFocus::Metacognitive => "metacognitive",
            HarvesterFocus::All => "all",
        };
        write!(f, "{label}")
    }
}

impl HarvesterFocus {
    pub fn directive(&self) -> &'static str {
        match self {
            HarvesterFocus::Factual => {
                "Prioritize factual terms/definitions first, then extract remaining concepts."
            }
            HarvesterFocus::Conceptual => {
                "Prioritize conceptual nodes and definitions first, then extract remaining items."
            }
            HarvesterFocus::Procedural => {
                "Prioritize procedural nodes and worked examples first, then extract remaining \
                 items."
            }
            HarvesterFocus::Metacognitive => {
                "Prioritize metacognitive strategies first, then extract remaining items."
            }
            HarvesterFocus::All => "Cover all node types evenly; completeness is the goal.",
        }
    }
}

impl fmt::Display for WeaverFocus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            WeaverFocus::Requires => "requires",
            WeaverFocus::Supports => "supports",
            WeaverFocus::Assesses => "assesses",
            WeaverFocus::All => "all",
        };
        write!(f, "{label}")
    }
}

impl WeaverFocus {
    pub fn directive(&self) -> &'static str {
        match self {
            WeaverFocus::Requires => {
                "Prioritize prerequisite wiring and DAG validation before adding other edges."
            }
            WeaverFocus::Supports => {
                "Prioritize supports (examples/analogies) and fadeability checks before other \
                 edges."
            }
            WeaverFocus::Assesses => {
                "Prioritize assessment alignment edges and evidence coverage before other edges."
            }
            WeaverFocus::All => {
                "Cover requires, supports, assesses, precedes, and anchors; close gaps \
                 methodically."
            }
        }
    }
}

mod prompts;

/// Message used by the LLM gateway to execute a tool invocation within the
/// file reader context.
pub struct ExecuteTool {
    pub identifier: String,
    pub arguments:  Value,
}

pub fn make_conversation_id(actor_name: &str) -> String {
    format!("{actor_name}#{}", Uuid::new_v4())
}

/// Actor that exposes local filesystem utilities to LLM collaborators.
#[derive(Actor)]
pub struct Reader<M: ModeSpec> {
    gateway:            ActorRef<LLMGateway>,
    model:              Arc<String>,
    root:               Arc<PathBuf>,
    course_commit:      Arc<String>,
    metrics:            Arc<GatewayMetrics>,
    graph:              ActorRef<crate::graph::manager::GraphManager>,
    dedup:              ActorRef<DeduplicationAgent>,
    analysis_cache:     Arc<AnalysisCache>,
    rerun:              Option<ActorRef<crate::rerun_sink::RerunSink>>,
    depth:              usize,
    max_subdelegations: usize,
    actor_name:         Arc<String>,
    conversation_id:    Arc<String>,
    cancellation_token: CancellationToken,
    cancel_reason:      Option<String>,
    _mode:              PhantomData<M>,
}

#[derive(Clone)]
struct ReaderDeps {
    gateway:        ActorRef<LLMGateway>,
    model:          Arc<String>,
    root:           Arc<PathBuf>,
    course_commit:  Arc<String>,
    metrics:        Arc<GatewayMetrics>,
    graph:          ActorRef<crate::graph::manager::GraphManager>,
    dedup:          ActorRef<DeduplicationAgent>,
    analysis_cache: Arc<AnalysisCache>,
    rerun:          Option<ActorRef<crate::rerun_sink::RerunSink>>,
    cancellation:   CancellationToken,
}

impl<M: ModeSpec> Reader<M> {
    /// Build a new reader using `OPENAI_MODEL`/`OPENAI_API_BASE`.
    pub fn from_env(
        root: impl AsRef<Path>,
        gateway: ActorRef<LLMGateway>,
        metrics: Arc<GatewayMetrics>,
        graph: ActorRef<crate::graph::manager::GraphManager>,
        dedup: ActorRef<DeduplicationAgent>,
        rerun: Option<ActorRef<crate::rerun_sink::RerunSink>>,
        course_commit: impl Into<String>,
    ) -> Result<Self> {
        Self::from_env_with_limit_and_cancellation(
            root,
            gateway,
            metrics,
            graph,
            dedup,
            rerun,
            DEFAULT_MAX_SUBDELEGATIONS,
            course_commit,
            CancellationToken::new(),
        )
    }

    /// Build a new reader with a custom delegation limit.
    #[allow(clippy::too_many_arguments)]
    pub fn from_env_with_limit(
        root: impl AsRef<Path>,
        gateway: ActorRef<LLMGateway>,
        metrics: Arc<GatewayMetrics>,
        graph: ActorRef<crate::graph::manager::GraphManager>,
        dedup: ActorRef<DeduplicationAgent>,
        rerun: Option<ActorRef<crate::rerun_sink::RerunSink>>,
        max_subdelegations: usize,
        course_commit: impl Into<String>,
    ) -> Result<Self> {
        Self::from_env_with_limit_and_cancellation(
            root,
            gateway,
            metrics,
            graph,
            dedup,
            rerun,
            max_subdelegations,
            course_commit,
            CancellationToken::new(),
        )
    }

    /// Build a new reader with explicit cancellation control.
    #[allow(clippy::too_many_arguments)]
    pub fn from_env_with_limit_and_cancellation(
        root: impl AsRef<Path>,
        gateway: ActorRef<LLMGateway>,
        metrics: Arc<GatewayMetrics>,
        graph: ActorRef<crate::graph::manager::GraphManager>,
        dedup: ActorRef<DeduplicationAgent>,
        rerun: Option<ActorRef<crate::rerun_sink::RerunSink>>,
        max_subdelegations: usize,
        course_commit: impl Into<String>,
        cancellation_token: CancellationToken,
    ) -> Result<Self> {
        let model = Arc::new(
            env::var("OPENAI_MODEL")
                .map_err(|_| anyhow!("OPENAI_MODEL environment variable must be set"))?,
        );

        let root =
            Arc::new(root.as_ref().canonicalize().with_context(|| {
                format!("failed to canonicalize root {}", root.as_ref().display())
            })?);

        let analysis_cache = Arc::new(AnalysisCache::new());
        let course_commit = Arc::new(course_commit.into());

        let deps = ReaderDeps {
            gateway,
            model,
            root,
            course_commit,
            metrics,
            graph,
            dedup,
            analysis_cache,
            rerun,
            cancellation: cancellation_token,
        };
        Ok(Self::new(deps, 0, max_subdelegations))
    }

    fn new(deps: ReaderDeps, depth: usize, max_subdelegations: usize) -> Self {
        let suffix = M::SUFFIX;
        let actor_name = if depth == 0 {
            format!("FileReader/Root{suffix}")
        } else {
            format!("FileReader/Delegate{depth}{suffix}")
        };
        let conversation_id = make_conversation_id(&actor_name);
        Self {
            gateway: deps.gateway,
            model: deps.model,
            root: deps.root,
            course_commit: deps.course_commit,
            metrics: deps.metrics,
            graph: deps.graph,
            dedup: deps.dedup,
            analysis_cache: deps.analysis_cache,
            rerun: deps.rerun,
            depth,
            max_subdelegations,
            actor_name: Arc::new(actor_name),
            conversation_id: Arc::new(conversation_id),
            cancellation_token: deps.cancellation,
            cancel_reason: None,
            _mode: PhantomData,
        }
    }

    fn system_prompt(&self) -> String {
        prompts::build_system_prompt(M::MODE, self.root.as_ref(), self.course_commit.as_ref())
    }

    pub fn workspace_root(&self) -> &Path {
        self.root.as_ref()
    }

    pub fn tool_identifiers() -> Result<Vec<&'static str>> {
        tool_identifiers_for_mode(M::MODE)
    }
}

pub type FileReader = Reader<InteractiveSpec>;
pub type InteractiveReader = Reader<InteractiveSpec>;
pub type HarvesterReader = Reader<HarvesterSpec>;
pub type WeaverReader = Reader<WeaverSpec>;

pub fn tool_identifiers_for_mode(mode: AgentMode) -> Result<Vec<&'static str>> {
    let all_tools = llm::all_tools()?;

    let tools: Vec<&'static str> = match mode {
        AgentMode::Interactive => all_tools.iter().map(|meta| meta.id).collect(),
        AgentMode::Harvester => {
            let allowed = [
                "graph_insert_knowledge",
                "graph_update_knowledge",
                "graph_insert_teaching_step",
                "graph_update_teaching_step",
                "delegate_tasks",
                "graph_list_nodes_by_tag",
                "graph_list_nodes_by_kind",
                "graph_list_tags",
                "graph_course_commit",
                "graph_get_node",
                "graph_neighbors",
                "graph_first_principles",
                "locate_snippet",
                "list_directory",
                "read_file_full",
                "read_file_range",
                "search_text",
            ];
            all_tools
                .iter()
                .filter(|meta| allowed.contains(&meta.id))
                .map(|meta| meta.id)
                .collect()
        }
        AgentMode::Weaver => {
            let allowed = [
                "graph_add_requires",
                "graph_add_supports",
                "graph_add_assesses",
                "graph_add_precedes",
                "graph_add_anchors",
                "delegate_tasks",
                "graph_get_node",
                "graph_list_nodes_by_tag",
                "graph_list_nodes_by_kind",
                "graph_list_tags",
                "graph_search_nodes",
                "graph_course_commit",
                "graph_neighbors",
                "graph_first_principles",
                "graph_dag_check",
                "graph_lo_alignment_summary",
                "graph_gap_summary",
                "locate_snippet",
                "list_directory",
                "read_file_full",
                "read_file_range",
                "search_text",
            ];
            all_tools
                .iter()
                .filter(|meta| allowed.contains(&meta.id))
                .map(|meta| meta.id)
                .collect()
        }
    };

    Ok(tools)
}

#[derive(Clone)]
pub(crate) struct DelegateBatchCtx {
    pub gateway:            ActorRef<LLMGateway>,
    pub model:              Arc<String>,
    pub workspace_root:     Arc<PathBuf>,
    pub course_commit:      Arc<String>,
    pub metrics:            Arc<GatewayMetrics>,
    pub graph:              ActorRef<crate::graph::manager::GraphManager>,
    pub dedup:              ActorRef<DeduplicationAgent>,
    pub analysis_cache:     Arc<AnalysisCache>,
    pub rerun:              Option<ActorRef<crate::rerun_sink::RerunSink>>,
    pub depth:              usize,
    pub max_subdelegations: usize,
    pub mode:               AgentMode,
    pub cancellation:       CancellationToken,
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

    if ctx.cancellation.is_cancelled() {
        return Ok(json!({
            "type": "delegation_cancelled",
            "depth": ctx.depth + 1,
            "reason": "parent cancelled",
        }));
    }

    let total = tasks.len();
    let limit = MAX_PARALLEL_DELEGATIONS.min(total);

    let cancellation = ctx.cancellation.clone();
    let results = tokio::select! {
        _ = cancellation.cancelled() => {
            return Ok(json!({
                "type": "delegation_cancelled",
                "depth": ctx.depth + 1,
                "reason": "cancelled during execution",
            }));
        }
        results = async {
            stream::iter(tasks.into_iter())
                .map(|task| {
                    let deps = ReaderDeps {
                        gateway:        ctx.gateway.clone(),
                        model:          Arc::clone(&ctx.model),
                        root:           Arc::clone(&ctx.workspace_root),
                        metrics:        Arc::clone(&ctx.metrics),
                        graph:          ctx.graph.clone(),
                        dedup:          ctx.dedup.clone(),
                        analysis_cache: Arc::clone(&ctx.analysis_cache),
                        rerun:          ctx.rerun.clone(),
                        course_commit:  Arc::clone(&ctx.course_commit),
                        cancellation:   ctx.cancellation.clone(),
                    };
                    let depth = ctx.depth;
                    let max_subdelegations = ctx.max_subdelegations;
                    async move {
                        if deps.cancellation.is_cancelled() {
                            return json!({
                                "task": task,
                                "status": "error",
                                "error": "delegated task cancelled",
                            });
                        }
                        let prompt = task.clone();
                        let result = match ctx.mode {
                            AgentMode::Harvester => {
                                let child = Reader::<HarvesterSpec>::new(
                                    deps.clone(),
                                    depth + 1,
                                    max_subdelegations,
                                );
                                Reader::<HarvesterSpec>::spawn(child)
                                    .ask(FileReaderQuery { prompt })
                                    .await
                            }
                            AgentMode::Weaver => {
                                let child = Reader::<WeaverSpec>::new(
                                    deps.clone(),
                                    depth + 1,
                                    max_subdelegations,
                                );
                                Reader::<WeaverSpec>::spawn(child)
                                    .ask(FileReaderQuery { prompt })
                                    .await
                            }
                            AgentMode::Interactive => {
                                let child = Reader::<InteractiveSpec>::new(
                                    deps.clone(),
                                    depth + 1,
                                    max_subdelegations,
                                );
                                Reader::<InteractiveSpec>::spawn(child)
                                    .ask(FileReaderQuery { prompt })
                                    .await
                            }
                        };
                        match result {
                            Ok(content) => {
                                json!({
                                    "task": task,
                                    "status": "ok",
                                    "content": content,
                                })
                            }
                            Err(err) => {
                                let error = err.to_string();
                                json!({
                                    "task": task,
                                    "status": "error",
                                    "error": error,
                                })
                            }
                        }
                    }
                })
                .buffered(limit)
                .collect::<Vec<Value>>()
                .await
        } => results,
    };

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

pub struct CancelWork {
    pub reason: String,
}

impl<M: ModeSpec> Message<FileReaderQuery> for Reader<M> {
    type Reply = DelegatedReply<Result<String>>;

    async fn handle(
        &mut self,
        FileReaderQuery { prompt }: FileReaderQuery,
        ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        if self.cancellation_token.is_cancelled() {
            return ctx.spawn(async { Err(anyhow!("FileReader cancelled")) });
        }
        let tool_host = M::wrap_tool_host(ctx.actor_ref().clone());
        let gateway = self.gateway.clone();
        let system_prompt = self.system_prompt();
        let model = self.model.clone();
        let actor_name = (*self.actor_name).clone();
        let conversation_id = (*self.conversation_id).clone();
        let rerun = self.rerun.clone();
        let cancellation = self.cancellation_token.clone();
        let cancel_reason = self.cancel_reason.clone();

        ctx.spawn(async move {
            let cancel_msg = cancel_reason.unwrap_or_else(|| "FileReader cancelled".to_string());
            if cancellation.is_cancelled() {
                return Err(anyhow!(cancel_msg.clone()));
            }
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
                tool_ids: Self::tool_identifiers()?,
                max_iterations: MAX_TOOL_ITERATIONS,
                tool_host,
                actor_name,
                conversation_id,
                rerun,
                cancellation_token: cancellation.clone(),
            };

            let reply = tokio::select! {
                _ = cancellation.cancelled() => {
                    Err(anyhow!(cancel_msg.clone()))
                }
                response = gateway.ask(request) => response.map_err(|err| anyhow!(err.to_string())),
            }?;
            Ok(reply)
        })
    }
}

impl<M: ModeSpec> Message<CancelWork> for Reader<M> {
    type Reply = Result<()>;

    async fn handle(
        &mut self,
        CancelWork { reason }: CancelWork,
        ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        self.cancel_reason = Some(reason.clone());
        self.cancellation_token.cancel();
        ctx.stop();
        Ok(())
    }
}

impl<M: ModeSpec> Message<ExecuteTool> for Reader<M> {
    type Reply = Result<ToolOutput, ToolExecutionError>;

    async fn handle(
        &mut self,
        ExecuteTool {
            identifier,
            arguments,
        }: ExecuteTool,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        if self.cancellation_token.is_cancelled() {
            let reason = self
                .cancel_reason
                .clone()
                .unwrap_or_else(|| "FileReader cancelled".to_string());
            return Err(llm::ToolExecutionError::Internal(anyhow!(reason)));
        }
        let meta = match llm::lookup_tool(&identifier)
            .map_err(|err| ToolExecutionError::system(err.into()))?
        {
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
            dedup:              self.dedup.clone(),
            actor_name:         Arc::clone(&self.actor_name),
            conversation_id:    Arc::clone(&self.conversation_id),
            rerun:              self.rerun.clone(),
            analysis_cache:     Arc::clone(&self.analysis_cache),
            mode:               M::MODE,
            course_commit:      Arc::clone(&self.course_commit),
            cancellation:       self.cancellation_token.clone(),
        };

        let tool = (meta.parse)(arguments, &state).map_err(ToolExecutionError::from)?;

        tool.execute().await
    }
}
