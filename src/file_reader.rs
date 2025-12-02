use std::{
    env, fmt,
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
pub struct FileReader {
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
    mode:               AgentMode,
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
    mode:           AgentMode,
}

impl FileReader {
    /// Build a new [`FileReader`] using `OPENAI_MODEL`/`OPENAI_API_BASE`.
    pub fn from_env(
        root: impl AsRef<Path>,
        gateway: ActorRef<LLMGateway>,
        metrics: Arc<GatewayMetrics>,
        graph: ActorRef<crate::graph::manager::GraphManager>,
        dedup: ActorRef<DeduplicationAgent>,
        rerun: Option<ActorRef<crate::rerun_sink::RerunSink>>,
        course_commit: impl Into<String>,
    ) -> Result<Self> {
        Self::from_env_with_limit_and_mode(
            root,
            gateway,
            metrics,
            graph,
            dedup,
            rerun,
            DEFAULT_MAX_SUBDELEGATIONS,
            AgentMode::Interactive,
            course_commit,
        )
    }

    /// Build a new [`FileReader`] using `OPENAI_MODEL`/`OPENAI_API_BASE` with a
    /// specific mode.
    #[allow(clippy::too_many_arguments)]
    pub fn from_env_with_mode(
        root: impl AsRef<Path>,
        gateway: ActorRef<LLMGateway>,
        metrics: Arc<GatewayMetrics>,
        graph: ActorRef<crate::graph::manager::GraphManager>,
        dedup: ActorRef<DeduplicationAgent>,
        rerun: Option<ActorRef<crate::rerun_sink::RerunSink>>,
        mode: AgentMode,
        course_commit: impl Into<String>,
    ) -> Result<Self> {
        Self::from_env_with_limit_and_mode(
            root,
            gateway,
            metrics,
            graph,
            dedup,
            rerun,
            DEFAULT_MAX_SUBDELEGATIONS,
            mode,
            course_commit,
        )
    }

    /// Build a new [`FileReader`] with a custom delegation limit.
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
        Self::from_env_with_limit_and_mode(
            root,
            gateway,
            metrics,
            graph,
            dedup,
            rerun,
            max_subdelegations,
            AgentMode::Interactive,
            course_commit,
        )
    }

    /// Build a new [`FileReader`] with a custom delegation limit and explicit
    /// mode.
    #[allow(clippy::too_many_arguments)]
    pub fn from_env_with_limit_and_mode(
        root: impl AsRef<Path>,
        gateway: ActorRef<LLMGateway>,
        metrics: Arc<GatewayMetrics>,
        graph: ActorRef<crate::graph::manager::GraphManager>,
        dedup: ActorRef<DeduplicationAgent>,
        rerun: Option<ActorRef<crate::rerun_sink::RerunSink>>,
        max_subdelegations: usize,
        mode: AgentMode,
        course_commit: impl Into<String>,
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
            mode,
        };
        Ok(Self::new(deps, 0, max_subdelegations))
    }

    fn new(deps: ReaderDeps, depth: usize, max_subdelegations: usize) -> Self {
        let suffix = match deps.mode {
            AgentMode::Harvester => "/Harvester",
            AgentMode::Weaver => "/Weaver",
            AgentMode::Interactive => "",
        };
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
            mode: deps.mode,
        }
    }

    fn system_prompt(&self) -> String {
        prompts::build_system_prompt(self.mode, self.root.as_ref(), self.course_commit.as_ref())
    }

    pub fn workspace_root(&self) -> &Path {
        self.root.as_ref()
    }

    pub fn tool_identifiers() -> Result<Vec<&'static str>> {
        Self::tool_identifiers_for_mode(AgentMode::Interactive)
    }

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
            let deps = ReaderDeps {
                gateway:        ctx.gateway.clone(),
                model:          Arc::clone(&ctx.model),
                root:           Arc::clone(&ctx.workspace_root),
                metrics:        Arc::clone(&ctx.metrics),
                graph:          ctx.graph.clone(),
                dedup:          ctx.dedup.clone(),
                analysis_cache: Arc::clone(&ctx.analysis_cache),
                rerun:          ctx.rerun.clone(),
                mode:           ctx.mode,
                course_commit:  Arc::clone(&ctx.course_commit),
            };
            let depth = ctx.depth;
            let max_subdelegations = ctx.max_subdelegations;
            async move {
                let child = FileReader::new(deps, depth + 1, max_subdelegations);
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
        let actor_name = (*self.actor_name).clone();
        let conversation_id = (*self.conversation_id).clone();
        let rerun = self.rerun.clone();
        let mode = self.mode;

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
                tool_ids: Self::tool_identifiers_for_mode(mode)?,
                max_iterations: MAX_TOOL_ITERATIONS,
                tool_host,
                actor_name,
                conversation_id,
                rerun,
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
            mode:               self.mode,
            course_commit:      Arc::clone(&self.course_commit),
        };

        let tool = (meta.parse)(arguments, &state).map_err(ToolExecutionError::from)?;

        tool.execute().await
    }
}
