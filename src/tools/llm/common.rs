use std::{future::Future, marker::PhantomData, pin::Pin, sync::Arc};

use async_trait::async_trait;
use kameo::{error::SendError, message::Message, prelude::ActorRef};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use crate::{
    graph::manager::{GraphManager, GraphMeta},
    schema::types::KnowledgeType,
    tools::llm::{
        CallState, ToolExecutionError, ToolExecutionResult, ToolInputError, ToolInputResult,
        ToolInstance, ToolOutput, ToolPayloadMode, apply_preview_cost, build_cost_preview,
        estimate_tokens_from_characters, payload_size_bytes, prepare_payload_estimates,
    },
};

/// Minimal pagination metadata used to attach hints and normalize outputs.
#[derive(Debug, Clone, Copy)]
pub struct Page {
    pub offset:   usize,
    pub limit:    usize,
    pub has_more: bool,
}

/// Normalized payload returned by tool compute closures.
#[derive(Debug)]
pub struct ToolRunPayload {
    pub body:          Value,
    pub approx_bytes:  Option<u64>,
    pub preview:       Option<Value>,
    pub preview_hints: Vec<String>,
    pub page:          Option<Page>,
}

impl ToolRunPayload {
    pub fn new(body: Value) -> Self {
        Self {
            body,
            approx_bytes: None,
            preview: None,
            preview_hints: Vec::new(),
            page: None,
        }
    }
}

/// Attach graph meta to a payload object if possible.
pub fn attach_meta(mut payload: Value, meta: &GraphMeta) -> Value {
    if let Value::Object(ref mut obj) = payload {
        obj.insert("meta".to_string(), serde_json::to_value(meta).unwrap_or(Value::Null));
    }
    payload
}

/// Fetch graph metadata for a tool call.
pub async fn graph_meta(graph: &ActorRef<GraphManager>) -> Result<GraphMeta, ToolExecutionError> {
    graph
        .ask(crate::graph::manager::GetGraphMeta)
        .await
        .map_err(|err| ToolExecutionError::Internal(anyhow::anyhow!("{err:?}")))
}

/// Generic tool runner that centralizes preview/body rendering, cost
/// estimation, meta attachment, and pagination hints.
pub struct ToolRunner<'a> {
    tool:          &'static str,
    state:         &'a CallState,
    mode:          ToolPayloadMode,
    meta:          Option<GraphMeta>,
    preview_hints: Vec<String>,
}

impl<'a> ToolRunner<'a> {
    pub fn new(tool: &'static str, state: &'a CallState) -> Self {
        Self {
            tool,
            state,
            mode: ToolPayloadMode::Preview,
            meta: None,
            preview_hints: Vec::new(),
        }
    }

    pub fn with_mode(mut self, mode: ToolPayloadMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_meta(mut self, meta: GraphMeta) -> Self {
        self.meta = Some(meta);
        self
    }

    pub fn hint(mut self, hint: impl Into<String>) -> Self {
        self.preview_hints.push(hint.into());
        self
    }

    pub fn hints(mut self, hints: impl Into<Vec<String>>) -> Self {
        self.preview_hints.extend(hints.into());
        self
    }

    pub async fn run<F, Fut>(self, compute: F) -> ToolExecutionResult
    where
        F: FnOnce(ToolPayloadMode) -> Fut,
        Fut: Future<Output = Result<ToolRunPayload, ToolExecutionError>> + Send + 'static,
    {
        let mut payload = compute(self.mode).await?;

        let approx_bytes =
            payload
                .approx_bytes
                .unwrap_or_else(|| match (self.mode, payload.preview.as_ref()) {
                    (ToolPayloadMode::Preview, Some(preview)) => payload_size_bytes(preview),
                    _ => payload_size_bytes(&payload.body),
                });

        let mut hints = self.preview_hints;
        hints.append(&mut payload.preview_hints);
        if let Some(page) = &payload.page
            && page.has_more
        {
            hints.push("More results available; increase limit or adjust offset.".to_string());
        }

        let estimates =
            prepare_payload_estimates(&self.state.metrics, self.state.model.as_str(), approx_bytes);

        match self.mode {
            ToolPayloadMode::Preview => {
                let mut preview = payload.preview.unwrap_or_else(|| {
                    build_cost_preview(
                        self.tool,
                        approx_bytes,
                        estimates.safe_tokens,
                        hints.clone(),
                    )
                });
                ensure_cost_block(&mut preview, approx_bytes, estimates.safe_tokens, hints);
                let preview_tokens =
                    estimate_tokens_from_characters(payload_size_bytes(&preview) as usize);
                apply_preview_cost(
                    &mut preview,
                    &self.state.metrics,
                    self.state.model.as_str(),
                    self.state.conversation_id.as_str(),
                    preview_tokens,
                    estimates.safe_tokens,
                );
                let with_meta = if let Some(meta) = &self.meta {
                    attach_meta(preview, meta)
                } else {
                    preview
                };
                let hint = payload_size_bytes(&with_meta);
                Ok(ToolOutput::with_byte_hint(with_meta, hint))
            }
            ToolPayloadMode::Body => {
                let mut body = payload.body;
                if let Some(page) = payload.page
                    && let Value::Object(ref mut obj) = body
                {
                    obj.entry("offset".to_string())
                        .or_insert(json!(page.offset));
                    obj.entry("limit".to_string()).or_insert(json!(page.limit));
                    obj.entry("has_more".to_string())
                        .or_insert(json!(page.has_more));
                }
                let with_meta = if let Some(meta) = &self.meta {
                    attach_meta(body, meta)
                } else {
                    body
                };
                let size = payload_size_bytes(&with_meta);
                Ok(ToolOutput::with_byte_hint(with_meta, size))
            }
        }
    }
}

fn ensure_cost_block(
    payload: &mut Value,
    bytes_total: u64,
    approx_tokens: Option<u64>,
    hints: Vec<String>,
) {
    match payload {
        Value::Object(map) => {
            map.entry("cost".to_string()).or_insert_with(|| {
                json!({
                    "bytes_total": bytes_total,
                    "approx_tokens": approx_tokens,
                    "preview_tokens": Value::Null,
                    "remaining_tokens": Value::Null,
                    "remaining_ratio": Value::Null
                })
            });
            map.entry("hints".to_string())
                .or_insert_with(|| json!(hints));
        }
        _ => {
            let data = std::mem::take(payload);
            *payload = json!({
                "type": "preview",
                "tool": "<unknown>",
                "data": data,
                "cost": {
                    "bytes_total": bytes_total,
                    "approx_tokens": approx_tokens,
                    "preview_tokens": Value::Null,
                    "remaining_tokens": Value::Null,
                    "remaining_ratio": Value::Null
                },
                "hints": hints,
            });
        }
    }
}

/// Typed slug helper that encodes expected knowledge type at compile time.
#[derive(Debug)]
pub struct Slug<T> {
    raw:      String,
    _phantom: PhantomData<T>,
}

impl<T> Slug<T> {
    pub fn new<S: Into<String>>(raw: S) -> Self {
        Self {
            raw:      raw.into(),
            _phantom: PhantomData,
        }
    }

    pub fn as_str(&self) -> &str {
        &self.raw
    }

    pub fn into_string(self) -> String {
        self.raw
    }
}

impl<T> Clone for Slug<T> {
    fn clone(&self) -> Self {
        Self {
            raw:      self.raw.clone(),
            _phantom: PhantomData,
        }
    }
}

pub trait SlugKind {
    fn validate(
        graph: &crate::graph::CurriculumGraph,
        id: crate::graph::NodeId,
        slug: &str,
        tool: &'static str,
    ) -> Result<(), ToolExecutionError>;
}

pub struct LearningOutcome;
pub struct AssessmentItem;
pub struct AnyKnowledge;
pub struct TeachingStep;

impl SlugKind for LearningOutcome {
    fn validate(
        graph: &crate::graph::CurriculumGraph,
        id: crate::graph::NodeId,
        slug: &str,
        tool: &'static str,
    ) -> Result<(), ToolExecutionError> {
        super::graph_tools::common::ensure_knowledge_type(
            graph,
            id,
            slug,
            KnowledgeType::LearningOutcome,
            tool,
        )
    }
}

impl SlugKind for AssessmentItem {
    fn validate(
        graph: &crate::graph::CurriculumGraph,
        id: crate::graph::NodeId,
        slug: &str,
        tool: &'static str,
    ) -> Result<(), ToolExecutionError> {
        super::graph_tools::common::ensure_knowledge_type(
            graph,
            id,
            slug,
            KnowledgeType::AssessmentItem,
            tool,
        )
    }
}

impl SlugKind for AnyKnowledge {
    fn validate(
        graph: &crate::graph::CurriculumGraph,
        id: crate::graph::NodeId,
        slug: &str,
        tool: &'static str,
    ) -> Result<(), ToolExecutionError> {
        super::graph_tools::common::ensure_any_knowledge(graph, id, slug, tool)
    }
}

impl SlugKind for TeachingStep {
    fn validate(
        graph: &crate::graph::CurriculumGraph,
        id: crate::graph::NodeId,
        slug: &str,
        tool: &'static str,
    ) -> Result<(), ToolExecutionError> {
        super::graph_tools::common::ensure_teaching_step(graph, id, slug, tool)
    }
}

pub async fn resolve_typed<K: SlugKind>(
    graph: &ActorRef<GraphManager>,
    slug: Slug<K>,
    tool: &'static str,
) -> Result<crate::graph::NodeId, ToolExecutionError> {
    let id =
        super::graph_tools::common::resolve_slug(graph, slug.clone().into_string(), tool).await?;
    let graph_snapshot: Arc<crate::graph::CurriculumGraph> = graph
        .ask(crate::graph::commands::GetGraph)
        .await
        .map_err(super::graph_tools::common::map_send_err_inf)?;
    K::validate(&graph_snapshot, id, slug.as_str(), tool)?;
    Ok(id)
}

/// Reusable graph action trait to standardize mutation previews and responses.
#[async_trait]
pub trait GraphAction: Send + Sync {
    type Msg: Send + 'static;
    type Reply;
    type Err;

    fn tool(&self) -> &'static str;
    fn apply(&self) -> bool;
    fn build_message(&self) -> Self::Msg;
    fn map_ok(&self, reply: Self::Reply) -> Value;
    fn map_err(&self, err: SendError<Self::Msg, Self::Err>) -> ToolExecutionError;

    fn preview_payload(&self) -> Value {
        json!({
            "type": "graph_command",
            "tool": self.tool(),
            "status": "preview",
            "apply": false,
            "hint": "Set apply=true to execute this mutation"
        })
    }
}

pub async fn run_graph_action<A, Msg>(action: &A, state: &CallState) -> ToolExecutionResult
where
    A: GraphAction<
            Msg = Msg,
            Reply = <MessageReply<Msg> as kameo::Reply>::Ok,
            Err = <MessageReply<Msg> as kameo::Reply>::Error,
        > + ?Sized,
    Msg: Send + 'static,
    GraphManager: Message<Msg>,
    <GraphManager as Message<Msg>>::Reply: Send,
    <<GraphManager as Message<Msg>>::Reply as kameo::Reply>::Ok: Send,
    <<GraphManager as Message<Msg>>::Reply as kameo::Reply>::Error: Send,
{
    let meta = graph_meta(&state.graph).await?;
    let mode = if action.apply() {
        ToolPayloadMode::Body
    } else {
        ToolPayloadMode::Preview
    };

    match mode {
        ToolPayloadMode::Preview => {
            let preview = action.preview_payload();
            ToolRunner::new(action.tool(), state)
                .with_mode(mode)
                .with_meta(meta)
                .hint("Set apply=true to execute this mutation.")
                .run(move |_| async move { Ok(ToolRunPayload::new(preview)) })
                .await
        }
        ToolPayloadMode::Body => {
            let msg = action.build_message();
            let reply: <MessageReply<Msg> as kameo::Reply>::Ok =
                state.graph.ask(msg).await.map_err(|e| action.map_err(e))?;
            let body = action.map_ok(reply);
            ToolRunner::new(action.tool(), state)
                .with_mode(mode)
                .with_meta(meta)
                .hint("Set apply=true to execute this mutation.")
                .run(move |_| async move { Ok(ToolRunPayload::new(body)) })
                .await
        }
    }
}

type MessageReply<Msg> = <GraphManager as Message<Msg>>::Reply;

pub type GraphActionPreflight<A> = dyn Fn(&A, &CallState) -> Pin<Box<dyn Future<Output = Result<(), ToolExecutionError>> + Send>>
    + Send
    + Sync;

pub type ArgsPreflight<Args> = Arc<
    dyn Fn(
            &Args,
            &CallState,
        ) -> Pin<Box<dyn Future<Output = Result<(), ToolExecutionError>> + Send>>
        + Send
        + Sync,
>;

/// ToolInstance wrapper over a GraphAction.
pub struct GraphActionInstance<A>
where
    A: GraphAction + 'static,
{
    action:    Arc<A>,
    state:     CallState,
    preflight: Option<Arc<GraphActionPreflight<A>>>,
}

impl<A> GraphActionInstance<A>
where
    A: GraphAction + 'static,
{
    pub fn new(
        action: A,
        state: CallState,
        preflight: Option<Arc<GraphActionPreflight<A>>>,
    ) -> Self {
        Self {
            action: Arc::new(action),
            state,
            preflight,
        }
    }
}

#[async_trait]
impl<A, Msg> ToolInstance for GraphActionInstance<A>
where
    A: GraphAction<
            Msg = Msg,
            Reply = <MessageReply<Msg> as kameo::Reply>::Ok,
            Err = <MessageReply<Msg> as kameo::Reply>::Error,
        > + Send
        + Sync
        + 'static,
    Msg: Send + 'static,
    GraphManager: Message<Msg>,
    MessageReply<Msg>: Send,
    <MessageReply<Msg> as kameo::Reply>::Ok: Send,
    <MessageReply<Msg> as kameo::Reply>::Error: Send,
{
    async fn execute(&self) -> ToolExecutionResult {
        if let Some(preflight) = &self.preflight
            && self.action.apply()
        {
            preflight(self.action.as_ref(), &self.state).await?;
        }
        run_graph_action(self.action.as_ref(), &self.state).await
    }
}

/// Parse raw JSON args into a strongly typed args struct.
pub fn parse_args<T>(tool: &'static str, raw: Value) -> ToolInputResult<T>
where
    T: DeserializeOwned,
{
    serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
        tool,
        message: err.to_string(),
    })
}

/// Generic pagination macro to clamp limit/offset and return a page slice with
/// metadata.
#[macro_export]
macro_rules! paginate {
    ($items:expr, $limit:expr, $offset:expr) => {{
        let mut items = $items;
        let len = items.len();
        let limit = $limit.unwrap_or(50).clamp(1, 200);
        let offset = $offset.unwrap_or(0).min(len);
        let end = (offset + limit).min(len);
        let has_more = end < len;
        let page = items.drain(offset..end).collect::<Vec<_>>();
        (
            page,
            $crate::tools::llm::common::Page {
                offset,
                limit,
                has_more,
            },
        )
    }};
}

/// Macro to build graph tools from GraphAction implementations.
#[macro_export]
macro_rules! graph_action_tool {
    (
        $meta_fn:ident,
        id: $id_const:expr,
        description: $description:expr,
        args: $args_ty:ty,
        prepare: $prep:expr,
        build: $build:expr,
        ok: $ok:expr,
        map_err: $map_err:expr
    ) => {
        $crate::graph_action_tool!(
            $meta_fn,
            id: $id_const,
            description: $description,
            args: $args_ty,
            prepare: $prep,
            build: $build,
            ok: $ok,
            map_err: $map_err,
            preflight: None,
            mutate: None
        );
    };
    (
        $meta_fn:ident,
        id: $id_const:expr,
        description: $description:expr,
        args: $args_ty:ty,
        prepare: $prep:expr,
        build: $build:expr,
        ok: $ok:expr,
        map_err: $map_err:expr,
        preflight: $preflight:expr,
        mutate: $mutate:expr
    ) => {
        pub(super) fn $meta_fn() -> $crate::tools::llm::ToolPrototype {
            $crate::tools::llm::ToolPrototype {
                id:          $id_const,
                description: $description,
                schema:      $crate::tools::llm::schema_for_args::<$args_ty>(),
                parse:       |raw, state| {
                    let args: $args_ty = ($prep)(raw)?;
                    let raw_args =
                        serde_json::to_value(&args).expect("failed to serialize graph action args");
                    $crate::tools::llm::graph_tools::common::parse_graph_command::<$args_ty, _>(
                        $id_const, raw_args, state, $build, $ok, $map_err, $preflight, $mutate,
                    )
                },
            }
        }
    };
}

/// Macro to declare tool ID constants and keep CURATED_TOOL_IDS in sync.
#[macro_export]
macro_rules! tool_ids {
    ($vis:vis $name:ident { $($const_name:ident = $value:expr;)+ }) => {
        $(pub const $const_name: &str = $value;)+
        $vis const $name: &[&str] = &[
            $($const_name,)+
        ];
        const _: () = {
            $crate::tools::llm::assert_unique_tool_ids($name);
        };
    };
}

/// Skeleton macro for analysis tools; expands to a ToolPrototype with a
/// ToolRunner-backed instance.
#[macro_export]
macro_rules! analysis_tool {
    (
        $meta_fn:ident,
        id: $id_const:expr,
        description: $description:expr,
        args: $args_ty:ty,
        prepare: $prep:expr,
        runner: $runner:expr
    ) => {
        pub(super) fn $meta_fn() -> $crate::tools::llm::ToolPrototype {
            $crate::tools::llm::ToolPrototype {
                id:          $id_const,
                description: $description,
                schema:      $crate::tools::llm::schema_for_args::<$args_ty>(),
                parse:       |raw, state| {
                    let args: $args_ty = ($prep)(raw)?;
                    Ok(Box::new(($runner)(args, state)))
                },
            }
        }
    };
}

/// Macro for non-graph utilities that only need args parsing and a runner. The
/// `prepare` closure receives `(raw, state)` so callers can enforce depth or
/// workspace limits during parsing.
#[macro_export]
macro_rules! basic_tool {
    (
        $meta_fn:ident,
        id: $id_const:expr,
        description: $description:expr,
        args: $args_ty:ty,
        prepare: $prep:expr,
        runner: $runner:expr
    ) => {
        pub(super) fn $meta_fn() -> $crate::tools::llm::ToolPrototype {
            $crate::tools::llm::ToolPrototype {
                id:          $id_const,
                description: $description,
                schema:      $crate::tools::llm::schema_for_args::<$args_ty>(),
                parse:       |raw, state| {
                    let args: $args_ty = ($prep)(raw, state)?;
                    Ok(Box::new(($runner)(args, state)))
                },
            }
        }
    };
}
