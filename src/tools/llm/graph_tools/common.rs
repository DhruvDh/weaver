use anyhow::anyhow;
use async_trait::async_trait;
use kameo::{
    error::{Infallible, SendError},
    prelude::ActorRef,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    graph::{CurriculumGraph, GraphError, NodeId, NodeKind},
    schema::types::KnowledgeType,
    tools::llm::{
        CallState, ToolExecutionError, ToolInputError, ToolInputResult, ToolInstance, ToolOutput,
        apply_preview_cost, estimate_tokens_from_characters, payload_size_bytes,
        prepare_payload_estimates,
    },
};

/// Map graph errors into tool-facing errors.
pub(crate) fn map_graph_err(err: GraphError, tool: &'static str) -> ToolExecutionError {
    match err {
        GraphError::MissingSlug(ref slug) => {
            ToolExecutionError::Input(ToolInputError::InvalidPayload {
                tool,
                message: format!("slug `{}` not found", slug),
            })
        }
        GraphError::InvalidEndpoints { .. } | GraphError::Schema(_) => {
            ToolExecutionError::Input(ToolInputError::InvalidPayload {
                tool,
                message: err.to_string(),
            })
        }
        GraphError::RequiresCycle { cycle_slugs } => {
            ToolExecutionError::Input(ToolInputError::InvalidPayload {
                tool,
                message: format!(
                    "requires edge would create a cycle along: {}",
                    cycle_slugs.join(" -> ")
                ),
            })
        }
        GraphError::InvariantTimeout { timeout_ms } => ToolExecutionError::Internal(anyhow!(
            "graph invariant validation timed out after {timeout_ms} ms"
        )),
        GraphError::InvariantTaskFailed { message } => {
            ToolExecutionError::Internal(anyhow!(message))
        }
        GraphError::InvariantViolation { violations } => {
            let joined = violations
                .iter()
                .map(|v| format!("{}: {}", v.code.as_str(), v.message))
                .collect::<Vec<_>>()
                .join("; ");
            ToolExecutionError::Input(ToolInputError::InvalidPayload {
                tool,
                message: format!("graph invariants violated: {joined}"),
            })
        }
    }
}

pub(crate) fn map_send_err<M>(
    err: SendError<M, GraphError>,
    tool: &'static str,
) -> ToolExecutionError {
    match err {
        SendError::HandlerError(e) => map_graph_err(e, tool),
        other => ToolExecutionError::Internal(anyhow!("{:?}", other)),
    }
}

pub(crate) fn map_send_err_inf<M>(err: SendError<M, Infallible>) -> ToolExecutionError {
    ToolExecutionError::Internal(anyhow!("{:?}", err))
}

pub(crate) fn map_send_err_anyhow<M>(err: SendError<M, anyhow::Error>) -> ToolExecutionError {
    match err {
        SendError::HandlerError(e) => ToolExecutionError::Internal(e),
        other => ToolExecutionError::Internal(anyhow!("{:?}", other)),
    }
}

/// Ensure a node has the expected knowledge type for tools that require it.
pub(crate) fn ensure_knowledge_type(
    graph: &CurriculumGraph,
    id: NodeId,
    slug: &str,
    expected: KnowledgeType,
    tool: &'static str,
) -> Result<(), ToolExecutionError> {
    match &graph[id].kind {
        NodeKind::Knowledge(k) if k.knowledge_type == expected => Ok(()),
        _ => Err(ToolExecutionError::Input(ToolInputError::InvalidPayload {
            tool,
            message: format!(
                "slug `{}` is not a {:?} node (required by this tool)",
                slug, expected
            ),
        })),
    }
}

/// Generic, minimal boilerplate tool wrapper for simple graph commands that are
/// just an actor message + a JSON success payload.
type MsgReply<Msg> = <crate::graph::manager::GraphManager as kameo::message::Message<Msg>>::Reply;

pub(crate) struct GraphCommandTool<Args, Msg>
where
    Msg: Send + 'static,
    crate::graph::manager::GraphManager: kameo::message::Message<Msg>,
{
    args:    Args,
    graph:   ActorRef<crate::graph::manager::GraphManager>,
    state:   CallState,
    build:   fn(&Args) -> Msg,
    map_ok:  fn(&Args, <MsgReply<Msg> as kameo::Reply>::Ok) -> Value,
    map_err: fn(SendError<Msg, <MsgReply<Msg> as kameo::Reply>::Error>) -> ToolExecutionError,
    tool:    &'static str,
}

impl<Args, Msg> GraphCommandTool<Args, Msg>
where
    Msg: Send + 'static,
    crate::graph::manager::GraphManager: kameo::message::Message<Msg>,
{
    pub(crate) fn new(
        args: Args,
        graph: ActorRef<crate::graph::manager::GraphManager>,
        state: CallState,
        build: fn(&Args) -> Msg,
        map_ok: fn(&Args, <MsgReply<Msg> as kameo::Reply>::Ok) -> Value,
        map_err: fn(SendError<Msg, <MsgReply<Msg> as kameo::Reply>::Error>) -> ToolExecutionError,
        tool: &'static str,
    ) -> Self {
        Self {
            args,
            graph,
            state,
            build,
            map_ok,
            map_err,
            tool,
        }
    }
}

#[async_trait]
impl<Args, Msg> ToolInstance for GraphCommandTool<Args, Msg>
where
    Args: Send + Sync + MaybeApply + 'static,
    Msg: Send + 'static,
    crate::graph::manager::GraphManager: kameo::message::Message<Msg>,
{
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let meta = graph_meta(&self.graph).await?;
        if !self.args.apply_flag() {
            let preview = json!({
                "type": "graph_command",
                "tool": self.tool,
                "status": "preview",
                "apply": false,
                "hint": "Set apply=true to execute this mutation"
            });
            return Ok(preview_with_cost(self.tool, preview, &meta, &self.state));
        }
        let msg = (self.build)(&self.args);
        let reply: <MsgReply<Msg> as kameo::Reply>::Ok =
            self.graph.ask(msg).await.map_err(|e| (self.map_err)(e))?;
        let payload = attach_meta((self.map_ok)(&self.args, reply), &meta);
        Ok(apply_with_byte_hint(payload))
    }
}

pub(crate) fn parse_graph_command<Args, Msg>(
    tool: &'static str,
    raw: Value,
    state: &CallState,
    build: fn(&Args) -> Msg,
    map_ok: fn(&Args, <MsgReply<Msg> as kameo::Reply>::Ok) -> Value,
    map_err: fn(SendError<Msg, <MsgReply<Msg> as kameo::Reply>::Error>) -> ToolExecutionError,
) -> ToolInputResult<Box<dyn ToolInstance>>
where
    Args: for<'de> Deserialize<'de> + JsonSchema + Clone + Send + Sync + MaybeApply + 'static,
    Msg: Send + 'static,
    crate::graph::manager::GraphManager: kameo::message::Message<Msg>,
{
    let args: Args = serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
        tool,
        message: err.to_string(),
    })?;

    Ok(Box::new(GraphCommandTool::new(
        args,
        state.graph.clone(),
        state.clone(),
        build,
        map_ok,
        map_err,
        tool,
    )))
}

/// Deserialize args, then apply a builder closure, mapping errors into
/// InvalidPayload.
pub(crate) fn parse_args_with_builder<Args, Build>(
    tool: &'static str,
    raw: Value,
    build: Build,
) -> ToolInputResult<Args>
where
    Args: for<'de> Deserialize<'de>,
    Build: FnOnce(Args) -> ToolInputResult<Args>,
{
    let input: Args =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool,
            message: err.to_string(),
        })?;
    build(input)
}

pub(crate) fn default_confidence() -> f32 {
    1.0
}

/// Paginate a list with limit/offset and return the slice plus metadata.
/// Clamps limit to [1, 200] and offset to the vector length to avoid panics.
pub(crate) fn paginate<T>(
    mut items: Vec<T>,
    limit: Option<usize>,
    offset: Option<usize>,
) -> (Vec<T>, usize, usize, bool) {
    let len = items.len();
    let limit = limit.unwrap_or(50).clamp(1, 200);
    let offset = offset.unwrap_or(0).min(len);
    let end = (offset + limit).min(len);
    let has_more = end < len;
    (items.drain(offset..end).collect(), offset, limit, has_more)
}

/// Resolve a slug to a NodeId through GraphManager to keep slug lookups
/// centralized.
pub(crate) async fn resolve_slug(
    graph: &ActorRef<crate::graph::manager::GraphManager>,
    slug: String,
    tool: &'static str,
) -> Result<crate::graph::NodeId, ToolExecutionError> {
    graph
        .ask(crate::graph::manager::ResolveSlug { slug })
        .await
        .map_err(|e| map_send_err(e, tool))
}

/// Batch slug resolution to reduce actor round-trips.
pub(crate) async fn resolve_slugs(
    graph: &ActorRef<crate::graph::manager::GraphManager>,
    slugs: Vec<String>,
    tool: &'static str,
) -> Result<Vec<crate::graph::NodeId>, ToolExecutionError> {
    graph
        .ask(crate::graph::manager::ResolveSlugs { slugs })
        .await
        .map_err(|e| map_send_err(e, tool))
}

pub(crate) trait MaybeApply {
    fn apply_flag(&self) -> bool {
        true
    }
}

pub(crate) async fn graph_meta(
    graph: &ActorRef<crate::graph::manager::GraphManager>,
) -> Result<crate::graph::manager::GraphMeta, ToolExecutionError> {
    graph
        .ask(crate::graph::manager::GetGraphMeta)
        .await
        .map_err(map_send_err_inf)
}

pub(crate) fn attach_meta(
    mut payload: serde_json::Value,
    meta: &crate::graph::manager::GraphMeta,
) -> serde_json::Value {
    if let serde_json::Value::Object(ref mut obj) = payload {
        obj.insert(
            "meta".to_string(),
            serde_json::to_value(meta).unwrap_or(serde_json::Value::Null),
        );
    }
    payload
}

fn preview_with_cost(
    _tool: &'static str,
    payload: serde_json::Value,
    meta: &crate::graph::manager::GraphMeta,
    state: &CallState,
) -> ToolOutput {
    let approx_bytes = payload_size_bytes(&payload);
    let estimates = prepare_payload_estimates(&state.metrics, state.model.as_str(), approx_bytes);
    let mut with_cost = payload;
    if let serde_json::Value::Object(ref mut obj) = with_cost {
        obj.insert(
            "cost".to_string(),
            json!({
                "bytes_total": approx_bytes,
                "approx_tokens": estimates.safe_tokens,
                "preview_tokens": Value::Null,
                "remaining_tokens": Value::Null,
                "remaining_ratio": Value::Null,
            }),
        );
    }
    let preview_tokens = estimate_tokens_from_characters(payload_size_bytes(&with_cost) as usize);
    apply_preview_cost(
        &mut with_cost,
        &state.metrics,
        state.model.as_str(),
        state.conversation_id.as_str(),
        preview_tokens,
        estimates.safe_tokens,
    );
    let with_meta = attach_meta(with_cost, meta);
    let hint = payload_size_bytes(&with_meta);
    ToolOutput::with_byte_hint(with_meta, hint)
}

fn apply_with_byte_hint(payload: serde_json::Value) -> ToolOutput {
    let size = payload_size_bytes(&payload);
    ToolOutput::with_byte_hint(payload, size)
}
