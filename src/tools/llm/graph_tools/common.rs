use anyhow::anyhow;
use async_trait::async_trait;
use kameo::{error::SendError, prelude::ActorRef};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::{
    graph::{CurriculumGraph, GraphError, NodeId, NodeKind},
    schema::types::KnowledgeType,
    tools::llm::{
        CallState, ToolExecutionError, ToolInputError, ToolInputResult, ToolInstance, ToolOutput,
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
        GraphError::InvariantViolation { violations } => {
            ToolExecutionError::Input(ToolInputError::InvalidPayload {
                tool,
                message: format!("graph invariants violated: {}", violations.join("; ")),
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

pub(crate) fn map_send_err_inf<M>(
    err: SendError<M, std::convert::Infallible>,
) -> ToolExecutionError {
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
    build:   fn(&Args) -> Msg,
    map_ok:  fn(&Args, <MsgReply<Msg> as kameo::Reply>::Ok) -> Value,
    map_err: fn(SendError<Msg, <MsgReply<Msg> as kameo::Reply>::Error>) -> ToolExecutionError,
}

impl<Args, Msg> GraphCommandTool<Args, Msg>
where
    Msg: Send + 'static,
    crate::graph::manager::GraphManager: kameo::message::Message<Msg>,
{
    pub(crate) fn new(
        args: Args,
        graph: ActorRef<crate::graph::manager::GraphManager>,
        build: fn(&Args) -> Msg,
        map_ok: fn(&Args, <MsgReply<Msg> as kameo::Reply>::Ok) -> Value,
        map_err: fn(SendError<Msg, <MsgReply<Msg> as kameo::Reply>::Error>) -> ToolExecutionError,
    ) -> Self {
        Self {
            args,
            graph,
            build,
            map_ok,
            map_err,
        }
    }
}

#[async_trait]
impl<Args, Msg> ToolInstance for GraphCommandTool<Args, Msg>
where
    Args: Send + Sync + 'static,
    Msg: Send + 'static,
    crate::graph::manager::GraphManager: kameo::message::Message<Msg>,
{
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let msg = (self.build)(&self.args);
        let reply: <MsgReply<Msg> as kameo::Reply>::Ok =
            self.graph.ask(msg).await.map_err(|e| (self.map_err)(e))?;

        Ok(ToolOutput::new((self.map_ok)(&self.args, reply)))
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
    Args: for<'de> Deserialize<'de> + JsonSchema + Clone + Send + Sync + 'static,
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
        build,
        map_ok,
        map_err,
    )))
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
