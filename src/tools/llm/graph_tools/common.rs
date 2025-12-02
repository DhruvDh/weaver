use std::sync::Arc;

use anyhow::anyhow;
use async_trait::async_trait;
use kameo::{
    error::{Infallible, SendError},
    prelude::ActorRef,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::{
    graph::{CurriculumGraph, GraphError, NodeId, NodeKind},
    schema::types::KnowledgeType,
    tools::llm::{CallState, ToolExecutionError, ToolInputError, ToolInputResult, ToolInstance},
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
        GraphError::AmbiguousSlug { slug, matches } => {
            let options = matches.join(", ");
            ToolExecutionError::Input(ToolInputError::InvalidPayload {
                tool,
                message: format!(
                    "slug `{}` is ambiguous; specify full slug. Candidates: {}",
                    slug, options
                ),
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
        GraphError::Operational(crate::graph::GraphOperationalError::InvariantTimeout {
            timeout_ms,
        }) => ToolExecutionError::Internal(anyhow!(
            "graph invariant validation timed out after {timeout_ms} ms"
        )),
        GraphError::Operational(crate::graph::GraphOperationalError::InvariantTaskFailed {
            message,
        }) => ToolExecutionError::Internal(anyhow!(message)),
        GraphError::Operational(crate::graph::GraphOperationalError::Poisoned) => {
            ToolExecutionError::Internal(anyhow!(
                "graph service is poisoned after a rollback failure; restart and retry"
            ))
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

/// Ensure a slug resolves to some Knowledge node (any knowledge_type).
pub(crate) fn ensure_any_knowledge(
    graph: &CurriculumGraph,
    id: NodeId,
    slug: &str,
    tool: &'static str,
) -> Result<(), ToolExecutionError> {
    match &graph[id].kind {
        NodeKind::Knowledge(_) => Ok(()),
        _ => Err(ToolExecutionError::Input(ToolInputError::InvalidPayload {
            tool,
            message: format!("slug `{}` is not a knowledge node (required by this tool)", slug),
        })),
    }
}

/// Ensure a slug resolves to a TeachingStep node.
pub(crate) fn ensure_teaching_step(
    graph: &CurriculumGraph,
    id: NodeId,
    slug: &str,
    tool: &'static str,
) -> Result<(), ToolExecutionError> {
    match &graph[id].kind {
        NodeKind::TeachingStep(_) => Ok(()),
        _ => Err(ToolExecutionError::Input(ToolInputError::InvalidPayload {
            tool,
            message: format!("slug `{}` is not a teaching_step node (required by this tool)", slug),
        })),
    }
}

/// Generic adapter from args/build/map closures into the GraphAction trait so
/// existing command parsers can stay simple.
struct GraphActionAdapter<Args, Msg>
where
    Msg: Send + 'static,
    crate::graph::manager::GraphManager: kameo::message::Message<Msg>,
{
    args:    Args,
    build:   fn(&Args) -> Msg,
    map_ok:  fn(&Args, <MsgReply<Msg> as kameo::Reply>::Ok) -> Value,
    map_err: fn(SendError<Msg, <MsgReply<Msg> as kameo::Reply>::Error>) -> ToolExecutionError,
    tool:    &'static str,
}

type MsgReply<Msg> = <crate::graph::manager::GraphManager as kameo::message::Message<Msg>>::Reply;

#[async_trait]
impl<Args, Msg> crate::tools::llm::common::GraphAction for GraphActionAdapter<Args, Msg>
where
    Args: MaybeApply + Send + Sync + 'static,
    Msg: Send + 'static,
    crate::graph::manager::GraphManager: kameo::message::Message<Msg>,
{
    type Msg = Msg;
    type Reply = <MsgReply<Msg> as kameo::Reply>::Ok;
    type Err = <MsgReply<Msg> as kameo::Reply>::Error;

    fn tool(&self) -> &'static str {
        self.tool
    }

    fn apply(&self) -> bool {
        self.args.apply_flag()
    }

    fn build_message(&self) -> Self::Msg {
        (self.build)(&self.args)
    }

    fn map_ok(&self, reply: Self::Reply) -> Value {
        (self.map_ok)(&self.args, reply)
    }

    fn map_err(&self, err: SendError<Self::Msg, Self::Err>) -> ToolExecutionError {
        (self.map_err)(err)
    }
}

pub(crate) fn parse_graph_command<Args, Msg>(
    tool: &'static str,
    raw: Value,
    state: &CallState,
    build: fn(&Args) -> Msg,
    map_ok: fn(&Args, <MsgReply<Msg> as kameo::Reply>::Ok) -> Value,
    map_err: fn(SendError<Msg, <MsgReply<Msg> as kameo::Reply>::Error>) -> ToolExecutionError,
    preflight: Option<crate::tools::llm::common::ArgsPreflight<Args>>,
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

    let action = GraphActionAdapter {
        args,
        build,
        map_ok,
        map_err,
        tool,
    };

    let preflight = preflight.map(|pf| {
        Arc::new(move |action: &GraphActionAdapter<Args, Msg>, state: &CallState| {
            pf(&action.args, state)
        }) as Arc<crate::tools::llm::common::GraphActionPreflight<_>>
    });

    Ok(Box::new(crate::tools::llm::common::GraphActionInstance::new(
        action,
        state.clone(),
        preflight,
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

/// Resolve a slug to a NodeId through GraphManager to keep slug lookups
/// centralized.
pub(crate) async fn resolve_slug(
    graph: &ActorRef<crate::graph::manager::GraphManager>,
    slug: String,
    tool: &'static str,
) -> Result<crate::graph::NodeId, ToolExecutionError> {
    graph
        .ask(crate::graph::commands::ResolveSlug { slug })
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
        .ask(crate::graph::commands::ResolveSlugs { slugs })
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
