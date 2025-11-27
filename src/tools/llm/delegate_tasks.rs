use std::{path::PathBuf, sync::Arc};

use async_trait::async_trait;
use bon::Builder;
use kameo::prelude::ActorRef;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;
use tracing::info;

use super::{
    CallState, ToolExecutionError, ToolInputError, ToolInputResult, ToolInstance, ToolOutput,
    ToolPrototype, depth_exceeded, schema_for_args, trim_optional,
};
use crate::{
    constants::MAX_PARALLEL_DELEGATIONS,
    file_reader::run_delegate_batch_with_state,
    llm_gateway::{GatewayMetrics, LLMGateway},
};

const IDENTIFIER: &str = "delegate_tasks";
const DESCRIPTION: &str = "Delegate one or more tasks to child FileReader agents via the `tasks` \
                           array. Wrap single tasks in an array when needed.";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DelegateTasksArgs {
    #[schemars(
        length(min = 1),
        description = "Precise, educational description with motivation and acceptance criteria \
                       of tasks to delegate in parallel."
    )]
    #[builder(with = |raw_tasks: Vec<String>| -> ToolInputResult<_> {
        let tasks: Vec<_> = raw_tasks
            .into_iter()
            .filter_map(|entry| trim_optional(Some(entry)))
            .collect();
        if tasks.is_empty() {
            Err(ToolInputError::EmptyCollection {
                tool: IDENTIFIER,
                item: "task",
            })
        } else {
            Ok(tasks)
        }
    })]
    pub tasks: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DelegateTasksPayload {
    tasks: Vec<String>,
}

pub(super) fn delegate_tasks_meta() -> ToolPrototype {
    ToolPrototype {
        id:          IDENTIFIER,
        description: DESCRIPTION,
        schema:      schema_for_args::<DelegateTasksArgs>(),
        parse:       parse_delegate_tasks,
    }
}

fn parse_delegate_tasks(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn ToolInstance>> {
    if state.depth >= state.max_subdelegations {
        return Err(depth_exceeded(IDENTIFIER, state.depth, state.max_subdelegations));
    }

    let payload: DelegateTasksPayload =
        serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
            tool:    IDENTIFIER,
            message: err.to_string(),
        })?;

    let args = DelegateTasksArgs::builder().tasks(payload.tasks)?.build();

    Ok(Box::new(DelegateTasksTool {
        args,
        depth: state.depth,
        max_subdelegations: state.max_subdelegations,
        workspace_root: Arc::clone(&state.workspace_root),
        gateway: state.gateway.clone(),
        model: Arc::clone(&state.model),
        metrics: Arc::clone(&state.metrics),
        graph: state.graph.clone(),
        rerun: state.rerun.clone(),
    }))
}

struct DelegateTasksTool {
    args:               DelegateTasksArgs,
    depth:              usize,
    max_subdelegations: usize,
    workspace_root:     Arc<PathBuf>,
    gateway:            ActorRef<LLMGateway>,
    model:              Arc<String>,
    metrics:            Arc<GatewayMetrics>,
    graph:              ActorRef<crate::graph::manager::GraphManager>,
    rerun:              Option<ActorRef<crate::rerun_sink::RerunSink>>,
}

#[async_trait]
impl ToolInstance for DelegateTasksTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let next_depth = self.depth + 1;
        info!(
            "tool_call delegate_tasks depth={} tasks={} max_concurrency={}",
            next_depth,
            self.args.tasks.len(),
            MAX_PARALLEL_DELEGATIONS
        );

        let ctx = crate::file_reader::DelegateBatchCtx {
            gateway:            self.gateway.clone(),
            model:              Arc::clone(&self.model),
            workspace_root:     Arc::clone(&self.workspace_root),
            metrics:            Arc::clone(&self.metrics),
            graph:              self.graph.clone(),
            rerun:              self.rerun.clone(),
            depth:              self.depth,
            max_subdelegations: self.max_subdelegations,
        };

        let result = run_delegate_batch_with_state(ctx, self.args.tasks.clone()).await?;

        Ok(ToolOutput::new(result))
    }
}
