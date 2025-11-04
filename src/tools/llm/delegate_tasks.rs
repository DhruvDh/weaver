use std::{path::PathBuf, sync::Arc};

use anyhow::Result;
use async_trait::async_trait;
use bon::Builder;
use kameo::prelude::ActorRef;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;
use tracing::info;

use super::{
    CallState, Tool, ToolInputError, ToolInputResult, ToolMeta, depth_exceeded, schema_for_args,
    trim_optional,
};
use crate::{
    constants::MAX_PARALLEL_DELEGATIONS, file_reader::run_delegate_batch_with_state,
    llm_gateway::LLMGateway,
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

pub(super) fn delegate_tasks_meta() -> ToolMeta {
    ToolMeta {
        id:          IDENTIFIER,
        description: DESCRIPTION,
        schema:      schema_for_args::<DelegateTasksArgs>(),
        parse:       parse_delegate_tasks,
    }
}

fn parse_delegate_tasks(raw: Value, state: &CallState) -> ToolInputResult<Box<dyn Tool>> {
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
    }))
}

struct DelegateTasksTool {
    args:               DelegateTasksArgs,
    depth:              usize,
    max_subdelegations: usize,
    workspace_root:     Arc<PathBuf>,
    gateway:            ActorRef<LLMGateway>,
    model:              Arc<String>,
}

#[async_trait]
impl Tool for DelegateTasksTool {
    fn id(&self) -> &'static str {
        IDENTIFIER
    }

    async fn execute(&self) -> Result<Value> {
        let next_depth = self.depth + 1;
        info!(
            "tool_call delegate_tasks depth={} tasks={} max_concurrency={}",
            next_depth,
            self.args.tasks.len(),
            MAX_PARALLEL_DELEGATIONS
        );

        run_delegate_batch_with_state(
            self.gateway.clone(),
            Arc::clone(&self.model),
            Arc::clone(&self.workspace_root),
            self.depth,
            self.max_subdelegations,
            self.args.tasks.clone(),
        )
        .await
    }
}
