use std::sync::Arc;

use async_trait::async_trait;
use bon::Builder;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::info;

use super::{
    CallState, ToolExecutionError, ToolInputError, ToolInputResult, ToolInstance, ToolOutput,
    ToolPayloadMode,
    common::{ToolRunPayload, ToolRunner},
    depth_exceeded, payload_size_bytes, trim_optional,
};
use crate::{
    constants::{MAX_DELEGATED_TASK_LEN, MAX_DELEGATED_TASKS, MAX_PARALLEL_DELEGATIONS},
    file_reader::run_delegate_batch_with_state,
};

const IDENTIFIER: &str = "delegate_tasks";
const DESCRIPTION: &str = "Delegate one or more tasks to child FileReader agents via the `tasks` \
                           array. Wrap single tasks in an array when needed.";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DelegateTasksArgs {
    #[schemars(
        length(min = 1, max = MAX_DELEGATED_TASKS),
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
        } else if tasks.len() > MAX_DELEGATED_TASKS {
            Err(ToolInputError::InvalidPayload {
                tool:    IDENTIFIER,
                message: format!(
                    "tasks exceeds max count {MAX_DELEGATED_TASKS} (received {})",
                    tasks.len()
                ),
            })
        } else {
            for task in &tasks {
                super::ensure_max_len(task, MAX_DELEGATED_TASK_LEN, IDENTIFIER, "task")?;
            }
            Ok(tasks)
        }
    })]
    pub tasks:      Vec<String>,
    #[serde(default)]
    #[builder(default = false)]
    #[schemars(
        description = "When true, return full child outputs; default false returns a summary \
                       preview to avoid context blowups."
    )]
    pub fetch_body: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DelegateTasksPayload {
    tasks:      Vec<String>,
    #[serde(default)]
    fetch_body: bool,
}

crate::basic_tool!(
    delegate_tasks_meta,
    id: IDENTIFIER,
    description: DESCRIPTION,
    args: DelegateTasksArgs,
    prepare: |raw, state: &CallState| {
        if state.depth >= state.max_subdelegations {
            return Err(depth_exceeded(IDENTIFIER, state.depth, state.max_subdelegations));
        }
        let payload: DelegateTasksPayload =
            serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
                tool:    IDENTIFIER,
                message: err.to_string(),
            })?;
        let args = DelegateTasksArgs::builder()
            .tasks(payload.tasks)?
            .fetch_body(payload.fetch_body)
            .build();
        Ok(args)
    },
    runner: |args: DelegateTasksArgs, state: &CallState| DelegateTasksTool {
        args,
        state: state.clone(),
    }
);

struct DelegateTasksTool {
    args:  DelegateTasksArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for DelegateTasksTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let next_depth = self.state.depth + 1;
        info!(
            "tool_call delegate_tasks depth={} tasks={} max_concurrency={}",
            next_depth,
            self.args.tasks.len(),
            MAX_PARALLEL_DELEGATIONS
        );

        let ctx = crate::file_reader::DelegateBatchCtx {
            gateway:            self.state.gateway.clone(),
            model:              Arc::clone(&self.state.model),
            workspace_root:     Arc::clone(&self.state.workspace_root),
            metrics:            Arc::clone(&self.state.metrics),
            graph:              self.state.graph.clone(),
            dedup:              self.state.dedup.clone(),
            analysis_cache:     Arc::clone(&self.state.analysis_cache),
            rerun:              self.state.rerun.clone(),
            depth:              self.state.depth,
            max_subdelegations: self.state.max_subdelegations,
            mode:               self.state.mode,
            course_commit:      Arc::clone(&self.state.course_commit),
        };

        let result = run_delegate_batch_with_state(ctx, self.args.tasks.clone()).await?;

        let requested = result
            .get("requested")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.args.tasks.len() as u64);
        let max_concurrency = result
            .get("max_concurrency")
            .and_then(|v| v.as_u64())
            .unwrap_or(MAX_PARALLEL_DELEGATIONS as u64);
        let results = result
            .get("results")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let ok_count = results
            .iter()
            .filter(|entry| entry.get("status") == Some(&Value::String("ok".to_string())))
            .count();
        let err_count = results.len().saturating_sub(ok_count);

        let preview_results: Vec<Value> = results
            .iter()
            .map(|entry| {
                json!({
                    "task": entry.get("task").cloned().unwrap_or(Value::Null),
                    "status": entry.get("status").cloned().unwrap_or(Value::Null),
                    "error": entry.get("error").cloned().unwrap_or(Value::Null),
                })
            })
            .collect();

        let preview = json!({
            "type": "delegation_batch_preview",
            "tool": IDENTIFIER,
            "depth": next_depth,
            "requested": requested,
            "max_concurrency": max_concurrency,
            "completed": results.len(),
            "ok": ok_count,
            "errors": err_count,
            "results": preview_results,
        });

        let full = result;
        let approx = payload_size_bytes(&full);
        let preview_size = payload_size_bytes(&preview);
        let mode = ToolPayloadMode::from_fetch_flag(self.args.fetch_body);

        let hints = vec![
            format!("Delegated {} task(s) at depth {}", requested, next_depth),
            "Set fetch_body=true to include child outputs; preview only shows statuses."
                .to_string(),
        ];

        ToolRunner::new(IDENTIFIER, &self.state)
            .with_mode(mode)
            .hints(hints)
            .run(move |mode| async move {
                match mode {
                    ToolPayloadMode::Preview => Ok(ToolRunPayload {
                        body:          preview.clone(),
                        approx_bytes:  Some(preview_size),
                        preview:       Some(preview),
                        preview_hints: Vec::new(),
                        page:          None,
                    }),
                    ToolPayloadMode::Body => Ok(ToolRunPayload {
                        body:          full,
                        approx_bytes:  Some(approx),
                        preview:       Some(json!({
                            "type": "preview",
                            "tool": IDENTIFIER,
                            "requested": requested,
                            "hint": "Set fetch_body=true to include child outputs."
                        })),
                        preview_hints: Vec::new(),
                        page:          None,
                    }),
                }
            })
            .await
    }
}
