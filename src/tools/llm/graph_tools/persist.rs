use std::{
    env,
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
};

use anyhow::anyhow;
use bon::Builder;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::info;

use super::common::{MaybeApply, map_send_err_anyhow, parse_args_with_builder};
use crate::{
    graph::{commands::LoadSnapshot, manager::SaveSnapshot},
    tools::llm::{
        CallState, ToolExecutionError, ToolInputError, ToolPrototype, require_string,
        resolve_workspace_path,
    },
};

fn command_ok(tool: &'static str, extra: serde_json::Value) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    map.insert("type".into(), json!("graph_command"));
    map.insert("tool".into(), json!(tool));
    map.insert("status".into(), json!("ok"));
    if let serde_json::Value::Object(obj) = extra {
        for (k, v) in obj {
            map.insert(k, v);
        }
    }
    serde_json::Value::Object(map)
}

// ---------- Persistence tools ----------

const SAVE_SNAPSHOT: &str = "graph_save_now";
const LOAD_SNAPSHOT: &str = "graph_load_snapshot";

#[derive(Debug, Clone, Builder, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SaveSnapshotArgs {
    #[serde(default)]
    pub path:  Option<String>,
    #[serde(default)]
    #[builder(default = false)]
    pub apply: bool,
}

impl MaybeApply for SaveSnapshotArgs {
    fn apply_flag(&self) -> bool {
        self.apply
    }
}

fn build_save_snapshot(args: &SaveSnapshotArgs) -> SaveSnapshot {
    SaveSnapshot {
        path: PathBuf::from(resolve_snapshot_path(args.path.clone())),
    }
}

fn ok_save_snapshot(args: &SaveSnapshotArgs, _: ()) -> serde_json::Value {
    let path = resolve_snapshot_path(args.path.clone());
    command_ok(SAVE_SNAPSHOT, json!({"path": path}))
}

crate::graph_action_tool!(
    save_snapshot_meta,
    id: SAVE_SNAPSHOT,
    description: "Persist the in-memory graph to disk immediately. Use only when explicitly asked \
                  to save a snapshot; autosave runs separately. Path defaults to \
                  GRAPH_SNAPSHOT_PATH or graph_snapshot.json.",
    args: SaveSnapshotArgs,
    prepare: |raw| super::common::parse_args_with_builder(SAVE_SNAPSHOT, raw, |args: SaveSnapshotArgs| Ok(args)),
    build: |args: &SaveSnapshotArgs| build_save_snapshot(args),
    ok: |args: &SaveSnapshotArgs, reply: ()| {
        let path = resolve_snapshot_path(args.path.clone());
        info!(tool = SAVE_SNAPSHOT, path = %path, "graph save snapshot");
        ok_save_snapshot(args, reply)
    },
    map_err: map_send_err_anyhow,
    preflight: Some(Arc::new(
        |args: &SaveSnapshotArgs, state: &CallState| -> Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<(), crate::tools::llm::ToolExecutionError>
                    > + Send,
            >,
        > {
            let provided = args.path.clone();
            let root = Arc::clone(&state.workspace_root);
            Box::pin(async move { preflight_snapshot_path(provided, SAVE_SNAPSHOT, root).await })
        },
    ))
);

#[derive(Debug, Clone, Builder, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LoadSnapshotArgs {
    #[builder(with = |v: String| -> crate::tools::llm::ToolInputResult<_> {
        crate::tools::llm::require_string(v, LOAD_SNAPSHOT, "path")
    })]
    pub path:  String,
    #[serde(default)]
    #[builder(default = false)]
    pub apply: bool,
}

impl MaybeApply for LoadSnapshotArgs {
    fn apply_flag(&self) -> bool {
        self.apply
    }
}

fn build_load_snapshot(args: &LoadSnapshotArgs) -> LoadSnapshot {
    LoadSnapshot {
        path: PathBuf::from(args.path.clone()),
    }
}

fn map_load_snapshot_err(
    err: kameo::error::SendError<LoadSnapshot, anyhow::Error>,
) -> ToolExecutionError {
    match err {
        kameo::error::SendError::HandlerError(e) => {
            if let Some(io) = e.downcast_ref::<std::io::Error>()
                && io.kind() == std::io::ErrorKind::NotFound
            {
                return ToolExecutionError::Input(ToolInputError::InvalidPayload {
                    tool:    LOAD_SNAPSHOT,
                    message: e.to_string(),
                });
            }
            let msg = e.to_string();
            if msg.contains("snapshot version") {
                return ToolExecutionError::Input(ToolInputError::InvalidPayload {
                    tool:    LOAD_SNAPSHOT,
                    message: msg,
                });
            }
            ToolExecutionError::Internal(e)
        }
        other => ToolExecutionError::Internal(anyhow!("{:?}", other)),
    }
}

crate::graph_action_tool!(
    load_snapshot_meta,
    id: LOAD_SNAPSHOT,
    description: "Load a graph snapshot from disk, replacing the in-memory graph. Use only when \
                  explicitly asked to restore from a snapshot path.",
    args: LoadSnapshotArgs,
    prepare: |raw| {
        let args = parse_args_with_builder(LOAD_SNAPSHOT, raw, |mut input: LoadSnapshotArgs| {
            input.path = require_string(input.path, LOAD_SNAPSHOT, "path")?;
            Ok(input)
        })?;
        if !Path::new(&args.path).exists() {
            return Err(ToolInputError::InvalidPayload {
                tool:    LOAD_SNAPSHOT,
                message: format!("snapshot file `{}` not found", args.path),
            });
        }
        Ok(args)
    },
    build: |args: &LoadSnapshotArgs| build_load_snapshot(args),
    ok: |args: &LoadSnapshotArgs, _| {
        info!(tool = LOAD_SNAPSHOT, path = %args.path, "graph load snapshot");
        command_ok(LOAD_SNAPSHOT, json!({"path": args.path}))
    },
    map_err: map_load_snapshot_err,
    preflight: Some(Arc::new(
        |args: &LoadSnapshotArgs, state: &CallState| -> Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<(), crate::tools::llm::ToolExecutionError>
                    > + Send,
            >,
        > {
            let provided = Some(args.path.clone());
            let root = Arc::clone(&state.workspace_root);
            Box::pin(async move { preflight_snapshot_path(provided, LOAD_SNAPSHOT, root).await })
        },
    ))
);

fn resolve_snapshot_path(path: Option<String>) -> String {
    path.or_else(|| env::var("GRAPH_SNAPSHOT_PATH").ok())
        .unwrap_or_else(|| "graph_snapshot.json".to_string())
}

async fn preflight_snapshot_path(
    provided: Option<String>,
    tool: &'static str,
    workspace_root: Arc<PathBuf>,
) -> Result<(), ToolExecutionError> {
    let path = resolve_snapshot_path(provided);
    resolve_workspace_path(workspace_root.as_ref(), Path::new(&path), tool)
        .map_err(ToolExecutionError::Input)?;
    Ok(())
}

pub(super) fn tool_prototypes() -> Vec<ToolPrototype> {
    vec![save_snapshot_meta(), load_snapshot_meta()]
}
