use std::{
    env,
    path::{Path, PathBuf},
};

use anyhow::anyhow;
use bon::Builder;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tracing::info;

use super::common::{
    MaybeApply, map_send_err_anyhow, parse_args_with_builder, parse_graph_command,
};
use crate::{
    graph::manager::{LoadSnapshot, SaveSnapshot},
    tools::llm::{
        CallState, ToolExecutionError, ToolInputError, ToolPrototype, require_string,
        schema_for_args,
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

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
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

pub(super) fn save_snapshot_meta() -> ToolPrototype {
    ToolPrototype {
        id:          SAVE_SNAPSHOT,
        description: "Persist the in-memory graph to disk immediately. Use only when explicitly \
                      asked to save a snapshot; autosave runs separately. Path defaults to \
                      GRAPH_SNAPSHOT_PATH or graph_snapshot.json.",
        schema:      schema_for_args::<SaveSnapshotArgs>(),
        parse:       parse_save_snapshot,
    }
}

fn build_save_snapshot(args: &SaveSnapshotArgs) -> SaveSnapshot {
    SaveSnapshot {
        path: PathBuf::from(resolve_snapshot_path(args.path.clone())),
    }
}

fn ok_save_snapshot(args: &SaveSnapshotArgs, _: ()) -> Value {
    let path = resolve_snapshot_path(args.path.clone());
    command_ok(SAVE_SNAPSHOT, json!({"path": path}))
}

fn parse_save_snapshot(
    raw: Value,
    state: &CallState,
) -> crate::tools::llm::ToolInputResult<Box<dyn crate::tools::llm::ToolInstance>> {
    parse_graph_command::<SaveSnapshotArgs, SaveSnapshot>(
        SAVE_SNAPSHOT,
        raw,
        state,
        build_save_snapshot,
        |args, reply| {
            let path = resolve_snapshot_path(args.path.clone());
            info!(tool = SAVE_SNAPSHOT, path = %path, "graph save snapshot");
            ok_save_snapshot(args, reply)
        },
        map_send_err_anyhow,
    )
}

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

pub(super) fn load_snapshot_meta() -> ToolPrototype {
    ToolPrototype {
        id:          LOAD_SNAPSHOT,
        description: "Load a graph snapshot from disk, replacing the in-memory graph. Use only \
                      when explicitly asked to restore from a snapshot path.",
        schema:      schema_for_args::<LoadSnapshotArgs>(),
        parse:       parse_load_snapshot,
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

fn parse_load_snapshot(
    raw: Value,
    state: &CallState,
) -> crate::tools::llm::ToolInputResult<Box<dyn crate::tools::llm::ToolInstance>> {
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
    parse_graph_command::<LoadSnapshotArgs, LoadSnapshot>(
        LOAD_SNAPSHOT,
        serde_json::to_value(&args).expect("serialize"),
        state,
        build_load_snapshot,
        |args, _| {
            info!(tool = LOAD_SNAPSHOT, path = %args.path, "graph load snapshot");
            command_ok(LOAD_SNAPSHOT, json!({"path": args.path}))
        },
        map_load_snapshot_err,
    )
}

fn resolve_snapshot_path(path: Option<String>) -> String {
    path.or_else(|| env::var("GRAPH_SNAPSHOT_PATH").ok())
        .unwrap_or_else(|| "graph_snapshot.json".to_string())
}

pub(super) fn tool_prototypes() -> Vec<ToolPrototype> {
    vec![save_snapshot_meta(), load_snapshot_meta()]
}
