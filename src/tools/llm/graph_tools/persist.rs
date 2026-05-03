use std::{
    env,
    path::{Component, Path, PathBuf},
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
        CallState, ToolExecutionError, ToolInputError, ToolInputResult, ToolPrototype,
        require_string,
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
    preflight: None,
    mutate: Some(|args: &mut SaveSnapshotArgs, state: &CallState| {
        let resolved = resolve_snapshot_save_path(state.workspace_root.as_ref(), args.path.clone(), SAVE_SNAPSHOT)?;
        args.path = Some(resolved.display().to_string());
        Ok(())
    })
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
        parse_args_with_builder(LOAD_SNAPSHOT, raw, |mut input: LoadSnapshotArgs| {
            input.path = require_string(input.path, LOAD_SNAPSHOT, "path")?;
            Ok(input)
        })
    },
    build: |args: &LoadSnapshotArgs| build_load_snapshot(args),
    ok: |args: &LoadSnapshotArgs, _| {
        info!(tool = LOAD_SNAPSHOT, path = %args.path, "graph load snapshot");
        command_ok(LOAD_SNAPSHOT, json!({"path": args.path}))
    },
    map_err: map_load_snapshot_err,
    preflight: None,
    mutate: Some(|args: &mut LoadSnapshotArgs, state: &CallState| {
        let resolved = resolve_snapshot_load_path(state.workspace_root.as_ref(), &args.path, LOAD_SNAPSHOT)?;
        args.path = resolved.display().to_string();
        Ok(())
    })
);

fn resolve_snapshot_path(path: Option<String>) -> String {
    path.or_else(|| env::var("GRAPH_SNAPSHOT_PATH").ok())
        .unwrap_or_else(|| "graph_snapshot.json".to_string())
}

fn resolve_snapshot_save_path(
    workspace_root: &Path,
    provided: Option<String>,
    tool: &'static str,
) -> ToolInputResult<PathBuf> {
    let raw = resolve_snapshot_path(provided);
    reject_parent_dir_components(&raw, tool)?;
    let root = canonical_workspace_root(workspace_root, tool)?;
    let candidate = absolute_or_workspace_path(&root, Path::new(&raw));
    let parent = candidate
        .parent()
        .ok_or_else(|| ToolInputError::InvalidPath {
            tool,
            path: candidate.display().to_string(),
            message: "snapshot path has no parent directory".to_string(),
        })?;
    let canonical_parent = parent
        .canonicalize()
        .map_err(|err| ToolInputError::InvalidPath {
            tool,
            path: parent.display().to_string(),
            message: err.to_string(),
        })?;
    if !canonical_parent.starts_with(&root) {
        return Err(ToolInputError::InvalidPath {
            tool,
            path: canonical_parent.display().to_string(),
            message: format!("escapes workspace root {}", root.display()),
        });
    }
    let file_name = candidate
        .file_name()
        .ok_or_else(|| ToolInputError::InvalidPath {
            tool,
            path: candidate.display().to_string(),
            message: "snapshot path must include a file name".to_string(),
        })?;
    let resolved = canonical_parent.join(file_name);
    if resolved.exists() {
        let canonical = resolved
            .canonicalize()
            .map_err(|err| ToolInputError::InvalidPath {
                tool,
                path: resolved.display().to_string(),
                message: err.to_string(),
            })?;
        if !canonical.starts_with(&root) {
            return Err(ToolInputError::InvalidPath {
                tool,
                path: canonical.display().to_string(),
                message: format!("escapes workspace root {}", root.display()),
            });
        }
        if !canonical.is_file() {
            return Err(ToolInputError::InvalidPath {
                tool,
                path: canonical.display().to_string(),
                message: "snapshot path must be a file".to_string(),
            });
        }
        return Ok(canonical);
    }
    Ok(resolved)
}

fn resolve_snapshot_load_path(
    workspace_root: &Path,
    provided: &str,
    tool: &'static str,
) -> ToolInputResult<PathBuf> {
    reject_parent_dir_components(provided, tool)?;
    let root = canonical_workspace_root(workspace_root, tool)?;
    let candidate = absolute_or_workspace_path(&root, Path::new(provided));
    let canonical = candidate
        .canonicalize()
        .map_err(|err| ToolInputError::InvalidPath {
            tool,
            path: candidate.display().to_string(),
            message: err.to_string(),
        })?;
    if !canonical.starts_with(&root) {
        return Err(ToolInputError::InvalidPath {
            tool,
            path: canonical.display().to_string(),
            message: format!("escapes workspace root {}", root.display()),
        });
    }
    let meta = canonical
        .metadata()
        .map_err(|err| ToolInputError::InvalidPath {
            tool,
            path: canonical.display().to_string(),
            message: err.to_string(),
        })?;
    if !meta.is_file() {
        return Err(ToolInputError::InvalidPath {
            tool,
            path: canonical.display().to_string(),
            message: "snapshot path must be a file".to_string(),
        });
    }
    Ok(canonical)
}

fn canonical_workspace_root(workspace_root: &Path, tool: &'static str) -> ToolInputResult<PathBuf> {
    workspace_root
        .canonicalize()
        .map_err(|err| ToolInputError::InvalidPath {
            tool,
            path: workspace_root.display().to_string(),
            message: err.to_string(),
        })
}

fn absolute_or_workspace_path(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

fn reject_parent_dir_components(path: &str, tool: &'static str) -> ToolInputResult<()> {
    if Path::new(path)
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(ToolInputError::InvalidPath {
            tool,
            path: path.to_string(),
            message: "path must not contain `..` components".to_string(),
        });
    }
    Ok(())
}

pub(super) fn tool_prototypes() -> Vec<ToolPrototype> {
    vec![save_snapshot_meta(), load_snapshot_meta()]
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use uuid::Uuid;

    use super::*;

    fn temp_workspace(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("weaver-{name}-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).expect("create temp workspace");
        root
    }

    #[test]
    fn save_snapshot_allows_new_file_inside_workspace() {
        let root = temp_workspace("snapshot-save-new");
        let resolved =
            resolve_snapshot_save_path(&root, Some("snapshot.json".into()), SAVE_SNAPSHOT)
                .expect("resolve save path");

        assert_eq!(resolved, root.canonicalize().unwrap().join("snapshot.json"));
        assert!(!resolved.exists());
    }

    #[test]
    fn save_snapshot_rejects_missing_parent() {
        let root = temp_workspace("snapshot-save-missing-parent");
        let err =
            resolve_snapshot_save_path(&root, Some("missing/snapshot.json".into()), SAVE_SNAPSHOT)
                .expect_err("missing parent rejected");

        assert!(err.to_string().contains("No such file"));
    }

    #[test]
    fn load_snapshot_requires_existing_workspace_file() {
        let root = temp_workspace("snapshot-load-existing");
        let path = root.join("snapshot.json");
        fs::write(&path, "{}").expect("write snapshot");

        let resolved =
            resolve_snapshot_load_path(&root, "snapshot.json", LOAD_SNAPSHOT).expect("load path");

        assert_eq!(resolved, path.canonicalize().unwrap());
    }

    #[test]
    fn load_snapshot_rejects_missing_file() {
        let root = temp_workspace("snapshot-load-missing");

        let err =
            resolve_snapshot_load_path(&root, "snapshot.json", LOAD_SNAPSHOT).expect_err("missing");

        assert!(err.to_string().contains("No such file"));
    }

    #[test]
    fn snapshot_paths_reject_parent_dir_escape() {
        let root = temp_workspace("snapshot-parent-escape");

        assert!(
            resolve_snapshot_save_path(&root, Some("../snapshot.json".into()), SAVE_SNAPSHOT)
                .is_err()
        );
        assert!(resolve_snapshot_load_path(&root, "../snapshot.json", LOAD_SNAPSHOT).is_err());
    }

    #[test]
    fn snapshot_paths_reject_absolute_outside_workspace() {
        let root = temp_workspace("snapshot-absolute-root");
        let outside = temp_workspace("snapshot-absolute-outside").join("snapshot.json");
        fs::write(&outside, "{}").expect("write outside snapshot");

        assert!(
            resolve_snapshot_save_path(&root, Some(outside.display().to_string()), SAVE_SNAPSHOT)
                .is_err()
        );
        assert!(
            resolve_snapshot_load_path(&root, &outside.display().to_string(), LOAD_SNAPSHOT)
                .is_err()
        );
    }

    #[cfg(unix)]
    #[test]
    fn snapshot_paths_reject_symlink_escape() {
        let root = temp_workspace("snapshot-symlink-root");
        let outside = temp_workspace("snapshot-symlink-outside");
        let outside_file = outside.join("snapshot.json");
        fs::write(&outside_file, "{}").expect("write outside snapshot");
        std::os::unix::fs::symlink(&outside_file, root.join("linked.json"))
            .expect("create file symlink");
        std::os::unix::fs::symlink(&outside, root.join("linked_dir")).expect("create dir symlink");

        assert!(resolve_snapshot_load_path(&root, "linked.json", LOAD_SNAPSHOT).is_err());
        assert!(
            resolve_snapshot_save_path(&root, Some("linked.json".into()), SAVE_SNAPSHOT).is_err()
        );
        assert!(
            resolve_snapshot_save_path(
                &root,
                Some("linked_dir/snapshot.json".into()),
                SAVE_SNAPSHOT
            )
            .is_err()
        );
    }
}
