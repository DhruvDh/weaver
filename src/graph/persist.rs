use std::{
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use tokio::{
    fs::{self, File},
    io::AsyncWriteExt,
    task,
};

use crate::graph::CurriculumGraph;

pub const SNAPSHOT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSnapshot {
    pub version:       u32,
    pub saved_at_sec:  u64,
    pub course_commit: String,
    pub graph:         CurriculumGraph,
}

impl GraphSnapshot {
    pub fn new(graph: CurriculumGraph, course_commit: String) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            version: SNAPSHOT_VERSION,
            saved_at_sec: now,
            course_commit,
            graph,
        }
    }
}

/// Serialize the graph to disk (JSON).
pub async fn save_graph(
    graph: &CurriculumGraph,
    path: impl AsRef<Path>,
    course_commit: &str,
) -> anyhow::Result<()> {
    let snapshot = GraphSnapshot::new(graph.clone(), course_commit.to_string());

    // Serialize in a blocking task to avoid hogging async executors on large
    // graphs.
    let data = task::spawn_blocking(move || {
        serde_json::to_vec_pretty(&snapshot).context("serialize graph snapshot")
    })
    .await
    .context("snapshot serialization task panicked")??;
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .await
            .with_context(|| format!("create snapshot directory {}", parent.display()))?;
    }

    let tmp_path = path.with_extension("tmp");
    {
        let mut file = File::create(&tmp_path)
            .await
            .with_context(|| format!("create temp snapshot {}", tmp_path.display()))?;
        file.write_all(&data)
            .await
            .with_context(|| format!("write temp snapshot {}", tmp_path.display()))?;
        file.sync_all()
            .await
            .with_context(|| format!("fsync temp snapshot {}", tmp_path.display()))?;
    }
    fs::rename(&tmp_path, path)
        .await
        .with_context(|| format!("rename temp snapshot to {}", path.display()))
}

/// Load a snapshot from disk and reconstruct GraphService.
pub async fn load_graph(path: impl AsRef<Path>) -> anyhow::Result<GraphSnapshot> {
    let data = fs::read(path.as_ref())
        .await
        .with_context(|| format!("read snapshot {}", path.as_ref().display()))?;
    let snapshot: GraphSnapshot =
        serde_json::from_slice(&data).context("deserialize graph snapshot")?;
    if snapshot.version != SNAPSHOT_VERSION {
        anyhow::bail!(
            "snapshot version {} unsupported (current {})",
            snapshot.version,
            SNAPSHOT_VERSION
        );
    }
    Ok(snapshot)
}
