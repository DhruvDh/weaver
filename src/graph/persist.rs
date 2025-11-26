use std::{
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use tokio::fs;

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
    pub fn new(graph: CurriculumGraph) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            version: SNAPSHOT_VERSION,
            saved_at_sec: now,
            course_commit: std::env::var("GRAPH_COURSE_COMMIT").unwrap_or_default(),
            graph,
        }
    }
}

/// Serialize the graph to disk (JSON).
pub async fn save_graph(graph: &CurriculumGraph, path: impl AsRef<Path>) -> anyhow::Result<()> {
    let snapshot = GraphSnapshot::new(graph.clone());
    let data = serde_json::to_vec_pretty(&snapshot).context("serialize graph snapshot")?;
    fs::write(path.as_ref(), data)
        .await
        .with_context(|| format!("write snapshot to {}", path.as_ref().display()))
}

/// Load a snapshot from disk and reconstruct GraphService.
pub async fn load_graph(path: impl AsRef<Path>) -> anyhow::Result<CurriculumGraph> {
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
    Ok(snapshot.graph)
}
