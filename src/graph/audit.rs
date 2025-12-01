//! Experimental event-sourced hooks for graph mutations. Intended for future
//! auditability without changing core storage paths.
use std::{
    borrow::Cow,
    sync::{Arc, RwLock},
};

use kameo::prelude::ActorRef;
use serde::Serialize;
use serde_json::Value;
use tokio::task;
use tracing::debug;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MutationKind {
    InsertKnowledge,
    UpdateKnowledge,
    InsertTeachingStep,
    UpdateTeachingStep,
    AddEdge { edge: &'static str },
    RemoveEdge { edge: &'static str },
    RemoveNode,
    RenameNode,
    InstallGraph,
    PruneRequires,
    SetStrictQuality { strict: bool },
}

impl MutationKind {
    pub fn label(&self) -> Cow<'static, str> {
        match self {
            MutationKind::InsertKnowledge => Cow::Borrowed("insert_knowledge"),
            MutationKind::UpdateKnowledge => Cow::Borrowed("update_knowledge"),
            MutationKind::InsertTeachingStep => Cow::Borrowed("insert_teaching_step"),
            MutationKind::UpdateTeachingStep => Cow::Borrowed("update_teaching_step"),
            MutationKind::AddEdge { edge } => Cow::Owned(format!("add_edge.{edge}")),
            MutationKind::RemoveEdge { edge } => Cow::Owned(format!("remove_edge.{edge}")),
            MutationKind::RemoveNode => Cow::Borrowed("remove_node"),
            MutationKind::RenameNode => Cow::Borrowed("rename_node"),
            MutationKind::InstallGraph => Cow::Borrowed("install_graph"),
            MutationKind::PruneRequires => Cow::Borrowed("prune_requires"),
            MutationKind::SetStrictQuality { .. } => Cow::Borrowed("set_strict_quality"),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MutationEvent {
    pub id:            Uuid,
    pub graph_version: u64,
    pub timestamp_ms:  u128,
    pub kind:          MutationKind,
    pub payload:       Value,
}

impl MutationEvent {
    pub fn new(kind: MutationKind, graph_version: u64, payload: Value) -> Self {
        Self {
            id: Uuid::new_v4(),
            graph_version,
            timestamp_ms: current_millis(),
            kind,
            payload,
        }
    }
}

fn current_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or_default()
}

pub type SharedMutationSink = Arc<dyn MutationEventSink + Send + Sync>;

pub trait MutationEventSink: Send + Sync {
    fn record(&self, event: &MutationEvent);
}

#[derive(Default)]
pub struct NoopMutationSink;

impl MutationEventSink for NoopMutationSink {
    fn record(&self, _: &MutationEvent) {}
}

#[derive(Default)]
pub struct InMemoryMutationLog {
    events: RwLock<Vec<MutationEvent>>,
}

impl InMemoryMutationLog {
    pub fn push(&self, event: MutationEvent) {
        if let Ok(mut guard) = self.events.write() {
            guard.push(event);
        }
    }

    pub fn snapshot(&self) -> Vec<MutationEvent> {
        self.events
            .read()
            .map(|guard| guard.clone())
            .unwrap_or_default()
    }
}

impl MutationEventSink for InMemoryMutationLog {
    fn record(&self, event: &MutationEvent) {
        self.push(event.clone());
    }
}

#[derive(Clone)]
pub struct RerunMutationSink {
    rerun:       ActorRef<crate::rerun_sink::RerunSink>,
    path_prefix: String,
}

impl RerunMutationSink {
    pub fn new(rerun: ActorRef<crate::rerun_sink::RerunSink>) -> Self {
        Self {
            rerun,
            path_prefix: "audit/mutations".to_string(),
        }
    }

    pub fn with_prefix(
        rerun: ActorRef<crate::rerun_sink::RerunSink>,
        path_prefix: impl Into<String>,
    ) -> Self {
        Self {
            rerun,
            path_prefix: path_prefix.into(),
        }
    }
}

impl MutationEventSink for RerunMutationSink {
    fn record(&self, event: &MutationEvent) {
        let payload = match serde_json::to_string(event) {
            Ok(body) => body,
            Err(err) => {
                debug!(error = %err, "failed to serialize mutation event");
                return;
            }
        };
        let nanos: i64 = event
            .timestamp_ms
            .saturating_mul(1_000_000)
            .min(i64::MAX as u128) as i64;
        let path = format!("{}/{}", self.path_prefix, event.kind.label());
        let msg = crate::rerun_sink::LogText {
            path,
            value: payload,
            time_ns: Some(nanos),
        };
        let sink = self.rerun.clone();
        task::spawn(async move {
            let _ = sink.tell(msg).await;
        });
    }
}
