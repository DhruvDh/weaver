//! Experimental event-sourced hooks for graph mutations. Intended for future
//! auditability without changing core storage paths.
use std::sync::{Arc, RwLock};

use serde::Serialize;
use serde_json::Value;
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
