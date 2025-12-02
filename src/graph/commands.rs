use std::path::PathBuf;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    graph::{
        AnchorsAttrs, AssessesAttrs, EdgeConflict, EdgeKind, KnowledgeNode, PrecedesAttrs,
        RequiresAttrs, SupportsAttrs, TeachingStepNode,
    },
    schema::types::KnowledgeType,
};

#[derive(Clone, Debug, Serialize)]
pub struct Neighbor {
    pub neighbor_slug: String,
    pub edge_kind:     String,
    pub direction:     String,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKindFilter {
    Requires,
    Supports,
    Assesses,
    Precedes,
    Anchors,
}

#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NeighborDirection {
    Incoming,
    Outgoing,
    Both,
}

#[derive(Clone, Debug, Serialize)]
pub struct NodeSummary {
    pub slug:           String,
    pub title:          String,
    pub kind:           String,
    pub knowledge_type: Option<KnowledgeType>,
    pub tags:           Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct NodeSearchResult {
    pub slug:           String,
    pub title:          String,
    pub kind:           String,
    pub knowledge_type: Option<KnowledgeType>,
    pub tags:           Vec<String>,
    pub score:          f32,
}

#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NodeKindSelector {
    Knowledge { knowledge_type: KnowledgeType },
    TeachingStep,
    AnyKnowledge,
}

pub struct InsertKnowledge {
    pub slug:    String,
    pub payload: KnowledgeNode,
    pub tags:    Vec<String>,
}

pub struct UpdateKnowledge {
    pub slug:    String,
    pub payload: KnowledgeNode,
    pub tags:    Vec<String>,
}

pub struct InsertTeachingStep {
    pub slug:    String,
    pub payload: TeachingStepNode,
    pub tags:    Vec<String>,
}

pub struct UpdateTeachingStep {
    pub slug:    String,
    pub payload: TeachingStepNode,
    pub tags:    Vec<String>,
}

pub struct AddRequires {
    pub from:       String,
    pub to:         String,
    pub attrs:      RequiresAttrs,
    pub confidence: f32,
}

pub struct AddSupports {
    pub from:       String,
    pub to:         String,
    pub attrs:      SupportsAttrs,
    pub confidence: f32,
}

pub struct AddAssesses {
    pub from:       String,
    pub to:         String,
    pub attrs:      AssessesAttrs,
    pub confidence: f32,
}

pub struct AddPrecedes {
    pub from:       String,
    pub to:         String,
    pub attrs:      PrecedesAttrs,
    pub confidence: f32,
}

pub struct AddAnchors {
    pub from:       String,
    pub to:         String,
    pub attrs:      AnchorsAttrs,
    pub confidence: f32,
}

pub struct GetNode {
    pub slug: String,
}

pub struct GetGraph;
pub struct GetGraphWithVersion;
pub struct GetGraphVersion;

pub struct Neighbors {
    pub slug:      String,
    pub edge_kind: Option<EdgeKindFilter>,
    pub direction: Option<NeighborDirection>,
}

pub struct ListNodesByTag {
    pub tag: String,
}

pub struct ListNodesByKind {
    pub selector: NodeKindSelector,
}

pub struct ListTags;

pub struct SearchNodes {
    pub query: String,
    pub limit: usize,
}

pub struct RenameNode {
    pub old_slug: String,
    pub new_slug: String,
}

pub struct RemoveNode {
    pub slug: String,
}

pub struct MergeNodes {
    pub canonical: String,
    pub duplicate: String,
}

pub struct ResolveSlug {
    pub slug: String,
}

pub struct ResolveSlugs {
    pub slugs: Vec<String>,
}

pub struct LoadSnapshot {
    pub path: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct EdgeConflictView {
    pub edge_id:    u32,
    pub from_slug:  String,
    pub to_slug:    String,
    pub kind:       EdgeKind,
    pub confidence: f32,
    pub conflicts:  Vec<EdgeConflict>,
}

pub struct GetEdgeConflicts;

#[derive(Clone, Debug)]
pub struct ResolveEdgeConflict {
    pub edge_id:         u32,
    pub resolved_kind:   EdgeKind,
    pub confidence:      Option<f32>,
    pub clear_conflicts: bool,
}
