use std::{
    collections::{HashMap, HashSet},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Result, anyhow};
use kameo::prelude::*;
use petgraph::Direction;
use serde::Serialize;
use tracing::{info, warn};

use crate::{
    constants::DEDUP_CLUSTER_DISTANCE,
    graph::{
        self, EdgeKind, MergeSummary,
        commands::{GetGraphWithVersion, MergeNodes},
        dedup::{hamming_distance, simhash},
        manager::GraphManager,
    },
    schema::types::KnowledgeType,
};

#[derive(Debug, Clone)]
pub struct RunDeduplication {
    pub auto_merge_threshold: f64,
    pub dry_run:              bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct MergeCandidate {
    pub nodes:                 Vec<String>,
    pub recommended_canonical: String,
    pub confidence:            f64,
    pub rationale:             String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MergeRecord {
    pub merged_from:      String,
    pub merged_into:      String,
    pub timestamp:        u64,
    pub edges_redirected: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct DeduplicationReport {
    pub graph_version:     u64,
    pub clusters_analyzed: usize,
    pub duplicates_found:  usize,
    pub auto_merged:       Vec<MergeRecord>,
    pub pending_review:    Vec<MergeCandidate>,
}

#[derive(Actor)]
pub struct DeduplicationAgent {
    graph: ActorRef<GraphManager>,
}

impl DeduplicationAgent {
    pub fn new(graph: ActorRef<GraphManager>) -> Self {
        Self { graph }
    }
}

impl Message<RunDeduplication> for DeduplicationAgent {
    type Reply = Result<DeduplicationReport>;

    async fn handle(
        &mut self,
        RunDeduplication {
            auto_merge_threshold,
            dry_run,
        }: RunDeduplication,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let (graph, graph_version) = self
            .graph
            .ask(GetGraphWithVersion)
            .await
            .map_err(|err| anyhow!("{err:?}"))?;
        let mut fingerprints: Vec<(graph::NodeId, KnowledgeType, u64)> = Vec::new();
        for node in graph.node_indices() {
            if let graph::NodeKind::Knowledge(k) = &graph[node].kind {
                fingerprints.push((node, k.knowledge_type, simhash(&k.statement)));
            }
        }

        let clusters = cluster_by_similarity(&fingerprints, DEDUP_CLUSTER_DISTANCE);
        let fp_map: HashMap<graph::NodeId, u64> =
            fingerprints.iter().map(|(id, _, fp)| (*id, *fp)).collect();

        let mut pending = Vec::new();
        let mut merged = Vec::new();

        for cluster in clusters {
            let slugs: Vec<String> = cluster.iter().map(|id| graph[*id].slug.clone()).collect();
            if slugs.is_empty() {
                continue;
            }
            let canonical_slug = slugs
                .iter()
                .min()
                .cloned()
                .unwrap_or_else(|| graph[cluster[0]].slug.clone());
            let confidence = cluster_confidence(&cluster, &fp_map);
            let rationale = format!("simhash cluster (size {})", cluster.len());
            pending.push(MergeCandidate {
                nodes: slugs.clone(),
                recommended_canonical: canonical_slug.clone(),
                confidence,
                rationale: rationale.clone(),
            });

            if dry_run || confidence < auto_merge_threshold {
                continue;
            }

            if !cluster_can_auto_merge(&graph, &cluster) {
                continue;
            }

            for slug in slugs.iter().filter(|s| **s != canonical_slug) {
                match self
                    .graph
                    .ask(MergeNodes {
                        canonical: canonical_slug.clone(),
                        duplicate: slug.clone(),
                    })
                    .await
                {
                    Ok(MergeSummary {
                        merged_from,
                        merged_into,
                        edges_redirected,
                    }) => merged.push(MergeRecord {
                        merged_from,
                        merged_into,
                        timestamp: now_secs(),
                        edges_redirected,
                    }),
                    Err(kameo::error::SendError::HandlerError(err)) => {
                        warn!(error = %err, canonical = %canonical_slug, duplicate = %slug, "dedup merge failed")
                    }
                    Err(err) => {
                        warn!(error = ?err, canonical = %canonical_slug, duplicate = %slug, "dedup merge send failed")
                    }
                }
            }
        }

        info!(
            graph_version,
            clusters = pending.len(),
            auto_merged = merged.len(),
            "deduplication agent completed"
        );
        let duplicates_found: usize = pending
            .iter()
            .map(|c| c.nodes.len().saturating_sub(1))
            .sum();

        Ok(DeduplicationReport {
            graph_version,
            clusters_analyzed: pending.len(),
            duplicates_found,
            auto_merged: merged,
            pending_review: pending,
        })
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn cluster_by_similarity(
    entries: &[(graph::NodeId, KnowledgeType, u64)],
    threshold: u32,
) -> Vec<Vec<graph::NodeId>> {
    let mut clusters = Vec::new();
    let mut visited = HashSet::new();
    for (idx, (id, kind, fp)) in entries.iter().enumerate() {
        if visited.contains(id) {
            continue;
        }
        let mut cluster = vec![*id];
        for (other, other_kind, other_fp) in entries.iter().skip(idx + 1) {
            if other_kind != kind {
                continue;
            }
            if hamming_distance(*fp, *other_fp) <= threshold {
                cluster.push(*other);
                visited.insert(*other);
            }
        }
        visited.insert(*id);
        if cluster.len() > 1 {
            clusters.push(cluster);
        }
    }
    clusters
}

fn cluster_confidence(cluster: &[graph::NodeId], fps: &HashMap<graph::NodeId, u64>) -> f64 {
    let mut best = 0.0;
    for (i, lhs) in cluster.iter().enumerate() {
        let Some(lhs_fp) = fps.get(lhs) else {
            continue;
        };
        for rhs in cluster.iter().skip(i + 1) {
            if let Some(rhs_fp) = fps.get(rhs) {
                let dist = hamming_distance(*lhs_fp, *rhs_fp) as f64;
                let sim = 1.0 - dist / 64.0;
                if sim > best {
                    best = sim;
                }
            }
        }
    }
    best
}

fn cluster_can_auto_merge(graph: &graph::CurriculumGraph, cluster: &[graph::NodeId]) -> bool {
    for node in cluster {
        let graph::NodeKind::Knowledge(k) = &graph[*node].kind else {
            return false;
        };
        if k.knowledge_type.is_learning_outcome() || k.knowledge_type.is_assessment_item() {
            return false;
        }
        let has_assesses = graph
            .edges_directed(*node, Direction::Incoming)
            .chain(graph.edges_directed(*node, Direction::Outgoing))
            .any(|edge| matches!(edge.weight().kind, EdgeKind::Assesses(_)));
        if has_assesses {
            return false;
        }
    }
    true
}
