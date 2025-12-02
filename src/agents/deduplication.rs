use std::{
    collections::HashMap,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Result, anyhow};
use kameo::prelude::*;
use petgraph::Direction;
use serde::Serialize;
use strsim::jaro_winkler;
use tokio::task;
use tracing::{info, warn};

use crate::{
    constants::{DEDUP_CLUSTER_SIMILARITY, DEDUP_MAX_AUTO_MERGES_PER_RUN},
    graph::{
        self, EdgeKind, MergeSummary,
        commands::{
            EdgeConflictView, GetEdgeConflicts, GetGraphVersion, GetGraphWithVersion, MergeNodes,
            ResolveEdgeConflict, ResolveSlugs,
        },
        manager::GraphManager,
    },
    rerun_sink::RerunSink,
    schema::types::KnowledgeType,
};

const STATEMENT_PREVIEW_CHARS: usize = 200;

#[derive(Debug, Clone)]
pub struct RunDeduplication {
    pub auto_merge_threshold: f64,
    pub dry_run:              bool,
}

#[derive(Debug, Clone)]
pub struct ListMergeCandidates {
    pub include_statements: bool,
}

#[derive(Debug, Clone)]
pub struct MergeNodesCommand {
    pub canonical: String,
    pub duplicate: String,
    pub apply:     bool,
}

#[derive(Debug, Clone)]
pub struct ListEdgeConflicts {
    pub limit:  Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct ResolveEdgeConflictCommand {
    pub edge_id:         u32,
    pub resolved_kind:   EdgeKind,
    pub confidence:      Option<f32>,
    pub clear_conflicts: bool,
    pub apply:           bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeCandidate {
    pub slug:           String,
    pub title:          String,
    pub statement:      String,
    pub knowledge_type: KnowledgeType,
}

#[derive(Debug, Clone, Serialize)]
pub struct MergeCandidate {
    pub nodes:                 Vec<NodeCandidate>,
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
pub struct MergeCandidatesResult {
    pub graph_version: u64,
    pub candidates:    Vec<MergeCandidate>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EdgeConflictsResult {
    pub graph_version: u64,
    pub conflicts:     Vec<EdgeConflictView>,
    pub offset:        usize,
    pub limit:         usize,
    pub has_more:      bool,
    pub total:         usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct DeduplicationReport {
    pub graph_version:         u64,
    pub clusters_analyzed:     usize,
    pub duplicates_found:      usize,
    pub auto_merged:           Vec<MergeRecord>,
    pub auto_merged_clusters:  usize,
    pub skipped_clusters:      usize,
    pub skipped_due_to_budget: usize,
    pub pending_review:        Vec<MergeCandidate>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MergeStatus {
    Preview,
    Applied,
}

#[derive(Debug, Clone, Serialize)]
pub struct MergeOperationResult {
    pub canonical:     String,
    pub duplicate:     String,
    pub graph_version: u64,
    pub status:        MergeStatus,
    pub record:        Option<MergeRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EdgeConflictResolutionReport {
    pub attempted:     usize,
    pub resolved:      Vec<EdgeConflictResolved>,
    pub failed:        Vec<EdgeConflictError>,
    pub skipped:       usize,
    pub dry_run:       bool,
    pub graph_version: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct EdgeConflictResolved {
    pub edge_id:   u32,
    pub from_slug: String,
    pub to_slug:   String,
}

#[derive(Debug, Clone, Serialize)]
pub struct EdgeConflictError {
    pub edge_id:   u32,
    pub from_slug: String,
    pub to_slug:   String,
    pub message:   String,
}

#[derive(Actor)]
pub struct DeduplicationAgent {
    graph: ActorRef<GraphManager>,
    rerun: Option<ActorRef<RerunSink>>,
}

impl DeduplicationAgent {
    pub fn new(graph: ActorRef<GraphManager>, rerun: Option<ActorRef<RerunSink>>) -> Self {
        Self { graph, rerun }
    }

    fn build_clusters(&self, graph: &graph::CurriculumGraph) -> Vec<Cluster> {
        let mut entries = Vec::new();
        for node in graph.node_indices() {
            let graph::NodeKind::Knowledge(k) = &graph[node].kind else {
                continue;
            };
            entries.push(NodeEntry::new(
                node,
                graph[node].slug.clone(),
                k.title.clone(),
                k.statement.clone(),
                k.knowledge_type,
            ));
        }
        let mut clusters = cluster_by_similarity(&entries, DEDUP_CLUSTER_SIMILARITY);
        for cluster in &mut clusters {
            cluster.nodes.sort_by(|a, b| a.slug.cmp(&b.slug));
            cluster.canonical_slug = cluster
                .nodes
                .iter()
                .map(|n| n.slug.clone())
                .min()
                .unwrap_or_default();
        }
        clusters.sort_by(|a, b| a.canonical_slug.cmp(&b.canonical_slug));
        clusters
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

        let clusters = self.build_clusters(&graph);
        let mut pending = Vec::with_capacity(clusters.len());
        let mut merged = Vec::new();
        let mut auto_merged_clusters = 0usize;
        let mut skipped_clusters = 0usize;
        let mut skipped_due_to_budget = 0usize;
        let mut merge_budget = DEDUP_MAX_AUTO_MERGES_PER_RUN;

        for cluster in clusters {
            let candidate = cluster.to_candidate(None);
            let canonical_slug = candidate.recommended_canonical.clone();
            let duplicates: Vec<_> = candidate
                .nodes
                .iter()
                .filter(|node| node.slug != canonical_slug)
                .collect();
            pending.push(candidate.clone());

            if duplicates.is_empty() || dry_run || cluster.confidence < auto_merge_threshold {
                skipped_clusters = skipped_clusters.saturating_add(1);
                continue;
            }

            if !cluster_can_auto_merge(&graph, &cluster.nodes) {
                skipped_clusters = skipped_clusters.saturating_add(1);
                continue;
            }

            let mut merged_this_cluster = false;
            for duplicate in duplicates {
                if merge_budget == 0 {
                    skipped_due_to_budget = skipped_due_to_budget.saturating_add(1);
                    continue;
                }
                merge_budget = merge_budget.saturating_sub(1);
                match self
                    .graph
                    .ask(MergeNodes {
                        canonical: canonical_slug.clone(),
                        duplicate: duplicate.slug.clone(),
                    })
                    .await
                {
                    Ok(MergeSummary {
                        merged_from,
                        merged_into,
                        edges_redirected,
                    }) => {
                        merged.push(MergeRecord {
                            merged_from,
                            merged_into,
                            timestamp: now_secs(),
                            edges_redirected,
                        });
                        merged_this_cluster = true;
                    }
                    Err(kameo::error::SendError::HandlerError(err)) => {
                        warn!(
                            error = %err,
                            canonical = %canonical_slug,
                            duplicate = %duplicate.slug,
                            "dedup merge failed"
                        );
                    }
                    Err(err) => {
                        warn!(
                            error = ?err,
                            canonical = %canonical_slug,
                            duplicate = %duplicate.slug,
                            "dedup merge send failed"
                        );
                    }
                }
            }

            if merged_this_cluster {
                auto_merged_clusters = auto_merged_clusters.saturating_add(1);
            } else {
                skipped_clusters = skipped_clusters.saturating_add(1);
            }
        }

        let duplicates_found: usize = pending
            .iter()
            .map(|c| c.nodes.len().saturating_sub(1))
            .sum();

        info!(
            graph_version,
            clusters = pending.len(),
            auto_merged = merged.len(),
            auto_merged_clusters,
            skipped_clusters,
            skipped_due_to_budget,
            merge_budget_remaining = merge_budget,
            "deduplication agent completed"
        );
        log_scalar(&self.rerun, "metrics/dedup/clusters", pending.len() as f64);
        log_scalar(&self.rerun, "metrics/dedup/auto_merged", merged.len() as f64);
        log_scalar(&self.rerun, "metrics/dedup/auto_merged_clusters", auto_merged_clusters as f64);
        log_scalar(&self.rerun, "metrics/dedup/skipped_clusters", skipped_clusters as f64);
        log_scalar(
            &self.rerun,
            "metrics/dedup/skipped_due_to_budget",
            skipped_due_to_budget as f64,
        );

        Ok(DeduplicationReport {
            graph_version,
            clusters_analyzed: pending.len(),
            duplicates_found,
            auto_merged: merged,
            auto_merged_clusters,
            skipped_clusters,
            skipped_due_to_budget,
            pending_review: pending,
        })
    }
}

impl Message<ListMergeCandidates> for DeduplicationAgent {
    type Reply = Result<MergeCandidatesResult>;

    async fn handle(
        &mut self,
        ListMergeCandidates { include_statements }: ListMergeCandidates,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let (graph, graph_version) = self
            .graph
            .ask(GetGraphWithVersion)
            .await
            .map_err(|err| anyhow!("{err:?}"))?;
        let preview_len = if include_statements {
            None
        } else {
            Some(STATEMENT_PREVIEW_CHARS)
        };
        let candidates = self
            .build_clusters(&graph)
            .into_iter()
            .map(|cluster| cluster.to_candidate(preview_len))
            .collect();

        Ok(MergeCandidatesResult {
            graph_version,
            candidates,
        })
    }
}

impl Message<MergeNodesCommand> for DeduplicationAgent {
    type Reply = Result<MergeOperationResult>;

    async fn handle(
        &mut self,
        MergeNodesCommand {
            canonical,
            duplicate,
            apply,
        }: MergeNodesCommand,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        if canonical == duplicate {
            return Err(anyhow!("canonical and duplicate must differ"));
        }

        self.graph
            .ask(ResolveSlugs {
                slugs: vec![canonical.clone(), duplicate.clone()],
            })
            .await
            .map_err(|err| anyhow!("{err:?}"))?;

        if !apply {
            let graph_version = self
                .graph
                .ask(GetGraphVersion)
                .await
                .map_err(|err| anyhow!("{err:?}"))?;
            return Ok(MergeOperationResult {
                canonical,
                duplicate,
                graph_version,
                status: MergeStatus::Preview,
                record: None,
            });
        }

        let MergeSummary {
            merged_from,
            merged_into,
            edges_redirected,
        } = self
            .graph
            .ask(MergeNodes {
                canonical: canonical.clone(),
                duplicate: duplicate.clone(),
            })
            .await
            .map_err(|err| anyhow!("{err:?}"))?;

        let graph_version = self
            .graph
            .ask(GetGraphVersion)
            .await
            .map_err(|err| anyhow!("{err:?}"))?;

        Ok(MergeOperationResult {
            canonical: merged_into.clone(),
            duplicate: merged_from.clone(),
            graph_version,
            status: MergeStatus::Applied,
            record: Some(MergeRecord {
                merged_from,
                merged_into,
                timestamp: now_secs(),
                edges_redirected,
            }),
        })
    }
}

impl Message<ListEdgeConflicts> for DeduplicationAgent {
    type Reply = Result<EdgeConflictsResult>;

    async fn handle(
        &mut self,
        ListEdgeConflicts { limit, offset }: ListEdgeConflicts,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let (graph_version, conflicts) = futures::try_join!(
            async {
                self.graph
                    .ask(GetGraphVersion)
                    .await
                    .map_err(|err| anyhow!("{err:?}"))
            },
            async {
                self.graph
                    .ask(GetEdgeConflicts)
                    .await
                    .map_err(|err| anyhow!("{err:?}"))
            }
        )?;

        let mut conflicts = conflicts;
        let total = conflicts.len();
        let limit = limit.unwrap_or(50).clamp(1, 200);
        let offset = offset.unwrap_or(0).min(total);
        let end = (offset + limit).min(total);
        let has_more = end < total;
        let page = conflicts.drain(offset..end).collect();

        Ok(EdgeConflictsResult {
            graph_version,
            conflicts: page,
            offset,
            limit,
            has_more,
            total,
        })
    }
}

impl Message<ResolveEdgeConflictCommand> for DeduplicationAgent {
    type Reply = Result<EdgeConflictResolutionReport>;

    async fn handle(
        &mut self,
        ResolveEdgeConflictCommand {
            edge_id,
            resolved_kind,
            confidence,
            clear_conflicts,
            apply,
        }: ResolveEdgeConflictCommand,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let mut report = EdgeConflictResolutionReport {
            attempted:     0,
            resolved:      Vec::new(),
            failed:        Vec::new(),
            skipped:       0,
            dry_run:       !apply,
            graph_version: 0,
        };

        if !apply {
            report.skipped = 1;
            report.graph_version = self
                .graph
                .ask(GetGraphVersion)
                .await
                .map_err(|err| anyhow!("{err:?}"))?;
            log_edge_conflict_metrics(&self.rerun, &report);
            return Ok(report);
        }

        report.attempted = 1;
        let graph = self
            .graph
            .ask(GetGraphWithVersion)
            .await
            .map_err(|err| anyhow!("{err:?}"))?;

        let current_graph = graph.0;
        let before_version = graph.1;
        let idx = petgraph::stable_graph::EdgeIndex::new(edge_id as usize);
        let endpoints = current_graph
            .edge_endpoints(idx)
            .ok_or_else(|| anyhow!("edge_id {edge_id} not found"))?;
        let from_slug = current_graph[endpoints.0].slug.clone();
        let to_slug = current_graph[endpoints.1].slug.clone();

        match self
            .graph
            .ask(ResolveEdgeConflict {
                edge_id,
                resolved_kind,
                confidence,
                clear_conflicts,
            })
            .await
        {
            Ok(()) => {
                let graph_version = self
                    .graph
                    .ask(GetGraphVersion)
                    .await
                    .map_err(|err| anyhow!("{err:?}"))?;
                report.graph_version = graph_version;
                report.resolved.push(EdgeConflictResolved {
                    edge_id,
                    from_slug,
                    to_slug,
                });
                info!(
                    edge_id,
                    from = %report.resolved[0].from_slug,
                    to = %report.resolved[0].to_slug,
                    before_version,
                    graph_version,
                    "resolved edge conflict"
                );
            }
            Err(kameo::error::SendError::HandlerError(err)) => {
                report.graph_version = before_version;
                report.failed.push(EdgeConflictError {
                    edge_id,
                    from_slug,
                    to_slug,
                    message: err.to_string(),
                });
                warn!(
                    edge_id,
                    from = %report.failed[0].from_slug,
                    to = %report.failed[0].to_slug,
                    error = %err,
                    "failed to resolve edge conflict"
                );
            }
            Err(err) => {
                report.graph_version = before_version;
                report.failed.push(EdgeConflictError {
                    edge_id,
                    from_slug,
                    to_slug,
                    message: err.to_string(),
                });
                warn!(
                    edge_id,
                    from = %report.failed[0].from_slug,
                    to = %report.failed[0].to_slug,
                    error = ?err,
                    "resolve edge conflict send failed"
                );
            }
        }

        info!(
            edge_id,
            attempted = report.attempted,
            resolved = report.resolved.len(),
            failed = report.failed.len(),
            skipped = report.skipped,
            dry_run = report.dry_run,
            graph_version = report.graph_version,
            "edge conflict resolution completed"
        );

        log_edge_conflict_metrics(&self.rerun, &report);

        Ok(report)
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn log_scalar(rerun: &Option<ActorRef<RerunSink>>, path: impl Into<String>, value: f64) {
    if let Some(sink) = rerun {
        let msg = crate::rerun_sink::LogScalar {
            path: path.into(),
            value,
            time_ns: None,
        };
        let sink = sink.clone();
        task::spawn(async move {
            let _ = sink.tell(msg).await;
        });
    }
}

fn log_edge_conflict_metrics(
    rerun: &Option<ActorRef<RerunSink>>,
    report: &EdgeConflictResolutionReport,
) {
    log_scalar(rerun, "metrics/dedup/edge_conflicts/attempted", report.attempted as f64);
    log_scalar(rerun, "metrics/dedup/edge_conflicts/resolved", report.resolved.len() as f64);
    log_scalar(rerun, "metrics/dedup/edge_conflicts/failed", report.failed.len() as f64);
    log_scalar(rerun, "metrics/dedup/edge_conflicts/skipped", report.skipped as f64);
}

fn normalize_statement(statement: &str) -> String {
    statement
        .split_whitespace()
        .map(|part| part.to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}

fn node_similarity(lhs: &NodeEntry, rhs: &NodeEntry) -> f64 {
    jaro_winkler(&lhs.normalized_statement, &rhs.normalized_statement)
}

fn cluster_by_similarity(entries: &[NodeEntry], threshold: f64) -> Vec<Cluster> {
    let mut ordered = entries.to_vec();
    ordered.sort_by(|a, b| a.slug.cmp(&b.slug));

    if ordered.is_empty() {
        return Vec::new();
    }

    let mut dsu = DisjointSet::new(ordered.len());
    for (lhs_idx, lhs) in ordered.iter().enumerate() {
        for (rhs_idx, rhs) in ordered.iter().enumerate().skip(lhs_idx + 1) {
            if lhs.knowledge_type != rhs.knowledge_type {
                continue;
            }
            if node_similarity(lhs, rhs) >= threshold {
                dsu.union(lhs_idx, rhs_idx);
            }
        }
    }

    let mut buckets: HashMap<usize, Vec<NodeEntry>> = HashMap::new();
    for (idx, entry) in ordered.into_iter().enumerate() {
        let root = dsu.find(idx);
        buckets.entry(root).or_default().push(entry);
    }

    let mut clusters: Vec<Cluster> = buckets
        .into_values()
        .filter_map(|mut nodes| {
            if nodes.len() < 2 {
                return None;
            }
            nodes.sort_by(|a, b| a.slug.cmp(&b.slug));
            let confidence = cluster_confidence(&nodes);
            let canonical_slug = nodes
                .iter()
                .map(|n| n.slug.clone())
                .min()
                .unwrap_or_default();
            Some(Cluster {
                nodes,
                confidence,
                canonical_slug,
                rationale: String::new(),
            })
        })
        .collect();

    for cluster in &mut clusters {
        cluster.rationale = format!("strsim cluster (size {})", cluster.nodes.len());
    }

    clusters.sort_by(|a, b| a.canonical_slug.cmp(&b.canonical_slug));
    clusters
}

fn cluster_confidence(cluster: &[NodeEntry]) -> f64 {
    let mut best = 0.0;
    for (idx, lhs) in cluster.iter().enumerate() {
        for rhs in cluster.iter().skip(idx + 1) {
            let sim = node_similarity(lhs, rhs);
            if sim > best {
                best = sim;
            }
        }
    }
    best
}

fn cluster_can_auto_merge(graph: &graph::CurriculumGraph, cluster: &[NodeEntry]) -> bool {
    for node in cluster {
        if node.knowledge_type.is_learning_outcome() || node.knowledge_type.is_assessment_item() {
            return false;
        }
        let has_assesses = graph
            .edges_directed(node.id, Direction::Incoming)
            .chain(graph.edges_directed(node.id, Direction::Outgoing))
            .any(|edge| matches!(edge.weight().kind, EdgeKind::Assesses(_)));
        if has_assesses {
            return false;
        }
    }
    true
}

fn truncate_statement(statement: &str, max_len: usize) -> String {
    if statement.chars().count() <= max_len {
        return statement.to_string();
    }
    let mut truncated: String = statement.chars().take(max_len.saturating_sub(1)).collect();
    truncated.push('…');
    truncated
}

#[derive(Debug, Clone)]
struct NodeEntry {
    id:                   graph::NodeId,
    slug:                 String,
    title:                String,
    statement:            String,
    knowledge_type:       KnowledgeType,
    normalized_statement: String,
}

impl NodeEntry {
    fn new(
        id: graph::NodeId,
        slug: String,
        title: String,
        statement: String,
        knowledge_type: KnowledgeType,
    ) -> Self {
        Self {
            id,
            slug,
            title,
            statement: statement.clone(),
            knowledge_type,
            normalized_statement: normalize_statement(&statement),
        }
    }
}

#[derive(Debug, Clone)]
struct Cluster {
    nodes:          Vec<NodeEntry>,
    confidence:     f64,
    canonical_slug: String,
    rationale:      String,
}

impl Cluster {
    fn to_candidate(&self, preview_len: Option<usize>) -> MergeCandidate {
        let nodes = self
            .nodes
            .iter()
            .map(|node| NodeCandidate {
                slug:           node.slug.clone(),
                title:          node.title.clone(),
                statement:      preview_len
                    .map(|len| truncate_statement(&node.statement, len))
                    .unwrap_or_else(|| node.statement.clone()),
                knowledge_type: node.knowledge_type,
            })
            .collect();

        MergeCandidate {
            nodes,
            recommended_canonical: self.canonical_slug.clone(),
            confidence: self.confidence,
            rationale: self.rationale.clone(),
        }
    }
}

#[derive(Debug, Clone)]
struct DisjointSet {
    parent: Vec<usize>,
    rank:   Vec<usize>,
}

impl DisjointSet {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank:   vec![0; size],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            let root = self.find(self.parent[x]);
            self.parent[x] = root;
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let mut root_a = self.find(a);
        let mut root_b = self.find(b);
        if root_a == root_b {
            return;
        }
        let rank_a = self.rank[root_a];
        let rank_b = self.rank[root_b];
        if rank_a < rank_b || (rank_a == rank_b && root_a > root_b) {
            std::mem::swap(&mut root_a, &mut root_b);
        }
        self.parent[root_b] = root_a;
        if rank_a == rank_b {
            self.rank[root_a] = self.rank[root_a].saturating_add(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::types::KnowledgeType;

    fn entry(id: usize, slug: &str, statement: &str) -> NodeEntry {
        NodeEntry::new(
            graph::NodeId::new(id),
            slug.into(),
            slug.into(),
            statement.into(),
            KnowledgeType::Conceptual,
        )
    }

    #[test]
    fn clusters_union_transitive_similarities() {
        let a = entry(0, "c.a", "north south east");
        let b = entry(1, "c.b", "north central east");
        let c = entry(2, "c.c", "central west east");

        let sim_ab = node_similarity(&a, &b);
        let sim_bc = node_similarity(&b, &c);
        let sim_ac = node_similarity(&a, &c);

        let threshold = (sim_ab.min(sim_bc) - 0.01).max(0.0);
        assert!(
            sim_ab >= threshold && sim_bc >= threshold,
            "adjacent similarities should clear the threshold"
        );
        assert!(
            sim_ac < threshold,
            "non-adjacent similarity should fall below the threshold to exercise transitivity"
        );

        let clusters = cluster_by_similarity(&[a, b, c], threshold);
        assert_eq!(clusters.len(), 1, "transitive similarities should merge");
        let cluster = &clusters[0];
        assert_eq!(cluster.nodes.len(), 3);
        assert_eq!(cluster.canonical_slug, "c.a");
    }

    #[test]
    fn clusters_are_deterministic_and_sorted() {
        let entries = vec![
            entry(0, "c.delta", "second cluster value"),
            entry(1, "c.charlie", "second cluster value"),
            entry(2, "c.alpha", "shared one"),
            entry(3, "c.bravo", "shared one"),
        ];
        let clusters = cluster_by_similarity(&entries, 0.95);
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].canonical_slug, "c.alpha");
        assert_eq!(clusters[1].canonical_slug, "c.charlie");
        let first_nodes: Vec<_> = clusters[0].nodes.iter().map(|n| n.slug.as_str()).collect();
        assert_eq!(first_nodes, vec!["c.alpha", "c.bravo"]);
    }
}
