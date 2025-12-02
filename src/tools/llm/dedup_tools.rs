use anyhow::anyhow;
use async_trait::async_trait;
use bon::Builder;
use kameo::{error::SendError, prelude::ActorRef};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    agents::deduplication::{
        DeduplicationAgent, EdgeConflictResolutionReport, EdgeConflictsResult, ListEdgeConflicts,
        ListMergeCandidates, MergeCandidate, MergeCandidatesResult, MergeNodesCommand,
        MergeOperationResult, MergeStatus, ResolveEdgeConflictCommand,
    },
    graph::commands::EdgeConflictView,
    tools::llm::{
        CallState, ToolExecutionError, ToolInstance, ToolOutput, ToolPayloadMode, ToolPrototype,
        common::{ToolRunPayload, ToolRunner},
        payload_size_bytes, require_string,
    },
};

const DEDUP_LIST_EDGE_CONFLICTS: &str = "dedup_list_edge_conflicts";
const DEDUP_RESOLVE_EDGE_CONFLICT: &str = "dedup_resolve_edge_conflict";
const DEDUP_MERGE_NODES: &str = "dedup_merge_nodes";
const DEDUP_LIST_NODE_CANDIDATES: &str = "dedup_list_node_candidates";

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ListNodeCandidatesArgs {
    #[serde(default)]
    #[schemars(description = "Maximum merge-candidate clusters to return (default 25, max 200).")]
    pub limit:      Option<usize>,
    #[serde(default)]
    #[schemars(description = "Offset into the candidate list (default 0).")]
    pub offset:     Option<usize>,
    #[serde(default = "crate::tools::llm::default_false")]
    #[schemars(
        description = "Set fetch_body=true to include full statements; default returns previews \
                       to conserve tokens."
    )]
    pub fetch_body: bool,
}

crate::analysis_tool!(
    dedup_list_node_candidates_meta,
    id: DEDUP_LIST_NODE_CANDIDATES,
    description: "List clustered node merge candidates discovered by the deduplication agent.",
    args: ListNodeCandidatesArgs,
    prepare: |raw| crate::tools::llm::graph_tools::common::parse_args_with_builder(
        DEDUP_LIST_NODE_CANDIDATES,
        raw,
        |mut input: ListNodeCandidatesArgs| {
            input.limit = input.limit.or(Some(25));
            Ok(input)
        },
    ),
    runner: |args: ListNodeCandidatesArgs, state: &CallState| ListNodeCandidatesTool {
        args,
        state: state.clone(),
    }
);

#[derive(Debug, Clone, Builder, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ListEdgeConflictsArgs {
    #[serde(default)]
    #[schemars(description = "Maximum conflicts to return (default 50, max 200).")]
    pub limit:  Option<usize>,
    #[serde(default)]
    #[schemars(description = "Offset into the conflict list (default 0).")]
    pub offset: Option<usize>,
}

crate::analysis_tool!(
    dedup_list_edge_conflicts_meta,
    id: DEDUP_LIST_EDGE_CONFLICTS,
    description: "List edges carrying queued conflict payloads (for deduplication).",
    args: ListEdgeConflictsArgs,
    prepare: |raw| crate::tools::llm::graph_tools::common::parse_args_with_builder(
        DEDUP_LIST_EDGE_CONFLICTS,
        raw,
        |input: ListEdgeConflictsArgs| Ok(input),
    ),
    runner: |args: ListEdgeConflictsArgs, state: &CallState| ListEdgeConflictsTool {
        args,
        state: state.clone(),
    }
);

struct ListNodeCandidatesTool {
    args:  ListNodeCandidatesArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for ListNodeCandidatesTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let dedup: &ActorRef<DeduplicationAgent> = &self.state.dedup;
        let result: MergeCandidatesResult = dedup
            .ask(ListMergeCandidates {
                include_statements: self.args.fetch_body,
            })
            .await
            .map_err(|err| map_dedup_err(err, DEDUP_LIST_NODE_CANDIDATES))?;

        let candidates = result.candidates;
        let total = candidates.len();
        let (page, page_meta) = crate::paginate!(candidates, self.args.limit, self.args.offset);
        let truncated = !self.args.fetch_body;
        let preview_candidates: Vec<_> = page.iter().map(render_candidate_preview).collect();
        let rendered: Vec<_> = page
            .iter()
            .map(|candidate| render_candidate(candidate, self.args.fetch_body, truncated))
            .collect();
        let preview = json!({
            "type": "graph_view",
            "tool": DEDUP_LIST_NODE_CANDIDATES,
            "graph_version": result.graph_version,
            "total_clusters": total,
            "preview_only": !self.args.fetch_body,
            "offset": page_meta.offset,
            "limit": page_meta.limit,
            "has_more": page_meta.has_more,
            "candidates": preview_candidates,
        });
        let body = json!({
            "type": "graph_view",
            "tool": DEDUP_LIST_NODE_CANDIDATES,
            "graph_version": result.graph_version,
            "total_clusters": total,
            "offset": page_meta.offset,
            "limit": page_meta.limit,
            "has_more": page_meta.has_more,
            "candidates": rendered,
        });

        let mode = ToolPayloadMode::from_fetch_flag(self.args.fetch_body);
        let byte_estimate = payload_size_bytes(&body);
        ToolRunner::new(DEDUP_LIST_NODE_CANDIDATES, &self.state)
            .with_mode(mode)
            .with_meta(super::common::graph_meta(&self.state.graph).await?)
            .hint("Set fetch_body=true to include full statements.")
            .run(move |mode| async move {
                let mut payload = ToolRunPayload {
                    body:          body.clone(),
                    approx_bytes:  Some(byte_estimate),
                    preview:       None,
                    preview_hints: Vec::new(),
                    page:          Some(page_meta),
                };
                if mode == ToolPayloadMode::Preview {
                    payload.preview = Some(preview.clone());
                }
                Ok(payload)
            })
            .await
    }
}

struct ListEdgeConflictsTool {
    args:  ListEdgeConflictsArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for ListEdgeConflictsTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let dedup: &ActorRef<DeduplicationAgent> = &self.state.dedup;
        let result: EdgeConflictsResult = dedup
            .ask(ListEdgeConflicts {
                limit:  self.args.limit,
                offset: self.args.offset,
            })
            .await
            .map_err(|err| map_dedup_err(err, DEDUP_LIST_EDGE_CONFLICTS))?;

        let rendered: Vec<_> = result.conflicts.iter().map(render_edge_conflict).collect();

        ToolRunner::new(DEDUP_LIST_EDGE_CONFLICTS, &self.state)
            .with_mode(ToolPayloadMode::Body)
            .with_meta(super::common::graph_meta(&self.state.graph).await?)
            .run(move |_| async move {
                Ok(ToolRunPayload {
                    body:          json!({
                        "type": "graph_view",
                        "tool": DEDUP_LIST_EDGE_CONFLICTS,
                        "graph_version": result.graph_version,
                        "offset": result.offset,
                        "limit": result.limit,
                        "has_more": result.has_more,
                        "total": result.total,
                        "conflicts": rendered,
                    }),
                    approx_bytes:  None,
                    preview:       None,
                    preview_hints: Vec::new(),
                    page:          Some(crate::tools::llm::common::Page {
                        offset:   result.offset,
                        limit:    result.limit,
                        has_more: result.has_more,
                    }),
                })
            })
            .await
    }
}

#[derive(Debug, Clone, Builder, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResolveEdgeConflictArgs {
    #[schemars(description = "Edge id (index) of the conflict to resolve.")]
    pub edge_id:         u32,
    #[schemars(description = "Merged edge payload to apply.")]
    pub resolved_kind:   crate::graph::EdgeKind,
    #[serde(default)]
    #[schemars(
        description = "Optional confidence to set on the merged edge (defaults to existing)."
    )]
    pub confidence:      Option<f32>,
    #[serde(default = "crate::tools::llm::default_false")]
    #[schemars(description = "Set apply=true to commit; default false previews the change.")]
    pub apply:           bool,
    #[serde(default = "default_clear_conflicts")]
    #[schemars(description = "Whether to clear queued conflicts after applying the resolution.")]
    pub clear_conflicts: bool,
}

const fn default_clear_conflicts() -> bool {
    true
}

crate::analysis_tool!(
    dedup_resolve_edge_conflict_meta,
    id: DEDUP_RESOLVE_EDGE_CONFLICT,
    description: "Resolve a conflicting edge payload by supplying the merged EdgeKind payload. Clears queued conflicts when apply=true (default).",
    args: ResolveEdgeConflictArgs,
    prepare: |raw| crate::tools::llm::graph_tools::common::parse_args_with_builder(
        DEDUP_RESOLVE_EDGE_CONFLICT,
        raw,
        |input: ResolveEdgeConflictArgs| Ok(input),
    ),
    runner: |args: ResolveEdgeConflictArgs, state: &CallState| ResolveEdgeConflictTool {
        args,
        state: state.clone(),
    }
);

struct ResolveEdgeConflictTool {
    args:  ResolveEdgeConflictArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for ResolveEdgeConflictTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let mode = ToolPayloadMode::from_fetch_flag(self.args.apply);
        let dedup = self.state.dedup.clone();
        let args = self.args.clone();

        ToolRunner::new(DEDUP_RESOLVE_EDGE_CONFLICT, &self.state)
            .with_mode(mode)
            .with_meta(super::common::graph_meta(&self.state.graph).await?)
            .hint("Set apply=true to execute this mutation.")
            .run(move |mode| {
                let args = args.clone();
                let dedup = dedup.clone();
                async move {
                    let outcome = dedup
                        .ask(ResolveEdgeConflictCommand {
                            edge_id:         args.edge_id,
                            resolved_kind:   args.resolved_kind.clone(),
                            confidence:      args.confidence,
                            clear_conflicts: args.clear_conflicts,
                            apply:           args.apply,
                        })
                        .await
                        .map_err(|err| map_dedup_err(err, DEDUP_RESOLVE_EDGE_CONFLICT))?;

                    let mut payload = ToolRunPayload::new(render_edge_resolution_body(
                        args.edge_id,
                        &outcome,
                        args.clear_conflicts,
                    ));
                    if matches!(mode, ToolPayloadMode::Preview) {
                        payload.preview =
                            Some(render_edge_resolution_preview(args.edge_id, &outcome));
                    }
                    Ok(payload)
                }
            })
            .await
    }
}

#[derive(Debug, Clone, Builder, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MergeNodesArgs {
    #[schemars(description = "Canonical node slug to merge into.")]
    #[builder(with = |value: String| -> crate::tools::llm::ToolInputResult<_> {
        require_string(value, DEDUP_MERGE_NODES, "canonical")
    })]
    pub canonical: String,
    #[schemars(description = "Duplicate node slug to merge from.")]
    #[builder(with = |value: String| -> crate::tools::llm::ToolInputResult<_> {
        require_string(value, DEDUP_MERGE_NODES, "duplicate")
    })]
    pub duplicate: String,
    #[serde(default = "crate::tools::llm::default_false")]
    #[schemars(description = "Set apply=true to perform the merge; otherwise returns a preview.")]
    pub apply:     bool,
}

crate::analysis_tool!(
    dedup_merge_nodes_meta,
    id: DEDUP_MERGE_NODES,
    description: "Merge a duplicate node into a canonical node (deduplication helper).",
    args: MergeNodesArgs,
    prepare: |raw| crate::tools::llm::graph_tools::common::parse_args_with_builder(
        DEDUP_MERGE_NODES,
        raw,
        |mut input: MergeNodesArgs| {
            input.canonical = require_string(input.canonical, DEDUP_MERGE_NODES, "canonical")?;
            input.duplicate = require_string(input.duplicate, DEDUP_MERGE_NODES, "duplicate")?;
            Ok(input)
        },
    ),
    runner: |args: MergeNodesArgs, state: &CallState| MergeNodesTool {
        args,
        state: state.clone(),
    }
);

struct MergeNodesTool {
    args:  MergeNodesArgs,
    state: CallState,
}

#[async_trait]
impl ToolInstance for MergeNodesTool {
    async fn execute(&self) -> Result<ToolOutput, ToolExecutionError> {
        let meta = super::common::graph_meta(&self.state.graph).await?;
        let dedup = self.state.dedup.clone();
        let args = self.args.clone();
        let mode = ToolPayloadMode::from_fetch_flag(self.args.apply);

        ToolRunner::new(DEDUP_MERGE_NODES, &self.state)
            .with_mode(mode)
            .with_meta(meta)
            .hint("Set apply=true to execute this mutation.")
            .run(move |mode| {
                let args = args.clone();
                let dedup = dedup.clone();
                async move {
                    let outcome: MergeOperationResult = dedup
                        .ask(MergeNodesCommand {
                            canonical: args.canonical.clone(),
                            duplicate: args.duplicate.clone(),
                            apply:     args.apply,
                        })
                        .await
                        .map_err(|err| map_dedup_err(err, DEDUP_MERGE_NODES))?;

                    let mut payload = ToolRunPayload::new(render_merge_body(&outcome));
                    if matches!(mode, ToolPayloadMode::Preview) {
                        payload.preview = Some(render_merge_preview(&outcome));
                    }
                    Ok(payload)
                }
            })
            .await
    }
}

pub fn tool_prototypes() -> Vec<ToolPrototype> {
    vec![
        dedup_list_node_candidates_meta(),
        dedup_list_edge_conflicts_meta(),
        dedup_resolve_edge_conflict_meta(),
        dedup_merge_nodes_meta(),
    ]
}

fn render_candidate_preview(candidate: &MergeCandidate) -> serde_json::Value {
    json!({
        "recommended_canonical": candidate.recommended_canonical,
        "confidence": candidate.confidence,
        "rationale": candidate.rationale,
        "node_count": candidate.nodes.len(),
        "nodes": candidate.nodes.iter().map(|node| json!({
            "slug": node.slug,
            "title": node.title,
            "knowledge_type": node.knowledge_type,
        })).collect::<Vec<_>>(),
    })
}

fn render_candidate(
    candidate: &MergeCandidate,
    include_statement: bool,
    truncated: bool,
) -> serde_json::Value {
    let nodes: Vec<_> = candidate
        .nodes
        .iter()
        .map(|node| {
            let mut rendered = json!({
                "slug": node.slug,
                "title": node.title,
                "knowledge_type": node.knowledge_type,
            });
            if let Some(obj) = rendered.as_object_mut() {
                if include_statement {
                    obj.insert("statement".to_string(), json!(node.statement));
                } else if !node.statement.is_empty() {
                    obj.insert("statement_preview".to_string(), json!(node.statement));
                }
                if truncated {
                    obj.insert("statement_truncated".to_string(), json!(!include_statement));
                }
            }
            rendered
        })
        .collect();

    json!({
        "recommended_canonical": candidate.recommended_canonical,
        "confidence": candidate.confidence,
        "rationale": candidate.rationale,
        "nodes": nodes,
    })
}

fn render_merge_preview(outcome: &MergeOperationResult) -> serde_json::Value {
    json!({
        "type": "graph_command",
        "tool": DEDUP_MERGE_NODES,
        "status": "preview",
        "canonical": outcome.canonical,
        "duplicate": outcome.duplicate,
        "graph_version": outcome.graph_version,
        "hint": "Set apply=true to execute this mutation.",
    })
}

fn render_merge_body(outcome: &MergeOperationResult) -> serde_json::Value {
    match (&outcome.status, &outcome.record) {
        (MergeStatus::Applied, Some(record)) => json!({
            "type": "graph_command",
            "tool": DEDUP_MERGE_NODES,
            "status": "ok",
            "merged_into": record.merged_into,
            "merged_from": record.merged_from,
            "edges_redirected": record.edges_redirected,
            "timestamp": record.timestamp,
            "graph_version": outcome.graph_version,
        }),
        _ => render_merge_preview(outcome),
    }
}

fn render_edge_conflict(entry: &EdgeConflictView) -> serde_json::Value {
    json!({
        "edge_id": entry.edge_id,
        "from_slug": entry.from_slug,
        "to_slug": entry.to_slug,
        "kind": entry.kind,
        "confidence": entry.confidence,
        "conflicts": entry.conflicts,
    })
}

fn render_edge_resolution_preview(
    edge_id: u32,
    report: &EdgeConflictResolutionReport,
) -> serde_json::Value {
    json!({
        "type": "graph_command",
        "tool": DEDUP_RESOLVE_EDGE_CONFLICT,
        "status": "preview",
        "edge_id": edge_id,
        "graph_version": report.graph_version,
        "attempted": report.attempted,
        "resolved": report.resolved,
        "failed": report.failed,
        "skipped": report.skipped,
        "dry_run": report.dry_run,
        "hint": "Set apply=true to execute this mutation."
    })
}

fn render_edge_resolution_body(
    edge_id: u32,
    report: &EdgeConflictResolutionReport,
    cleared: bool,
) -> serde_json::Value {
    let status = if !report.failed.is_empty() {
        "error"
    } else if !report.resolved.is_empty() {
        "ok"
    } else if report.dry_run {
        "preview"
    } else {
        "skipped"
    };

    json!({
        "type": "graph_command",
        "tool": DEDUP_RESOLVE_EDGE_CONFLICT,
        "status": status,
        "edge_id": edge_id,
        "graph_version": report.graph_version,
        "attempted": report.attempted,
        "resolved": report.resolved,
        "failed": report.failed,
        "skipped": report.skipped,
        "dry_run": report.dry_run,
        "cleared_conflicts": cleared,
    })
}

fn map_dedup_err<A, E: std::fmt::Debug>(
    err: SendError<A, E>,
    tool: &'static str,
) -> ToolExecutionError {
    match err {
        SendError::HandlerError(e) => ToolExecutionError::Execution(anyhow!("{e:?}")),
        other => ToolExecutionError::Internal(anyhow!("{tool} send error: {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Arc};

    use kameo::{actor::Spawn, prelude::ActorRef};

    use super::*;
    use crate::{
        constants::DEFAULT_MAX_SUBDELEGATIONS,
        file_reader::AgentMode,
        graph::{
            CaseTag, EdgeKind, IntroductionScope, KnowledgeNode, SupportsAttrs, SupportsSpec,
            commands::{GetEdgeConflicts, GetGraphVersion, ResolveSlug},
            manager::{GraphManager, GraphManagerState},
            service::GraphService,
        },
        llm_gateway::{GatewayMetrics, LLMGateway, LLMGatewayState},
        schema::types::{IntendedEffect, SourceRef, SupportKind},
        tools::llm::analysis_cache::AnalysisCache,
    };

    fn sample_node(title: &str, statement: &str) -> KnowledgeNode {
        KnowledgeNode {
            title: title.into(),
            statement: statement.into(),
            knowledge_type: crate::schema::types::KnowledgeType::Conceptual,
            source_refs: vec![crate::schema::types::SourceRef {
                path:       "dummy".into(),
                start_line: 1,
                end_line:   1,
                revision:   "deadbeef".into(),
            }],
            confidence: 1.0,
            rubric_criteria: vec![],
            construct_irrelevant_demands: vec![],
            grain_level: None,
            intrinsic_load: None,
            introduction_scope: IntroductionScope::InCourse,
        }
    }

    async fn setup_call_state(
        nodes: &[(&str, &str)],
    ) -> (
        CallState,
        ActorRef<GraphManager>,
        ActorRef<DeduplicationAgent>,
        ActorRef<LLMGateway>,
    ) {
        setup_call_state_with(nodes, |_| {}).await
    }

    async fn setup_call_state_with<F>(
        nodes: &[(&str, &str)],
        build: F,
    ) -> (
        CallState,
        ActorRef<GraphManager>,
        ActorRef<DeduplicationAgent>,
        ActorRef<LLMGateway>,
    )
    where
        F: FnOnce(&mut GraphService),
    {
        let mut svc = GraphService::new();
        for (idx, (title, statement)) in nodes.iter().enumerate() {
            let slug = format!("c.node_{idx}");
            svc.add_knowledge_node(slug, sample_node(title, statement), vec![])
                .expect("insert knowledge");
        }
        build(&mut svc);

        let state = GraphManagerState::new(
            svc.snapshot_graph_owned(),
            "deadbeef".into(),
            false,
            svc.graph_version(),
            2_000,
            false,
        );
        let graph_actor = GraphManager::spawn(state);
        let dedup_actor =
            DeduplicationAgent::spawn(DeduplicationAgent::new(graph_actor.clone(), None));

        let gateway_instance =
            LLMGateway::from(LLMGatewayState::new(GatewayMetrics::default().to_state()));
        let metrics = gateway_instance.metrics();
        let gateway_actor = LLMGateway::spawn(gateway_instance);

        let call_state = CallState {
            depth: 0,
            max_subdelegations: DEFAULT_MAX_SUBDELEGATIONS,
            workspace_root: Arc::new(PathBuf::from(".")),
            gateway: gateway_actor.clone(),
            model: Arc::new("test-model".into()),
            metrics,
            graph: graph_actor.clone(),
            dedup: dedup_actor.clone(),
            actor_name: Arc::new("test-actor".into()),
            conversation_id: Arc::new("test-convo".into()),
            rerun: None,
            analysis_cache: Arc::new(AnalysisCache::new()),
            mode: AgentMode::Interactive,
            course_commit: Arc::new("deadbeef".into()),
        };

        (call_state, graph_actor, dedup_actor, gateway_actor)
    }

    async fn shutdown(
        graph: ActorRef<GraphManager>,
        dedup: ActorRef<DeduplicationAgent>,
        gateway: ActorRef<LLMGateway>,
    ) {
        let _ = dedup.stop_gracefully().await;
        let _ = graph.stop_gracefully().await;
        let _ = gateway.stop_gracefully().await;
    }

    #[tokio::test]
    async fn list_candidates_preview_and_body() -> anyhow::Result<()> {
        let (state, graph, dedup, gateway) =
            setup_call_state(&[("A", "Loops repeat work"), ("B", "Loop repeats work")]).await;

        let preview_tool = ListNodeCandidatesTool {
            args:  ListNodeCandidatesArgs {
                limit:      None,
                offset:     None,
                fetch_body: false,
            },
            state: state.clone(),
        };
        let preview = preview_tool
            .execute()
            .await
            .expect("candidate preview tool should succeed");
        let candidates = preview
            .payload
            .get("candidates")
            .and_then(|c| c.as_array())
            .expect("candidates array");
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0]["node_count"], json!(2));
        let first_node = candidates[0]["nodes"][0]
            .as_object()
            .expect("node preview object");
        assert!(!first_node.contains_key("statement"), "preview should omit full statements");

        let body_tool = ListNodeCandidatesTool {
            args: ListNodeCandidatesArgs {
                limit:      None,
                offset:     None,
                fetch_body: true,
            },
            state,
        };
        let body = body_tool
            .execute()
            .await
            .expect("candidate body tool should succeed");
        let body_candidates = body
            .payload
            .get("candidates")
            .and_then(|c| c.as_array())
            .expect("body candidates array");
        let rendered_node = body_candidates[0]["nodes"][0]
            .as_object()
            .expect("node body");
        assert!(rendered_node.get("statement").is_some(), "body should include statements");

        shutdown(graph, dedup, gateway).await;
        Ok(())
    }

    #[tokio::test]
    async fn merge_nodes_tool_preview_and_apply() -> anyhow::Result<()> {
        let (state, graph, dedup, gateway) =
            setup_call_state(&[("A", "Loops repeat work"), ("B", "Loop repeats work")]).await;
        let canonical = "c.node_0".to_string();
        let duplicate = "c.node_1".to_string();
        let before_version: u64 = graph.ask(GetGraphVersion).await?;

        let preview_tool = MergeNodesTool {
            args:  MergeNodesArgs {
                canonical: canonical.clone(),
                duplicate: duplicate.clone(),
                apply:     false,
            },
            state: state.clone(),
        };
        let preview = preview_tool
            .execute()
            .await
            .expect("merge preview tool should succeed");
        assert_eq!(preview.payload["status"], json!("preview"));
        let duplicate_exists = graph
            .ask(ResolveSlug {
                slug: duplicate.clone(),
            })
            .await
            .is_ok();
        assert!(duplicate_exists, "preview should not mutate the graph");

        let apply_tool = MergeNodesTool {
            args: MergeNodesArgs {
                canonical: canonical.clone(),
                duplicate: duplicate.clone(),
                apply:     true,
            },
            state,
        };
        let applied = apply_tool
            .execute()
            .await
            .expect("merge apply tool should succeed");
        assert_eq!(applied.payload["status"], json!("ok"));

        let after_merge_version: u64 = graph.ask(GetGraphVersion).await?;
        assert!(
            graph
                .ask(ResolveSlug {
                    slug: duplicate.clone(),
                })
                .await
                .is_err(),
            "duplicate should be removed after merge"
        );
        assert!(
            graph
                .ask(ResolveSlug {
                    slug: canonical.clone(),
                })
                .await
                .is_ok(),
            "canonical should remain after merge"
        );
        assert!(after_merge_version > before_version, "merge should bump graph version");

        shutdown(graph, dedup, gateway).await;
        Ok(())
    }

    #[tokio::test]
    async fn resolve_edge_conflict_via_agent_tools() -> anyhow::Result<()> {
        let revision = "deadbeef";
        let evidence = SourceRef {
            path:       "dummy".into(),
            start_line: 1,
            end_line:   1,
            revision:   revision.into(),
        };
        let attrs_a = SupportsAttrs {
            support_kind:    SupportKind::WorkedExample,
            intended_effect: IntendedEffect::Motivate,
            case_tag:        Some(CaseTag::Typical),
            coverage_tags:   vec![],
            evidence_refs:   vec![evidence.clone()],
        };
        let attrs_b = SupportsAttrs {
            support_kind:    SupportKind::Analogy,
            intended_effect: IntendedEffect::Motivate,
            case_tag:        Some(CaseTag::Edge),
            coverage_tags:   vec![],
            evidence_refs:   vec![evidence],
        };

        let (state, graph, dedup, gateway) =
            setup_call_state_with(&[("Alpha", "Alpha concept"), ("Beta", "Beta concept")], |svc| {
                let from = svc.node_by_slug("c.node_0").unwrap();
                let to = svc.node_by_slug("c.node_1").unwrap();
                svc.add_edge::<SupportsSpec>(from, to, attrs_a.clone(), 1.0)
                    .unwrap();
                svc.add_edge::<SupportsSpec>(from, to, attrs_b.clone(), 0.9)
                    .unwrap();
            })
            .await;

        let list_tool = ListEdgeConflictsTool {
            args:  ListEdgeConflictsArgs {
                limit:  None,
                offset: None,
            },
            state: state.clone(),
        };
        let listed = list_tool
            .execute()
            .await
            .expect("list edge conflicts should succeed");
        let conflicts = listed
            .payload
            .get("conflicts")
            .and_then(|c| c.as_array())
            .expect("conflicts array");
        assert_eq!(conflicts.len(), 1, "should surface one conflict");
        let edge_id = conflicts[0]["edge_id"].as_u64().unwrap() as u32;

        let preview_tool = ResolveEdgeConflictTool {
            args:  ResolveEdgeConflictArgs {
                edge_id,
                resolved_kind: EdgeKind::Supports(attrs_a.clone()),
                confidence: None,
                apply: false,
                clear_conflicts: true,
            },
            state: state.clone(),
        };
        let preview = preview_tool
            .execute()
            .await
            .expect("preview resolve should succeed");
        assert_eq!(preview.payload["status"], json!("preview"));
        let remaining = graph.ask(GetEdgeConflicts).await?;
        assert_eq!(remaining.len(), 1, "preview should not mutate conflicts");

        let apply_tool = ResolveEdgeConflictTool {
            args: ResolveEdgeConflictArgs {
                edge_id,
                resolved_kind: EdgeKind::Supports(attrs_a.clone()),
                confidence: Some(1.0),
                apply: true,
                clear_conflicts: true,
            },
            state,
        };
        let applied = apply_tool
            .execute()
            .await
            .expect("apply resolve should succeed");
        assert_eq!(applied.payload["status"], json!("ok"));
        assert_eq!(applied.payload["attempted"], json!(1));
        assert_eq!(applied.payload["failed"], json!([]));
        let after_conflicts = graph.ask(GetEdgeConflicts).await?;
        assert!(after_conflicts.is_empty(), "apply should clear edge conflicts");

        shutdown(graph, dedup, gateway).await;
        Ok(())
    }
}
