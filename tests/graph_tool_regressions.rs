use std::{path::PathBuf, sync::Arc};

use kameo::prelude::*;
use serde_json::json;
use tokio_util::sync::CancellationToken;
use weaver::{
    agents::deduplication::DeduplicationAgent,
    file_reader::AgentMode,
    graph::{
        GraphService, KnowledgeNode, RequiresAttrs, RequiresSpec,
        manager::{GetGraphMeta, GraphManager, GraphManagerState},
    },
    llm_gateway::{GatewayMetrics, LLMGateway, LLMGatewayState},
    schema::types::{KnowledgeType, SourceRef, Strength},
    tools::llm::{self, CallState},
};

fn source_ref() -> SourceRef {
    SourceRef {
        path:       "dummy".into(),
        start_line: 1,
        end_line:   1,
        revision:   "deadbeef".into(),
    }
}

fn knowledge(title: impl Into<String>) -> KnowledgeNode {
    let title = title.into();
    KnowledgeNode {
        title: title.clone(),
        statement: title,
        knowledge_type: KnowledgeType::Conceptual,
        source_refs: vec![source_ref()],
        confidence: 1.0,
        rubric_criteria: vec![],
        construct_irrelevant_demands: vec![],
        grain_level: None,
        intrinsic_load: None,
        introduction_scope: weaver::graph::IntroductionScope::InCourse,
    }
}

fn spawn_tool_state(
    svc: GraphService,
) -> (
    CallState,
    ActorRef<GraphManager>,
    ActorRef<DeduplicationAgent>,
    ActorRef<LLMGateway>,
) {
    let state = GraphManagerState::new(
        svc.snapshot_graph_owned(),
        "deadbeef".into(),
        false,
        svc.graph_version(),
        2_000,
        false,
    );
    let graph_actor = GraphManager::spawn(state);
    let dedup_actor = DeduplicationAgent::spawn(DeduplicationAgent::new(graph_actor.clone(), None));
    let gateway_instance =
        LLMGateway::from(LLMGatewayState::new(GatewayMetrics::default().to_state()));
    let metrics = gateway_instance.metrics();
    let gateway_actor = LLMGateway::spawn(gateway_instance);
    let call_state = CallState {
        depth: 0,
        max_subdelegations: weaver::constants::DEFAULT_MAX_SUBDELEGATIONS,
        workspace_root: Arc::new(PathBuf::from(".")),
        gateway: gateway_actor.clone(),
        model: Arc::new("test-model".into()),
        metrics,
        graph: graph_actor.clone(),
        dedup: dedup_actor.clone(),
        actor_name: Arc::new("test-actor".into()),
        conversation_id: Arc::new("test-convo".into()),
        rerun: None,
        analysis_cache: Arc::new(
            weaver::tools::llm::graph_tools::analysis_cache::AnalysisCache::new(),
        ),
        mode: AgentMode::Interactive,
        course_commit: Arc::new("deadbeef".into()),
        cancellation: CancellationToken::new(),
    };
    (call_state, graph_actor, dedup_actor, gateway_actor)
}

async fn stop_actors(
    graph: ActorRef<GraphManager>,
    dedup: ActorRef<DeduplicationAgent>,
    gateway: ActorRef<LLMGateway>,
) {
    let _ = dedup.stop_gracefully().await;
    let _ = gateway.stop_gracefully().await;
    let _ = graph.stop_gracefully().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn apply_graph_tool_response_reports_post_mutation_graph_version() -> anyhow::Result<()> {
    let svc = GraphService::new();
    let (call_state, graph, dedup, gateway) = spawn_tool_state(svc);
    let tool = llm::lookup_tool("graph_insert_knowledge")?.expect("tool registered");
    let instance = (tool.parse)(
        json!({
            "slug": "c.meta",
            "title": "Meta",
            "statement": "Meta",
            "knowledge_type": "conceptual",
            "source_refs": [{"path": "dummy", "start_line": 1, "end_line": 1}],
            "apply": true
        }),
        &call_state,
    )?;

    let output = instance.execute().await?;
    let response_version = output.payload["meta"]["graph_version"]
        .as_u64()
        .expect("response graph version");
    let meta = graph.ask(GetGraphMeta).await?.expect("get graph meta");

    assert_eq!(response_version, meta.graph_version);
    stop_actors(graph, dedup, gateway).await;
    Ok(())
}

fn keystone_service(node_count: usize) -> anyhow::Result<GraphService> {
    let mut svc = GraphService::new();
    for i in 0..node_count {
        svc.add_knowledge_node(format!("c.k{i}"), knowledge(format!("K{i}")), vec![])?;
    }
    for i in 0..node_count.saturating_sub(1) {
        let from = svc.node_by_slug(&format!("c.k{i}"))?;
        let to = svc.node_by_slug(&format!("c.k{}", i + 1))?;
        svc.add_edge::<RequiresSpec>(
            from,
            to,
            RequiresAttrs {
                strength:      Strength::Necessary,
                rationale:     "chain".into(),
                evidence_refs: vec![source_ref()],
            },
            1.0,
        )?;
    }
    Ok(svc)
}

#[tokio::test(flavor = "multi_thread")]
async fn graph_keystone_accepts_no_args_with_default_body_limit() -> anyhow::Result<()> {
    let (call_state, graph, dedup, gateway) = spawn_tool_state(keystone_service(25)?);
    let tool = llm::lookup_tool("graph_keystone")?.expect("tool registered");
    let instance = (tool.parse)(json!({}), &call_state)?;

    let output = instance.execute().await?;

    assert_eq!(output.payload["scores"].as_array().unwrap().len(), 20);
    assert_eq!(output.payload["limit"].as_u64(), Some(20));
    assert_eq!(output.payload["has_more"].as_bool(), Some(true));
    stop_actors(graph, dedup, gateway).await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn graph_keystone_accepts_fetch_body_and_limit() -> anyhow::Result<()> {
    let (call_state, graph, dedup, gateway) = spawn_tool_state(keystone_service(25)?);
    let tool = llm::lookup_tool("graph_keystone")?.expect("tool registered");
    let instance = (tool.parse)(json!({"fetch_body": true, "limit": 10}), &call_state)?;

    let output = instance.execute().await?;

    assert_eq!(output.payload["scores"].as_array().unwrap().len(), 10);
    assert_eq!(output.payload["limit"].as_u64(), Some(10));
    assert_eq!(output.payload["offset"].as_u64(), Some(0));
    assert_eq!(output.payload["has_more"].as_bool(), Some(true));
    stop_actors(graph, dedup, gateway).await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn graph_keystone_rejects_unknown_fields() -> anyhow::Result<()> {
    let (call_state, graph, dedup, gateway) = spawn_tool_state(GraphService::new());
    let tool = llm::lookup_tool("graph_keystone")?.expect("tool registered");

    assert!((tool.parse)(json!({"bogus": true}), &call_state).is_err());
    stop_actors(graph, dedup, gateway).await;
    Ok(())
}
