use std::fs;

use kameo_persistence::PersistentActor;
use url::Url;
use uuid::Uuid;
use weaver::llm_gateway::{
    GatewayMetrics, GetGatewayMetrics, LLMGateway, LLMGatewayState, PersistGatewaySnapshot,
};

#[tokio::test(flavor = "multi_thread")]
async fn llm_gateway_persists_metrics() -> anyhow::Result<()> {
    let state_dir = std::env::temp_dir().join(format!("weaver-gateway-state-{}", Uuid::new_v4()));
    fs::create_dir_all(&state_dir)?;
    let state_url = Url::from_directory_path(&state_dir)
        .map_err(|_| anyhow::anyhow!("invalid gateway state url"))?;

    let gateway = LLMGateway::from(LLMGatewayState::new(GatewayMetrics::default().to_state()));

    let actor = LLMGateway::spawn_persistent(state_url.clone(), gateway).await?;
    actor.ask(PersistGatewaySnapshot).await?;
    actor.stop_gracefully().await.expect("stop gateway");
    actor.wait_for_shutdown().await;
    drop(actor);

    let restored = LLMGateway::respawn_persistent(state_url.clone()).await?;
    let metrics = restored.ask(GetGatewayMetrics).await.unwrap();

    // Metrics object is available after restore.
    assert!(metrics.latest_prompt_tokens().is_none());

    restored
        .stop_gracefully()
        .await
        .expect("stop restored gateway");
    restored.wait_for_shutdown().await;

    Ok(())
}
