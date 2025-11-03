use std::{
    env,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow};
use async_openai::{
    Client,
    config::OpenAIConfig,
    error::OpenAIError,
    types::{
        ChatCompletionRequestMessage, ChatCompletionTool, CreateChatCompletionRequest,
        CreateChatCompletionRequestArgs, CreateChatCompletionResponse,
    },
};
use kameo::{prelude::*, reply::DelegatedReply};
use rand::{Rng, thread_rng};
use tokio::{
    sync::Semaphore,
    time::{sleep, timeout},
};
use tracing::{debug, warn};

use crate::{
    constants::{
        LLM_MAX_CONCURRENT_REQUESTS, LLM_MAX_RETRIES, REQUEST_TIMEOUT_SECS, RETRY_BASE_DELAY_MS,
        RETRY_MAX_EXP, RETRY_MAX_JITTER_MS,
    },
    tool_registry::{self, ToolName},
};

#[derive(Clone)]
struct GatewayConfig {
    max_retries:   usize,
    base_delay_ms: u64,
    max_jitter_ms: u64,
    timeout:       Duration,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            max_retries:   LLM_MAX_RETRIES,
            base_delay_ms: RETRY_BASE_DELAY_MS,
            max_jitter_ms: RETRY_MAX_JITTER_MS,
            timeout:       Duration::from_secs(REQUEST_TIMEOUT_SECS),
        }
    }
}

#[derive(Actor)]
pub struct LLMGateway {
    client:    Client<OpenAIConfig>,
    semaphore: Arc<Semaphore>,
    config:    GatewayConfig,
}

impl LLMGateway {
    /// Build a gateway using environment configuration.
    pub fn from_env() -> Result<Self> {
        let mut config = OpenAIConfig::default();
        if let Ok(url) = env::var("OPENAI_API_BASE") {
            config = config.with_api_base(url);
        }
        let client = Client::with_config(config);
        Ok(Self {
            client,
            semaphore: Arc::new(Semaphore::new(LLM_MAX_CONCURRENT_REQUESTS)),
            config: GatewayConfig::default(),
        })
    }

    fn build_request(
        request: &ChatCompletionRequest,
        tools: Vec<ChatCompletionTool>,
    ) -> Result<CreateChatCompletionRequest> {
        CreateChatCompletionRequestArgs::default()
            .model(request.model.clone())
            .messages(request.messages.clone())
            .temperature(request.temperature)
            .top_p(request.top_p)
            .tools(tools)
            .build()
            .context("failed to build chat completion request")
    }

    fn compute_backoff_ms(attempt: usize, cfg: &GatewayConfig) -> u64 {
        let exp = (attempt as u32).min(RETRY_MAX_EXP);
        let multiplier = 1u64.checked_shl(exp).unwrap_or(u64::MAX);
        let base = cfg.base_delay_ms.saturating_mul(multiplier).min(60_000);
        if cfg.max_jitter_ms == 0 {
            base
        } else {
            let jitter = thread_rng().gen_range(0..=cfg.max_jitter_ms);
            base.saturating_add(jitter)
        }
    }

    fn should_retry(err: &OpenAIError) -> bool {
        match err {
            OpenAIError::Reqwest(req_err) => {
                if req_err.is_timeout() || req_err.is_connect() {
                    return true;
                }
                if let Some(status) = req_err.status() {
                    return status.is_server_error() || status.as_u16() == 429;
                }
                false
            }
            OpenAIError::ApiError(_) => false,
            OpenAIError::JSONDeserialize(_, _) => false,
            OpenAIError::InvalidArgument(_) => false,
            _ => true,
        }
    }

    async fn execute_with_retries(
        client: Client<OpenAIConfig>,
        request: ChatCompletionRequest,
        config: GatewayConfig,
    ) -> Result<CreateChatCompletionResponse> {
        let mut attempt = 0usize;

        loop {
            attempt += 1;
            let tools = tool_registry::chat_tools(&request.tool_names)
                .context("failed to render tool specifications for request")?;
            let payload = Self::build_request(&request, tools)?;
            debug!(attempt, "llm_gateway dispatching chat completion");

            let started = Instant::now();
            let response = timeout(config.timeout, client.chat().create(payload)).await;

            match response {
                Ok(Ok(resp)) => {
                    if let Some(usage) = resp.usage.as_ref() {
                        debug!(
                            prompt_tokens = usage.prompt_tokens,
                            completion_tokens = usage.completion_tokens,
                            total_tokens = usage.total_tokens,
                            "llm_gateway usage" // TODO: integrate with metrics sink
                        );
                    }
                    debug!(elapsed_ms = started.elapsed().as_millis(), "llm_gateway request ok");
                    return Ok(resp);
                }
                Ok(Err(err)) => {
                    debug!(attempt, error = %err, "llm_gateway request error");
                    if attempt >= config.max_retries || !Self::should_retry(&err) {
                        return Err(anyhow!(err));
                    }
                    let backoff = Self::compute_backoff_ms(attempt, &config);
                    sleep(Duration::from_millis(backoff)).await;
                }
                Err(_) => {
                    warn!(attempt, timeout_secs = config.timeout.as_secs(), "llm_gateway timeout");
                    if attempt >= config.max_retries {
                        return Err(anyhow!(
                            "chat completion timed out after {} attempts",
                            attempt
                        ));
                    }
                    let backoff = Self::compute_backoff_ms(attempt, &config);
                    sleep(Duration::from_millis(backoff)).await;
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct ChatCompletionRequest {
    pub model:       String,
    pub messages:    Vec<ChatCompletionRequestMessage>,
    pub temperature: f32,
    pub top_p:       f32,
    pub tool_names:  Vec<ToolName>,
}

impl Message<ChatCompletionRequest> for LLMGateway {
    type Reply = DelegatedReply<Result<CreateChatCompletionResponse>>;

    async fn handle(
        &mut self,
        msg: ChatCompletionRequest,
        ctx: &mut kameo::message::Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let client = self.client.clone();
        let semaphore = Arc::clone(&self.semaphore);
        let config = self.config.clone();
        ctx.spawn(async move {
            let _permit = semaphore
                .acquire_owned()
                .await
                .map_err(|_| anyhow!("llm gateway shutting down"))?;
            Self::execute_with_retries(client, msg, config).await
        })
    }
}
