use std::{
    env,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow};
use async_openai::{
    Client,
    config::OpenAIConfig,
    error::OpenAIError,
    types::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestToolMessageArgs, ChatCompletionTool, CreateChatCompletionRequest,
        CreateChatCompletionRequestArgs, CreateChatCompletionResponse,
    },
};
use kameo::{error::SendError, prelude::*, reply::DelegatedReply};
use rand::{Rng, thread_rng};
use serde_json::{self, json};
use tokio::{
    sync::Semaphore,
    time::{sleep, timeout},
};
use tracing::{debug, error, warn};

use crate::{
    constants::{
        LLM_MAX_CONCURRENT_REQUESTS, LLM_MAX_RETRIES, REQUEST_TIMEOUT_SECS, RETRY_BASE_DELAY_MS,
        RETRY_MAX_EXP, RETRY_MAX_JITTER_MS,
    },
    file_reader::{ExecuteTool, FileReader},
    tools::llm,
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

#[derive(Default, Debug)]
pub struct GatewayMetrics {
    total_calls:       AtomicU64,
    total_tokens:      AtomicU64,
    prompt_tokens:     AtomicU64,
    completion_tokens: AtomicU64,
    last_latency_ms:   AtomicU64,
}

impl GatewayMetrics {
    fn record_completion(&self, prompt: u32, completion: u32, total: u32, latency: Duration) {
        self.total_calls.fetch_add(1, Ordering::Relaxed);
        self.prompt_tokens
            .fetch_add(prompt as u64, Ordering::Relaxed);
        self.completion_tokens
            .fetch_add(completion as u64, Ordering::Relaxed);
        self.total_tokens.fetch_add(total as u64, Ordering::Relaxed);
        self.last_latency_ms
            .store(latency.as_millis() as u64, Ordering::Relaxed);
        debug!(
            prompt_tokens = prompt,
            completion_tokens = completion,
            total_tokens = total,
            latency_ms = latency.as_millis(),
            "llm_gateway completion"
        );
    }

    pub fn snapshot(&self) -> GatewayMetricsSnapshot {
        GatewayMetricsSnapshot {
            total_calls:       self.total_calls.load(Ordering::Relaxed),
            total_tokens:      self.total_tokens.load(Ordering::Relaxed),
            prompt_tokens:     self.prompt_tokens.load(Ordering::Relaxed),
            completion_tokens: self.completion_tokens.load(Ordering::Relaxed),
            last_latency_ms:   self.last_latency_ms.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GatewayMetricsSnapshot {
    pub total_calls:       u64,
    pub total_tokens:      u64,
    pub prompt_tokens:     u64,
    pub completion_tokens: u64,
    pub last_latency_ms:   u64,
}

#[derive(Actor)]
pub struct LLMGateway {
    client:    Client<OpenAIConfig>,
    semaphore: Arc<Semaphore>,
    config:    GatewayConfig,
    metrics:   Arc<GatewayMetrics>,
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
            metrics: Arc::new(Default::default()),
        })
    }

    pub fn with_metrics(mut self, metrics: Arc<GatewayMetrics>) -> Self {
        self.metrics = metrics;
        self
    }

    pub fn metrics(&self) -> Arc<GatewayMetrics> {
        Arc::clone(&self.metrics)
    }

    fn build_request(
        model: &str,
        messages: &[ChatCompletionRequestMessage],
        temperature: f32,
        top_p: f32,
        tools: Vec<ChatCompletionTool>,
    ) -> Result<CreateChatCompletionRequest> {
        CreateChatCompletionRequestArgs::default()
            .model(model.to_string())
            .messages(messages.to_vec())
            .temperature(temperature)
            .top_p(top_p)
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
            OpenAIError::ApiError(api_err) => {
                // async-openai surfaces HTTP status failures (including 429/5xx)
                // via ApiError without exposing the numeric status code. Prefer
                // to treat these as retryable unless the error code clearly
                // indicates a caller-side issue.
                let non_retryable = [
                    "invalid_api_key",
                    "account_deactivated",
                    "invalid_request_error",
                    "context_length_exceeded",
                    "insufficient_quota",
                    "billing_not_active",
                ];
                api_err
                    .code
                    .as_deref()
                    .into_iter()
                    .chain(api_err.r#type.as_deref())
                    .all(|value| !non_retryable.contains(&value))
            }
            OpenAIError::JSONDeserialize(_, _) => false,
            OpenAIError::InvalidArgument(_) => false,
            _ => true,
        }
    }

    async fn call_with_retry<F>(
        client: &Client<OpenAIConfig>,
        config: &GatewayConfig,
        metrics: &Arc<GatewayMetrics>,
        iteration: usize,
        mut build_payload: F,
    ) -> Result<CreateChatCompletionResponse>
    where
        F: FnMut() -> Result<CreateChatCompletionRequest>,
    {
        let mut attempt = 0usize;
        loop {
            attempt += 1;
            let payload = build_payload()?;
            let started = Instant::now();
            let call = timeout(config.timeout, client.chat().create(payload)).await;
            match call {
                Ok(Ok(resp)) => {
                    if let Some(usage) = resp.usage.as_ref() {
                        metrics.record_completion(
                            usage.prompt_tokens,
                            usage.completion_tokens,
                            usage.total_tokens,
                            started.elapsed(),
                        );
                        debug!(
                            prompt_tokens = usage.prompt_tokens,
                            completion_tokens = usage.completion_tokens,
                            total_tokens = usage.total_tokens,
                            "llm_gateway usage"
                        );
                    }
                    debug!(
                        elapsed_ms = started.elapsed().as_millis(),
                        iteration, attempt, "llm_gateway request ok"
                    );
                    return Ok(resp);
                }
                Ok(Err(err)) => {
                    warn!(iteration, attempt, error = %err, "llm_gateway request error");
                    if attempt >= config.max_retries || !Self::should_retry(&err) {
                        return Err(anyhow!(err));
                    }
                }
                Err(_) => {
                    warn!(
                        iteration,
                        attempt,
                        timeout_secs = config.timeout.as_secs(),
                        "llm_gateway timeout"
                    );
                    if attempt >= config.max_retries {
                        return Err(anyhow!(
                            "chat completion timed out after {} attempts",
                            attempt
                        ));
                    }
                }
            }
            let backoff = Self::compute_backoff_ms(attempt, config);
            sleep(Duration::from_millis(backoff)).await;
        }
    }

    fn push_tool_payload(
        messages: &mut Vec<ChatCompletionRequestMessage>,
        call_id: &str,
        payload: serde_json::Value,
    ) -> Result<()> {
        let tool_msg = ChatCompletionRequestToolMessageArgs::default()
            .tool_call_id(call_id.to_string())
            .content(payload.to_string())
            .build()
            .context("failed to build tool response message")?
            .into();
        messages.push(tool_msg);
        Ok(())
    }

    fn tool_error_payload(
        tool: &str,
        code: &str,
        arguments: serde_json::Value,
        message: impl Into<String>,
    ) -> serde_json::Value {
        json!({
            "type": "tool_error",
            "tool": tool,
            "code": code,
            "arguments": arguments,
            "message": message.into(),
        })
    }

    async fn run_conversation(
        client: Client<OpenAIConfig>,
        request: ChatCompletionRequest,
        config: GatewayConfig,
        metrics: Arc<GatewayMetrics>,
    ) -> Result<String> {
        let mut messages = request.messages.clone();
        let tools = llm::tool_specs(&request.tool_ids)
            .context("failed to render tool specifications for request")?;
        for iteration in 0..request.max_iterations {
            debug!(iteration, "Starting LLM tool iteration");
            let response = Self::call_with_retry(&client, &config, &metrics, iteration, || {
                Self::build_request(
                    request.model.as_str(),
                    &messages,
                    request.temperature,
                    request.top_p,
                    tools.clone(),
                )
            })
            .await?;

            let mut choices = response.choices.into_iter();
            let message = choices
                .next()
                .ok_or_else(|| anyhow!("chat completion returned no choices"))?
                .message;

            debug!(
                iteration,
                role = ?message.role,
                has_content = message.content.as_ref().map(|c| !c.trim().is_empty()),
                tool_call_count = message.tool_calls.as_ref().map(|c| c.len()),
                content = ?message.content,
                refusal = ?message.refusal,
                "Assistant message received"
            );

            if let Some(tool_calls) = message.tool_calls.clone()
                && !tool_calls.is_empty()
            {
                let assistant_msg = ChatCompletionRequestAssistantMessageArgs::default()
                    .tool_calls(tool_calls.clone())
                    .build()
                    .context("failed to build assistant tool-call message")?
                    .into();
                messages.push(assistant_msg);

                for call in tool_calls {
                    let tool_name = call.function.name.clone();
                    let call_id = call.id.clone();
                    let raw_arguments = call.function.arguments.clone();
                    let parsed_args =
                        match serde_json::from_str::<serde_json::Value>(&raw_arguments) {
                            Ok(value) => value,
                            Err(err) => {
                                warn!(
                                    iteration,
                                    tool = tool_name.as_str(),
                                    error = %err,
                                    "tool arguments were not valid JSON"
                                );
                                let payload = Self::tool_error_payload(
                                    &tool_name,
                                    "invalid_json",
                                    json!(raw_arguments),
                                    format!("invalid JSON arguments: {err}"),
                                );
                                Self::push_tool_payload(&mut messages, &call_id, payload)?;
                                continue;
                            }
                        };

                    if llm::lookup_tool(tool_name.as_str()).is_none() {
                        warn!(
                            iteration,
                            tool = tool_name.as_str(),
                            "tool identifier not registered"
                        );
                        let payload = Self::tool_error_payload(
                            &tool_name,
                            "unsupported_tool",
                            parsed_args.clone(),
                            format!("unsupported tool: {}", tool_name),
                        );
                        Self::push_tool_payload(&mut messages, &call_id, payload)?;
                        continue;
                    }

                    debug!(
                        iteration,
                        tool = tool_name.as_str(),
                        "Executing assistant-requested tool"
                    );
                    let result = request
                        .tool_host
                        .ask(ExecuteTool {
                            identifier: tool_name.clone(),
                            arguments:  parsed_args.clone(),
                        })
                        .await;
                    let payload = match result {
                        Ok(value) => value,
                        Err(SendError::HandlerError(err)) => match err {
                            llm::ToolExecutionError::Input(input_err) => {
                                warn!(
                                    iteration,
                                    tool = tool_name.as_str(),
                                    error = %input_err,
                                    "tool reported invalid input"
                                );
                                Self::tool_error_payload(
                                    &tool_name,
                                    match &input_err {
                                        llm::ToolInputError::MissingField { .. } => "missing_field",
                                        llm::ToolInputError::EmptyField { .. } => "empty_field",
                                        llm::ToolInputError::BelowMinimum { .. } => "below_minimum",
                                        llm::ToolInputError::InvalidRange { .. } => "invalid_range",
                                        llm::ToolInputError::EmptyCollection { .. } => {
                                            "empty_collection"
                                        }
                                        llm::ToolInputError::UnsupportedTool { .. } => {
                                            "unsupported_tool"
                                        }
                                        llm::ToolInputError::InvalidPayload { .. } => {
                                            "invalid_payload"
                                        }
                                        llm::ToolInputError::DepthExceeded { .. } => {
                                            "depth_exceeded"
                                        }
                                        llm::ToolInputError::InvalidPath { .. } => "invalid_path",
                                    },
                                    parsed_args.clone(),
                                    input_err.to_string(),
                                )
                            }
                            llm::ToolExecutionError::Internal(internal_err) => {
                                error!(
                                    iteration,
                                    tool = tool_name.as_str(),
                                    error = ?internal_err,
                                    "tool execution failed"
                                );
                                let payload = Self::tool_error_payload(
                                    &tool_name,
                                    "internal_error",
                                    parsed_args.clone(),
                                    internal_err.to_string(),
                                );
                                Self::push_tool_payload(&mut messages, &call_id, payload)?;
                                continue;
                            }
                        },
                        Err(other) => {
                            return Err(anyhow!("tool host communication failed: {other:?}"));
                        }
                    };
                    Self::push_tool_payload(&mut messages, &call_id, payload)?;
                }
                continue;
            }

            if let Some(content) = message.content {
                let trimmed = content.trim();
                if trimmed.is_empty() {
                    debug!(iteration, "Assistant content was empty; continuing");
                } else {
                    debug!(iteration, "Assistant returned final content");
                    return Ok(content);
                }
            }

            debug!(iteration, "Assistant response had no tool calls and no content; continuing");
        }

        Err(anyhow!(
            "LLM tool loop did not terminate with a message after {} iterations",
            request.max_iterations
        ))
    }
}

#[derive(Clone, Debug)]
pub struct ChatCompletionRequest {
    pub model:          Arc<String>,
    pub messages:       Vec<ChatCompletionRequestMessage>,
    pub temperature:    f32,
    pub top_p:          f32,
    pub tool_ids:       Vec<&'static str>,
    pub max_iterations: usize,
    pub tool_host:      ActorRef<FileReader>,
}

impl Message<ChatCompletionRequest> for LLMGateway {
    type Reply = DelegatedReply<Result<String>>;

    async fn handle(
        &mut self,
        msg: ChatCompletionRequest,
        ctx: &mut kameo::message::Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let client = self.client.clone();
        let semaphore = Arc::clone(&self.semaphore);
        let config = self.config.clone();
        let metrics = Arc::clone(&self.metrics);
        ctx.spawn(async move {
            let _permit = semaphore
                .acquire_owned()
                .await
                .map_err(|_| anyhow!("llm gateway shutting down"))?;
            Self::run_conversation(client, msg, config, metrics).await
        })
    }
}
