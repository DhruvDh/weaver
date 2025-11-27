use std::{
    collections::HashMap,
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
        ChatCompletionMessageToolCall, ChatCompletionRequestAssistantMessageArgs,
        ChatCompletionRequestMessage, ChatCompletionRequestToolMessageArgs,
        ChatCompletionResponseMessage, ChatCompletionTool, CompletionUsage,
        CreateChatCompletionRequest, CreateChatCompletionRequestArgs, CreateChatCompletionResponse,
    },
};
use kameo::{error::SendError, prelude::*, reply::DelegatedReply};
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use rand::{Rng, rng};
use reqwest::Client as HttpClient;
use serde_json::{self, json};
use tokio::{
    sync::Semaphore,
    time::{sleep, timeout},
};
use tracing::{debug, error, info, warn};

use crate::{
    constants::{
        LLM_MAX_CONCURRENT_REQUESTS, LLM_MAX_RETRIES, REQUEST_TIMEOUT_SECS, RETRY_BASE_DELAY_MS,
        RETRY_MAX_BACKOFF_MS, RETRY_MAX_EXP,
    },
    file_reader::{ExecuteTool, FileReader},
    rerun_sink::{LogScalar, RerunSink},
    tools::llm::{self, ToolOutput},
};

#[derive(Clone)]
struct GatewayConfig {
    max_retries:   usize,
    base_delay_ms: u64,
    timeout:       Duration,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            max_retries:   LLM_MAX_RETRIES,
            base_delay_ms: RETRY_BASE_DELAY_MS,
            timeout:       Duration::from_secs(REQUEST_TIMEOUT_SECS),
        }
    }
}

#[derive(Default)]
struct IterationState {
    pending_bytes:       u64,
    latest_total_tokens: u64,
    last_prompt_total:   u32,
}

impl IterationState {
    fn record_usage(
        &mut self,
        metrics: &GatewayMetrics,
        conversation_id: &str,
        model: &str,
        usage: &CompletionUsage,
    ) {
        self.latest_total_tokens = usage.total_tokens as u64;
        let prompt_total = usage.prompt_tokens;
        let prompt_delta = prompt_total.saturating_sub(self.last_prompt_total);
        self.last_prompt_total = prompt_total;
        metrics.record_conversation_prompt_tokens(conversation_id, prompt_total as u64);
        if prompt_delta > 0 && self.pending_bytes > 0 {
            metrics.observe_payload_bytes(model, self.pending_bytes, prompt_delta as u64);
            self.pending_bytes = 0;
        }
    }

    fn finalize_content(
        &self,
        metrics: &GatewayMetrics,
        actor_name: &str,
        message: &ChatCompletionResponseMessage,
    ) -> Option<String> {
        let content = message.content.as_ref()?.trim();
        if content.is_empty() {
            return None;
        }
        metrics.record_conversation_tokens(actor_name, self.latest_total_tokens);
        message.content.clone()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct TokenEstimator {
    observed_bytes:  u64,
    observed_tokens: u64,
}

impl TokenEstimator {
    fn update(&mut self, bytes: u64, tokens: u64) {
        self.observed_bytes = self.observed_bytes.saturating_add(bytes);
        self.observed_tokens = self.observed_tokens.saturating_add(tokens);
    }

    fn predict(&self, bytes: u64) -> Option<u64> {
        if self.observed_bytes == 0 || bytes == 0 {
            return None;
        }
        let numerator = (self.observed_tokens as u128).saturating_mul(bytes as u128);
        let denominator = self.observed_bytes as u128;
        Some((numerator / denominator) as u64)
    }
}

#[derive(Default, Debug)]
struct ConversationAccumulator {
    total_tokens: u64,
    count:        u64,
}

#[derive(Default, Debug)]
pub struct GatewayMetrics {
    total_calls:                AtomicU64,
    total_tokens:               AtomicU64,
    prompt_tokens:              AtomicU64,
    completion_tokens:          AtomicU64,
    last_latency_ms:            AtomicU64,
    last_prompt_tokens:         AtomicU64,
    last_completion_tokens:     AtomicU64,
    estimators:                 RwLock<HashMap<String, TokenEstimator>>,
    conversation_stats:         RwLock<HashMap<String, ConversationAccumulator>>,
    conversation_prompt_tokens: RwLock<HashMap<String, u64>>,
    context_limits:             RwLock<HashMap<String, u32>>,
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
        self.last_prompt_tokens
            .store(prompt as u64, Ordering::Relaxed);
        self.last_completion_tokens
            .store(completion as u64, Ordering::Relaxed);
        debug!(
            prompt_tokens = prompt,
            completion_tokens = completion,
            total_tokens = total,
            latency_ms = latency.as_millis(),
            "llm_gateway completion"
        );
    }

    pub fn latest_prompt_tokens(&self) -> Option<u64> {
        let tokens = self.last_prompt_tokens.load(Ordering::Relaxed);
        if tokens == 0 { None } else { Some(tokens) }
    }

    pub fn reset_conversation_prompt_tokens(&self, conversation: &str) {
        let mut guard = self.conversation_prompt_tokens.write();
        guard.remove(conversation);
    }

    pub fn record_conversation_prompt_tokens(&self, conversation: &str, tokens: u64) {
        let mut guard = self.conversation_prompt_tokens.write();
        guard.insert(conversation.to_string(), tokens);
    }

    pub fn latest_prompt_tokens_for_conversation(&self, conversation: &str) -> Option<u64> {
        self.conversation_prompt_tokens
            .read()
            .get(conversation)
            .copied()
            .filter(|value| *value > 0)
    }

    fn observe_payload_bytes(&self, model: &str, bytes: u64, prompt_delta: u64) {
        if bytes == 0 || prompt_delta == 0 {
            return;
        }
        let mut estimators = self.estimators.write();
        let estimator = estimators.entry(model.to_string()).or_default();
        estimator.update(bytes, prompt_delta);
    }

    pub fn estimate_tokens(&self, model: &str, bytes: u64) -> Option<u64> {
        if bytes == 0 {
            return None;
        }
        let estimators = self.estimators.read();
        estimators
            .get(model)
            .and_then(|estimator| estimator.predict(bytes))
    }

    fn record_conversation_tokens(&self, actor: &str, tokens: u64) {
        let mut stats = self.conversation_stats.write();
        let entry = stats.entry(actor.to_string()).or_default();
        entry.total_tokens = entry.total_tokens.saturating_add(tokens);
        entry.count = entry.count.saturating_add(1);
    }

    pub fn log_summary(&self) {
        let total_calls = self.total_calls.load(Ordering::Relaxed);
        let total_tokens = self.total_tokens.load(Ordering::Relaxed);
        let prompt_tokens = self.prompt_tokens.load(Ordering::Relaxed);
        let completion_tokens = self.completion_tokens.load(Ordering::Relaxed);
        info!(
            total_calls,
            total_tokens, prompt_tokens, completion_tokens, "llm_gateway summary"
        );
        let stats = self.conversation_stats.read();
        for (actor, acc) in stats.iter() {
            if acc.count == 0 {
                continue;
            }
            let avg = (acc.total_tokens as f64) / (acc.count as f64);
            info!(
                actor = actor.as_str(),
                conversations = acc.count,
                total_tokens = acc.total_tokens,
                avg_tokens = avg,
                "llm_gateway conversation usage"
            );
        }
    }

    pub fn context_limit(&self, model: &str) -> u32 {
        if let Some(limit) = self.context_limits.read().get(model) {
            return *limit;
        }
        let fallback = *DEFAULT_CONTEXT_LIMIT;
        let mut guard = self.context_limits.write();
        *guard.entry(model.to_string()).or_insert(fallback)
    }
}

struct ConversationPromptGuard {
    metrics:         Arc<GatewayMetrics>,
    conversation_id: String,
}

impl ConversationPromptGuard {
    fn new(metrics: Arc<GatewayMetrics>, conversation_id: String) -> Self {
        Self {
            metrics,
            conversation_id,
        }
    }
}

impl Drop for ConversationPromptGuard {
    fn drop(&mut self) {
        self.metrics
            .reset_conversation_prompt_tokens(&self.conversation_id);
    }
}

static DEFAULT_CONTEXT_LIMIT: Lazy<u32> = Lazy::new(|| {
    env::var("WEAVER_CONTEXT_LIMIT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(131_072)
});

async fn log_scalar(rerun: &Option<ActorRef<RerunSink>>, path: impl Into<String>, value: f64) {
    if let Some(sink) = rerun {
        let msg = LogScalar {
            path: path.into(),
            value,
            time_ns: None,
        };
        let sink = sink.clone();
        tokio::spawn(async move {
            let _ = sink.tell(msg).await;
        });
    }
}

async fn log_completion_metrics(
    rerun: &Option<ActorRef<RerunSink>>,
    base: &str,
    elapsed_ms: f64,
    usage: &CompletionUsage,
) {
    let paths = [
        (format!("{base}/latency/completion_ms"), elapsed_ms),
        (format!("{base}/tokens/prompt"), usage.prompt_tokens as f64),
        (format!("{base}/tokens/completion"), usage.completion_tokens as f64),
        (format!("{base}/tokens/total"), usage.total_tokens as f64),
    ];
    for (p, v) in paths {
        log_scalar(rerun, p, v).await;
    }
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
        let http_client = HttpClient::builder()
            .user_agent("weaver-llm-gateway")
            .build()
            .context("failed to build http client")?;
        let client = Client::with_config(config).with_http_client(http_client);
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
        let exp = attempt.saturating_sub(1).min(RETRY_MAX_EXP as usize) as u32;
        let multiplier = 1u64.checked_shl(exp).unwrap_or(u64::MAX);
        let cap = cfg
            .base_delay_ms
            .saturating_mul(multiplier)
            .min(RETRY_MAX_BACKOFF_MS);
        if cap == 0 {
            return 0;
        }
        let mut rng = rng();
        rng.random_range(0..=cap)
    }

    fn should_retry(err: &OpenAIError) -> bool {
        match err {
            OpenAIError::Reqwest(req_err) => {
                if req_err.is_timeout()
                    || req_err.is_connect()
                    || req_err.is_body()
                    || req_err.is_decode()
                {
                    return true;
                }
                if let Some(status) = req_err.status() {
                    let code = status.as_u16();
                    return status.is_server_error() || code == 429 || code == 408;
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
            OpenAIError::JSONDeserialize(_, _) => true,
            OpenAIError::InvalidArgument(_) => false,
            _ => true,
        }
    }

    async fn handle_tool_calls(
        iteration: usize,
        tool_host: &ActorRef<FileReader>,
        tool_calls: Vec<ChatCompletionMessageToolCall>,
        messages: &mut Vec<ChatCompletionRequestMessage>,
        state: &mut IterationState,
    ) -> Result<()> {
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
            let parsed_args = match serde_json::from_str::<serde_json::Value>(&raw_arguments) {
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
                    let bytes = Self::push_tool_payload(messages, &call_id, payload)?;
                    state.pending_bytes = state.pending_bytes.saturating_add(bytes);
                    continue;
                }
            };

            if llm::lookup_tool(tool_name.as_str()).is_none() {
                warn!(iteration, tool = tool_name.as_str(), "tool identifier not registered");
                let payload = Self::tool_error_payload(
                    &tool_name,
                    "unsupported_tool",
                    parsed_args.clone(),
                    format!("unsupported tool: {}", tool_name),
                );
                let bytes = Self::push_tool_payload(messages, &call_id, payload)?;
                state.pending_bytes = state.pending_bytes.saturating_add(bytes);
                continue;
            }

            debug!(iteration, tool = tool_name.as_str(), "Executing assistant-requested tool");
            let result = tool_host
                .ask(ExecuteTool {
                    identifier: tool_name.clone(),
                    arguments:  parsed_args.clone(),
                })
                .await;
            let tool_output = match result {
                Ok(value) => value,
                Err(SendError::HandlerError(err)) => match err {
                    llm::ToolExecutionError::Input(input_err) => {
                        warn!(
                            iteration,
                            tool = tool_name.as_str(),
                            error = %input_err,
                            "tool reported invalid input"
                        );
                        let payload = Self::tool_error_payload(
                            &tool_name,
                            match &input_err {
                                llm::ToolInputError::MissingField { .. } => "missing_field",
                                llm::ToolInputError::EmptyField { .. } => "empty_field",
                                llm::ToolInputError::BelowMinimum { .. } => "below_minimum",
                                llm::ToolInputError::InvalidRange { .. } => "invalid_range",
                                llm::ToolInputError::EmptyCollection { .. } => "empty_collection",
                                llm::ToolInputError::UnsupportedTool { .. } => "unsupported_tool",
                                llm::ToolInputError::InvalidPayload { .. } => "invalid_payload",
                                llm::ToolInputError::DepthExceeded { .. } => "depth_exceeded",
                                llm::ToolInputError::InvalidPath { .. } => "invalid_path",
                            },
                            parsed_args.clone(),
                            input_err.to_string(),
                        );
                        ToolOutput::new(payload)
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
                        ToolOutput::new(payload)
                    }
                },
                Err(other) => {
                    return Err(anyhow!("tool host communication failed: {other:?}"));
                }
            };
            let contribution_hint = tool_output.byte_hint;
            let payload = tool_output.payload;
            let measured = Self::push_tool_payload(messages, &call_id, payload)?;
            let contribution = contribution_hint.unwrap_or(measured);
            state.pending_bytes = state.pending_bytes.saturating_add(contribution);
        }
        Ok(())
    }

    async fn call_with_retry<F>(
        client: Client<OpenAIConfig>,
        config: &GatewayConfig,
        metrics: &Arc<GatewayMetrics>,
        rerun: Option<ActorRef<RerunSink>>,
        iteration: usize,
        model_label: &str,
        mut build_payload: F,
    ) -> Result<CreateChatCompletionResponse>
    where
        F: FnMut() -> Result<CreateChatCompletionRequest>,
    {
        let mut attempt = 0usize;
        loop {
            attempt += 1;
            let base = format!("metrics/llm/{}", model_label);
            log_scalar(&rerun, format!("{base}/attempt"), attempt as f64).await;
            let payload = build_payload()?;
            let started = Instant::now();
            let client = client.clone();
            let call =
                timeout(config.timeout, async move { client.chat().create(payload).await }).await;
            match call {
                Ok(Ok(resp)) => {
                    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                    log_scalar(&rerun, format!("{base}/latency/request_ms"), elapsed_ms).await;
                    if let Some(usage) = resp.usage.as_ref() {
                        metrics.record_completion(
                            usage.prompt_tokens,
                            usage.completion_tokens,
                            usage.total_tokens,
                            started.elapsed(),
                        );
                        log_completion_metrics(&rerun, &base, elapsed_ms, usage).await;
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
                    warn!(
                        iteration,
                        attempt,
                        error = %err,
                        "llm_gateway request error"
                    );
                    log_scalar(&rerun, format!("{base}/errors/api"), 1.0).await;
                    let retryable = Self::should_retry(&err);
                    if attempt >= config.max_retries || !retryable {
                        return Err(anyhow!(err));
                    }
                    let backoff = Self::compute_backoff_ms(attempt, config);
                    log_scalar(&rerun, format!("{base}/backoff_ms"), backoff as f64).await;
                    sleep(Duration::from_millis(backoff)).await;
                }
                Err(_) => {
                    warn!(
                        iteration,
                        attempt,
                        timeout_secs = config.timeout.as_secs(),
                        "llm_gateway timeout"
                    );
                    log_scalar(&rerun, format!("{base}/errors/timeout"), 1.0).await;
                    if attempt >= config.max_retries {
                        return Err(anyhow!(
                            "chat completion timed out after {} attempts",
                            attempt
                        ));
                    }
                    // Timeout has no server hint.
                    let backoff = Self::compute_backoff_ms(attempt, config);
                    log_scalar(&rerun, format!("{base}/backoff_ms"), backoff as f64).await;
                    sleep(Duration::from_millis(backoff)).await;
                }
            }
        }
    }

    fn push_tool_payload(
        messages: &mut Vec<ChatCompletionRequestMessage>,
        call_id: &str,
        payload: serde_json::Value,
    ) -> Result<u64> {
        let content = payload.to_string();
        let bytes = content.len() as u64;
        let tool_msg = ChatCompletionRequestToolMessageArgs::default()
            .tool_call_id(call_id.to_string())
            .content(content)
            .build()
            .context("failed to build tool response message")?
            .into();
        messages.push(tool_msg);
        Ok(bytes)
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
        let actor_name = request.actor_name.clone();
        let conversation_id = request.conversation_id.clone();
        metrics.reset_conversation_prompt_tokens(&conversation_id);
        let _prompt_guard =
            ConversationPromptGuard::new(Arc::clone(&metrics), conversation_id.clone());
        let mut iter_state = IterationState::default();
        let tools = llm::tool_specs(&request.tool_ids)
            .context("failed to render tool specifications for request")?;
        for iteration in 0..request.max_iterations {
            debug!(iteration, "Starting LLM tool iteration");
            let response = Self::call_with_retry(
                client.clone(),
                &config,
                &metrics,
                request.rerun.clone(),
                iteration,
                request.model.as_str(),
                || {
                    Self::build_request(
                        request.model.as_str(),
                        &messages,
                        request.temperature,
                        request.top_p,
                        tools.clone(),
                    )
                },
            )
            .await?;

            if let Some(usage) = response.usage.as_ref() {
                iter_state.record_usage(&metrics, &conversation_id, request.model.as_str(), usage);
            }

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

            if let Some(tool_calls) = message.tool_calls.clone().filter(|calls| !calls.is_empty()) {
                Self::handle_tool_calls(
                    iteration,
                    &request.tool_host,
                    tool_calls,
                    &mut messages,
                    &mut iter_state,
                )
                .await?;
                continue;
            }

            if let Some(content) = iter_state.finalize_content(&metrics, &actor_name, &message) {
                debug!(iteration, "Assistant returned final content");
                return Ok(content);
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
    pub model:           Arc<String>,
    pub messages:        Vec<ChatCompletionRequestMessage>,
    pub temperature:     f32,
    pub top_p:           f32,
    pub tool_ids:        Vec<&'static str>,
    pub max_iterations:  usize,
    pub tool_host:       ActorRef<FileReader>,
    pub actor_name:      String,
    pub conversation_id: String,
    pub rerun:           Option<ActorRef<RerunSink>>,
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
