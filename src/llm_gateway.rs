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
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestToolMessageArgs, ChatCompletionTool, CreateChatCompletionRequest,
        CreateChatCompletionRequestArgs,
    },
};
use kameo::{error::SendError, prelude::*, reply::DelegatedReply};
use rand::{Rng, thread_rng};
use serde_json::{self, json};
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
    file_reader::{ExecuteTool, FileReader},
    tools::llm::{self, ToolName},
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
            OpenAIError::ApiError(_) => false,
            OpenAIError::JSONDeserialize(_, _) => false,
            OpenAIError::InvalidArgument(_) => false,
            _ => true,
        }
    }

    async fn run_conversation(
        client: Client<OpenAIConfig>,
        request: ChatCompletionRequest,
        config: GatewayConfig,
    ) -> Result<String> {
        let mut messages = request.messages.clone();
        for iteration in 0..request.max_iterations {
            debug!(iteration, "Starting LLM tool iteration");
            let mut attempt = 0usize;
            let response = loop {
                attempt += 1;
                let tools = llm::tool_specs(&request.tool_names)
                    .context("failed to render tool specifications for request")?;
                let payload = Self::build_request(
                    &request.model,
                    &messages,
                    request.temperature,
                    request.top_p,
                    tools,
                )?;
                let started = Instant::now();
                let call = timeout(config.timeout, client.chat().create(payload)).await;
                match call {
                    Ok(Ok(resp)) => {
                        if let Some(usage) = resp.usage.as_ref() {
                            debug!(
                                prompt_tokens = usage.prompt_tokens,
                                completion_tokens = usage.completion_tokens,
                                total_tokens = usage.total_tokens,
                                "llm_gateway usage" // TODO: integrate with metrics sink
                            );
                        }
                        debug!(
                            elapsed_ms = started.elapsed().as_millis(),
                            iteration, "llm_gateway request ok"
                        );
                        break resp;
                    }
                    Ok(Err(err)) => {
                        warn!(iteration, attempt, error = %err, "llm_gateway request error");
                        if attempt >= config.max_retries || !Self::should_retry(&err) {
                            return Err(anyhow!(err));
                        }
                        let backoff = Self::compute_backoff_ms(attempt, &config);
                        sleep(Duration::from_millis(backoff)).await;
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
                        let backoff = Self::compute_backoff_ms(attempt, &config);
                        sleep(Duration::from_millis(backoff)).await;
                    }
                }
            };

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
                                let payload = json!({
                                    "type": "tool_error",
                                    "tool": tool_name,
                                    "arguments": raw_arguments,
                                    "message": format!("invalid JSON arguments: {err}"),
                                });
                                let tool_msg = ChatCompletionRequestToolMessageArgs::default()
                                    .tool_call_id(call.id.clone())
                                    .content(payload.to_string())
                                    .build()
                                    .context("failed to build tool response message")?
                                    .into();
                                messages.push(tool_msg);
                                continue;
                            }
                        };

                    let invocation = match llm::parse_invocation(
                        call.function.name.as_str(),
                        parsed_args.clone(),
                    ) {
                        Ok(invocation) => invocation,
                        Err(err) => {
                            warn!(
                                iteration,
                                tool = tool_name.as_str(),
                                error = %err,
                                "tool argument validation failed"
                            );
                            let payload = json!({
                                "type": "tool_error",
                                "tool": tool_name,
                                "arguments": parsed_args,
                                "message": err.to_string(),
                            });
                            let tool_msg = ChatCompletionRequestToolMessageArgs::default()
                                .tool_call_id(call.id.clone())
                                .content(payload.to_string())
                                .build()
                                .context("failed to build tool response message")?
                                .into();
                            messages.push(tool_msg);
                            continue;
                        }
                    };

                    debug!(
                        iteration,
                        tool = tool_name.as_str(),
                        "Executing assistant-requested tool"
                    );
                    let result = request.tool_host.ask(ExecuteTool { invocation }).await;
                    let payload = match result {
                        Ok(value) => value,
                        Err(SendError::HandlerError(err)) => {
                            warn!(
                                iteration,
                                tool = tool_name.as_str(),
                                error = %err,
                                "tool execution failed"
                            );
                            json!({
                                "type": "tool_error",
                                "tool": tool_name,
                                "arguments": parsed_args,
                                "message": err.to_string(),
                            })
                        }
                        Err(other) => {
                            return Err(anyhow!(other));
                        }
                    };
                    let tool_msg = ChatCompletionRequestToolMessageArgs::default()
                        .tool_call_id(call.id.clone())
                        .content(payload.to_string())
                        .build()
                        .context("failed to build tool response message")?
                        .into();
                    messages.push(tool_msg);
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
    pub model:          String,
    pub messages:       Vec<ChatCompletionRequestMessage>,
    pub temperature:    f32,
    pub top_p:          f32,
    pub tool_names:     Vec<ToolName>,
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
        ctx.spawn(async move {
            let _permit = semaphore
                .acquire_owned()
                .await
                .map_err(|_| anyhow!("llm gateway shutting down"))?;
            Self::run_conversation(client, msg, config).await
        })
    }
}
