use std::env;

use anyhow::Result;
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::{CreateChatCompletionRequest, CreateChatCompletionResponse},
};
use kameo::prelude::*;

/// Centralized gateway for all LLM chat-completion requests.
#[derive(Actor)]
pub struct LLMGateway {
    client: Client<OpenAIConfig>,
}

impl LLMGateway {
    /// Build a gateway using environment variables (OPENAI_MODEL handled by
    /// callers).
    pub fn from_env() -> Result<Self> {
        let mut config = OpenAIConfig::default();
        if let Ok(url) = env::var("OPENAI_API_BASE") {
            config = config.with_api_base(url);
        }
        let client = Client::with_config(config);
        Ok(Self { client })
    }
}

/// Message requesting a chat completion through the gateway.
pub struct ChatCompletionRequest {
    pub request: CreateChatCompletionRequest,
}

impl Message<ChatCompletionRequest> for LLMGateway {
    type Reply = Result<CreateChatCompletionResponse>;

    async fn handle(
        &mut self,
        ChatCompletionRequest { request }: ChatCompletionRequest,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        // TODO: add structured tracing/metrics for outbound LLM calls.
        self.client.chat().create(request).await.map_err(Into::into)
    }
}
