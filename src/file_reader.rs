use std::{env, path::PathBuf};

use anyhow::{anyhow, Result};
use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestSystemMessageContent,
        ChatCompletionRequestUserMessageArgs, ChatCompletionRequestUserMessageContent,
        CreateChatCompletionRequestArgs,
    },
    Client,
};
use kameo::prelude::*;
use uuid::Uuid;

/// `FileReader` actor loads file contents and can delegate questions to an LLM.
#[derive(Actor)]
pub struct FileReader {
    client: Client<OpenAIConfig>,
    model:  String,
}

impl FileReader {
    /// Build a new `FileReader` using environment configuration.
    pub fn from_env() -> Result<Self> {
        let model = env::var("OPENAI_MODEL")
            .map_err(|_| anyhow!("OPENAI_MODEL environment variable must be set"))?;

        let mut config = OpenAIConfig::default();
        if let Ok(url) = env::var("OPENAI_API_BASE") {
            config = config.with_api_base(url);
        }

        Ok(Self {
            client: Client::with_config(config),
            model,
        })
    }
}

/// Message instructing the actor to read the file at `path`.
pub struct ReadFile(pub PathBuf);

impl Message<ReadFile> for FileReader {
    type Reply = Result<String>;

    async fn handle(
        &mut self,
        ReadFile(path): ReadFile,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let contents = tokio::fs::read_to_string(path).await?;
        Ok(contents)
    }
}

/// Message requesting a test chat completion round-trip with the LLM.
pub struct TestChat {
    pub prompt: String,
}

impl Message<TestChat> for FileReader {
    type Reply = Result<String>;

    async fn handle(
        &mut self,
        TestChat { prompt }: TestChat,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let nonce = Uuid::new_v4();
        let prompt_with_nonce = format!(
            "{prompt}\n\
             Load-test token: {nonce}\n\
             \n\
             Craft an in-depth technical briefing (minimum 600 words) for fellow Weaver agents.\n\
             Structure the answer into clearly labeled sections covering:\n\
               1. Overview of the local repository layout with emphasis on 'uncc_cs2-pretext-project' subdirectories.\n\
               2. Recommended strategies for parallel, cache-friendly file summarization workflows.\n\
               3. Potential pitfalls when crawling large PreTeXt projects along with mitigation playbooks.\n\
               4. Creative heuristics for prioritizing files by pedagogical impact and novelty.\n\
               5. Checklist-style operational runbook for handing findings back to coordinating agents.\n\
             \n\
             Each section should contain multiple paragraphs rich in detail, concrete suggestions, and hypothetical examples.\n\
             Close with a reflective postscript that echoes the load-test token {nonce} twice to verify response uniqueness.\n",
            prompt = prompt,
            nonce = nonce
        );

        let request = CreateChatCompletionRequestArgs::default()
            .model(self.model.clone())
            .messages([
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(ChatCompletionRequestSystemMessageContent::Text(
                        "You are a verbose, detail-obsessed file-reading assistant. All responses must exceed 600 words and follow every structural directive in the latest user prompt."
                            .to_string(),
                    ))
                    .build()?
                    .into(),
                ChatCompletionRequestUserMessageArgs::default()
                    .content(ChatCompletionRequestUserMessageContent::Text(prompt_with_nonce))
                    .build()?
                    .into(),
            ])
            .build()?;

        let response = self.client.chat().create(request).await?;

        let reply = response
            .choices
            .into_iter()
            .filter_map(|choice| choice.message.content)
            .next()
            .ok_or_else(|| anyhow!("chat completion returned no content"))?;

        Ok(reply)
    }
}
