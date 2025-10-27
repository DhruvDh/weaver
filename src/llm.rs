use std::env;

use async_openai::{
    Client,
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs, ResponseFormat, ResponseFormatJsonSchema,
    },
};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::model::{EdgeProposal, InventoryEntry, NodeKind, NodeProposal};

/// Errors surfaced when interacting with the LLM backend.
#[derive(Debug, Error)]
pub enum LlmError {
    #[error("LLM usage disabled")]
    Disabled,
    #[error("missing OPENAI_API_KEY in environment")]
    MissingApiKey,
    #[error("LLM call failed: {0}")]
    RequestFailed(String),
    #[error("failed to parse LLM response: {0}")]
    InvalidResponse(String),
}

/// Client used by generators to reach the LLM backend.
#[derive(Debug, Clone)]
pub struct LlmClient {
    client: Client<OpenAIConfig>,
    model:  String,
}

impl LlmClient {
    pub fn new(use_llm: bool) -> Result<Self, LlmError> {
        if !use_llm {
            return Err(LlmError::Disabled);
        }

        let api_key = env::var("OPENAI_API_KEY").map_err(|_| LlmError::MissingApiKey)?;
        let base_url = env::var("OPENAI_BASE_URL").ok();
        let model = env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());

        let mut openai_config = OpenAIConfig::new().with_api_key(api_key);
        if let Some(url) = base_url {
            openai_config = openai_config.with_api_base(url);
        }

        let client = Client::with_config(openai_config);

        Ok(Self { client, model })
    }

    pub async fn generate_nodes(
        &self,
        concepts: usize,
        learning_outcomes: usize,
    ) -> Result<Vec<NodeProposal>, LlmError> {
        let system_prompt = r#"You produce placeholder educational nodes for a learning network.
Rules:
- Emit pure JSON matching the provided schema exactly.
- Each node is a standalone statement that can be understood without citations.
- "kind" must be either "Concept" or "LearningOutcome".
- "granularity" must be "Sentence".
- Avoid duplicates; vary vocabulary.
- Learning outcomes MUST start with "I can " or "Students can ".
- Match the requested counts for each node type."#;

        let user_prompt = format!(
            "Produce exactly {concepts} Concept nodes and {learning_outcomes} LearningOutcome \
             nodes. Return ONLY JSON that satisfies the schema.",
            concepts = concepts,
            learning_outcomes = learning_outcomes
        );

        let system_message = ChatCompletionRequestSystemMessageArgs::default()
            .content(system_prompt)
            .build()
            .map_err(|err| LlmError::RequestFailed(err.to_string()))?;
        let user_message = ChatCompletionRequestUserMessageArgs::default()
            .content(user_prompt)
            .build()
            .map_err(|err| LlmError::RequestFailed(err.to_string()))?;

        let schema = schema_for!(NodeBatchPayload);
        let schema_value = serde_json::to_value(&schema)
            .map_err(|err| LlmError::RequestFailed(err.to_string()))?;

        let response_format = ResponseFormat::JsonSchema {
            json_schema: ResponseFormatJsonSchema {
                name:        "node_batch".into(),
                description: Some("List of node proposals".into()),
                schema:      Some(schema_value),
                strict:      Some(true),
            },
        };

        let request = CreateChatCompletionRequestArgs::default()
            .model(self.model.clone())
            .messages(vec![system_message.into(), user_message.into()])
            .temperature(0.2)
            .response_format(response_format)
            .build()
            .map_err(|err| LlmError::RequestFailed(err.to_string()))?;

        let response = self
            .client
            .chat()
            .create(request)
            .await
            .map_err(|err| LlmError::RequestFailed(err.to_string()))?;

        let content = response
            .choices
            .first()
            .and_then(|choice| choice.message.content.clone())
            .ok_or_else(|| LlmError::InvalidResponse("missing content".into()))?;

        let batch: NodeBatchPayload = serde_json::from_str(&content)
            .map_err(|err| LlmError::InvalidResponse(err.to_string()))?;

        Ok(batch.nodes)
    }

    pub async fn generate_edges(
        &self,
        inventory: &[InventoryEntry],
        target_edges: usize,
    ) -> Result<Vec<EdgeProposal>, LlmError> {
        let inventory_items: Vec<InventoryItem> = inventory
            .iter()
            .map(|(id, kind, level, text, tags)| InventoryItem {
                id:    *id,
                kind:  kind.clone(),
                level: *level,
                text:  text.clone(),
                tags:  tags.clone(),
            })
            .collect();
        let inventory_json = serde_json::to_string_pretty(&inventory_items)
            .map_err(|err| LlmError::InvalidResponse(err.to_string()))?;

        let system_prompt = r#"You propose placeholder edges among existing nodes.
Rules:
- Emit pure JSON matching the provided schema exactly.
- Edge kinds: "PrerequisiteFor" or "Supports" (use exact casing).
- Use from_id and to_id copied exactly from the provided inventory of UUIDs.
- For "PrerequisiteFor", prefer foundational → advanced concepts or concept → learning outcome.
- Include a concise rationale string for every edge.
- Aim for the requested number of edges; it is OK to return fewer but avoid duplicates."#;

        let user_prompt = format!(
            "Accepted nodes (JSON array):\n{}\nRequested edge count: {}\nReturn ONLY JSON that \
             satisfies the schema.",
            inventory_json, target_edges
        );

        let system_message = ChatCompletionRequestSystemMessageArgs::default()
            .content(system_prompt)
            .build()
            .map_err(|err| LlmError::RequestFailed(err.to_string()))?;
        let user_message = ChatCompletionRequestUserMessageArgs::default()
            .content(user_prompt)
            .build()
            .map_err(|err| LlmError::RequestFailed(err.to_string()))?;

        let schema = schema_for!(EdgeBatchPayload);
        let schema_value = serde_json::to_value(&schema)
            .map_err(|err| LlmError::RequestFailed(err.to_string()))?;

        let response_format = ResponseFormat::JsonSchema {
            json_schema: ResponseFormatJsonSchema {
                name:        "edge_batch".into(),
                description: Some("List of edge proposals".into()),
                schema:      Some(schema_value),
                strict:      Some(true),
            },
        };

        let request = CreateChatCompletionRequestArgs::default()
            .model(self.model.clone())
            .messages(vec![system_message.into(), user_message.into()])
            .temperature(0.2)
            .response_format(response_format)
            .build()
            .map_err(|err| LlmError::RequestFailed(err.to_string()))?;

        let response = self
            .client
            .chat()
            .create(request)
            .await
            .map_err(|err| LlmError::RequestFailed(err.to_string()))?;

        let content = response
            .choices
            .first()
            .and_then(|choice| choice.message.content.clone())
            .ok_or_else(|| LlmError::InvalidResponse("missing content".into()))?;

        let batch: EdgeBatchPayload = serde_json::from_str(&content)
            .map_err(|err| LlmError::InvalidResponse(err.to_string()))?;

        Ok(batch.edges)
    }
}

#[derive(Debug, Clone, Serialize)]
struct InventoryItem {
    id:    uuid::Uuid,
    kind:  NodeKind,
    level: u8,
    text:  String,
    tags:  Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct NodeBatchPayload {
    nodes: Vec<NodeProposal>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct EdgeBatchPayload {
    edges: Vec<EdgeProposal>,
}
