use anyhow::{Context, Result, bail};
use async_openai::types::{
    ChatCompletionTool, ChatCompletionToolArgs, ChatCompletionToolType, FunctionObjectArgs,
};
use serde::Deserialize;
use serde_json::{Value, json};

/// Enumeration of all tools available to LLM collaborators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ToolName {
    ListDirectory,
    ReadFileFull,
    ReadFileRange,
    SearchText,
    DelegateSubtask,
}

impl ToolName {
    pub const fn identifier(self) -> &'static str {
        match self {
            ToolName::ListDirectory => "list_directory",
            ToolName::ReadFileFull => "read_file_full",
            ToolName::ReadFileRange => "read_file_range",
            ToolName::SearchText => "search_text",
            ToolName::DelegateSubtask => "delegate_subtask",
        }
    }

    pub const fn description(self) -> &'static str {
        match self {
            ToolName::ListDirectory => {
                "List the entries of a directory relative to the workspace root."
            }
            ToolName::ReadFileFull => "Read the full contents of a UTF-8 text file.",
            ToolName::ReadFileRange => {
                "Read a specific inclusive line range from a UTF-8 text file."
            }
            ToolName::SearchText => "Run a regex search (ripgrep-style) within the workspace.",
            ToolName::DelegateSubtask => {
                "Delegate one or more subtasks to child FileReader agents via the `subtasks` \
                 array. Wrap single subtasks in an array when needed."
            }
        }
    }

    pub fn from_identifier(name: &str) -> Option<Self> {
        match name {
            "list_directory" => Some(ToolName::ListDirectory),
            "read_file_full" => Some(ToolName::ReadFileFull),
            "read_file_range" => Some(ToolName::ReadFileRange),
            "search_text" => Some(ToolName::SearchText),
            "delegate_subtask" => Some(ToolName::DelegateSubtask),
            _ => None,
        }
    }
}

const DEFAULT_TOOL_ORDER: [ToolName; 5] = [
    ToolName::ListDirectory,
    ToolName::ReadFileFull,
    ToolName::ReadFileRange,
    ToolName::SearchText,
    ToolName::DelegateSubtask,
];

/// Returns the canonical tool order used by default sessions.
pub const fn all_tools() -> &'static [ToolName; 5] {
    &DEFAULT_TOOL_ORDER
}

fn schema_for(tool: ToolName) -> Value {
    match tool {
        ToolName::ListDirectory => json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to workspace root. Defaults to \".\""
                }
            }
        }),
        ToolName::ReadFileFull => json!({
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to workspace root."
                }
            }
        }),
        ToolName::ReadFileRange => json!({
            "type": "object",
            "required": ["path", "start_line", "end_line"],
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to workspace root."
                },
                "start_line": {
                    "type": "integer",
                    "minimum": 1
                },
                "end_line": {
                    "type": "integer",
                    "minimum": 1
                }
            }
        }),
        ToolName::SearchText => json!({
            "type": "object",
            "required": ["pattern"],
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Rust-style regular expression."
                },
                "path": {
                    "type": "string",
                    "description": "Optional directory to scope the search. Defaults to root."
                }
            }
        }),
        ToolName::DelegateSubtask => json!({
            "type": "object",
            "required": ["subtasks"],
            "properties": {
                "subtask": {
                    "type": "string",
                    "description": "Instruction for a delegated FileReader agent."
                },
                "subtasks": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Precise, educational description with motivation and acceptance criteria of subtasks to delegate in parallel."
                }
            }
        }),
    }
}

/// Constructs OpenAI tool specs for the supplied tool list.
pub fn chat_tools(names: &[ToolName]) -> Result<Vec<ChatCompletionTool>> {
    names
        .iter()
        .map(|tool| {
            ChatCompletionToolArgs::default()
                .r#type(ChatCompletionToolType::Function)
                .function(
                    FunctionObjectArgs::default()
                        .name(tool.identifier())
                        .description(tool.description())
                        .parameters(schema_for(*tool))
                        .build()
                        .context("failed to build tool function schema")?,
                )
                .build()
                .context("failed to build chat tool specification")
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListDirectoryArgs {
    pub path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadFileFullArgs {
    pub path: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadFileRangeArgs {
    pub path:       String,
    pub start_line: usize,
    pub end_line:   usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchTextArgs {
    pub pattern: String,
    pub path:    Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DelegateSubtaskArgs {
    pub subtasks: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolArgs {
    ListDirectory(ListDirectoryArgs),
    ReadFileFull(ReadFileFullArgs),
    ReadFileRange(ReadFileRangeArgs),
    SearchText(SearchTextArgs),
    DelegateSubtask(DelegateSubtaskArgs),
}

impl ToolArgs {
    pub fn name(&self) -> ToolName {
        match self {
            ToolArgs::ListDirectory(_) => ToolName::ListDirectory,
            ToolArgs::ReadFileFull(_) => ToolName::ReadFileFull,
            ToolArgs::ReadFileRange(_) => ToolName::ReadFileRange,
            ToolArgs::SearchText(_) => ToolName::SearchText,
            ToolArgs::DelegateSubtask(_) => ToolName::DelegateSubtask,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ListDirectoryRaw {
    path: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadFileFullRaw {
    path: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadFileRangeRaw {
    path:       String,
    start_line: usize,
    end_line:   usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchTextRaw {
    pattern: String,
    path:    Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DelegateSubtaskRaw {
    subtask:  Option<String>,
    #[serde(default)]
    subtasks: Vec<String>,
}

fn trim_optional(text: Option<String>) -> Option<String> {
    text.and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn trim_required(value: String, field: &str, tool: ToolName) -> Result<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        bail!("{} requires a non-empty `{}` argument", tool.identifier(), field);
    }
    Ok(trimmed.to_string())
}

/// Parse and validate tool arguments returning strongly typed payloads.
pub fn parse_args(tool: ToolName, value: &Value) -> Result<ToolArgs> {
    match tool {
        ToolName::ListDirectory => {
            let raw: ListDirectoryRaw = serde_json::from_value(value.clone())
                .context("list_directory expects an object")?;
            let path = trim_optional(raw.path);
            Ok(ToolArgs::ListDirectory(ListDirectoryArgs { path }))
        }
        ToolName::ReadFileFull => {
            let raw: ReadFileFullRaw = serde_json::from_value(value.clone())
                .context("read_file_full expects an object with `path`")?;
            let path = trim_required(raw.path, "path", tool)?;
            Ok(ToolArgs::ReadFileFull(ReadFileFullArgs { path }))
        }
        ToolName::ReadFileRange => {
            let raw: ReadFileRangeRaw = serde_json::from_value(value.clone())
                .context("read_file_range expects `path`, `start_line`, `end_line`")?;
            if raw.start_line == 0 {
                bail!("read_file_range requires `start_line` >= 1");
            }
            if raw.end_line == 0 {
                bail!("read_file_range requires `end_line` >= 1");
            }
            if raw.end_line < raw.start_line {
                bail!("read_file_range requires `end_line` >= `start_line`");
            }
            let path = trim_required(raw.path, "path", tool)?;
            Ok(ToolArgs::ReadFileRange(ReadFileRangeArgs {
                path,
                start_line: raw.start_line,
                end_line: raw.end_line,
            }))
        }
        ToolName::SearchText => {
            let raw: SearchTextRaw = serde_json::from_value(value.clone())
                .context("search_text expects an object with `pattern`")?;
            let pattern = trim_required(raw.pattern, "pattern", tool)?;
            let path = trim_optional(raw.path);
            Ok(ToolArgs::SearchText(SearchTextArgs { pattern, path }))
        }
        ToolName::DelegateSubtask => {
            let raw: DelegateSubtaskRaw = serde_json::from_value(value.clone())
                .context("delegate_subtask expects `subtasks` array")?;
            let mut subtasks = Vec::new();
            if let Some(single) = raw.subtask
                && let Some(trimmed) = trim_optional(Some(single))
            {
                subtasks.push(trimmed);
            }
            for entry in raw.subtasks {
                if let Some(trimmed) = trim_optional(Some(entry)) {
                    subtasks.push(trimmed);
                }
            }
            if subtasks.is_empty() {
                bail!("delegate_subtask requires at least one non-empty subtask");
            }
            Ok(ToolArgs::DelegateSubtask(DelegateSubtaskArgs { subtasks }))
        }
    }
}
