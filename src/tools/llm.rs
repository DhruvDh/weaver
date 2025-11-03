use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use async_openai::types::{
    ChatCompletionTool, ChatCompletionToolArgs, ChatCompletionToolType, FunctionObjectArgs,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use tracing::info;
use typed_builder::TypedBuilder;

use crate::{
    constants::DEFAULT_PARALLEL_DELEGATIONS,
    tools::{filesystem, search},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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

    pub fn schema(self) -> Value {
        match self {
            ToolName::ListDirectory => list_directory::ListDirectory::schema(),
            ToolName::ReadFileFull => read_file_full::ReadFileFull::schema(),
            ToolName::ReadFileRange => read_file_range::ReadFileRange::schema(),
            ToolName::SearchText => search_text::SearchText::schema(),
            ToolName::DelegateSubtask => delegate_subtask::DelegateSubtask::schema(),
        }
    }

    pub fn parse(self, value: Value) -> Result<ToolInvocation> {
        match self {
            ToolName::ListDirectory => {
                Ok(ToolInvocation::ListDirectory(list_directory::ListDirectory::parse(value)?))
            }
            ToolName::ReadFileFull => {
                Ok(ToolInvocation::ReadFileFull(read_file_full::ReadFileFull::parse(value)?))
            }
            ToolName::ReadFileRange => {
                Ok(ToolInvocation::ReadFileRange(read_file_range::ReadFileRange::parse(value)?))
            }
            ToolName::SearchText => {
                Ok(ToolInvocation::SearchText(search_text::SearchText::parse(value)?))
            }
            ToolName::DelegateSubtask => Ok(ToolInvocation::DelegateSubtask(
                delegate_subtask::DelegateSubtask::parse(value)?,
            )),
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

pub const fn all_tools() -> &'static [ToolName; 5] {
    &DEFAULT_TOOL_ORDER
}

pub fn tool_specs(names: &[ToolName]) -> Result<Vec<ChatCompletionTool>> {
    names
        .iter()
        .map(|tool| {
            ChatCompletionToolArgs::default()
                .r#type(ChatCompletionToolType::Function)
                .function(
                    FunctionObjectArgs::default()
                        .name(tool.identifier())
                        .description(tool.description())
                        .parameters(tool.schema())
                        .build()
                        .context("failed to build tool schema")?,
                )
                .build()
                .context("failed to build tool specification")
        })
        .collect()
}

#[derive(Debug, Clone)]
pub enum ToolInvocation {
    ListDirectory(list_directory::ListDirectory),
    ReadFileFull(read_file_full::ReadFileFull),
    ReadFileRange(read_file_range::ReadFileRange),
    SearchText(search_text::SearchText),
    DelegateSubtask(delegate_subtask::DelegateSubtask),
}

impl ToolInvocation {
    pub fn name(&self) -> ToolName {
        match self {
            ToolInvocation::ListDirectory(_) => ToolName::ListDirectory,
            ToolInvocation::ReadFileFull(_) => ToolName::ReadFileFull,
            ToolInvocation::ReadFileRange(_) => ToolName::ReadFileRange,
            ToolInvocation::SearchText(_) => ToolName::SearchText,
            ToolInvocation::DelegateSubtask(_) => ToolName::DelegateSubtask,
        }
    }

    pub fn into_action(self) -> Box<dyn ToolAction> {
        match self {
            ToolInvocation::ListDirectory(tool) => Box::new(tool),
            ToolInvocation::ReadFileFull(tool) => Box::new(tool),
            ToolInvocation::ReadFileRange(tool) => Box::new(tool),
            ToolInvocation::SearchText(tool) => Box::new(tool),
            ToolInvocation::DelegateSubtask(tool) => Box::new(tool),
        }
    }
}

#[async_trait]
pub trait ToolHost: Send + Sync {
    fn depth(&self) -> usize;
    fn max_subdelegations(&self) -> usize;
    fn workspace_root(&self) -> &Path;
    fn resolve_path(&self, relative: &str) -> Result<PathBuf>;
    fn render_relative_path(&self, path: &Path) -> String;
    async fn list_directory(&self, path: &Path) -> Result<Vec<filesystem::DirEntryInfo>>;
    async fn read_file_full(&self, path: &Path) -> Result<String>;
    async fn read_file_range(
        &self,
        path: &Path,
        start_line: usize,
        end_line: usize,
    ) -> Result<filesystem::FileRange>;
    async fn search_recursive(
        &self,
        scope: &Path,
        pattern: &str,
    ) -> Result<Vec<search::SearchMatch>>;
    async fn delegate_subtasks(&self, subtasks: Vec<String>) -> Result<Value>;
}

#[async_trait]
pub trait ToolAction: Send {
    async fn execute(self: Box<Self>, host: &dyn ToolHost) -> Result<Value>;
}

pub fn parse_invocation(identifier: &str, value: Value) -> Result<ToolInvocation> {
    let name = ToolName::from_identifier(identifier)
        .ok_or_else(|| anyhow!("unsupported tool call: {}", identifier))?;
    name.parse(value)
}

fn normalize_optional(input: Option<String>) -> Option<String> {
    input.and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn normalize_required(value: String, field: &str, tool: ToolName) -> Result<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        bail!("{} requires a non-empty `{}` argument", tool.identifier(), field);
    }
    Ok(trimmed.to_string())
}

mod list_directory {
    use super::*;

    #[derive(Debug, Clone, TypedBuilder, Deserialize)]
    #[builder(field_defaults(setter(into)))]
    #[serde(deny_unknown_fields)]
    pub struct ListDirectory {
        #[serde(default)]
        #[builder(default)]
        pub path: Option<String>,
    }

    impl ListDirectory {
        pub fn schema() -> Value {
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to workspace root. Defaults to \".\""
                    }
                }
            })
        }

        pub fn parse(value: Value) -> Result<Self> {
            let mut data: Self =
                serde_json::from_value(value).context("list_directory expects an object")?;
            data.path = super::normalize_optional(data.path);
            Ok(data)
        }
    }

    #[async_trait]
    impl ToolAction for ListDirectory {
        async fn execute(self: Box<Self>, host: &dyn ToolHost) -> Result<Value> {
            let Self { path } = *self;
            let relative = path.unwrap_or_else(|| ".".to_string());
            let resolved = host.resolve_path(&relative)?;
            let entries = host.list_directory(&resolved).await?;
            info!(
                "tool_call list_directory depth={} path={} entry_count={}",
                host.depth(),
                resolved.display(),
                entries.len()
            );
            let rendered = entries
                .into_iter()
                .map(|entry| {
                    json!({
                        "name": entry.name,
                        "path": host.render_relative_path(&entry.path),
                        "kind": entry.kind.as_str(),
                        "size": entry.size,
                    })
                })
                .collect::<Vec<_>>();
            Ok(json!({ "entries": rendered }))
        }
    }
}

mod read_file_full {
    use super::*;

    #[derive(Debug, Clone, TypedBuilder, Deserialize)]
    #[builder(field_defaults(setter(into)))]
    #[serde(deny_unknown_fields)]
    pub struct ReadFileFull {
        pub path: String,
    }

    impl ReadFileFull {
        pub fn schema() -> Value {
            json!({
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to workspace root."
                    }
                }
            })
        }

        pub fn parse(value: Value) -> Result<Self> {
            let data: Self = serde_json::from_value(value)
                .context("read_file_full expects an object with `path`")?;
            let path = super::normalize_required(data.path, "path", ToolName::ReadFileFull)?;
            Ok(Self { path })
        }
    }

    #[async_trait]
    impl ToolAction for ReadFileFull {
        async fn execute(self: Box<Self>, host: &dyn ToolHost) -> Result<Value> {
            let Self { path } = *self;
            let resolved = host.resolve_path(&path)?;
            let content = host.read_file_full(&resolved).await?;
            info!(
                "tool_call read_file_full depth={} path={} bytes={}",
                host.depth(),
                resolved.display(),
                content.len()
            );
            Ok(json!({
                "path": host.render_relative_path(&resolved),
                "content": content,
            }))
        }
    }
}

mod read_file_range {
    use super::*;

    #[derive(Debug, Clone, TypedBuilder, Deserialize)]
    #[builder(field_defaults(setter(into)))]
    #[serde(deny_unknown_fields)]
    pub struct ReadFileRange {
        pub path:       String,
        pub start_line: usize,
        pub end_line:   usize,
    }

    impl ReadFileRange {
        pub fn schema() -> Value {
            json!({
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
            })
        }

        pub fn parse(value: Value) -> Result<Self> {
            let data: Self = serde_json::from_value(value)
                .context("read_file_range expects `path`, `start_line`, `end_line`")?;
            if data.start_line == 0 {
                bail!("read_file_range requires `start_line` >= 1");
            }
            if data.end_line == 0 {
                bail!("read_file_range requires `end_line` >= 1");
            }
            if data.end_line < data.start_line {
                bail!("read_file_range requires `end_line` >= `start_line`");
            }
            let path = super::normalize_required(data.path, "path", ToolName::ReadFileRange)?;
            Ok(Self {
                path,
                start_line: data.start_line,
                end_line: data.end_line,
            })
        }
    }

    #[async_trait]
    impl ToolAction for ReadFileRange {
        async fn execute(self: Box<Self>, host: &dyn ToolHost) -> Result<Value> {
            let Self {
                path,
                start_line,
                end_line,
            } = *self;
            let resolved = host.resolve_path(&path)?;
            let range = host
                .read_file_range(&resolved, start_line, end_line)
                .await?;
            let filesystem::FileRange {
                path: range_path,
                start_line: first,
                end_line: last,
                text,
            } = range;
            let line_count = last.saturating_sub(first) + 1;
            info!(
                "tool_call read_file_range depth={} path={} start_line={} end_line={} \
                 line_count={}",
                host.depth(),
                range_path.display(),
                first,
                last,
                line_count
            );
            Ok(json!({
                "path": host.render_relative_path(&range_path),
                "start_line": first,
                "end_line": last,
                "content": text,
            }))
        }
    }
}

mod search_text {
    use super::*;

    #[derive(Debug, Clone, TypedBuilder, Deserialize)]
    #[builder(field_defaults(setter(into)))]
    #[serde(deny_unknown_fields)]
    pub struct SearchText {
        pub pattern: String,
        #[serde(default)]
        #[builder(default)]
        pub path:    Option<String>,
    }

    impl SearchText {
        pub fn schema() -> Value {
            json!({
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
            })
        }

        pub fn parse(value: Value) -> Result<Self> {
            let mut data: Self = serde_json::from_value(value)
                .context("search_text expects an object with `pattern`")?;
            let pattern = super::normalize_required(data.pattern, "pattern", ToolName::SearchText)?;
            data.path = super::normalize_optional(data.path);
            Ok(Self {
                pattern,
                path: data.path,
            })
        }
    }

    #[async_trait]
    impl ToolAction for SearchText {
        async fn execute(self: Box<Self>, host: &dyn ToolHost) -> Result<Value> {
            let Self { pattern, path } = *self;
            let scope = match path {
                Some(relative) => host.resolve_path(&relative)?,
                None => host.workspace_root().to_path_buf(),
            };
            let matches = host.search_recursive(&scope, &pattern).await?;
            info!(
                "tool_call search_text depth={} scope={} pattern={} match_count={}",
                host.depth(),
                scope.display(),
                pattern,
                matches.len()
            );
            let rendered = matches
                .into_iter()
                .map(|m| {
                    json!({
                        "path": host.render_relative_path(&m.path),
                        "line_number": m.line_number,
                        "line": m.context,
                    })
                })
                .collect::<Vec<_>>();
            Ok(json!({ "matches": rendered }))
        }
    }
}

mod delegate_subtask {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct DelegateSubtask {
        pub subtasks: Vec<String>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct DelegateRaw {
        #[serde(default)]
        subtask:  Option<String>,
        #[serde(default)]
        subtasks: Vec<String>,
    }

    impl DelegateSubtask {
        pub fn schema() -> Value {
            json!({
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
            })
        }

        pub fn parse(value: Value) -> Result<Self> {
            let raw: DelegateRaw =
                serde_json::from_value(value).context("delegate_subtask expects an object")?;
            let mut subtasks = Vec::new();
            if let Some(single) = super::normalize_optional(raw.subtask) {
                subtasks.push(single);
            }
            for entry in raw.subtasks {
                if let Some(trimmed) = super::normalize_optional(Some(entry)) {
                    subtasks.push(trimmed);
                }
            }
            if subtasks.is_empty() {
                bail!("delegate_subtask requires at least one non-empty subtask");
            }
            Ok(Self { subtasks })
        }
    }

    #[async_trait]
    impl ToolAction for DelegateSubtask {
        async fn execute(self: Box<Self>, host: &dyn ToolHost) -> Result<Value> {
            let Self { subtasks } = *self;
            if host.depth() >= host.max_subdelegations() {
                bail!(
                    "delegate_subtask limit reached (depth {} >= {})",
                    host.depth(),
                    host.max_subdelegations()
                );
            }
            info!(
                "tool_call delegate_subtask depth={} subtasks={} max_concurrency={}",
                host.depth() + 1,
                subtasks.len(),
                DEFAULT_PARALLEL_DELEGATIONS
            );
            host.delegate_subtasks(subtasks).await
        }
    }
}
