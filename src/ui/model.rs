use ratatui::text::{Line, Span, Text};
use tui_textarea::TextArea;
use uuid::Uuid;

const MAX_LOG_LINES: usize = 3;

fn render_markdown(content: &str) -> Text<'static> {
    let parsed = tui_markdown::from_str(content);
    let owned_lines: Vec<Line<'static>> = parsed
        .lines
        .into_iter()
        .map(|line| {
            let spans: Vec<Span<'static>> = line
                .spans
                .into_iter()
                .map(|span| Span::styled(span.content.into_owned(), span.style))
                .collect();
            let mut owned = Line::from(spans);
            owned.style = line.style;
            owned.alignment = line.alignment;
            owned
        })
        .collect();
    Text::from(owned_lines)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    Tool,
    System,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub id:           Uuid,
    pub role:         Role,
    pub content:      String,
    pub rendered:     Option<Text<'static>>,
    pub is_streaming: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppStatus {
    Idle,
    Streaming,
    ToolCall(String),
    Error(String),
}

pub struct AppState<'a> {
    pub messages:      Vec<ChatMessage>,
    pub textarea:      TextArea<'a>,
    pub logs:          std::collections::VecDeque<String>,
    pub scroll_offset: u16,
    pub status:        AppStatus,
    pub should_quit:   bool,
    pub spinner_idx:   usize,
}

impl<'a> AppState<'a> {
    pub fn new() -> Self {
        let mut textarea = TextArea::default();
        textarea.set_placeholder_text("Type a message... (Enter to send, Shift+Enter for newline)");

        Self {
            messages: Vec::new(),
            textarea,
            logs: std::collections::VecDeque::with_capacity(MAX_LOG_LINES),
            scroll_offset: 0,
            status: AppStatus::Idle,
            should_quit: false,
            spinner_idx: 0,
        }
    }

    pub fn add_message(&mut self, role: Role, content: String) {
        let rendered = match role {
            Role::Assistant | Role::System | Role::Tool => Some(render_markdown(&content)),
            Role::User => None,
        };

        self.messages.push(ChatMessage {
            id: Uuid::new_v4(),
            role,
            content,
            rendered,
            is_streaming: false,
        });
    }

    pub fn append_streaming_chunk(&mut self, chunk: &str) {
        if let Some(last) = self.messages.last_mut()
            && last.is_streaming
        {
            last.content.push_str(chunk);
            last.rendered = Some(render_markdown(&last.content));
            return;
        }

        let rendered = render_markdown(chunk);
        self.messages.push(ChatMessage {
            id:           Uuid::new_v4(),
            role:         Role::Assistant,
            content:      chunk.to_string(),
            rendered:     Some(rendered),
            is_streaming: true,
        });
    }

    pub fn finalize_stream(&mut self) {
        if let Some(last) = self.messages.last_mut() {
            last.is_streaming = false;
        }
    }

    pub fn push_log(&mut self, line: String) {
        if self.logs.len() == MAX_LOG_LINES {
            self.logs.pop_front();
        }
        self.logs.push_back(line);
    }

    pub fn clear_input(&mut self) {
        let mut textarea = TextArea::default();
        textarea.set_placeholder_text("Type a message... (Enter to send, Shift+Enter for newline)");
        self.textarea = textarea;
    }
}

impl<'a> Default for AppState<'a> {
    fn default() -> Self {
        Self::new()
    }
}
