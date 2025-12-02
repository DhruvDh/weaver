use std::time::Duration;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use futures::StreamExt;
use ratatui::{
    DefaultTerminal, Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};
use tokio::{sync::mpsc, time};

use super::{
    events::{BackendEvent, UiAction},
    model::{AppState, AppStatus, Role},
};

pub async fn run(
    mut terminal: DefaultTerminal,
    action_tx: mpsc::UnboundedSender<UiAction>,
    mut event_rx: mpsc::UnboundedReceiver<BackendEvent>,
    initial_prompt: Option<String>,
) -> Result<()> {
    let mut app = AppState::new();
    if let Some(prompt) = initial_prompt {
        app.add_message(Role::User, prompt.clone());
        app.status = AppStatus::Streaming;
        let _ = action_tx.send(UiAction::SendMessage(prompt));
    }

    let mut event_stream = event::EventStream::new();
    let mut tick_rate = time::interval(Duration::from_millis(100));

    loop {
        terminal.draw(|f| ui(f, &mut app))?;

        tokio::select! {
            Some(evt) = event_rx.recv() => {
                handle_backend_event(evt, &mut app);
            }
            Some(Ok(evt)) = event_stream.next() => {
                handle_input(evt, &action_tx, &mut app);
            }
            _ = tick_rate.tick() => {
                app.spinner_idx = (app.spinner_idx + 1) % 4;
            }
            else => break,
        }

        if app.should_quit {
            let _ = action_tx.send(UiAction::Exit);
            break;
        }
    }

    Ok(())
}

fn handle_backend_event(event: BackendEvent, app: &mut AppState<'_>) {
    match event {
        BackendEvent::TokenChunk(text) => {
            app.append_streaming_chunk(&text);
            app.status = AppStatus::Streaming;
        }
        BackendEvent::ToolCallStarted(tool) => {
            app.status = AppStatus::ToolCall(tool);
        }
        BackendEvent::ToolOutput(output) => {
            app.add_message(Role::Tool, output);
        }
        BackendEvent::RequestComplete => {
            app.status = AppStatus::Idle;
            app.finalize_stream();
        }
        BackendEvent::Error(err) => {
            let display = if err.len() > 120 {
                format!("{}...", &err[..117])
            } else {
                err.clone()
            };
            app.status = AppStatus::Error(display);
            app.add_message(Role::System, format!("LLM request failed: {err}"));
            app.finalize_stream();
        }
        BackendEvent::Log(line) => {
            app.push_log(line);
        }
    }
}

fn handle_input(event: Event, action_tx: &mpsc::UnboundedSender<UiAction>, app: &mut AppState<'_>) {
    if let Event::Key(key) = event
        && key.kind == KeyEventKind::Press
    {
        match (key.code, key.modifiers) {
            (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                app.should_quit = true;
            }
            (KeyCode::Char('x'), KeyModifiers::CONTROL) => {
                if app.status != AppStatus::Idle {
                    let _ = action_tx.send(UiAction::CancelRequest);
                    app.status = AppStatus::Idle;
                    app.add_message(Role::System, "Cancelled by user.".into());
                    app.finalize_stream();
                }
            }
            (KeyCode::Enter, KeyModifiers::NONE) => {
                let input = app.textarea.lines().join("\n");
                if !input.trim().is_empty() {
                    app.add_message(Role::User, input.clone());
                    app.status = AppStatus::Streaming;
                    app.scroll_offset = 0;
                    let _ = action_tx.send(UiAction::SendMessage(input));
                    app.clear_input();
                }
            }
            (KeyCode::Enter, KeyModifiers::SHIFT) => app.textarea.insert_newline(),
            (KeyCode::Up, _) => {
                if is_input_empty(app)
                    && let Some(pos) = app.messages.iter().rposition(|m| m.role == Role::User)
                    && let Some(msg) = app.messages.get(pos).cloned()
                {
                    app.messages.truncate(pos);
                    app.textarea = tui_textarea::TextArea::from(msg.content.lines());
                } else {
                    app.textarea.input(key);
                }
            }
            (KeyCode::PageUp, _) => {
                app.scroll_offset = app.scroll_offset.saturating_add(1);
            }
            (KeyCode::PageDown, _) => {
                app.scroll_offset = app.scroll_offset.saturating_sub(1);
            }
            _ => {
                app.textarea.input(key);
            }
        }
    }
}

fn ui(f: &mut Frame, app: &mut AppState<'_>) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),    // History
            Constraint::Length(1), // Status
            Constraint::Length(3), // Input
            Constraint::Length(4), // Logs
        ])
        .split(f.area());

    render_history(f, app, chunks[0]);
    render_status(f, app, chunks[1]);

    let input_block = Block::default().borders(Borders::ALL).title(" Input ");
    app.textarea.set_block(input_block);
    f.render_widget(&app.textarea, chunks[2]);

    render_logs(f, app, chunks[3]);
}

fn render_history(f: &mut Frame, app: &AppState<'_>, area: Rect) {
    let mut lines = Vec::new();

    for msg in &app.messages {
        let header = match msg.role {
            Role::User => {
                Line::from(vec![Span::styled("You", Style::default().fg(Color::Cyan).bold())])
            }
            Role::Assistant => Line::from(vec![Span::styled(
                "Weaver",
                Style::default().fg(Color::Green).bold(),
            )]),
            Role::Tool => {
                Line::from(vec![Span::styled("Tool", Style::default().fg(Color::Yellow))])
            }
            Role::System => {
                Line::from(vec![Span::styled("System", Style::default().fg(Color::Gray))])
            }
        };
        lines.push(header);

        if let Some(md) = &msg.rendered {
            for line in md.lines.iter() {
                lines.push(line.clone());
            }
        } else {
            for line in msg.content.lines() {
                lines.push(Line::from(line.to_string()));
            }
        }
        lines.push(Line::from(String::new()));
    }

    let total_height = lines.len() as u16;
    let view_height = area.height;
    let scroll_anchor = total_height.saturating_sub(view_height);
    let scroll_pos = scroll_anchor.saturating_sub(app.scroll_offset);

    let paragraph = Paragraph::new(lines)
        .wrap(Wrap { trim: true })
        .scroll((scroll_pos, 0));

    f.render_widget(paragraph, area);
}

fn render_status(f: &mut Frame, app: &AppState<'_>, area: Rect) {
    let spinner_chars = ["|", "/", "-", "\\"];
    let spinner = spinner_chars[app.spinner_idx % spinner_chars.len()];

    let content = match &app.status {
        AppStatus::Idle => Span::raw(" Ready "),
        AppStatus::Streaming => {
            Span::styled(format!(" {} Generating... ", spinner), Style::default().fg(Color::Green))
        }
        AppStatus::ToolCall(tool) => Span::styled(
            format!(" {} Running {tool} ", spinner),
            Style::default().fg(Color::Yellow),
        ),
        AppStatus::Error(err) => {
            Span::styled(format!(" Error: {err} "), Style::default().bg(Color::Red))
        }
    };

    f.render_widget(Paragraph::new(content), area);
}

fn render_logs(f: &mut Frame, app: &AppState<'_>, area: Rect) {
    let block = Block::default().borders(Borders::ALL).title(" Logs ");
    let mut lines = Vec::new();
    for log in app.logs.iter().rev().take(3).rev() {
        lines.push(Line::from(log.clone()));
    }
    let paragraph = Paragraph::new(lines).block(block).wrap(Wrap { trim: true });
    f.render_widget(paragraph, area);
}

fn is_input_empty(app: &AppState<'_>) -> bool {
    let lines = app.textarea.lines();
    lines.len() == 1 && lines.first().map(|l| l.is_empty()).unwrap_or(true)
}
