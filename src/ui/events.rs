#[derive(Debug)]
pub enum UiAction {
    SendMessage(String),
    CancelRequest,
    Exit,
}

#[derive(Debug)]
pub enum BackendEvent {
    TokenChunk(String),
    ToolCallStarted(String),
    ToolOutput(String),
    RequestComplete,
    Error(String),
    Log(String),
}
