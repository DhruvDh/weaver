pub mod events;
pub mod model;
pub mod tui;

pub use events::{BackendEvent, UiAction};
pub use model::{AppState, AppStatus, ChatMessage, Role};
