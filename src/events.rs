use uuid::Uuid;

use crate::model::{EdgeProposal, NodeKind, NodeProposal, Relation};

/// Domain-level events broadcast by the GraphAdder after every decision so
/// observers can remain in sync without depending on visualization details.
#[derive(Debug, Clone)]
pub enum DomainEvent {
    NodeAccepted {
        id:    Uuid,
        kind:  NodeKind,
        level: u8,
        tags:  Option<Vec<String>>,
        text:  String,
    },
    NodeRejected {
        proposal: NodeProposal,
        reason:   String,
    },
    EdgeAccepted {
        relation:  Relation,
        from:      Uuid,
        to:        Uuid,
        rationale: String,
    },
    EdgeRejected {
        proposal: EdgeProposal,
        reason:   String,
    },
    SummaryLine {
        message: String,
    },
}
