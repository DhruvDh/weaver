use thiserror::Error;

use super::types::{
    AssessmentScope, EvidenceLink, IntendedEffect, KnowledgeType, SourceRef, Strength, SupportKind,
};

#[derive(Debug, Error)]
pub enum SchemaError {
    #[error("{edge} edges cannot originate from {kind}")]
    InvalidEdgeFrom {
        edge: &'static str,
        kind: KnowledgeType,
    },
    #[error("{edge} edges cannot terminate at {kind}")]
    InvalidEdgeTo {
        edge: &'static str,
        kind: KnowledgeType,
    },
    #[error("{field} must not be empty")]
    EmptyField { field: &'static str },
    #[error("source_ref line range is invalid: start {start_line} > end {end_line}")]
    InvalidLineRange { start_line: u32, end_line: u32 },
    #[error("source_ref revision must be a 7-40 character hexadecimal git hash")]
    InvalidRevision,
    #[error("observation_features must contain at least one entry for {edge}")]
    MissingObservationFeature { edge: &'static str },
}

pub fn validate_source_ref(span: &SourceRef) -> Result<(), SchemaError> {
    if span.start_line == 0 {
        return Err(SchemaError::InvalidLineRange {
            start_line: span.start_line,
            end_line:   span.end_line,
        });
    }
    if span.end_line < span.start_line {
        return Err(SchemaError::InvalidLineRange {
            start_line: span.start_line,
            end_line:   span.end_line,
        });
    }
    ensure_non_empty(&span.path, "source_ref.path")?;
    ensure_non_empty(&span.revision, "source_ref.revision")?;
    if !is_git_hash(&span.revision) {
        return Err(SchemaError::InvalidRevision);
    }
    Ok(())
}

pub fn validate_requires(
    from: KnowledgeType,
    to: KnowledgeType,
    strength: Strength,
    rationale: &str,
    evidence_refs: &[SourceRef],
) -> Result<(), SchemaError> {
    ensure_non_empty(rationale, "requires.rationale")?;
    if evidence_refs.is_empty() {
        return Err(SchemaError::EmptyField {
            field: "requires.evidence_refs",
        });
    }
    match strength {
        Strength::Necessary | Strength::Strong | Strength::Helpful => {
            // enum exhaustiveness keeps this expression meaningful.
        }
    }

    if from.is_learning_outcome() || from.is_assessment_item() {
        return Err(SchemaError::InvalidEdgeFrom {
            edge: "requires",
            kind: from,
        });
    }
    if to.is_learning_outcome() {
        return Err(SchemaError::InvalidEdgeTo {
            edge: "requires",
            kind: to,
        });
    }
    for span in evidence_refs {
        validate_source_ref(span)?;
    }
    Ok(())
}

pub fn validate_supports(
    from: KnowledgeType,
    to: KnowledgeType,
    kind: SupportKind,
    intended_effect: IntendedEffect,
    evidence_refs: &[SourceRef],
) -> Result<(), SchemaError> {
    if evidence_refs.is_empty() {
        return Err(SchemaError::EmptyField {
            field: "supports.evidence_refs",
        });
    }
    if !from.is_instructional_knowledge() {
        return Err(SchemaError::InvalidEdgeFrom {
            edge: "supports",
            kind: from,
        });
    }
    if to.is_assessment_item() {
        let allowed =
            kind == SupportKind::RubricNote && matches!(intended_effect, IntendedEffect::Motivate);
        if !allowed {
            return Err(SchemaError::InvalidEdgeTo {
                edge: "supports",
                kind: to,
            });
        }
    }
    match kind {
        SupportKind::RubricNote
        | SupportKind::Analogy
        | SupportKind::Counterexample
        | SupportKind::MisconceptionFix
        | SupportKind::StrategyHint
        | SupportKind::WorkedExample => {
            // ensures exhaustive pattern; intentionally empty
        }
    }
    match intended_effect {
        IntendedEffect::ReduceExtraneousLoad
        | IntendedEffect::IncreaseGermaneLoad
        | IntendedEffect::Motivate
        | IntendedEffect::Contrast => {} // asserts enum exhaustiveness
    }
    for span in evidence_refs {
        validate_source_ref(span)?;
    }
    Ok(())
}

pub fn validate_assesses(
    from: KnowledgeType,
    to: KnowledgeType,
    evidence: &EvidenceLink,
) -> Result<(), SchemaError> {
    if !from.is_assessment_item() {
        return Err(SchemaError::InvalidEdgeFrom {
            edge: "assesses",
            kind: from,
        });
    }
    if !to.is_learning_outcome() {
        return Err(SchemaError::InvalidEdgeTo {
            edge: "assesses",
            kind: to,
        });
    }
    ensure_non_empty(&evidence.claim, "assesses.evidence_link.claim")?;
    if evidence.observation_features.is_empty()
        || evidence
            .observation_features
            .iter()
            .any(|entry| entry.trim().is_empty())
    {
        return Err(SchemaError::MissingObservationFeature { edge: "assesses" });
    }
    match evidence.scope {
        AssessmentScope::Target | AssessmentScope::Enabling => {}
    }
    Ok(())
}

fn ensure_non_empty(value: &str, field: &'static str) -> Result<(), SchemaError> {
    if value.trim().is_empty() {
        Err(SchemaError::EmptyField { field })
    } else {
        Ok(())
    }
}

fn is_git_hash(value: &str) -> bool {
    let len = value.len();
    if !(7..=40).contains(&len) {
        return false;
    }
    value.bytes().all(|b| b.is_ascii_hexdigit())
}
