use std::fmt;

use crate::{constants::MAX_SLUG_NAME_LEN, schema::types::KnowledgeType};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slug {
    raw:  String,
    kind: KnowledgeType,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SlugError {
    #[error("slug must include a kind prefix (e.g., c., p., lo.)")]
    MissingKindPrefix,
    #[error("invalid kind prefix `{0}`; expected one of f, c, p, m, lo, a")]
    InvalidKind(String),
    #[error("slug name must not be empty")]
    EmptyName,
    #[error("slug contains invalid characters: {0}")]
    InvalidCharacters(String),
    #[error("slug exceeds maximum length of 64 characters")]
    TooLong,
}

impl Slug {
    pub fn parse(raw: &str) -> Result<Self, SlugError> {
        let trimmed = raw.trim();
        let Some((kind_part, name_part)) = trimmed.split_once('.') else {
            return Err(SlugError::MissingKindPrefix);
        };
        let kind = parse_kind_prefix(kind_part)?;
        let normalized_name = normalize_name(name_part)?;
        let normalized = format!("{}.{}", kind_prefix(&kind), normalized_name);
        Ok(Self {
            raw: normalized,
            kind,
        })
    }

    pub fn generate(kind: KnowledgeType, name: &str) -> Self {
        let normalized_name = normalize_name(name).unwrap_or_else(|_| "unnamed".to_string());
        let raw = format!("{}.{}", kind_prefix(&kind), normalized_name);
        Self { raw, kind }
    }

    pub fn as_str(&self) -> &str {
        &self.raw
    }

    pub fn kind(&self) -> KnowledgeType {
        self.kind
    }
}

impl fmt::Display for Slug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.raw)
    }
}

fn parse_kind_prefix(raw: &str) -> Result<KnowledgeType, SlugError> {
    match raw.to_ascii_lowercase().as_str() {
        "f" => Ok(KnowledgeType::Factual),
        "c" => Ok(KnowledgeType::Conceptual),
        "p" => Ok(KnowledgeType::Procedural),
        "m" => Ok(KnowledgeType::Metacognitive),
        "lo" => Ok(KnowledgeType::LearningOutcome),
        "a" => Ok(KnowledgeType::AssessmentItem),
        other => Err(SlugError::InvalidKind(other.to_string())),
    }
}

fn kind_prefix(kind: &KnowledgeType) -> &'static str {
    match kind {
        KnowledgeType::Factual => "f",
        KnowledgeType::Conceptual => "c",
        KnowledgeType::Procedural => "p",
        KnowledgeType::Metacognitive => "m",
        KnowledgeType::LearningOutcome => "lo",
        KnowledgeType::AssessmentItem => "a",
    }
}

fn normalize_name(raw: &str) -> Result<String, SlugError> {
    let mut out = String::with_capacity(raw.len());
    let mut prev_is_underscore = false;
    let mut prev_is_alnum = false;

    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() {
            let is_upper = ch.is_ascii_uppercase();
            if is_upper && prev_is_alnum && !prev_is_underscore {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
            prev_is_underscore = false;
            prev_is_alnum = true;
            continue;
        }
        if ch == '_' || ch == '-' || ch.is_whitespace() {
            if !prev_is_underscore && !out.is_empty() {
                out.push('_');
                prev_is_underscore = true;
            }
            prev_is_alnum = false;
            continue;
        }
        return Err(SlugError::InvalidCharacters(ch.to_string()));
    }

    if prev_is_underscore {
        out.pop();
    }

    if out.is_empty() {
        return Err(SlugError::EmptyName);
    }

    if out.len() > MAX_SLUG_NAME_LEN {
        if let Some(idx) = out[..MAX_SLUG_NAME_LEN].rfind('_') {
            out.truncate(idx);
        } else {
            out.truncate(MAX_SLUG_NAME_LEN);
        }
        if out.is_empty() {
            return Err(SlugError::TooLong);
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_mixed_case_and_spaces() {
        let slug = Slug::parse("C.LoopBasics").unwrap();
        assert_eq!(slug.as_str(), "c.loop_basics");
    }

    #[test]
    fn rejects_invalid_characters() {
        let err = Slug::parse("c.loop$basics").unwrap_err();
        assert!(matches!(err, SlugError::InvalidCharacters(_)));
    }

    #[test]
    fn truncates_at_boundary() {
        let long = "c.".to_string() + &"word_".repeat(20);
        let slug = Slug::parse(&long).unwrap();
        assert!(slug.as_str().len() <= MAX_SLUG_NAME_LEN + 2);
    }
}
