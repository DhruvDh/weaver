use std::{collections::HashSet, hash::Hash};

use crate::{
    graph::{
        AnchorsAttrs, AssessesAttrs, CaseTag, EdgeConflict, EdgeKind, EdgePayload, PrecedesAttrs,
        RequiresAttrs, SupportsAttrs,
    },
    schema::types::{EvidenceLink, SourceRef, Strength},
};

pub enum MergeResult<T> {
    Merged(T),
    Conflict { existing: T, incoming: T },
}

impl<T> MergeResult<T> {
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> MergeResult<U> {
        match self {
            MergeResult::Merged(val) => MergeResult::Merged(f(val)),
            MergeResult::Conflict { existing, incoming } => MergeResult::Conflict {
                existing: f(existing),
                incoming: f(incoming),
            },
        }
    }
}

pub trait Mergeable: Sized {
    fn try_merge(self, other: Self) -> MergeResult<Self>;
}

pub fn merge_edge_payload(
    existing: EdgePayload,
    incoming: EdgePayload,
) -> MergeResult<EdgePayload> {
    let EdgePayload {
        kind: existing_kind,
        confidence: existing_conf,
        conflicts: mut existing_conflicts,
    } = existing;
    let EdgePayload {
        kind: incoming_kind,
        confidence: incoming_conf,
        conflicts: mut incoming_conflicts,
    } = incoming;

    let confidence = existing_conf.max(incoming_conf);
    existing_conflicts.append(&mut incoming_conflicts);

    match existing_kind.try_merge(incoming_kind) {
        MergeResult::Merged(kind) => MergeResult::Merged(EdgePayload {
            kind,
            confidence,
            conflicts: existing_conflicts,
        }),
        MergeResult::Conflict {
            existing: existing_kind,
            incoming: incoming_kind,
        } => {
            existing_conflicts.push(EdgeConflict {
                kind:       incoming_kind.clone(),
                confidence: incoming_conf,
            });
            MergeResult::Conflict {
                existing: EdgePayload {
                    kind: existing_kind,
                    confidence,
                    conflicts: existing_conflicts,
                },
                incoming: EdgePayload {
                    kind:       incoming_kind,
                    confidence: incoming_conf,
                    conflicts:  incoming_conflicts,
                },
            }
        }
    }
}

impl Mergeable for RequiresAttrs {
    fn try_merge(self, other: Self) -> MergeResult<Self> {
        MergeResult::Merged(Self {
            strength:      stronger_strength(self.strength, other.strength),
            rationale:     merge_rationale(self.rationale, other.rationale),
            evidence_refs: union_source_refs(&self.evidence_refs, &other.evidence_refs),
        })
    }
}

impl Mergeable for SupportsAttrs {
    fn try_merge(self, other: Self) -> MergeResult<Self> {
        if self.support_kind != other.support_kind {
            return MergeResult::Conflict {
                existing: self,
                incoming: other,
            };
        }
        if self.intended_effect != other.intended_effect {
            return MergeResult::Conflict {
                existing: self,
                incoming: other,
            };
        }
        MergeResult::Merged(Self {
            support_kind:    self.support_kind,
            intended_effect: self.intended_effect,
            case_tag:        merge_case_tag(self.case_tag, other.case_tag),
            coverage_tags:   union_vecs(self.coverage_tags, other.coverage_tags),
            evidence_refs:   union_source_refs(&self.evidence_refs, &other.evidence_refs),
        })
    }
}

impl Mergeable for EvidenceLink {
    fn try_merge(mut self, other: Self) -> MergeResult<Self> {
        if self.claim != other.claim || self.scope != other.scope {
            return MergeResult::Conflict {
                existing: self,
                incoming: other,
            };
        }
        self.observation_features =
            union_vecs(self.observation_features.clone(), other.observation_features);
        MergeResult::Merged(self)
    }
}

impl Mergeable for AssessesAttrs {
    fn try_merge(self, other: Self) -> MergeResult<Self> {
        match self.evidence_link.try_merge(other.evidence_link) {
            MergeResult::Merged(link) => MergeResult::Merged(Self {
                evidence_link: link,
            }),
            MergeResult::Conflict { existing, incoming } => MergeResult::Conflict {
                existing: Self {
                    evidence_link: existing,
                },
                incoming: Self {
                    evidence_link: incoming,
                },
            },
        }
    }
}

impl Mergeable for PrecedesAttrs {
    fn try_merge(self, other: Self) -> MergeResult<Self> {
        if self.episode == other.episode {
            MergeResult::Merged(self)
        } else {
            MergeResult::Conflict {
                existing: self,
                incoming: other,
            }
        }
    }
}

impl Mergeable for AnchorsAttrs {
    fn try_merge(self, other: Self) -> MergeResult<Self> {
        if self.impact == other.impact {
            MergeResult::Merged(self)
        } else {
            MergeResult::Conflict {
                existing: self,
                incoming: other,
            }
        }
    }
}

impl Mergeable for EdgeKind {
    fn try_merge(self, other: Self) -> MergeResult<Self> {
        match (self, other) {
            (EdgeKind::Requires(a), EdgeKind::Requires(b)) => {
                a.try_merge(b).map(EdgeKind::Requires)
            }
            (EdgeKind::Supports(a), EdgeKind::Supports(b)) => {
                a.try_merge(b).map(EdgeKind::Supports)
            }
            (EdgeKind::Assesses(a), EdgeKind::Assesses(b)) => {
                a.try_merge(b).map(EdgeKind::Assesses)
            }
            (EdgeKind::Precedes(a), EdgeKind::Precedes(b)) => {
                a.try_merge(b).map(EdgeKind::Precedes)
            }
            (EdgeKind::Anchors(a), EdgeKind::Anchors(b)) => a.try_merge(b).map(EdgeKind::Anchors),
            (existing, incoming) => MergeResult::Conflict { existing, incoming },
        }
    }
}

pub fn union_vecs<T>(a: Vec<T>, b: Vec<T>) -> Vec<T>
where
    T: Eq + Hash + Clone,
{
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for item in a.into_iter().chain(b.into_iter()) {
        if seen.insert(item.clone()) {
            out.push(item);
        }
    }
    out
}

pub fn union_source_refs(a: &[SourceRef], b: &[SourceRef]) -> Vec<SourceRef> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for span in a.iter().chain(b.iter()) {
        let key = (&span.path, span.start_line, span.end_line, &span.revision);
        if seen.insert(key) {
            out.push(span.clone());
        }
    }
    out
}

fn merge_rationale(primary: String, other: String) -> String {
    if primary.trim().is_empty() {
        return other;
    }
    if other.trim().is_empty() || primary.trim() == other.trim() {
        return primary;
    }
    format!("{primary}\n\n{other}")
}

fn merge_case_tag(a: Option<CaseTag>, b: Option<CaseTag>) -> Option<CaseTag> {
    match (a, b) {
        (Some(x), Some(y)) => Some(std::cmp::max(x, y)),
        (Some(x), None) | (None, Some(x)) => Some(x),
        (None, None) => None,
    }
}

fn stronger_strength(a: Strength, b: Strength) -> Strength {
    match (a, b) {
        (Strength::Necessary, _) | (_, Strength::Necessary) => Strength::Necessary,
        (Strength::Strong, _) | (_, Strength::Strong) => Strength::Strong,
        _ => Strength::Helpful,
    }
}
