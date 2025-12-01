//! Validation routines for graph invariants, organized by family so
//! `GraphService` can focus on storage and routing.
use std::{
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
    time::Instant,
};

use bitflags::bitflags;
use petgraph::Direction;
use tracing::warn;

use crate::{
    analysis,
    graph::{
        AnchorImpact, CurriculumGraph, EdgeKind, GraphError, InvariantCode, NodeId, NodeKind,
        TeachingPurpose,
    },
    schema::types::{AssessmentScope, KnowledgeType, SourceRef},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationSeverity {
    Warning,
    Error,
}

#[derive(Clone, Debug)]
pub struct ValidationIssue {
    pub code:              crate::graph::InvariantCode,
    pub severity:          ValidationSeverity,
    pub message:           String,
    pub promote_in_strict: bool,
}

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct InvariantFamilies: u16 {
        const STATEMENTS    = 1 << 0;
        const PROVENANCE    = 1 << 1;
        const REQUIRES_DAG  = 1 << 2;
        const FADEABILITY   = 1 << 3;
        const COVERAGE      = 1 << 4;
        const SUPPORTS      = 1 << 5;
        const PURITY        = 1 << 6;
        const DISCOURSE     = 1 << 7;
        const INTRODUCTIONS = 1 << 8;
        const ALL           = Self::STATEMENTS.bits()
            | Self::PROVENANCE.bits()
            | Self::REQUIRES_DAG.bits()
            | Self::FADEABILITY.bits()
            | Self::COVERAGE.bits()
            | Self::SUPPORTS.bits()
            | Self::PURITY.bits()
            | Self::DISCOURSE.bits()
            | Self::INTRODUCTIONS.bits();
    }
}

#[derive(Clone, Debug)]
pub enum ValidationScope {
    Full,
    Targeted {
        coverage_los:      Vec<NodeId>,
        skip_requires_dag: bool,
        skip_fadeability:  bool,
    },
}

#[derive(Clone, Debug)]
pub struct ValidationContext {
    pub strict:                bool,
    pub expected_revision:     Option<String>,
    pub rubric_prev:           HashMap<String, u64>,
    pub include_rubric_update: bool,
}

/// Lightweight helper for provenance checks to keep error text consistent.
pub struct Provenance<'a> {
    expected: Option<&'a str>,
}

impl<'a> Provenance<'a> {
    pub fn new(expected: Option<&'a str>) -> Self {
        Self { expected }
    }

    pub fn check(&self, spans: &[SourceRef], label: &str) -> Result<(), GraphError> {
        let Some(expected) = self.expected else {
            return Ok(());
        };
        for span in spans {
            if span.revision != expected {
                return Err(GraphError::Schema(format!(
                    "{label} source_ref revision `{}` must equal course_commit `{}`",
                    span.revision, expected
                )));
            }
        }
        Ok(())
    }
}

pub fn families_for_scope(
    mut families: InvariantFamilies,
    scope: &ValidationScope,
) -> InvariantFamilies {
    match scope {
        ValidationScope::Full => {}
        ValidationScope::Targeted {
            coverage_los,
            skip_requires_dag,
            skip_fadeability,
        } => {
            if *skip_requires_dag {
                families.remove(InvariantFamilies::REQUIRES_DAG);
            }
            if *skip_fadeability {
                families.remove(InvariantFamilies::FADEABILITY);
            }
            if coverage_los.is_empty() {
                families.remove(InvariantFamilies::COVERAGE);
            }
        }
    }
    families
}

pub fn run_invariants_for_graph(
    g: &CurriculumGraph,
    scope: ValidationScope,
    ctx: &ValidationContext,
    families: InvariantFamilies,
) -> Result<Option<HashMap<String, u64>>, GraphError> {
    let start = Instant::now();
    let slug_index: HashMap<String, NodeId> =
        g.node_indices().map(|n| (g[n].slug.clone(), n)).collect();
    let has_teaching_steps = g
        .node_indices()
        .any(|n| matches!(&g[n].kind, NodeKind::TeachingStep(_)));

    let (coverage_los, include_requires_dag, include_fadeability) = match &scope {
        ValidationScope::Full => (
            g.node_indices()
                .filter(|&n| {
                    matches!(
                        &g[n].kind,
                        NodeKind::Knowledge(k) if k.knowledge_type == KnowledgeType::LearningOutcome
                    )
                })
                .collect(),
            true,
            true,
        ),
        ValidationScope::Targeted {
            coverage_los,
            skip_requires_dag,
            skip_fadeability,
        } => (coverage_los.clone(), !skip_requires_dag, !skip_fadeability),
    };

    let include_requires_dag =
        include_requires_dag && families.contains(InvariantFamilies::REQUIRES_DAG);
    let include_fadeability =
        include_fadeability && families.contains(InvariantFamilies::FADEABILITY);
    let needs_coverage = !coverage_los.is_empty() && families.contains(InvariantFamilies::COVERAGE);

    let rubric_current = compute_rubric_hashes(g);
    let (first_principles, rubric_prev) =
        if needs_coverage || matches!(scope, ValidationScope::Full) || include_fadeability {
            (analysis::first_principles(g), ctx.rubric_prev.clone())
        } else {
            (Vec::new(), HashMap::new())
        };
    let fade_ctx = if include_fadeability {
        Some(analysis::FadeabilityContext::from_first_principles(g, &first_principles))
    } else {
        None
    };

    let mut issues = Vec::new();
    if families.contains(InvariantFamilies::STATEMENTS) {
        issues.extend(statements::validate(g));
    }
    if families.contains(InvariantFamilies::PROVENANCE) {
        issues.extend(provenance::validate(g, ctx.expected_revision.as_deref()));
    }
    if include_requires_dag || include_fadeability {
        issues.extend(requires::validate(
            g,
            include_requires_dag,
            include_fadeability,
            fade_ctx.as_ref(),
        ));
    }
    if needs_coverage {
        issues.extend(coverage::validate(
            g,
            &first_principles,
            &coverage_los,
            &rubric_prev,
            &rubric_current,
        ));
    }
    if families.contains(InvariantFamilies::SUPPORTS) {
        issues.extend(supports::validate(g));
    }
    if families.contains(InvariantFamilies::PURITY) {
        issues.extend(purity::validate(g, &slug_index));
    }
    if has_teaching_steps && families.contains(InvariantFamilies::DISCOURSE) {
        issues.extend(discourse::validate(g));
    }
    if has_teaching_steps && families.contains(InvariantFamilies::INTRODUCTIONS) {
        issues.extend(introductions::validate(g));
    }

    let mut errors: Vec<crate::graph::InvariantViolation> = Vec::new();
    let mut warnings: Vec<crate::graph::InvariantViolation> = Vec::new();
    for mut issue in issues {
        if ctx.strict && issue.promote_in_strict {
            issue.severity = ValidationSeverity::Error;
        }
        match issue.severity {
            ValidationSeverity::Error => errors.push(crate::graph::InvariantViolation {
                code:    issue.code,
                message: issue.message,
            }),
            ValidationSeverity::Warning => warnings.push(crate::graph::InvariantViolation {
                code:    issue.code,
                message: issue.message,
            }),
        }
    }

    if ctx.strict {
        errors.extend(warnings);
    } else {
        for w in warnings {
            warn!(
                target: "weaver.graph.invariants",
                code = %w.code.as_str(),
                "{message}",
                message = w.message
            );
        }
    }

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    tracing::debug!(
        target: "weaver.graph.validation",
        scope = %match scope {
            ValidationScope::Full => "full",
            ValidationScope::Targeted { .. } => "targeted",
        },
        coverage_los = coverage_los.len(),
        include_requires_dag,
        include_fadeability,
        elapsed_ms
    );

    if errors.is_empty() {
        if ctx.include_rubric_update {
            Ok(Some(rubric_current))
        } else {
            Ok(None)
        }
    } else {
        Err(GraphError::InvariantViolation { violations: errors })
    }
}

fn make_issue(
    code: crate::graph::InvariantCode,
    severity: ValidationSeverity,
    promote_in_strict: bool,
    message: String,
) -> ValidationIssue {
    ValidationIssue {
        code,
        severity,
        message,
        promote_in_strict,
    }
}

mod statements {
    use super::*;

    pub(super) fn validate(g: &CurriculumGraph) -> Vec<ValidationIssue> {
        g.node_indices()
            .filter_map(|n| match &g[n].kind {
                NodeKind::Knowledge(k) if k.statement.trim().is_empty() => Some(make_issue(
                    InvariantCode::StatementEmpty,
                    ValidationSeverity::Error,
                    true,
                    format!("knowledge `{}` has empty statement", g[n].slug),
                )),
                NodeKind::TeachingStep(ts) if ts.statement.trim().is_empty() => Some(make_issue(
                    InvariantCode::StatementEmpty,
                    ValidationSeverity::Error,
                    true,
                    format!("teaching_step `{}` has empty statement", g[n].slug),
                )),
                _ => None,
            })
            .collect()
    }
}

mod provenance {
    use super::*;

    pub(super) fn validate(
        g: &CurriculumGraph,
        expected_revision: Option<&str>,
    ) -> Vec<ValidationIssue> {
        let Some(expected) = expected_revision else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for n in g.node_indices() {
            match &g[n].kind {
                NodeKind::Knowledge(k) => {
                    for span in &k.source_refs {
                        if span.revision != expected {
                            out.push(make_issue(
                                InvariantCode::ProvenanceRevision,
                                ValidationSeverity::Error,
                                true,
                                format!(
                                    "knowledge `{}` source_ref revision `{}` must equal \
                                     course_commit `{}`",
                                    g[n].slug, span.revision, expected
                                ),
                            ));
                        }
                    }
                }
                NodeKind::TeachingStep(ts) => {
                    for span in &ts.source_refs {
                        if span.revision != expected {
                            out.push(make_issue(
                                InvariantCode::ProvenanceRevision,
                                ValidationSeverity::Error,
                                true,
                                format!(
                                    "teaching_step `{}` source_ref revision `{}` must equal \
                                     course_commit `{}`",
                                    g[n].slug, span.revision, expected
                                ),
                            ));
                        }
                    }
                }
            }
        }

        for edge in g.edge_indices() {
            if let Some((u, v)) = g.edge_endpoints(edge) {
                match &g[edge].kind {
                    EdgeKind::Requires(attrs) => {
                        for span in &attrs.evidence_refs {
                            if span.revision != expected {
                                out.push(make_issue(
                                    InvariantCode::ProvenanceRevision,
                                    ValidationSeverity::Error,
                                    true,
                                    format!(
                                        "{} -> {} evidence_ref revision `{}` must equal \
                                         course_commit `{}`",
                                        g[u].slug, g[v].slug, span.revision, expected
                                    ),
                                ));
                            }
                        }
                    }
                    EdgeKind::Supports(attrs) => {
                        for span in &attrs.evidence_refs {
                            if span.revision != expected {
                                out.push(make_issue(
                                    InvariantCode::ProvenanceRevision,
                                    ValidationSeverity::Error,
                                    true,
                                    format!(
                                        "{} -> {} evidence_ref revision `{}` must equal \
                                         course_commit `{}`",
                                        g[u].slug, g[v].slug, span.revision, expected
                                    ),
                                ));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        out
    }
}

mod requires {
    use super::*;

    pub(super) fn validate(
        g: &CurriculumGraph,
        include_dag: bool,
        include_fadeability: bool,
        fade_ctx: Option<&analysis::FadeabilityContext>,
    ) -> Vec<ValidationIssue> {
        let mut out = Vec::new();
        if include_dag && !analysis::requires_is_dag(g) {
            out.push(make_issue(
                InvariantCode::RequiresDag,
                ValidationSeverity::Error,
                true,
                "requires layer must remain acyclic".to_string(),
            ));
        }
        if include_fadeability {
            let ctx = fade_ctx
                .cloned()
                .unwrap_or_else(|| analysis::FadeabilityContext::compute(g));
            for issue in analysis::fadeability_issues_with_context(g, &ctx) {
                let assessment_slug = g[issue.assessment].slug.clone();
                let edges: Vec<String> = issue
                    .support_edges
                    .iter()
                    .filter_map(|e| g.edge_endpoints(*e))
                    .map(|(u, v)| format!("{} -> {}", g[u].slug, g[v].slug))
                    .collect();
                out.push(make_issue(
                    InvariantCode::Fadeability,
                    ValidationSeverity::Error,
                    true,
                    format!(
                        "assessment `{}` reachable only via supports that carry prerequisite \
                         load: [{}]",
                        assessment_slug,
                        edges.join("; ")
                    ),
                ));
            }
        }
        out
    }
}

mod coverage {
    use super::*;

    pub(super) fn validate(
        g: &CurriculumGraph,
        first_principles: &[NodeId],
        lo_nodes: &[NodeId],
        rubric_prev: &HashMap<String, u64>,
        rubric_current: &HashMap<String, u64>,
    ) -> Vec<ValidationIssue> {
        let mut out = Vec::new();
        for &lo_id in lo_nodes {
            let report = analysis::lo_reachability(g, lo_id, first_principles);
            if report.assessments.is_empty() {
                out.push(make_issue(
                    InvariantCode::LoTargetAssessment,
                    ValidationSeverity::Warning,
                    true,
                    format!(
                        "learning_outcome `{}` has no assesses(scope=target) assessments",
                        g[lo_id].slug
                    ),
                ));
            } else if !report
                .assessments
                .iter()
                .any(|a| a.reachable_from_first_principle)
            {
                let assessments = report
                    .assessments
                    .iter()
                    .map(|a| g[a.assessment].slug.clone())
                    .collect::<Vec<_>>()
                    .join(", ");
                out.push(make_issue(
                    InvariantCode::LoReachability,
                    ValidationSeverity::Warning,
                    true,
                    format!(
                        "learning_outcome `{}` not reachable from first principles via \
                         assessments [{}]",
                        g[lo_id].slug, assessments
                    ),
                ));
            }

            let report = analysis::coverage_report(g, lo_id);
            if !report.missing_criteria.is_empty() {
                out.push(make_issue(
                    InvariantCode::RubricCoverage,
                    ValidationSeverity::Warning,
                    true,
                    format!(
                        "learning_outcome `{}` missing coverage for rubric criteria: {}",
                        g[lo_id].slug,
                        report.missing_criteria.join(", ")
                    ),
                ));
            }
            if !report.unused_observation_features.is_empty() {
                out.push(make_issue(
                    InvariantCode::RubricUnusedObservationFeatures,
                    ValidationSeverity::Warning,
                    true,
                    format!(
                        "learning_outcome `{}` has observation_features not present in rubric: {}",
                        g[lo_id].slug,
                        report.unused_observation_features.join(", ")
                    ),
                ));
            }
            if let (Some(cur), Some(prev)) =
                (rubric_current.get(&g[lo_id].slug), rubric_prev.get(&g[lo_id].slug))
                && cur != prev
            {
                out.push(make_issue(
                    InvariantCode::RubricDrift,
                    ValidationSeverity::Warning,
                    true,
                    format!(
                        "learning_outcome `{}` rubric_criteria changed; coverage rechecked",
                        g[lo_id].slug
                    ),
                ));
            }
        }
        out
    }
}

mod supports {
    use super::*;

    pub(super) fn validate(g: &CurriculumGraph) -> Vec<ValidationIssue> {
        let mut out = Vec::new();
        for gap in analysis::example_gaps(g) {
            out.push(make_issue(
                InvariantCode::ExampleMinimums,
                ValidationSeverity::Warning,
                true,
                format!("{}: {}", g[gap.node].slug, gap.description),
            ));
        }
        for gap in analysis::procedural_practice_gaps(g) {
            out.push(make_issue(
                InvariantCode::ProceduralPractice,
                ValidationSeverity::Warning,
                true,
                format!(
                    "procedural `{}` lacks reachable assessment with assesses(scope=target)",
                    g[gap.node].slug
                ),
            ));
        }
        out
    }
}

mod purity {
    use super::*;

    pub(super) fn validate(
        g: &CurriculumGraph,
        slug_index: &HashMap<String, NodeId>,
    ) -> Vec<ValidationIssue> {
        let mut out = Vec::new();
        for edge in g.edge_indices() {
            if let EdgeKind::Assesses(attrs) = &g[edge].kind
                && attrs.evidence_link.scope == AssessmentScope::Target
                && let Some((assessment, lo)) = g.edge_endpoints(edge)
            {
                let intended = analysis::intended_knowledge_from_anchors(g, lo);
                if intended.is_empty() {
                    out.push(make_issue(
                        InvariantCode::PurityIntendedMissing,
                        ValidationSeverity::Warning,
                        true,
                        format!(
                            "purity check skipped for `{}` -> `{}` (no target anchors with \
                             intended knowledge)",
                            g[assessment].slug, g[lo].slug
                        ),
                    ));
                    continue;
                }

                let mut allowed = HashSet::new();
                if let NodeKind::Knowledge(k) = &g[assessment].kind {
                    for cid in &k.construct_irrelevant_demands {
                        if let Some(id) = slug_index.get(cid) {
                            allowed.insert(*id);
                        }
                    }
                }
                let mut extraneous = analysis::extraneous_knowledge(g, assessment, &intended);
                extraneous.retain(|n| !allowed.contains(n));
                if !extraneous.is_empty() {
                    let slugs: Vec<String> =
                        extraneous.iter().map(|n| g[*n].slug.clone()).collect();
                    out.push(make_issue(
                        InvariantCode::PurityExtraneous,
                        ValidationSeverity::Error,
                        true,
                        format!(
                            "purity violation: assessment `{}` -> LO `{}` requires extraneous \
                             knowledge [{}]",
                            g[assessment].slug,
                            g[lo].slug,
                            slugs.join(", ")
                        ),
                    ));
                }
            }
        }
        out
    }
}

mod discourse {
    use super::*;

    pub(super) fn validate(g: &CurriculumGraph) -> Vec<ValidationIssue> {
        let mut out = Vec::new();
        let episodes: HashSet<String> = g
            .node_indices()
            .filter_map(|n| {
                if let NodeKind::TeachingStep(ts) = &g[n].kind {
                    Some(ts.episode.clone())
                } else {
                    None
                }
            })
            .collect();
        for ep in episodes {
            for result in analysis::borrow_ahead(g, &ep) {
                if matches!(
                    &g[result.target].kind,
                    NodeKind::Knowledge(k) if k.knowledge_type == KnowledgeType::LearningOutcome
                ) {
                    continue;
                }
                let step_slug = g[result.step].slug.clone();
                let target_slug = g[result.target].slug.clone();
                match result.severity {
                    analysis::BorrowSeverity::CrossEpisode | analysis::BorrowSeverity::NoIntro => {
                        out.push(make_issue(
                            InvariantCode::BorrowAhead,
                            ValidationSeverity::Error,
                            true,
                            format!(
                                "borrow-ahead error in episode `{}`: step `{}` uses `{}` before \
                                 introduction (severity {:?})",
                                ep, step_slug, target_slug, result.severity
                            ),
                        ));
                    }
                    analysis::BorrowSeverity::InEpisode => out.push(make_issue(
                        InvariantCode::BorrowAhead,
                        ValidationSeverity::Warning,
                        true,
                        format!(
                            "borrow-ahead warning in episode `{}`: step `{}` uses `{}` before \
                             introduction",
                            ep, step_slug, target_slug
                        ),
                    )),
                    analysis::BorrowSeverity::Suppressed => {}
                }
            }
        }

        for orphan in analysis::discourse_orphans(g, None) {
            out.push(make_issue(
                InvariantCode::DiscourseOrphan,
                ValidationSeverity::Warning,
                true,
                format!(
                    "teaching_step `{}` is orphaned within its episode (no precedes links)",
                    g[orphan].slug
                ),
            ));
        }

        for n in g.node_indices() {
            if let NodeKind::TeachingStep(ts) = &g[n].kind {
                let has_anchor = g
                    .edges_directed(n, Direction::Outgoing)
                    .any(|e| matches!(&e.weight().kind, EdgeKind::Anchors(_)));
                let has_rationale = ts
                    .rationale
                    .as_ref()
                    .map(|r| !r.trim().is_empty())
                    .unwrap_or(false);
                let missing_anchor = !has_anchor;
                let missing_rationale = !has_rationale;
                if matches!(ts.purpose, TeachingPurpose::Use)
                    && missing_anchor
                    && !missing_rationale
                {
                    out.push(make_issue(
                        InvariantCode::BorrowAhead,
                        ValidationSeverity::Warning,
                        true,
                        format!(
                            "teaching_step `{}` has purpose=use but no anchors; cannot verify \
                             introduction order despite provided rationale",
                            g[n].slug
                        ),
                    ));
                }
                if missing_anchor && missing_rationale {
                    out.push(make_issue(
                        InvariantCode::TeachingStepAnchorOrRationale,
                        ValidationSeverity::Warning,
                        true,
                        format!(
                            "teaching_step `{}` must have at least one anchor or a rationale",
                            g[n].slug
                        ),
                    ));
                }
            }
        }

        out
    }
}

mod introductions {
    use super::*;

    pub(super) fn validate(g: &CurriculumGraph) -> Vec<ValidationIssue> {
        let mut out = Vec::new();
        for n in g.node_indices() {
            if let NodeKind::Knowledge(k) = &g[n].kind
                && k.knowledge_type.is_instructional_knowledge()
                && matches!(k.introduction_scope, crate::graph::IntroductionScope::InCourse)
            {
                let anchored = g
                    .edges_directed(n, Direction::Incoming)
                    .any(|e| matches!(&e.weight().kind, EdgeKind::Anchors(_)));
                if !anchored {
                    continue;
                }
                let has_intro = g.edges_directed(n, Direction::Incoming).any(|e| {
                    matches!(
                        &e.weight().kind,
                        EdgeKind::Anchors(attrs) if matches!(attrs.impact, AnchorImpact::Introduce)
                    )
                });
                if !has_intro {
                    out.push(make_issue(
                        InvariantCode::IntroduceAnchor,
                        ValidationSeverity::Warning,
                        true,
                        format!(
                            "knowledge `{}` (in_course) missing anchors(impact=introduce)",
                            g[n].slug
                        ),
                    ));
                }
            }
        }
        out
    }
}

pub fn compute_rubric_hashes(g: &CurriculumGraph) -> HashMap<String, u64> {
    g.node_indices()
        .filter_map(|n| {
            if let NodeKind::Knowledge(k) = &g[n].kind
                && k.knowledge_type == KnowledgeType::LearningOutcome
            {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                k.rubric_criteria.hash(&mut hasher);
                Some((g[n].slug.clone(), hasher.finish()))
            } else {
                None
            }
        })
        .collect()
}
