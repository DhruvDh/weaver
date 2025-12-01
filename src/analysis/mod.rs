//! Read-only analyses over the multiplex curriculum/teaching graph.
//!
//! Edge layers (all stored in one `CurriculumGraph`):
//! - `requires`: knowledge prerequisite DAG (acyclic by construction).
//! - `supports`: pedagogical supports/examples between knowledge nodes.
//! - `assesses`: assessment_item -> learning_outcome evidence links.
//! - `precedes`: discourse ordering between teaching steps (per-episode DAG).
//! - `anchors`: teaching_step -> knowledge/LO/assessment with impact semantics.
//!
//! Provided analyses include DAG/topo checks, first principles, LO
//! reachability/coverage, extraneous knowledge, example variety, keystone
//! scores, fadeability, procedural practice gaps, borrow-ahead/discourse
//! orphans, and alignment gaps. All functions operate on an immutable graph
//! snapshot and do not mutate state.

use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet, VecDeque},
};

use petgraph::{
    Direction,
    algo::{has_path_connecting, is_cyclic_directed, toposort},
    visit::EdgeRef,
};
use rayon::prelude::*;
use schemars::JsonSchema;

use crate::{
    graph::{self, AnchorImpact, CaseTag, CurriculumGraph, EdgeKind, NodeId, NodeKind, traversal},
    schema::types::{AssessmentScope, KnowledgeType, SupportKind},
};

/// A topological order over the `requires` layer.
#[derive(Debug, Clone, Copy)]
pub struct RequiresCycle;

pub fn requires_toposort(g: &CurriculumGraph) -> Result<Vec<NodeId>, RequiresCycle> {
    let view = petgraph::visit::EdgeFiltered::from_fn(g, |e| {
        matches!(e.weight().kind, EdgeKind::Requires(_))
    });
    toposort(&view, None).map_err(|_| RequiresCycle)
}

/// True when the `requires` layer is acyclic.
pub fn requires_is_dag(g: &CurriculumGraph) -> bool {
    let view = petgraph::visit::EdgeFiltered::from_fn(g, |e| {
        matches!(e.weight().kind, EdgeKind::Requires(_))
    });
    !is_cyclic_directed(&view)
}

/// Nodes with zero in-degree in the `requires` layer and instructional
/// knowledge kinds.
pub fn first_principles(g: &CurriculumGraph) -> Vec<NodeId> {
    g.node_indices()
        .filter(|&n| {
            matches!(
                &g[n].kind,
                NodeKind::Knowledge(k) if k.knowledge_type.is_instructional_knowledge()
            )
        })
        .filter(|&n| {
            g.edges_directed(n, Direction::Incoming)
                .all(|e| !matches!(e.weight().kind, EdgeKind::Requires(_)))
        })
        .collect()
}

/// Report for a single LO showing assessments and reachability from first
/// principles.
pub struct LoReachability {
    pub lo:          NodeId,
    pub assessments: Vec<AssessmentReach>,
}

pub struct AssessmentReach {
    pub assessment:                     NodeId,
    pub reachable_from_first_principle: bool,
}

pub fn lo_reachability(
    g: &CurriculumGraph,
    lo: NodeId,
    first_principles: &[NodeId],
) -> LoReachability {
    let assessments: Vec<_> = g
        .edges_directed(lo, Direction::Incoming)
        .filter_map(|edge| match &edge.weight().kind {
            EdgeKind::Assesses(attrs) => {
                if attrs.evidence_link.scope == AssessmentScope::Target {
                    Some(edge.source())
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect();

    let assessment_reports = assessments
        .into_iter()
        .map(|a| {
            let reachable = first_principles
                .iter()
                .any(|&fp| has_requires_path(g, fp, a));
            AssessmentReach {
                assessment:                     a,
                reachable_from_first_principle: reachable,
            }
        })
        .collect();

    LoReachability {
        lo,
        assessments: assessment_reports,
    }
}

/// For a given LO, compare rubric criteria vs. union of observation features
/// across target assesses edges.
pub struct CoverageReport {
    pub lo: NodeId,
    pub covered_criteria: Vec<String>,
    pub missing_criteria: Vec<String>,
    pub unused_observation_features: Vec<String>,
}

pub fn coverage_report(g: &CurriculumGraph, lo: NodeId) -> CoverageReport {
    fn canonical_feature(raw: &str) -> Option<String> {
        let filtered: String = raw
            .chars()
            .map(|c| {
                if c.is_alphanumeric() {
                    c.to_ascii_lowercase()
                } else {
                    ' '
                }
            })
            .collect();
        let normalized = filtered.split_whitespace().collect::<Vec<_>>().join(" ");
        if normalized.is_empty() {
            None
        } else {
            Some(normalized)
        }
    }

    let (rubric, mut observations) = match &g[lo].kind {
        NodeKind::Knowledge(k) if k.knowledge_type == KnowledgeType::LearningOutcome => {
            (k.rubric_criteria.clone(), Vec::new())
        }
        _ => (Vec::new(), Vec::new()),
    };

    for edge in g.edges_directed(lo, Direction::Incoming) {
        if let EdgeKind::Assesses(attrs) = &edge.weight().kind
            && attrs.evidence_link.scope == AssessmentScope::Target
        {
            observations.extend(attrs.evidence_link.observation_features.clone());
        }
    }

    let rubric_norm: HashMap<String, String> = rubric
        .into_iter()
        .filter_map(|item| canonical_feature(&item).map(|canon| (canon, item)))
        .collect();
    let obs_norm: HashMap<String, String> = observations
        .into_iter()
        .filter_map(|item| canonical_feature(&item).map(|canon| (canon, item)))
        .collect();

    let covered: HashSet<String> = rubric_norm
        .keys()
        .filter(|k| obs_norm.contains_key(*k))
        .filter_map(|k| rubric_norm.get(k).cloned())
        .collect();
    let missing: HashSet<String> = rubric_norm
        .iter()
        .filter(|(canon, _)| !obs_norm.contains_key(*canon))
        .map(|(_, original)| original.clone())
        .collect();
    let unused: HashSet<String> = obs_norm
        .iter()
        .filter(|(canon, _)| !rubric_norm.contains_key(*canon))
        .map(|(_, original)| original.clone())
        .collect();

    CoverageReport {
        lo,
        covered_criteria: covered.into_iter().collect(),
        missing_criteria: missing.into_iter().collect(),
        unused_observation_features: unused.into_iter().collect(),
    }
}

/// Extraneous knowledge set = requires-ancestors(assessment) \ intended set.
pub fn extraneous_knowledge(
    g: &CurriculumGraph,
    assessment: NodeId,
    intended: &HashSet<NodeId>,
) -> HashSet<NodeId> {
    let ancestors = requires_ancestors(g, assessment);
    ancestors.difference(intended).copied().collect()
}

/// Derive intended knowledge for an LO by inspecting target anchors. Any
/// teaching step that anchors to the LO with impact=target may also anchor to
/// specific knowledge nodes; those co-anchored knowledge nodes are treated as
/// the intended construct set.
pub fn intended_knowledge_from_anchors(g: &CurriculumGraph, lo: NodeId) -> HashSet<NodeId> {
    let mut intended = HashSet::new();
    for edge in g.edges_directed(lo, Direction::Incoming) {
        if let EdgeKind::Anchors(attrs) = &edge.weight().kind
            && matches!(attrs.impact, AnchorImpact::Target)
        {
            let step = edge.source();
            for out in g.edges_directed(step, Direction::Outgoing) {
                if let EdgeKind::Anchors(_) = &out.weight().kind
                    && matches!(&g[out.target()].kind, NodeKind::Knowledge(k) if k.knowledge_type.is_instructional_knowledge())
                {
                    intended.insert(out.target());
                }
            }
        }
    }
    intended
}

/// Betweenness-inspired keystone score over the requires DAG; higher scores
/// indicate nodes that sit on many shortest prerequisite paths.
pub struct KeystoneScore {
    pub node:      NodeId,
    pub score:     f64,
    pub in_reach:  usize,
    pub out_reach: usize,
}

pub fn keystone_scores(g: &CurriculumGraph) -> Vec<KeystoneScore> {
    let (in_map, out_map) = requires_reach_counts_all(g);
    let nodes: Vec<NodeId> = g
        .node_indices()
        .filter(|&n| {
            matches!(
                &g[n].kind,
                NodeKind::Knowledge(k) if k.knowledge_type.is_instructional_knowledge()
            )
        })
        .collect();
    let index: HashMap<NodeId, usize> = nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();
    let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];

    for edge in g.edge_indices() {
        if let EdgeKind::Requires(_) = &g[edge].kind
            && let Some((u, v)) = g.edge_endpoints(edge)
            && let (Some(&ui), Some(&vi)) = (index.get(&u), index.get(&v))
        {
            neighbors[ui].push(vi);
        }
    }

    let mut centrality = vec![0.0f64; nodes.len()];
    for s in 0..nodes.len() {
        let mut stack = Vec::new();
        let mut pred = vec![Vec::<usize>::new(); nodes.len()];
        let mut sigma = vec![0.0f64; nodes.len()];
        let mut dist = vec![usize::MAX; nodes.len()];
        sigma[s] = 1.0;
        dist[s] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(s);
        while let Some(v) = queue.pop_front() {
            stack.push(v);
            let dv = dist[v];
            for &w in &neighbors[v] {
                if dist[w] == usize::MAX {
                    dist[w] = dv + 1;
                    queue.push_back(w);
                }
                if dist[w] == dv + 1 {
                    sigma[w] += sigma[v];
                    pred[w].push(v);
                }
            }
        }
        let mut delta = vec![0.0f64; nodes.len()];
        while let Some(w) = stack.pop() {
            for &v in &pred[w] {
                if sigma[w] > 0.0 {
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
            }
            if w != s {
                centrality[w] += delta[w];
            }
        }
    }

    let mut scores: Vec<_> = nodes
        .iter()
        .enumerate()
        .map(|(idx, &n)| {
            let in_reach = *in_map.get(&n).unwrap_or(&0);
            let out_reach = *out_map.get(&n).unwrap_or(&0);
            KeystoneScore {
                node: n,
                score: centrality[idx],
                in_reach,
                out_reach,
            }
        })
        .collect();

    scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    scores
}

/// Fadeability test: assessments reachable only when supports are treated as
/// prerequisites.
pub struct FadeabilityIssue {
    pub assessment:    NodeId,
    pub support_edges: Vec<petgraph::stable_graph::EdgeIndex<u32>>,
}

#[derive(Clone, Debug)]
pub struct FadeabilityContext {
    pub first_principles:               Vec<NodeId>,
    pub reachable_requires:             HashSet<NodeId>,
    pub reachable_requires_or_supports: HashSet<NodeId>,
}

impl FadeabilityContext {
    pub fn from_first_principles(g: &CurriculumGraph, first_principles: &[NodeId]) -> Self {
        let fps = first_principles.to_vec();
        let reachable_requires =
            reachable_with_filter(g, &fps, |k| matches!(k, EdgeKind::Requires(_)));
        let reachable_requires_or_supports = reachable_with_filter(g, &fps, |k| {
            matches!(k, EdgeKind::Requires(_) | EdgeKind::Supports(_))
        });
        Self {
            first_principles: fps,
            reachable_requires,
            reachable_requires_or_supports,
        }
    }

    pub fn compute(g: &CurriculumGraph) -> Self {
        let fps = first_principles(g);
        Self::from_first_principles(g, &fps)
    }
}

pub fn fadeability_issues(g: &CurriculumGraph) -> Vec<FadeabilityIssue> {
    let ctx = FadeabilityContext::compute(g);
    fadeability_issues_with_context(g, &ctx)
}

pub fn fadeability_issues_with_context(
    g: &CurriculumGraph,
    ctx: &FadeabilityContext,
) -> Vec<FadeabilityIssue> {
    let assessments_support_only: Vec<NodeId> = g
        .node_indices()
        .filter(|&n| {
            matches!(
                &g[n].kind,
                NodeKind::Knowledge(k) if k.knowledge_type.is_assessment_item()
            )
        })
        .filter(|&a| {
            ctx.reachable_requires_or_supports.contains(&a) && !ctx.reachable_requires.contains(&a)
        })
        .collect();

    assessments_support_only
        .into_iter()
        .map(|assessment| FadeabilityIssue {
            assessment,
            support_edges: supports_on_paths_to_assessment(g, ctx, assessment),
        })
        .collect()
}

pub fn support_would_break_fadeability(
    g: &CurriculumGraph,
    ctx: &FadeabilityContext,
    from: NodeId,
    to: NodeId,
) -> bool {
    if !ctx.reachable_requires_or_supports.contains(&from) {
        // New support cannot be reached from first principles, so it cannot
        // introduce a reachable assessment.
        return false;
    }
    let after = reachable_with_virtual_support(g, &ctx.first_principles, from, to);
    after
        .into_iter()
        .filter(|&n| {
            matches!(
                &g[n].kind,
                NodeKind::Knowledge(k) if k.knowledge_type.is_assessment_item()
            )
        })
        .any(|assessment| {
            !ctx.reachable_requires_or_supports.contains(&assessment)
                && !ctx.reachable_requires.contains(&assessment)
        })
}

/// Example minimum + variety checks.
pub struct ExampleGap {
    pub node:        NodeId,
    pub description: String,
}

pub fn example_gaps(g: &CurriculumGraph) -> Vec<ExampleGap> {
    let nodes: Vec<NodeId> = g
        .node_indices()
        .filter(|&n| matches!(&g[n].kind, NodeKind::Knowledge(_)))
        .collect();

    let mut gaps: Vec<ExampleGap> = nodes
        .par_iter()
        .filter_map(|&n| {
            let knowledge = match &g[n].kind {
                NodeKind::Knowledge(k) => k,
                _ => return None,
            };
            let supports: Vec<_> = g
                .edges_directed(n, Direction::Incoming)
                .filter_map(|e| match &e.weight().kind {
                    EdgeKind::Supports(attrs) => Some(attrs.clone()),
                    _ => None,
                })
                .collect();

            let mut descs = Vec::new();
            match knowledge.knowledge_type {
                KnowledgeType::Procedural => {
                    let we_total = supports
                        .iter()
                        .filter(|s| s.support_kind == SupportKind::WorkedExample)
                        .count();
                    let typical = supports.iter().any(|s| {
                        s.support_kind == SupportKind::WorkedExample
                            && matches!(s.case_tag, Some(CaseTag::Typical))
                    });
                    let edge_case = supports.iter().any(|s| {
                        s.support_kind == SupportKind::WorkedExample
                            && matches!(s.case_tag, Some(CaseTag::Edge | CaseTag::ErrorCase))
                    });
                    if we_total < 2 || !typical || !edge_case {
                        descs.push(
                            "procedural nodes need >=2 worked examples (typical + edge/error)"
                                .to_string(),
                        );
                    }
                }
                KnowledgeType::Conceptual => {
                    let has_analogy = supports
                        .iter()
                        .any(|s| s.support_kind == SupportKind::Analogy);
                    let has_counter = supports
                        .iter()
                        .any(|s| s.support_kind == SupportKind::Counterexample);
                    if !(has_analogy || has_counter) {
                        descs.push(
                            "conceptual nodes need at least one analogy or counterexample"
                                .to_string(),
                        );
                    }
                }
                KnowledgeType::Factual => {
                    let has_example = supports.iter().any(|s| {
                        s.support_kind == SupportKind::WorkedExample
                            || s.support_kind == SupportKind::Counterexample
                    });
                    if !has_example {
                        descs.push("factual nodes need an example or counterexample".into());
                    }
                }
                KnowledgeType::Metacognitive => {
                    let has_hint = supports
                        .iter()
                        .any(|s| s.support_kind == SupportKind::StrategyHint);
                    if !has_hint {
                        descs.push("metacognitive nodes need a strategy hint support".into());
                    }
                }
                _ => {}
            }

            if matches!(knowledge.intrinsic_load, Some(graph::IntrinsicLoad::High)) {
                let has_support = !supports.is_empty();
                let has_coverage_tag = supports.iter().any(|s| !s.coverage_tags.is_empty());
                if !has_support || !has_coverage_tag {
                    descs.push(
                        "high intrinsic_load nodes should include rich supports with coverage_tags"
                            .to_string(),
                    );
                }
            }

            if descs.is_empty() {
                None
            } else {
                Some(ExampleGap {
                    node:        n,
                    description: descs.join("; "),
                })
            }
        })
        .collect();

    gaps.sort_by_key(|g| g.node.index());
    gaps
}

/// Borrow-ahead detection within an episode.
///
/// Semantics:
/// - If the target was introduced earlier in the *same* episode (reachable via
///   precedes), usage is allowed.
/// - If introduced only in other episodes, the severity is `CrossEpisode`.
/// - If introduction_scope is Prior/External, the finding is suppressed.
/// - If there is no introduction anywhere, severity is `NoIntro`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BorrowSeverity {
    Suppressed,   // prior/external scope, not flagged
    InEpisode,    // use before introduce in same episode
    CrossEpisode, // introduced elsewhere or only after this use
    NoIntro,      // no introduction anywhere
}

pub struct BorrowAhead {
    pub step:     NodeId,
    pub target:   NodeId,
    pub severity: BorrowSeverity,
}

pub fn borrow_ahead(g: &CurriculumGraph, episode: &str) -> Vec<BorrowAhead> {
    // Collect teaching steps in episode with their indices.
    let steps_in_episode: Vec<NodeId> = g
        .node_indices()
        .filter(|&n| matches!(&g[n].kind, NodeKind::TeachingStep(ts) if ts.episode == episode))
        .collect();

    // Precedes edges restricted to episode; build adjacency for reachability.
    let mut precedes_adj: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for &step in &steps_in_episode {
        for edge in g.edges_directed(step, Direction::Outgoing) {
            if let EdgeKind::Precedes(p) = &edge.weight().kind
                && matches!(&g[edge.target()].kind, NodeKind::TeachingStep(ts_to) if p.episode == episode && ts_to.episode == episode)
            {
                precedes_adj.entry(step).or_default().push(edge.target());
            }
        }
    }

    // Introductions keyed by target knowledge node (global).
    let mut introduces_all: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for n in g.node_indices() {
        if let NodeKind::TeachingStep(_) = &g[n].kind {
            for edge in g.edges_directed(n, Direction::Outgoing) {
                if let EdgeKind::Anchors(attrs) = &edge.weight().kind
                    && matches!(attrs.impact, AnchorImpact::Introduce)
                {
                    introduces_all.entry(edge.target()).or_default().push(n);
                }
            }
        }
    }

    let reachability = precedes_reachability(&steps_in_episode, &precedes_adj);

    let mut results = Vec::new();
    for &step in &steps_in_episode {
        for edge in g.edges_directed(step, Direction::Outgoing) {
            if let EdgeKind::Anchors(attrs) = &edge.weight().kind
                && matches!(
                    attrs.impact,
                    AnchorImpact::Use
                        | AnchorImpact::Motivate
                        | AnchorImpact::Refine
                        | AnchorImpact::Target
                )
            {
                let target = edge.target();

                let intro_steps_global = introduces_all.get(&target).cloned().unwrap_or_default();

                let intro_steps_in_episode: Vec<NodeId> = intro_steps_global
                    .iter()
                    .copied()
                    .filter(|s| matches!(&g[*s].kind, NodeKind::TeachingStep(ts) if ts.episode == episode))
                    .collect();

                let mut has_prior_intro = false;
                for intro in &intro_steps_in_episode {
                    if reachability
                        .get(intro)
                        .map(|set| set.contains(&step))
                        .unwrap_or(false)
                    {
                        has_prior_intro = true;
                        break;
                    }
                }
                if has_prior_intro {
                    continue;
                }

                // If introduction_scope is prior/external, downgrade severity to 0 (omit)
                let severity = match &g[target].kind {
                    NodeKind::Knowledge(k) => match k.introduction_scope {
                        crate::graph::IntroductionScope::Prior
                        | crate::graph::IntroductionScope::External => BorrowSeverity::Suppressed,
                        crate::graph::IntroductionScope::InCourse => {
                            if intro_steps_global.is_empty() {
                                BorrowSeverity::NoIntro
                            } else if intro_steps_in_episode.is_empty() {
                                BorrowSeverity::CrossEpisode
                            } else {
                                BorrowSeverity::InEpisode
                            }
                        }
                    },
                    _ => BorrowSeverity::InEpisode,
                };
                if !matches!(severity, BorrowSeverity::Suppressed) {
                    results.push(BorrowAhead {
                        step,
                        target,
                        severity,
                    });
                }
            }
        }
    }

    results
}

fn precedes_reachability(
    steps: &[NodeId],
    adj: &HashMap<NodeId, Vec<NodeId>>,
) -> HashMap<NodeId, HashSet<NodeId>> {
    let mut reach: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();

    for &start in steps {
        let mut seen = HashSet::new();
        let mut queue: VecDeque<NodeId> = VecDeque::new();

        if let Some(children) = adj.get(&start) {
            for &child in children {
                if seen.insert(child) {
                    queue.push_back(child);
                }
            }
        }

        while let Some(node) = queue.pop_front() {
            if let Some(children) = adj.get(&node) {
                for &child in children {
                    if seen.insert(child) {
                        queue.push_back(child);
                    }
                }
            }
        }

        reach.insert(start, seen);
    }

    reach
}

/// Procedural practice: each procedural node should reach an assessment that
/// targets an LO.
pub struct PracticeGap {
    pub node: NodeId,
}

pub fn procedural_practice_gaps(g: &CurriculumGraph) -> Vec<PracticeGap> {
    let procedural_nodes: Vec<NodeId> = g
        .node_indices()
        .filter(|&n| matches!(&g[n].kind, NodeKind::Knowledge(k) if k.knowledge_type == KnowledgeType::Procedural))
        .collect();

    let assessments: Vec<NodeId> = g
        .node_indices()
        .filter(|&n| matches!(&g[n].kind, NodeKind::Knowledge(k) if k.knowledge_type.is_assessment_item()))
        .collect();

    let view = traversal::requires_view(g);

    let mut gaps: Vec<PracticeGap> = procedural_nodes
        .par_iter()
        .filter_map(|&proc| {
            let ok = assessments.iter().any(|&a| {
                has_path_connecting(&view, proc, a, None)
                    && g.edges_directed(a, Direction::Outgoing).any(|e| {
                        matches!(
                            &e.weight().kind,
                            EdgeKind::Assesses(attrs) if attrs.evidence_link.scope == AssessmentScope::Target
                        )
                    })
            });
            if ok {
                None
            } else {
                Some(PracticeGap { node: proc })
            }
        })
        .collect();

    gaps.sort_by_key(|g| g.node.index());
    gaps
}

/// LOs with no incoming assesses(scope=target) edges.
pub fn lo_missing_target_assessments(g: &CurriculumGraph) -> Vec<NodeId> {
    g.node_indices()
        .filter(|&n| {
            matches!(
                &g[n].kind,
                NodeKind::Knowledge(k) if k.knowledge_type == KnowledgeType::LearningOutcome
            )
        })
        .filter(|&lo| {
            g.edges_directed(lo, Direction::Incoming).all(|e| {
                !matches!(
                    &e.weight().kind,
                    EdgeKind::Assesses(attrs)
                        if attrs.evidence_link.scope == AssessmentScope::Target
                )
            })
        })
        .collect()
}

/// Assessment items with no outgoing assesses edges.
pub fn orphan_assessments(g: &CurriculumGraph) -> Vec<NodeId> {
    g.node_indices()
        .filter(|&n| {
            matches!(
                &g[n].kind,
                NodeKind::Knowledge(k) if k.knowledge_type.is_assessment_item()
            )
        })
        .filter(|&a| {
            g.edges_directed(a, Direction::Outgoing)
                .all(|e| !matches!(&e.weight().kind, EdgeKind::Assesses(_)))
        })
        .collect()
}

/// Assessment items not reachable from any first-principle via requires*.
pub fn unreachable_assessments(g: &CurriculumGraph) -> Vec<NodeId> {
    let fps = first_principles(g);
    g.node_indices()
        .filter(|&n| {
            matches!(
                &g[n].kind,
                NodeKind::Knowledge(k) if k.knowledge_type.is_assessment_item()
            )
        })
        .filter(|&a| !fps.iter().any(|fp| has_requires_path(g, *fp, a)))
        .collect()
}

/// TeachingStep nodes with no in- or out-going precedes edges within their
/// episode.
pub fn discourse_orphans(g: &CurriculumGraph, episode: Option<&str>) -> Vec<NodeId> {
    g.node_indices()
        .filter(|&n| matches!(&g[n].kind, NodeKind::TeachingStep(_)))
        .filter(|&n| {
            if let NodeKind::TeachingStep(ts) = &g[n].kind {
                if let Some(ep) = episode
                    && ts.episode != ep
                {
                    return false;
                }
                let ep = &ts.episode;
                let has_in = g
                    .edges_directed(n, Direction::Incoming)
                    .any(|e| matches!(&e.weight().kind, EdgeKind::Precedes(p) if p.episode == *ep));
                let has_out = g
                    .edges_directed(n, Direction::Outgoing)
                    .any(|e| matches!(&e.weight().kind, EdgeKind::Precedes(p) if p.episode == *ep));
                !(has_in || has_out)
            } else {
                false
            }
        })
        .collect()
}

// ---------- helpers ----------

pub(crate) fn requires_ancestors(g: &CurriculumGraph, start: NodeId) -> HashSet<NodeId> {
    let mut seen = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(start);
    while let Some(node) = queue.pop_front() {
        for edge in g.edges_directed(node, Direction::Incoming) {
            if matches!(edge.weight().kind, EdgeKind::Requires(_)) {
                let pred = edge.source();
                if seen.insert(pred) {
                    queue.push_back(pred);
                }
            }
        }
    }
    seen
}

/// Compute in- and out-reach counts for every node in the requires layer using
/// a single topological pass (no per-node BFS). Assumes requires is a DAG; if a
/// cycle exists, falls back to empty maps.
fn requires_reach_counts_all(
    g: &CurriculumGraph,
) -> (HashMap<NodeId, usize>, HashMap<NodeId, usize>) {
    // Try toposort over requires edges; fall back if cyclic.
    let topo = requires_toposort(g).ok();
    let Some(order) = topo else {
        return (HashMap::new(), HashMap::new());
    };

    // Build predecessor and successor adjacency over requires edges.
    let mut succ: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    let mut pred: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for edge in g.edge_indices() {
        if let EdgeKind::Requires(_) = g[edge].kind
            && let Some((u, v)) = g.edge_endpoints(edge)
        {
            succ.entry(u).or_default().push(v);
            pred.entry(v).or_default().push(u);
        }
    }

    // out_reach: process reverse topological order.
    let mut out_sets: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();
    for &n in order.iter().rev() {
        let mut set = HashSet::new();
        if let Some(children) = succ.get(&n) {
            for &c in children {
                set.insert(c);
                if let Some(child_set) = out_sets.get(&c) {
                    set.extend(child_set.iter().copied());
                }
            }
        }
        out_sets.insert(n, set);
    }

    // in_reach: process forward topological order.
    let mut in_sets: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();
    for &n in &order {
        let mut set = HashSet::new();
        if let Some(parents) = pred.get(&n) {
            for &p in parents {
                set.insert(p);
                if let Some(parent_set) = in_sets.get(&p) {
                    set.extend(parent_set.iter().copied());
                }
            }
        }
        in_sets.insert(n, set);
    }

    let in_counts = in_sets
        .into_iter()
        .map(|(k, v)| (k, v.len()))
        .collect::<HashMap<_, _>>();
    let out_counts = out_sets
        .into_iter()
        .map(|(k, v)| (k, v.len()))
        .collect::<HashMap<_, _>>();

    (in_counts, out_counts)
}

fn has_requires_path(g: &CurriculumGraph, from: NodeId, to: NodeId) -> bool {
    let mut queue = VecDeque::new();
    let mut seen = HashSet::new();
    queue.push_back(from);
    while let Some(node) = queue.pop_front() {
        if node == to {
            return true;
        }
        if !seen.insert(node) {
            continue;
        }
        for edge in g.edges_directed(node, Direction::Outgoing) {
            if matches!(edge.weight().kind, EdgeKind::Requires(_)) {
                queue.push_back(edge.target());
            }
        }
    }
    false
}

fn reachable_with_filter(
    g: &CurriculumGraph,
    starts: &[NodeId],
    predicate: impl Fn(&EdgeKind) -> bool,
) -> HashSet<NodeId> {
    let mut seen = HashSet::new();
    let mut queue: VecDeque<NodeId> = VecDeque::new();
    for &s in starts {
        seen.insert(s);
        queue.push_back(s);
    }

    while let Some(node) = queue.pop_front() {
        for edge in g.edges_directed(node, Direction::Outgoing) {
            if predicate(&edge.weight().kind) {
                let tgt = edge.target();
                if seen.insert(tgt) {
                    queue.push_back(tgt);
                }
            }
        }
    }
    seen
}

fn reachable_with_virtual_support(
    g: &CurriculumGraph,
    starts: &[NodeId],
    from: NodeId,
    to: NodeId,
) -> HashSet<NodeId> {
    let mut seen = HashSet::new();
    let mut queue: VecDeque<NodeId> = VecDeque::new();
    for &s in starts {
        seen.insert(s);
        queue.push_back(s);
    }
    while let Some(node) = queue.pop_front() {
        for edge in g.edges_directed(node, Direction::Outgoing) {
            if matches!(edge.weight().kind, EdgeKind::Requires(_) | EdgeKind::Supports(_)) {
                let tgt = edge.target();
                if seen.insert(tgt) {
                    queue.push_back(tgt);
                }
            }
        }
        if node == from && seen.insert(to) {
            queue.push_back(to);
        }
    }
    seen
}

fn reverse_reachable_with_filter(
    g: &CurriculumGraph,
    target: NodeId,
    predicate: impl Fn(&EdgeKind) -> bool,
) -> HashSet<NodeId> {
    let mut seen = HashSet::new();
    let mut queue: VecDeque<NodeId> = VecDeque::new();
    seen.insert(target);
    queue.push_back(target);

    while let Some(node) = queue.pop_front() {
        for edge in g.edges_directed(node, Direction::Incoming) {
            if predicate(&edge.weight().kind) {
                let src = edge.source();
                if seen.insert(src) {
                    queue.push_back(src);
                }
            }
        }
    }
    seen
}

/// Return support edges that participate in at least one path from any first
/// principle to `assessment` when supports are allowed.
fn supports_on_paths_to_assessment(
    g: &CurriculumGraph,
    ctx: &FadeabilityContext,
    assessment: NodeId,
) -> Vec<petgraph::stable_graph::EdgeIndex<u32>> {
    let backwards = reverse_reachable_with_filter(g, assessment, |k| {
        matches!(k, EdgeKind::Requires(_) | EdgeKind::Supports(_))
    });

    g.edge_indices()
        .filter(|&e| matches!(g[e].kind, EdgeKind::Supports(_)))
        .filter(|&e| {
            if let Some((u, v)) = g.edge_endpoints(e) {
                ctx.reachable_requires_or_supports.contains(&u) && backwards.contains(&v)
            } else {
                false
            }
        })
        .collect()
}

pub struct ExtraneousReport {
    pub assessment:       NodeId,
    pub lo:               NodeId,
    pub extraneous_nodes: Vec<NodeId>,
}

pub fn extraneous_report(
    g: &CurriculumGraph,
    assessment: NodeId,
    lo: NodeId,
    intended: &std::collections::HashSet<NodeId>,
) -> ExtraneousReport {
    let extraneous_set = extraneous_knowledge(g, assessment, intended);
    ExtraneousReport {
        assessment,
        lo,
        extraneous_nodes: extraneous_set.into_iter().collect(),
    }
}
