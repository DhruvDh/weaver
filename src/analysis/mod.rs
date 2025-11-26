use std::collections::{HashMap, HashSet, VecDeque};

use petgraph::{
    Direction,
    algo::{is_cyclic_directed, toposort},
    graph::DiGraph,
    visit::EdgeRef,
};
use schemars::JsonSchema;

use crate::{
    graph::{AnchorImpact, CaseTag, CurriculumGraph, EdgeKind, NodeId, NodeKind},
    schema::types::{AssessmentScope, KnowledgeType, SupportKind},
};

/// A topological order over the `requires` layer.
#[derive(Debug, Clone, Copy)]
pub struct RequiresCycle;

pub fn requires_toposort(g: &CurriculumGraph) -> Result<Vec<NodeId>, RequiresCycle> {
    let (subgraph, map) = requires_only(g);
    let order = toposort(&subgraph, None).map_err(|_| RequiresCycle)?;
    let reversed: HashMap<usize, NodeId> = map
        .into_iter()
        .map(|(orig, idx)| (idx.index(), orig))
        .collect();
    Ok(order
        .into_iter()
        .map(|idx| reversed[&idx.index()])
        .collect())
}

/// True when the `requires` layer is acyclic.
pub fn requires_is_dag(g: &CurriculumGraph) -> bool {
    let (subgraph, _) = requires_only(g);
    !is_cyclic_directed(&subgraph)
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

    let rubric_set: HashSet<String> = rubric.iter().cloned().collect();
    let obs_set: HashSet<String> = observations.iter().cloned().collect();

    let covered: Vec<String> = rubric_set.intersection(&obs_set).cloned().collect();
    let missing: Vec<String> = rubric_set.difference(&obs_set).cloned().collect();
    let unused: Vec<String> = obs_set.difference(&rubric_set).cloned().collect();

    CoverageReport {
        lo,
        covered_criteria: covered,
        missing_criteria: missing,
        unused_observation_features: unused,
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

/// Approximate keystone score: |in_reach| * |out_reach| over requires layer.
pub struct KeystoneScore {
    pub node:      NodeId,
    pub score:     usize,
    pub in_reach:  usize,
    pub out_reach: usize,
}

pub fn keystone_scores(g: &CurriculumGraph) -> Vec<KeystoneScore> {
    let mut scores = Vec::new();
    for n in g.node_indices().filter(|&n| {
        matches!(
            &g[n].kind,
            NodeKind::Knowledge(k) if k.knowledge_type.is_instructional_knowledge()
        )
    }) {
        let (in_reach, out_reach) = requires_reach_counts(g, n);
        scores.push(KeystoneScore {
            node: n,
            score: in_reach * out_reach,
            in_reach,
            out_reach,
        });
    }
    scores.sort_by(|a, b| b.score.cmp(&a.score));
    scores
}

/// Fadeability test: assessments reachable only when supports are treated as
/// prerequisites.
pub struct FadeabilityIssue {
    pub assessment: NodeId,
}

pub fn fadeability_issues(g: &CurriculumGraph) -> Vec<FadeabilityIssue> {
    let fps = first_principles(g);
    let reachable_requires = reachable_assessments_requires_only(g, &fps);
    let reachable_with_supports = reachable_assessments_with_supports(g, &fps);
    reachable_with_supports
        .difference(&reachable_requires)
        .map(|&a| FadeabilityIssue { assessment: a })
        .collect()
}

/// Example minimum + variety checks.
pub struct ExampleGap {
    pub node:        NodeId,
    pub description: String,
}

pub fn example_gaps(g: &CurriculumGraph) -> Vec<ExampleGap> {
    let mut gaps = Vec::new();
    for n in g.node_indices() {
        let node = &g[n];
        let knowledge = match &node.kind {
            NodeKind::Knowledge(k) => k,
            _ => continue,
        };
        let supports: Vec<_> = g
            .edges_directed(n, Direction::Incoming)
            .filter_map(|e| match &e.weight().kind {
                EdgeKind::Supports(attrs) => Some(attrs),
                _ => None,
            })
            .collect();

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
                    gaps.push(ExampleGap {
                        node:        n,
                        description: "procedural nodes need >=2 worked examples (typical + \
                                      edge/error)"
                            .to_string(),
                    });
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
                    gaps.push(ExampleGap {
                        node:        n,
                        description: "conceptual nodes need at least one analogy or counterexample"
                            .to_string(),
                    });
                }
            }
            KnowledgeType::Factual => {
                let has_example = supports.iter().any(|s| {
                    s.support_kind == SupportKind::WorkedExample
                        || s.support_kind == SupportKind::Counterexample
                });
                if !has_example {
                    gaps.push(ExampleGap {
                        node:        n,
                        description: "factual nodes need an example or counterexample".into(),
                    });
                }
            }
            KnowledgeType::Metacognitive => {
                let has_hint = supports
                    .iter()
                    .any(|s| s.support_kind == SupportKind::StrategyHint);
                if !has_hint {
                    gaps.push(ExampleGap {
                        node:        n,
                        description: "metacognitive nodes need a strategy hint support".into(),
                    });
                }
            }
            _ => {}
        }

        // Additional CLT guard: high intrinsic load should have at least one support
        // with coverage_tags.
        if matches!(knowledge.intrinsic_load, Some(crate::graph::IntrinsicLoad::High)) {
            let has_support = !supports.is_empty();
            let has_coverage_tag = supports.iter().any(|s| !s.coverage_tags.is_empty());
            if !has_support || !has_coverage_tag {
                gaps.push(ExampleGap {
                    node:        n,
                    description: "high intrinsic_load nodes should include rich supports with \
                                  coverage_tags"
                        .to_string(),
                });
            }
        }
    }
    gaps
}

/// Borrow-ahead detection within an episode.
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

    // helper: is there an introduce step reachable before `use_step`?
    let precedes_reaches = |start: NodeId, goal: NodeId, adj: &HashMap<NodeId, Vec<NodeId>>| {
        let mut seen = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        while let Some(s) = queue.pop_front() {
            if s == goal {
                return true;
            }
            if !seen.insert(s) {
                continue;
            }
            if let Some(neigh) = adj.get(&s) {
                queue.extend(neigh.iter().copied());
            }
        }
        false
    };

    let mut results = Vec::new();
    for &step in &steps_in_episode {
        for edge in g.edges_directed(step, Direction::Outgoing) {
            if let EdgeKind::Anchors(attrs) = &edge.weight().kind
                && matches!(attrs.impact, AnchorImpact::Use)
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
                    if precedes_reaches(*intro, step, &precedes_adj) {
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

/// Procedural practice: each procedural node should reach an assessment that
/// targets an LO.
pub struct PracticeGap {
    pub node: NodeId,
}

pub fn procedural_practice_gaps(g: &CurriculumGraph) -> Vec<PracticeGap> {
    use petgraph::Direction;
    let mut gaps = Vec::new();
    let procedural_nodes: Vec<NodeId> = g
        .node_indices()
        .filter(|&n| matches!(&g[n].kind, NodeKind::Knowledge(k) if k.knowledge_type == KnowledgeType::Procedural))
        .collect();

    for proc in procedural_nodes {
        let mut reachable_assessments = Vec::new();
        // requires* forward
        let mut stack = vec![proc];
        let mut seen = std::collections::HashSet::new();
        while let Some(node) = stack.pop() {
            if !seen.insert(node) {
                continue;
            }
            if matches!(
                &g[node].kind,
                NodeKind::Knowledge(k) if k.knowledge_type.is_assessment_item()
            ) {
                reachable_assessments.push(node);
            }
            for edge in g
                .edges_directed(node, Direction::Outgoing)
                .filter(|e| matches!(e.weight().kind, EdgeKind::Requires(_)))
            {
                stack.push(edge.target());
            }
        }

        let mut ok = false;
        for a in reachable_assessments {
            for edge in g.edges_directed(a, Direction::Outgoing) {
                if let EdgeKind::Assesses(attrs) = &edge.weight().kind
                    && attrs.evidence_link.scope == AssessmentScope::Target
                {
                    ok = true;
                    break;
                }
            }
            if ok {
                break;
            }
        }
        if !ok {
            gaps.push(PracticeGap { node: proc });
        }
    }
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

fn requires_only(
    g: &CurriculumGraph,
) -> (DiGraph<(), ()>, HashMap<NodeId, petgraph::graph::NodeIndex>) {
    let mut sub = DiGraph::<(), ()>::with_capacity(g.node_count(), g.edge_count());
    let mut map = HashMap::new();
    for n in g.node_indices() {
        let idx = sub.add_node(());
        map.insert(n, idx);
    }
    for e in g.edge_indices() {
        if matches!(g[e].kind, EdgeKind::Requires(_)) {
            let (u, v) = g.edge_endpoints(e).expect("valid endpoints");
            let u2 = map[&u];
            let v2 = map[&v];
            sub.add_edge(u2, v2, ());
        }
    }
    (sub, map)
}

fn requires_ancestors(g: &CurriculumGraph, start: NodeId) -> HashSet<NodeId> {
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

fn requires_reach_counts(g: &CurriculumGraph, start: NodeId) -> (usize, usize) {
    let mut in_seen = HashSet::new();
    let mut out_seen = HashSet::new();
    {
        let mut queue = VecDeque::new();
        queue.push_back(start);
        while let Some(node) = queue.pop_front() {
            for edge in g.edges_directed(node, Direction::Incoming) {
                if matches!(edge.weight().kind, EdgeKind::Requires(_)) {
                    let pred = edge.source();
                    if in_seen.insert(pred) {
                        queue.push_back(pred);
                    }
                }
            }
        }
    }
    {
        let mut queue = VecDeque::new();
        queue.push_back(start);
        while let Some(node) = queue.pop_front() {
            for edge in g.edges_directed(node, Direction::Outgoing) {
                if matches!(edge.weight().kind, EdgeKind::Requires(_)) {
                    let succ = edge.target();
                    if out_seen.insert(succ) {
                        queue.push_back(succ);
                    }
                }
            }
        }
    }
    (in_seen.len(), out_seen.len())
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

fn reachable_assessments_requires_only(
    g: &CurriculumGraph,
    first_principles: &[NodeId],
) -> std::collections::HashSet<NodeId> {
    use petgraph::Direction;
    let mut reachable = std::collections::HashSet::new();
    let mut stack = first_principles.to_vec();
    let mut seen = std::collections::HashSet::new();
    while let Some(node) = stack.pop() {
        if !seen.insert(node) {
            continue;
        }
        if matches!(
            &g[node].kind,
            NodeKind::Knowledge(k) if k.knowledge_type.is_assessment_item()
        ) {
            reachable.insert(node);
        }
        for edge in g
            .edges_directed(node, Direction::Outgoing)
            .filter(|e| matches!(e.weight().kind, EdgeKind::Requires(_)))
        {
            stack.push(edge.target());
        }
    }
    reachable
}

fn reachable_assessments_with_supports(
    g: &CurriculumGraph,
    first_principles: &[NodeId],
) -> std::collections::HashSet<NodeId> {
    use petgraph::Direction;
    let mut reachable = std::collections::HashSet::new();
    let mut stack = first_principles.to_vec();
    let mut seen = std::collections::HashSet::new();
    while let Some(node) = stack.pop() {
        if !seen.insert(node) {
            continue;
        }
        if matches!(
            &g[node].kind,
            NodeKind::Knowledge(k) if k.knowledge_type.is_assessment_item()
        ) {
            reachable.insert(node);
        }
        for edge in g.edges_directed(node, Direction::Outgoing) {
            match edge.weight().kind {
                EdgeKind::Requires(_) | EdgeKind::Supports(_) => stack.push(edge.target()),
                _ => {}
            }
        }
    }
    reachable
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
