use std::{collections::HashSet, path::PathBuf, sync::Arc};

use kameo::{Reply, actor::ActorRef};
use tracing::warn;

use crate::{
    agents::deduplication::DeduplicationAgent,
    graph::manager::GraphManager,
    llm_gateway::{GatewayMetrics, LLMGateway},
    rerun_sink::RerunSink,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NicheStatus {
    NotStarted,
    InProgress,
    Cancelled,
    Done,
}

impl NicheStatus {
    pub fn is_done(&self) -> bool {
        matches!(self, NicheStatus::Done)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseKind {
    Harvest,
    Weave,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseStage {
    Idle,
    Harvesting,
    WaitingForWeave,
    Weaving,
    Completed,
    Stopped,
}

#[derive(Debug, Clone)]
pub enum StopReason {
    Explicit(String),
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StopReason::Explicit(reason) => write!(f, "{reason}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HarvestNiche {
    FactualConceptual,
    ProceduralExamples,
    Assessments,
    TeachingSteps,
    SupportsIllustrations,
    Metacognitive,
}

impl HarvestNiche {
    pub fn label(&self) -> &'static str {
        match self {
            HarvestNiche::FactualConceptual => "factual_conceptual",
            HarvestNiche::ProceduralExamples => "procedural_examples",
            HarvestNiche::Assessments => "assessments",
            HarvestNiche::TeachingSteps => "teaching_steps",
            HarvestNiche::SupportsIllustrations => "supports",
            HarvestNiche::Metacognitive => "metacognitive",
        }
    }

    pub fn spec_tag(&self) -> String {
        format!("spec:{}", self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeaveNiche {
    Requires,
    Supports,
    Assesses,
    TeachingSteps,
    CoverageGap,
    CleanupQa,
}

impl WeaveNiche {
    pub fn label(&self) -> &'static str {
        match self {
            WeaveNiche::Requires => "requires",
            WeaveNiche::Supports => "supports",
            WeaveNiche::Assesses => "assesses",
            WeaveNiche::TeachingSteps => "teaching_steps",
            WeaveNiche::CoverageGap => "coverage_gap",
            WeaveNiche::CleanupQa => "cleanup_qa",
        }
    }
}

pub const ALL_HARVEST_NICHES: &[HarvestNiche] = &[
    HarvestNiche::FactualConceptual,
    HarvestNiche::ProceduralExamples,
    HarvestNiche::Assessments,
    HarvestNiche::TeachingSteps,
    HarvestNiche::SupportsIllustrations,
    HarvestNiche::Metacognitive,
];

pub const ALL_WEAVE_NICHES: &[WeaveNiche] = &[
    WeaveNiche::Requires,
    WeaveNiche::Supports,
    WeaveNiche::Assesses,
    WeaveNiche::TeachingSteps,
    WeaveNiche::CoverageGap,
    WeaveNiche::CleanupQa,
];

#[derive(Debug, Clone)]
pub struct NicheOutcome<K> {
    pub niche:      K,
    pub attempts:   usize,
    pub status:     NicheStatus,
    pub last_error: Option<String>,
}

impl<K> NicheOutcome<K> {
    pub fn failures(&self) -> usize {
        if self.status.is_done() { 0 } else { 1 }
    }
}

#[derive(Debug, Clone)]
pub struct PhaseOutcome<K> {
    pub niches:    Vec<NicheOutcome<K>>,
    pub timed_out: bool,
}

impl<K> PhaseOutcome<K> {
    pub fn new(niches: Vec<NicheOutcome<K>>, timed_out: bool) -> Self {
        Self { niches, timed_out }
    }

    pub fn completed(&self) -> bool {
        self.niches.iter().all(|n| n.status.is_done())
    }

    pub fn failures(&self) -> usize {
        self.niches.iter().map(|n| n.failures()).sum::<usize>() + if self.timed_out { 1 } else { 0 }
    }
}

#[derive(Debug, Clone)]
pub struct HarvestReport {
    pub chapter:        PathBuf,
    pub chapter_tag:    String,
    pub tagged_nodes:   usize,
    pub harvest:        PhaseOutcome<HarvestNiche>,
    pub stop_reason:    Option<StopReason>,
    pub ready_to_weave: bool,
}

impl HarvestReport {
    pub fn failures(&self) -> usize {
        self.harvest.failures()
            + if self.tagged_nodes == 0 { 1 } else { 0 }
            + usize::from(self.stop_reason.is_some())
    }
}

#[derive(Debug, Clone)]
pub struct TwoPhaseSummary {
    pub chapter:      PathBuf,
    pub chapter_tag:  String,
    pub tagged_nodes: usize,
    pub harvest:      PhaseOutcome<HarvestNiche>,
    pub weave:        PhaseOutcome<WeaveNiche>,
    pub harvest_only: bool,
    pub stop_reason:  Option<StopReason>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Reply)]
pub struct PhaseStatus {
    pub phase:             PhaseStage,
    pub stop_reason:       Option<StopReason>,
    pub harvest_inflight:  usize,
    pub harvest_completed: usize,
    pub harvest_failures:  usize,
    pub weave_inflight:    usize,
    pub weave_completed:   usize,
    pub weave_failures:    usize,
}

#[derive(Clone)]
pub struct PhaseCtx {
    pub workspace_root: PathBuf,
    pub graph:          ActorRef<GraphManager>,
    pub gateway:        ActorRef<LLMGateway>,
    pub metrics:        Arc<GatewayMetrics>,
    pub dedup_agent:    ActorRef<DeduplicationAgent>,
    pub rerun:          Option<ActorRef<RerunSink>>,
    pub course_commit:  String,
}

impl PhaseCtx {
    pub fn log_persist_failure(&self, err: &anyhow::Error) {
        warn!(error = %err, "Final snapshot persist failed");
    }
}

#[derive(Clone)]
pub struct PhaseStatusRequest;

impl TwoPhaseSummary {
    pub fn harvest_failures(&self) -> usize {
        let missing_nodes = usize::from(self.tagged_nodes == 0);
        self.harvest.failures() + missing_nodes + usize::from(self.stop_reason.is_some())
    }

    pub fn weave_failures(&self) -> usize {
        self.weave.failures() + usize::from(self.stop_reason.is_some())
    }

    pub fn harvested(&self) -> usize {
        if self.harvest_success() { 1 } else { 0 }
    }

    pub fn weave_total(&self) -> usize {
        self.weave.niches.len()
    }

    pub fn harvest_success(&self) -> bool {
        self.stop_reason.is_none()
            && self.tagged_nodes > 0
            && self.harvest.failures() == 0
            && !self.harvest.timed_out
    }

    pub fn weave_success(&self) -> bool {
        self.stop_reason.is_none()
            && (!self.harvest_only || self.weave.failures() == 0)
            && !self.weave.timed_out
            && (self.weave.niches.is_empty() || self.weave.failures() == 0)
    }

    pub fn is_failure(&self) -> bool {
        self.harvest_failures() > 0 || self.weave_failures() > 0 || self.weave.timed_out
    }

    pub fn unique_tags(&self) -> HashSet<String> {
        let mut tags = HashSet::new();
        tags.insert(self.chapter_tag.clone());
        tags
    }
}
