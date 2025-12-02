use std::{
    collections::HashMap,
    hash::Hash,
    path::{Component, Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use anyhow::{Result, anyhow, bail};
use kameo::prelude::*;
use tokio::{
    sync::oneshot,
    task::{JoinHandle, JoinSet},
    time::sleep,
};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use super::{
    prompts::{harvest_prompt, weave_prompt},
    types::{
        ALL_HARVEST_NICHES, ALL_WEAVE_NICHES, HarvestNiche, HarvestReport, NicheOutcome,
        NicheStatus, PhaseCtx, PhaseOutcome, PhaseStage, PhaseStatus, PhaseStatusRequest,
        StopReason, TwoPhaseSummary, WeaveNiche,
    },
};
use crate::{
    constants::DEFAULT_MAX_SUBDELEGATIONS,
    file_reader::{CancelWork, FileReaderQuery, HarvesterReader, WeaverReader},
    graph::commands::ListNodesByTag,
};

const MAX_HARVEST_ATTEMPTS: usize = 3;
const MAX_WEAVE_ATTEMPTS: usize = 3;
const GRACE_PERIOD: Duration = Duration::from_secs(180);

pub struct StartTwoPhase {
    pub chapter:      PathBuf,
    pub cli_opts:     crate::app::Cli,
    pub handles:      crate::app::AppHandles,
    pub harvest_done: Option<oneshot::Sender<HarvestReport>>,
    pub weave_gate:   Option<oneshot::Receiver<()>>,
}

pub struct StopTwoPhase {
    pub reason: StopReason,
}

#[derive(Clone)]
enum ReaderHandle {
    Harvester(ActorRef<HarvesterReader>),
    Weaver(ActorRef<WeaverReader>),
}

impl ReaderHandle {
    async fn cancel(&self, reason: &str) {
        let msg = CancelWork {
            reason: reason.to_string(),
        };
        match self {
            ReaderHandle::Harvester(reader) => {
                let _ = reader.ask(msg).await;
            }
            ReaderHandle::Weaver(reader) => {
                let _ = reader.ask(msg).await;
            }
        }
    }
}

struct PhaseState<K>
where
    K: Copy + Eq + Hash,
{
    statuses:  HashMap<K, NicheOutcome<K>>,
    inflight:  HashMap<K, ReaderHandle>,
    join_set:  JoinSet<()>,
    timer:     Option<JoinHandle<()>>,
    grace:     Option<JoinHandle<()>>,
    timed_out: bool,
}

impl<K> PhaseState<K>
where
    K: Copy + Eq + Hash,
{
    fn new(niches: &[K]) -> Self {
        let mut statuses = HashMap::new();
        for niche in niches {
            statuses.insert(
                *niche,
                NicheOutcome {
                    niche:      *niche,
                    attempts:   0,
                    status:     NicheStatus::NotStarted,
                    last_error: None,
                },
            );
        }
        Self {
            statuses,
            inflight: HashMap::new(),
            join_set: JoinSet::new(),
            timer: None,
            grace: None,
            timed_out: false,
        }
    }

    fn reset(&mut self, niches: &[K]) {
        *self = Self::new(niches);
    }

    fn completed(&self) -> usize {
        self.statuses
            .values()
            .filter(|n| n.status.is_done())
            .count()
    }

    fn failures(&self) -> usize {
        self.statuses
            .values()
            .filter(|n| !n.status.is_done())
            .count()
    }
}

struct HarvestTaskFinished {
    niche:  HarvestNiche,
    result: Result<(), anyhow::Error>,
}

struct WeaveTaskFinished {
    niche:  WeaveNiche,
    result: Result<(), anyhow::Error>,
}

pub struct HarvestTimeoutFired;
pub struct WeaveTimeoutFired;
struct HarvestGraceExpired;
struct WeaveGraceExpired;
struct BeginWeave;
struct WeaveGateClosed;

#[derive(Actor)]
pub struct TwoPhaseOrchestrator {
    ctx:          Option<PhaseCtx>,
    chapter:      Option<PathBuf>,
    chapter_tag:  Option<String>,
    phase:        PhaseStage,
    harvest:      PhaseState<HarvestNiche>,
    weave:        PhaseState<WeaveNiche>,
    tagged_count: usize,
    harvest_only: bool,
    stop_reason:  Option<StopReason>,
    cancellation: Option<CancellationToken>,
    harvest_done: Option<oneshot::Sender<HarvestReport>>,
    weave_gate:   Option<oneshot::Receiver<()>>,
    completion:   Option<oneshot::Sender<Result<TwoPhaseSummary>>>,
}

impl TwoPhaseOrchestrator {
    pub fn new() -> Self {
        Self {
            ctx:          None,
            chapter:      None,
            chapter_tag:  None,
            phase:        PhaseStage::Idle,
            harvest:      PhaseState::new(ALL_HARVEST_NICHES),
            weave:        PhaseState::new(ALL_WEAVE_NICHES),
            tagged_count: 0,
            harvest_only: false,
            stop_reason:  None,
            cancellation: None,
            harvest_done: None,
            weave_gate:   None,
            completion:   None,
        }
    }

    fn reset_state(&mut self) {
        self.phase = PhaseStage::Harvesting;
        self.stop_reason = None;
        self.tagged_count = 0;
        self.harvest_only = false;
        self.harvest.reset(ALL_HARVEST_NICHES);
        self.weave.reset(ALL_WEAVE_NICHES);
        self.completion = None;
    }

    fn cancellation(&self) -> CancellationToken {
        self.cancellation
            .as_ref()
            .cloned()
            .unwrap_or_else(CancellationToken::new)
    }

    fn cancel_timeout(handle: &mut Option<JoinHandle<()>>) {
        if let Some(join) = handle.take() {
            join.abort();
        }
    }

    async fn trigger_stop(&mut self, reason: StopReason) {
        if self.stop_reason.is_some() {
            return;
        }
        self.stop_reason = Some(reason.clone());
        self.phase = PhaseStage::Stopped;
        if let Some(token) = &self.cancellation {
            token.cancel();
        }
        Self::cancel_timeout(&mut self.harvest.grace);
        Self::cancel_timeout(&mut self.weave.grace);
        self.abort_tasks(&reason.to_string()).await;
        self.finalize().await;
    }

    async fn abort_tasks(&mut self, reason: &str) {
        Self::mark_cancelled(&mut self.harvest.statuses, self.harvest.inflight.keys().copied());
        for reader in self.harvest.inflight.drain().map(|(_, r)| r) {
            reader.cancel(reason).await;
        }
        Self::mark_cancelled(&mut self.weave.statuses, self.weave.inflight.keys().copied());
        for reader in self.weave.inflight.drain().map(|(_, r)| r) {
            reader.cancel(reason).await;
        }
        self.harvest.join_set.abort_all();
        self.weave.join_set.abort_all();
        self.drain_join_sets().await;
    }

    fn spawn_harvest_grace(&mut self, actor: ActorRef<Self>) {
        let handle = tokio::spawn(async move {
            sleep(GRACE_PERIOD).await;
            let _ = actor.tell(HarvestGraceExpired).await;
        });
        self.harvest.grace = Some(handle);
    }

    fn spawn_weave_grace(&mut self, actor: ActorRef<Self>) {
        let handle = tokio::spawn(async move {
            sleep(GRACE_PERIOD).await;
            let _ = actor.tell(WeaveGraceExpired).await;
        });
        self.weave.grace = Some(handle);
    }

    fn ensure_status<K>(
        statuses: &mut HashMap<K, NicheOutcome<K>>,
        niche: K,
    ) -> &mut NicheOutcome<K>
    where
        K: Copy + Eq + Hash,
    {
        statuses.entry(niche).or_insert(NicheOutcome {
            niche,
            attempts: 0,
            status: NicheStatus::NotStarted,
            last_error: None,
        })
    }

    fn mark_cancelled<K, I>(statuses: &mut HashMap<K, NicheOutcome<K>>, niches: I)
    where
        K: Copy + Eq + Hash,
        I: IntoIterator<Item = K>,
    {
        for niche in niches {
            let status = Self::ensure_status(statuses, niche);
            if !status.status.is_done() {
                status.status = NicheStatus::Cancelled;
            }
        }
    }

    fn spawn_harvest_task(&mut self, orchestrator: ActorRef<Self>, niche: HarvestNiche) {
        let Some(phase_ctx) = &self.ctx else { return };
        if self.harvest.timed_out || self.stop_reason.is_some() {
            return;
        }
        {
            let status = Self::ensure_status(&mut self.harvest.statuses, niche);
            if status.status.is_done() || status.attempts >= MAX_HARVEST_ATTEMPTS {
                return;
            }
            status.attempts = status.attempts.saturating_add(1);
            status.status = NicheStatus::InProgress;
        }

        let cancellation = self.cancellation().child_token();
        let reader_actor = match HarvesterReader::from_env_with_limit_and_cancellation(
            phase_ctx.workspace_root.clone(),
            phase_ctx.gateway.clone(),
            Arc::clone(&phase_ctx.metrics),
            phase_ctx.graph.clone(),
            phase_ctx.dedup_agent.clone(),
            phase_ctx.rerun.clone(),
            DEFAULT_MAX_SUBDELEGATIONS,
            phase_ctx.course_commit.clone(),
            cancellation.clone(),
        ) {
            Ok(actor) => actor,
            Err(err) => {
                warn!(error = %err, niche = ?niche, "Failed to initialize harvester reader");
                let status = Self::ensure_status(&mut self.harvest.statuses, niche);
                status.last_error = Some(err.to_string());
                return;
            }
        };

        let reader = HarvesterReader::spawn(reader_actor);
        self.harvest
            .inflight
            .insert(niche, ReaderHandle::Harvester(reader.clone()));
        let chapter_path = self.chapter.clone().unwrap_or_default();
        let chapter_tag = self.chapter_tag.clone().unwrap_or_default();
        let ctx_for_task = phase_ctx.clone();
        self.harvest.join_set.spawn(async move {
            let result = harvest_niche_task(
                ctx_for_task,
                chapter_path,
                chapter_tag,
                niche,
                reader,
                cancellation,
            )
            .await;
            let _ = orchestrator
                .tell(HarvestTaskFinished { niche, result })
                .await;
        });
    }

    fn spawn_weave_task(&mut self, orchestrator: ActorRef<Self>, niche: WeaveNiche) {
        let Some(phase_ctx) = &self.ctx else { return };
        if self.weave.timed_out || self.stop_reason.is_some() {
            return;
        }
        {
            let status = Self::ensure_status(&mut self.weave.statuses, niche);
            if status.status.is_done() || status.attempts >= MAX_WEAVE_ATTEMPTS {
                return;
            }
            status.attempts = status.attempts.saturating_add(1);
            status.status = NicheStatus::InProgress;
        }

        let cancellation = self.cancellation().child_token();
        let reader_actor = match WeaverReader::from_env_with_limit_and_cancellation(
            phase_ctx.workspace_root.clone(),
            phase_ctx.gateway.clone(),
            Arc::clone(&phase_ctx.metrics),
            phase_ctx.graph.clone(),
            phase_ctx.dedup_agent.clone(),
            phase_ctx.rerun.clone(),
            DEFAULT_MAX_SUBDELEGATIONS,
            phase_ctx.course_commit.clone(),
            cancellation.clone(),
        ) {
            Ok(actor) => actor,
            Err(err) => {
                warn!(error = %err, niche = ?niche, "Failed to initialize weaver reader");
                let status = Self::ensure_status(&mut self.weave.statuses, niche);
                status.last_error = Some(err.to_string());
                return;
            }
        };

        let reader = WeaverReader::spawn(reader_actor);
        self.weave
            .inflight
            .insert(niche, ReaderHandle::Weaver(reader.clone()));
        let chapter_path = self.chapter.clone().unwrap_or_default();
        let chapter_tag = self.chapter_tag.clone().unwrap_or_default();
        self.weave.join_set.spawn(async move {
            let result =
                weave_niche_task(chapter_path, chapter_tag, niche, reader, cancellation).await;
            let _ = orchestrator.tell(WeaveTaskFinished { niche, result }).await;
        });
    }

    fn fill_harvest(&mut self, actor: &ActorRef<Self>) {
        if self.stop_reason.is_some() {
            return;
        }
        for niche in ALL_HARVEST_NICHES.iter().copied() {
            let status = Self::ensure_status(&mut self.harvest.statuses, niche);
            if status.status.is_done() || self.harvest.inflight.contains_key(&niche) {
                continue;
            }
            if status.attempts >= MAX_HARVEST_ATTEMPTS || self.harvest.timed_out {
                continue;
            }
            self.spawn_harvest_task(actor.clone(), niche);
        }
    }

    fn fill_weave(&mut self, actor: &ActorRef<Self>) {
        if self.stop_reason.is_some() {
            return;
        }
        for niche in ALL_WEAVE_NICHES.iter().copied() {
            let status = Self::ensure_status(&mut self.weave.statuses, niche);
            if status.status.is_done() || self.weave.inflight.contains_key(&niche) {
                continue;
            }
            if status.attempts >= MAX_WEAVE_ATTEMPTS || self.weave.timed_out {
                continue;
            }
            self.spawn_weave_task(actor.clone(), niche);
        }
    }

    async fn finalize(&mut self) {
        self.drain_join_sets().await;
        let summary = self.build_summary();
        let outcome = self.evaluate_summary(summary);
        if let Some(sender) = self.completion.take() {
            let _ = sender.send(outcome);
        }
    }

    fn build_summary(&self) -> TwoPhaseSummary {
        let harvest_outcome = PhaseOutcome::new(
            self.harvest.statuses.values().cloned().collect(),
            self.harvest.timed_out,
        );
        let weave_outcome = if self.harvest_only {
            PhaseOutcome::new(Vec::new(), false)
        } else {
            PhaseOutcome::new(self.weave.statuses.values().cloned().collect(), self.weave.timed_out)
        };
        TwoPhaseSummary {
            chapter:      self.chapter.clone().unwrap_or_default(),
            chapter_tag:  self.chapter_tag.clone().unwrap_or_default(),
            tagged_nodes: self.tagged_count,
            harvest:      harvest_outcome,
            weave:        weave_outcome,
            harvest_only: self.harvest_only,
            stop_reason:  self.stop_reason.clone(),
        }
    }

    fn evaluate_summary(&self, summary: TwoPhaseSummary) -> Result<TwoPhaseSummary> {
        if let Some(reason) = summary.stop_reason.clone() {
            return Err(anyhow!("Two-phase construction stopped: {reason}"));
        }
        if summary.harvest_failures() > 0 {
            return Err(anyhow!("Harvest phase reported failures"));
        }
        if !summary.harvest_only && summary.weave_failures() > 0 {
            return Err(anyhow!("Weave phase reported failures"));
        }
        Ok(summary)
    }

    async fn drain_join_sets(&mut self) {
        while let Some(res) = self.harvest.join_set.try_join_next() {
            if let Err(err) = res {
                warn!(error = ?err, "Harvest task failed while draining");
            }
        }
        while let Some(res) = self.weave.join_set.try_join_next() {
            if let Err(err) = res {
                warn!(error = ?err, "Weave task failed while draining");
            }
        }
    }

    async fn finish_harvest(&mut self, actor: &ActorRef<Self>) {
        Self::cancel_timeout(&mut self.harvest.timer);
        Self::cancel_timeout(&mut self.harvest.grace);
        if self.stop_reason.is_some() {
            self.finalize().await;
            return;
        }

        let tagged_count = if let (Some(ctx), Some(tag)) = (&self.ctx, &self.chapter_tag) {
            match ctx.graph.ask(ListNodesByTag { tag: tag.clone() }).await {
                Ok(tagged) => tagged.len(),
                Err(err) => {
                    warn!(error = %err, "Failed to list harvested nodes by tag");
                    0
                }
            }
        } else {
            0
        };
        self.tagged_count = tagged_count;

        let harvest_outcome = PhaseOutcome::new(
            self.harvest.statuses.values().cloned().collect(),
            self.harvest.timed_out,
        );
        if let Some(sender) = self.harvest_done.take() {
            // If we harvested anything, allow weaving to proceed even when the
            // harvest phase timed out or niches are incomplete so we can at
            // least wire the available nodes.
            let ready_to_weave = tagged_count > 0;
            let report = HarvestReport {
                chapter: self.chapter.clone().unwrap_or_default(),
                chapter_tag: self.chapter_tag.clone().unwrap_or_default(),
                tagged_nodes: tagged_count,
                harvest: harvest_outcome.clone(),
                stop_reason: self.stop_reason.clone(),
                ready_to_weave,
            };
            let _ = sender.send(report);
        }

        if tagged_count == 0 {
            warn!("Harvest produced no tagged nodes; skipping weave");
            self.harvest_only = true;
            self.phase = PhaseStage::Completed;
            self.finalize().await;
            return;
        }

        if self.weave_gate.is_none() {
            warn!("No weave gate provided; completing after harvest");
            self.harvest_only = true;
            self.phase = PhaseStage::Completed;
            self.finalize().await;
            return;
        }

        self.phase = PhaseStage::WaitingForWeave;
        self.wait_for_weave_gate(actor.clone());
    }

    fn wait_for_weave_gate(&mut self, actor: ActorRef<Self>) {
        if let Some(gate) = self.weave_gate.take() {
            tokio::spawn(async move {
                match gate.await {
                    Ok(()) => {
                        let _ = actor.tell(BeginWeave).await;
                    }
                    Err(_) => {
                        let _ = actor.tell(WeaveGateClosed).await;
                    }
                }
            });
        } else {
            tokio::spawn(async move {
                let _ = actor.tell(WeaveGateClosed).await;
            });
        }
    }

    async fn start_weave(&mut self, actor: &ActorRef<Self>) {
        if self.stop_reason.is_some() || self.harvest_only {
            self.finalize().await;
            return;
        }
        self.phase = PhaseStage::Weaving;
        self.fill_weave(actor);
        if self.weave.inflight.is_empty() {
            self.finish_weave().await;
        }
    }

    async fn finish_weave(&mut self) {
        Self::cancel_timeout(&mut self.weave.timer);
        Self::cancel_timeout(&mut self.weave.grace);
        self.phase = PhaseStage::Completed;
        self.finalize().await;
    }

    async fn handle_harvest_timeout(&mut self, actor: &ActorRef<Self>) {
        if matches!(self.phase, PhaseStage::Harvesting) {
            self.harvest.timed_out = true;
            warn!("Harvest phase timed out; stopping new harvest tasks");
            if let Some(token) = &self.cancellation {
                token.cancel();
            }
            Self::cancel_timeout(&mut self.harvest.timer);
            if self.harvest.inflight.is_empty() {
                self.finish_harvest(actor).await;
            } else {
                self.spawn_harvest_grace(actor.clone());
            }
        }
    }

    async fn handle_weave_timeout(&mut self, actor: &ActorRef<Self>) {
        if matches!(self.phase, PhaseStage::Weaving | PhaseStage::WaitingForWeave) {
            self.weave.timed_out = true;
            warn!("Weave phase timed out; stopping new weave tasks");
            if let Some(token) = &self.cancellation {
                token.cancel();
            }
            Self::cancel_timeout(&mut self.weave.timer);
            if self.weave.inflight.is_empty() {
                self.finish_weave().await;
            } else {
                self.spawn_weave_grace(actor.clone());
            }
        }
    }
}

impl Default for TwoPhaseOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

impl Message<StartTwoPhase> for TwoPhaseOrchestrator {
    type Reply = DelegatedReply<Result<TwoPhaseSummary>>;

    async fn handle(
        &mut self,
        StartTwoPhase {
            chapter,
            cli_opts,
            handles,
            harvest_done,
            weave_gate,
        }: StartTwoPhase,
        ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        if self.completion.is_some() {
            return ctx.spawn(async { bail!("Two-phase orchestrator already running") });
        }

        let course_commit = handles
            .graph
            .ask(crate::graph::manager::GetCourseCommit)
            .await
            .unwrap_or_default();
        let metrics = handles
            .gateway
            .ask(crate::llm_gateway::GetGatewayMetrics)
            .await
            .unwrap_or_else(|_| Arc::new(crate::llm_gateway::GatewayMetrics::default()));

        self.cancellation = Some(CancellationToken::new());
        self.ctx = Some(PhaseCtx {
            workspace_root: cli_opts.workspace.clone(),
            graph: handles.graph.clone(),
            gateway: handles.gateway.clone(),
            metrics: Arc::clone(&metrics),
            dedup_agent: handles.dedup.clone(),
            rerun: handles.rerun.clone(),
            course_commit,
        });
        self.chapter = Some(chapter.clone());
        self.chapter_tag =
            Some(format!("source:{}", chapter_scope_tag(&cli_opts.workspace, &chapter)));
        self.harvest_done = harvest_done;
        self.weave_gate = weave_gate;
        self.reset_state();
        let (tx, rx) = oneshot::channel();
        self.completion = Some(tx);

        let actor = ctx.actor_ref().clone();
        self.fill_harvest(&actor);
        if self.harvest.inflight.is_empty() {
            self.finish_harvest(&actor).await;
        }

        ctx.spawn(async move {
            rx.await
                .unwrap_or_else(|_| Err(anyhow!("Two-phase orchestrator stopped unexpectedly")))
        })
    }
}

impl Message<StopTwoPhase> for TwoPhaseOrchestrator {
    type Reply = Result<()>;

    async fn handle(
        &mut self,
        StopTwoPhase { reason }: StopTwoPhase,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        self.trigger_stop(reason).await;
        Ok(())
    }
}

impl Message<PhaseStatusRequest> for TwoPhaseOrchestrator {
    type Reply = PhaseStatus;

    async fn handle(
        &mut self,
        _msg: PhaseStatusRequest,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        PhaseStatus {
            phase:             self.phase,
            stop_reason:       self.stop_reason.clone(),
            harvest_inflight:  self.harvest.inflight.len(),
            harvest_completed: self.harvest.completed(),
            harvest_failures:  self.harvest.failures(),
            weave_inflight:    self.weave.inflight.len(),
            weave_completed:   self.weave.completed(),
            weave_failures:    self.weave.failures(),
        }
    }
}

impl Message<HarvestTaskFinished> for TwoPhaseOrchestrator {
    type Reply = ();

    async fn handle(
        &mut self,
        HarvestTaskFinished { niche, result }: HarvestTaskFinished,
        ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let actor = ctx.actor_ref().clone();
        if let Some(reader) = self.harvest.inflight.remove(&niche) {
            reader.cancel("harvest task completed").await;
        }
        if self.stop_reason.is_some() {
            self.drain_join_sets().await;
            return;
        }
        let status = Self::ensure_status(&mut self.harvest.statuses, niche);
        match result {
            Ok(()) => {
                info!(niche = %niche.label(), "Harvest niche complete");
                status.status = NicheStatus::Done;
                status.last_error = None;
            }
            Err(err) => {
                warn!(niche = %niche.label(), error = ?err, "Harvest niche failed");
                status.last_error = Some(err.to_string());
                status.status = NicheStatus::Cancelled;
            }
        }

        if self.harvest.timed_out {
            self.drain_join_sets().await;
            if self.harvest.inflight.is_empty() {
                self.finish_harvest(&actor).await;
            }
            return;
        }

        self.fill_harvest(&actor);
        if self.harvest.inflight.is_empty()
            && self
                .harvest
                .statuses
                .values()
                .all(|n| n.status.is_done() || n.attempts >= MAX_HARVEST_ATTEMPTS)
        {
            self.finish_harvest(&actor).await;
        }
        self.drain_join_sets().await;
    }
}

impl Message<WeaveTaskFinished> for TwoPhaseOrchestrator {
    type Reply = ();

    async fn handle(
        &mut self,
        WeaveTaskFinished { niche, result }: WeaveTaskFinished,
        ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let actor = ctx.actor_ref().clone();
        if let Some(reader) = self.weave.inflight.remove(&niche) {
            reader.cancel("weave task completed").await;
        }
        if self.stop_reason.is_some() {
            self.drain_join_sets().await;
            return;
        }
        let status = Self::ensure_status(&mut self.weave.statuses, niche);
        match result {
            Ok(()) => {
                info!(niche = %niche.label(), "Weave niche complete");
                status.status = NicheStatus::Done;
                status.last_error = None;
            }
            Err(err) => {
                warn!(niche = %niche.label(), error = ?err, "Weave niche failed");
                status.last_error = Some(err.to_string());
                status.status = NicheStatus::Cancelled;
            }
        }

        if self.weave.timed_out {
            self.drain_join_sets().await;
            if self.weave.inflight.is_empty() {
                self.finish_weave().await;
            }
            return;
        }

        self.fill_weave(&actor);
        if self.weave.inflight.is_empty()
            && self
                .weave
                .statuses
                .values()
                .all(|n| n.status.is_done() || n.attempts >= MAX_WEAVE_ATTEMPTS)
        {
            self.finish_weave().await;
        }
        self.drain_join_sets().await;
    }
}

impl Message<HarvestTimeoutFired> for TwoPhaseOrchestrator {
    type Reply = ();

    async fn handle(
        &mut self,
        _msg: HarvestTimeoutFired,
        ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let actor = ctx.actor_ref().clone();
        self.handle_harvest_timeout(&actor).await;
    }
}

impl Message<WeaveTimeoutFired> for TwoPhaseOrchestrator {
    type Reply = ();

    async fn handle(
        &mut self,
        _msg: WeaveTimeoutFired,
        ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let actor = ctx.actor_ref().clone();
        self.handle_weave_timeout(&actor).await;
    }
}

impl Message<HarvestGraceExpired> for TwoPhaseOrchestrator {
    type Reply = ();

    async fn handle(
        &mut self,
        _msg: HarvestGraceExpired,
        ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        if matches!(self.phase, PhaseStage::Harvesting) && self.harvest.timed_out {
            warn!("Harvest grace period expired; cancelling remaining harvest tasks");
            if let Some(token) = &self.cancellation {
                token.cancel();
            }
            Self::mark_cancelled(&mut self.harvest.statuses, self.harvest.inflight.keys().copied());
            for reader in self.harvest.inflight.drain().map(|(_, r)| r) {
                reader.cancel("harvest grace expired").await;
            }
            self.harvest.join_set.abort_all();
            self.drain_join_sets().await;
            let actor = ctx.actor_ref().clone();
            self.finish_harvest(&actor).await;
        }
    }
}

impl Message<WeaveGraceExpired> for TwoPhaseOrchestrator {
    type Reply = ();

    async fn handle(
        &mut self,
        _msg: WeaveGraceExpired,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        if matches!(self.phase, PhaseStage::Weaving) && self.weave.timed_out {
            warn!("Weave grace period expired; cancelling remaining weave tasks");
            if let Some(token) = &self.cancellation {
                token.cancel();
            }
            Self::mark_cancelled(&mut self.weave.statuses, self.weave.inflight.keys().copied());
            for reader in self.weave.inflight.drain().map(|(_, r)| r) {
                reader.cancel("weave grace expired").await;
            }
            self.weave.join_set.abort_all();
            self.drain_join_sets().await;
            self.finish_weave().await;
        }
    }
}

impl Message<BeginWeave> for TwoPhaseOrchestrator {
    type Reply = ();

    async fn handle(
        &mut self,
        _msg: BeginWeave,
        ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let actor = ctx.actor_ref().clone();
        self.start_weave(&actor).await;
    }
}

impl Message<WeaveGateClosed> for TwoPhaseOrchestrator {
    type Reply = ();

    async fn handle(
        &mut self,
        _msg: WeaveGateClosed,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        if matches!(self.phase, PhaseStage::WaitingForWeave) {
            warn!("Weave gate closed; finalizing after harvest-only run");
            self.harvest_only = true;
            self.phase = PhaseStage::Completed;
            self.finalize().await;
        }
    }
}

async fn harvest_niche_task(
    _ctx: PhaseCtx,
    chapter_path: PathBuf,
    chapter_tag: String,
    niche: HarvestNiche,
    reader: ActorRef<HarvesterReader>,
    cancellation: CancellationToken,
) -> Result<(), anyhow::Error> {
    if cancellation.is_cancelled() {
        bail!("Harvest cancelled");
    }
    let spec_tag = niche.spec_tag();
    let (focus_label, focus_directive, body) =
        harvest_prompt(&chapter_path, &chapter_tag, &spec_tag, niche);
    let prompt = format!(
        "PHASE 1: Harvest specialist for {chapter}\n- Apply tags: {chapter_tag} and {spec_tag} on \
         every node.\n- Use req:/sup:/ref: hints in tags for suspected prerequisites.\n- Do NOT \
         create edges.\n- Focus: {focus_label} ({focus_directive}).\n{body}",
        chapter = chapter_path.display(),
    );

    let result = reader.ask(FileReaderQuery { prompt }).await;
    if cancellation.is_cancelled() {
        bail!("Harvest cancelled");
    }
    result.map(|_| ()).map_err(anyhow::Error::from)
}

async fn weave_niche_task(
    chapter_path: PathBuf,
    chapter_tag: String,
    niche: WeaveNiche,
    reader: ActorRef<WeaverReader>,
    cancellation: CancellationToken,
) -> Result<(), anyhow::Error> {
    if cancellation.is_cancelled() {
        bail!("Weave cancelled");
    }
    let (focus_label, focus_directive, body) = weave_prompt(&chapter_path, &chapter_tag, niche);
    let prompt = format!(
        "PHASE 2: Weave specialist for {chapter}\n- Scope nodes with graph_list_nodes_by_tag \
         {chapter_tag}.\n- Do NOT create new nodes.\n- Focus: {focus_label} \
         ({focus_directive}).\n{body}",
        chapter = chapter_path.display(),
    );

    let result = reader.ask(FileReaderQuery { prompt }).await;
    if cancellation.is_cancelled() {
        bail!("Weave cancelled");
    }
    result.map(|_| ()).map_err(anyhow::Error::from)
}

pub(crate) fn chapter_scope_tag(workspace_root: &Path, chapter_path: &Path) -> String {
    let relative = if chapter_path.is_absolute() {
        pathdiff::diff_paths(chapter_path, workspace_root)
            .unwrap_or_else(|| chapter_path.to_path_buf())
    } else {
        chapter_path.to_path_buf()
    };

    let normalized = relative
        .components()
        .filter_map(|component| match component {
            Component::Normal(part) => Some(part.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/");

    if normalized.is_empty() {
        chapter_path.display().to_string()
    } else {
        normalized
    }
}
