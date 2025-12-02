use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    sync::Arc,
    time::Duration,
};

use anyhow::{Result, anyhow, bail};
use kameo::prelude::*;
use tokio::{sync::oneshot, task::JoinHandle, time::sleep};
use tracing::{info, warn};

use super::{
    orchestrator::{
        HarvestTimeoutFired, StartTwoPhase, StopTwoPhase, TwoPhaseOrchestrator, WeaveTimeoutFired,
        chapter_scope_tag,
    },
    types::{
        ALL_HARVEST_NICHES, ALL_WEAVE_NICHES, HarvestReport, NicheOutcome, NicheStatus, PhaseCtx,
        PhaseOutcome, StopReason, TwoPhaseSummary,
    },
    worker::{DedupPersistWorker, PersistOnlyRequest, RunDedupPersist},
};
use crate::{
    graph::manager::GetCourseCommit,
    llm_gateway::{GatewayMetrics, GetGatewayMetrics},
};

#[derive(Debug, Clone)]
pub enum ChildOutcome {
    Success {
        chapter: PathBuf,
        summary: TwoPhaseSummary,
    },
    Failure {
        chapter: PathBuf,
        error:   Arc<anyhow::Error>,
        summary: Option<TwoPhaseSummary>,
    },
}

#[derive(Debug, Clone)]
pub struct TwoPhaseSupervisorSummary {
    pub outcomes:    Vec<ChildOutcome>,
    pub stop_reason: Option<StopReason>,
}

impl TwoPhaseSupervisorSummary {
    pub fn successes(&self) -> usize {
        self.outcomes
            .iter()
            .filter(|outcome| matches!(outcome, ChildOutcome::Success { .. }))
            .count()
    }

    pub fn failures(&self) -> usize {
        self.outcomes
            .iter()
            .filter(|outcome| matches!(outcome, ChildOutcome::Failure { .. }))
            .count()
    }
}

pub struct StartTwoPhaseSupervision {
    pub chapters:     Vec<PathBuf>,
    pub cli_opts:     crate::app::Cli,
    pub handles:      Option<crate::app::AppHandles>,
    pub dedup_worker: Option<ActorRef<DedupPersistWorker>>,
}

pub struct StopAll {
    pub reason: StopReason,
}

struct HarvestPhaseTimedOut;
struct WeavePhaseTimedOut;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SupervisorStage {
    Harvesting,
    Deduplicating,
    Weaving,
    Completed,
    Stopped,
}

pub type ChildLauncher = Arc<
    dyn Fn(
            PathBuf,
            crate::app::Cli,
            Option<crate::app::AppHandles>,
            oneshot::Sender<HarvestReport>,
            oneshot::Receiver<HarvestReport>,
            oneshot::Sender<()>,
            oneshot::Receiver<()>,
            ActorRef<TwoPhaseSupervisor>,
        ) -> ChildHandle
        + Send
        + Sync,
>;

pub struct ChildHandle {
    pub join:         JoinHandle<()>,
    pub stop: Box<dyn Fn(StopReason) -> futures::future::BoxFuture<'static, ()> + Send + Sync>,
    pub harvest_rx:   Option<oneshot::Receiver<HarvestReport>>,
    pub weave_gate:   Option<oneshot::Sender<()>>,
    pub orchestrator: Option<ActorRef<TwoPhaseOrchestrator>>,
}

impl ChildHandle {
    async fn stop(&self, reason: StopReason) {
        (self.stop)(reason).await;
    }

    async fn wait_with_timeout(&mut self, duration: Duration) -> bool {
        match tokio::time::timeout(duration, &mut self.join).await {
            Ok(res) => {
                if let Err(err) = res {
                    warn!(error = ?err, "Two-phase child task failed while joining");
                }
                true
            }
            Err(_) => false,
        }
    }

    fn abort(&mut self) {
        self.join.abort();
    }

    async fn wait(self) {
        if let Err(err) = self.join.await {
            warn!(error = ?err, "Two-phase child task failed while joining");
        }
    }

    fn take_harvest(&mut self) -> Option<oneshot::Receiver<HarvestReport>> {
        self.harvest_rx.take()
    }

    fn start_weave(&mut self) {
        if let Some(tx) = self.weave_gate.take() {
            let _ = tx.send(());
        }
    }

    fn close_weave(&mut self) {
        // Keep the gate so weaving can be started later when a slot is
        // available.
    }

    async fn harvest_timeout(&self) {
        if let Some(orchestrator) = &self.orchestrator {
            let _ = orchestrator.tell(HarvestTimeoutFired).await;
        }
    }

    async fn weave_timeout(&self) {
        if let Some(orchestrator) = &self.orchestrator {
            let _ = orchestrator.tell(WeaveTimeoutFired).await;
        }
    }
}

#[derive(Actor)]
pub struct TwoPhaseSupervisor {
    max_concurrent:    usize,
    pending:           VecDeque<PathBuf>,
    inflight:          HashMap<PathBuf, ChildHandle>,
    weaving_active:    std::collections::HashSet<PathBuf>,
    outcomes:          Vec<ChildOutcome>,
    harvest_reports:   HashMap<PathBuf, Result<HarvestReport>>,
    completion:        Option<oneshot::Sender<Result<TwoPhaseSupervisorSummary>>>,
    stop_reason:       Option<StopReason>,
    harvest_timeout:   Option<Duration>,
    weave_timeout:     Option<Duration>,
    harvest_timer:     Option<JoinHandle<()>>,
    weave_timer:       Option<JoinHandle<()>>,
    harvest_timed_out: bool,
    weave_timed_out:   bool,
    cli_opts:          Option<crate::app::Cli>,
    handles:           Option<crate::app::AppHandles>,
    dedup_worker:      Option<ActorRef<DedupPersistWorker>>,
    dedup_threshold:   f64,
    phase_ctx:         Option<PhaseCtx>,
    launch_child:      ChildLauncher,
    require_handles:   bool,
    total_chapters:    usize,
    stage:             SupervisorStage,
}

impl TwoPhaseSupervisor {
    async fn build_phase_ctx(&self) -> Option<Result<PhaseCtx>> {
        let (Some(handles), Some(cli_opts)) = (&self.handles, &self.cli_opts) else {
            return None;
        };
        let course_commit = handles.graph.ask(GetCourseCommit).await.ok()?;
        let metrics = handles
            .gateway
            .ask(GetGatewayMetrics)
            .await
            .unwrap_or_else(|_| Arc::new(GatewayMetrics::default()));
        Some(Ok(PhaseCtx {
            workspace_root: cli_opts.workspace.clone(),
            graph: handles.graph.clone(),
            gateway: handles.gateway.clone(),
            metrics: Arc::clone(&metrics),
            dedup_agent: handles.dedup.clone(),
            rerun: handles.rerun.clone(),
            course_commit,
        }))
    }

    async fn run_dedup_barrier(&self) {
        let (Some(worker), Some(ctx)) = (&self.dedup_worker, &self.phase_ctx) else {
            warn!("Skipping dedup barrier: worker or context missing");
            return;
        };
        let threshold = self.dedup_threshold;
        if let Err(err) = worker
            .ask(RunDedupPersist {
                ctx: ctx.clone(),
                threshold,
                cancellation: None,
            })
            .await
        {
            warn!(error = %err, "Batch dedup/persist after harvests failed");
        } else {
            info!("Batch dedup/persist after harvests completed");
        }
    }

    async fn run_final_persist(&self) {
        let (Some(worker), Some(ctx)) = (&self.dedup_worker, &self.phase_ctx) else {
            return;
        };
        if let Err(err) = worker
            .ask(PersistOnlyRequest {
                ctx:          ctx.clone(),
                cancellation: None,
            })
            .await
        {
            warn!(error = %err, "Final persist after weaving failed");
        }
    }

    fn cancel_harvest_timer(&mut self) {
        if let Some(handle) = self.harvest_timer.take() {
            handle.abort();
        }
    }

    fn cancel_weave_timer(&mut self) {
        if let Some(handle) = self.weave_timer.take() {
            handle.abort();
        }
    }

    fn spawn_harvest_timer(&mut self, actor: ActorRef<Self>) {
        if let Some(duration) = self.harvest_timeout {
            self.cancel_harvest_timer();
            self.harvest_timer = Some(tokio::spawn(async move {
                sleep(duration).await;
                let _ = actor.tell(HarvestPhaseTimedOut).await;
            }));
        }
    }

    fn spawn_weave_timer(&mut self, actor: ActorRef<Self>) {
        if let Some(duration) = self.weave_timeout {
            self.cancel_weave_timer();
            self.weave_timer = Some(tokio::spawn(async move {
                sleep(duration).await;
                let _ = actor.tell(WeavePhaseTimedOut).await;
            }));
        }
    }

    fn synthesize_skipped_chapter(&mut self, chapter: PathBuf, reason: &str) {
        let workspace = self
            .cli_opts
            .as_ref()
            .map(|c| c.workspace.clone())
            .unwrap_or_default();
        let chapter_tag = format!("source:{}", chapter_scope_tag(&workspace, &chapter));
        let harvest_outcome = PhaseOutcome::new(
            ALL_HARVEST_NICHES
                .iter()
                .copied()
                .map(|niche| NicheOutcome {
                    niche,
                    attempts: 0,
                    status: NicheStatus::NotStarted,
                    last_error: None,
                })
                .collect(),
            true,
        );
        let weave_outcome = PhaseOutcome::new(
            ALL_WEAVE_NICHES
                .iter()
                .copied()
                .map(|niche| NicheOutcome {
                    niche,
                    attempts: 0,
                    status: NicheStatus::NotStarted,
                    last_error: None,
                })
                .collect(),
            true,
        );
        let stop_reason = Some(StopReason::Explicit(reason.to_string()));
        let report = HarvestReport {
            chapter:        chapter.clone(),
            chapter_tag:    chapter_tag.clone(),
            tagged_nodes:   0,
            harvest:        harvest_outcome.clone(),
            stop_reason:    stop_reason.clone(),
            ready_to_weave: false,
        };
        self.harvest_reports.insert(chapter.clone(), Ok(report));
        self.outcomes.push(ChildOutcome::Failure {
            chapter: chapter.clone(),
            error:   Arc::new(anyhow!(reason.to_string())),
            summary: Some(TwoPhaseSummary {
                chapter,
                chapter_tag,
                tagged_nodes: 0,
                harvest: harvest_outcome,
                weave: weave_outcome,
                harvest_only: true,
                stop_reason,
            }),
        });
    }

    pub fn new(max_concurrent: usize) -> Self {
        Self::with_launcher_internal(
            max_concurrent,
            Arc::new(Self::launch_orchestrator_child),
            true,
        )
    }

    #[allow(dead_code)]
    pub fn with_test_launcher(max_concurrent: usize, launch_child: ChildLauncher) -> Self {
        Self::with_launcher_internal(max_concurrent, launch_child, false)
    }

    fn with_launcher_internal(
        max_concurrent: usize,
        launch_child: ChildLauncher,
        require_handles: bool,
    ) -> Self {
        Self {
            max_concurrent,
            pending: VecDeque::new(),
            inflight: HashMap::new(),
            weaving_active: std::collections::HashSet::new(),
            outcomes: Vec::new(),
            harvest_reports: HashMap::new(),
            completion: None,
            stop_reason: None,
            harvest_timeout: None,
            weave_timeout: None,
            harvest_timer: None,
            weave_timer: None,
            harvest_timed_out: false,
            weave_timed_out: false,
            cli_opts: None,
            handles: None,
            dedup_worker: None,
            dedup_threshold: 0.0,
            phase_ctx: None,
            launch_child,
            require_handles,
            total_chapters: 0,
            stage: SupervisorStage::Harvesting,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_orchestrator_child(
        chapter: PathBuf,
        cli_opts: crate::app::Cli,
        handles: Option<crate::app::AppHandles>,
        harvest_tx: oneshot::Sender<HarvestReport>,
        harvest_rx: oneshot::Receiver<HarvestReport>,
        weave_gate_tx: oneshot::Sender<()>,
        weave_gate_rx: oneshot::Receiver<()>,
        supervisor: ActorRef<Self>,
    ) -> ChildHandle {
        let Some(handles) = handles else {
            let join = tokio::spawn(async move {
                let _ = supervisor
                    .tell(ChildFinished {
                        chapter,
                        result: Err(anyhow!(
                            "App handles missing; cannot launch two-phase orchestrator"
                        )),
                    })
                    .await;
            });
            let stop = Box::new(|_reason: StopReason| -> futures::future::BoxFuture<'static, ()> {
                Box::pin(async {})
            });
            return ChildHandle {
                join,
                stop,
                harvest_rx: Some(harvest_rx),
                weave_gate: Some(weave_gate_tx),
                orchestrator: None,
            };
        };

        let orchestrator = TwoPhaseOrchestrator::spawn(TwoPhaseOrchestrator::new());
        let orchestrator_for_join = orchestrator.clone();
        let orchestrator_for_stop = orchestrator.clone();
        let join = tokio::spawn(async move {
            let result = orchestrator_for_join
                .ask(StartTwoPhase {
                    chapter: chapter.clone(),
                    cli_opts,
                    handles,
                    harvest_done: Some(harvest_tx),
                    weave_gate: Some(weave_gate_rx),
                })
                .await
                .map_err(|err| anyhow!(err));
            let _ = supervisor.tell(ChildFinished { chapter, result }).await;
        });
        let stop = Box::new(move |reason: StopReason| -> futures::future::BoxFuture<'static, ()> {
            let orchestrator = orchestrator_for_stop.clone();
            Box::pin(async move {
                let _ = orchestrator.tell(StopTwoPhase { reason }).await;
            })
        });

        ChildHandle {
            join,
            stop,
            harvest_rx: Some(harvest_rx),
            weave_gate: Some(weave_gate_tx),
            orchestrator: Some(orchestrator),
        }
    }

    fn spawn_next(&mut self, actor: &ActorRef<Self>) {
        if !matches!(self.stage, SupervisorStage::Harvesting) || self.harvest_timed_out {
            return;
        }
        let Some(cli_opts) = self.cli_opts.clone() else {
            return;
        };
        let handles = self.handles.clone();
        while self
            .inflight
            .keys()
            .filter(|chapter| !self.harvest_reports.contains_key(*chapter))
            .count()
            < self.max_concurrent
        {
            if let Some(chapter) = self.pending.pop_front() {
                let launcher = Arc::clone(&self.launch_child);
                let (harvest_tx, harvest_rx) = oneshot::channel();
                let (weave_tx, weave_rx) = oneshot::channel();
                let handle = launcher(
                    chapter.clone(),
                    cli_opts.clone(),
                    handles.clone(),
                    harvest_tx,
                    harvest_rx,
                    weave_tx,
                    weave_rx,
                    actor.clone(),
                );
                self.register_harvest_listener(actor.clone(), chapter.clone(), handle);
            } else {
                break;
            }
        }
    }

    fn register_harvest_listener(
        &mut self,
        actor: ActorRef<Self>,
        chapter: PathBuf,
        mut handle: ChildHandle,
    ) {
        let chapter_key = chapter.clone();
        if let Some(rx) = handle.take_harvest() {
            let supervisor = actor.clone();
            let chapter_for_msg = chapter.clone();
            tokio::spawn(async move {
                let result = rx
                    .await
                    .map_err(|_| anyhow!("Harvest report channel closed"));
                let _ = supervisor
                    .tell(ChildHarvestFinished {
                        chapter: chapter_for_msg,
                        result,
                    })
                    .await;
            });
        }
        self.inflight.insert(chapter_key, handle);
    }

    async fn finish(&mut self) {
        if !self.inflight.is_empty() {
            return;
        }
        self.cancel_harvest_timer();
        self.cancel_weave_timer();
        if self.stop_reason.is_none() && self.stage != SupervisorStage::Stopped {
            self.run_final_persist().await;
        }
        let summary = self.build_summary();
        let outcome = self.evaluate_summary(summary);
        if let Some(sender) = self.completion.take() {
            let _ = sender.send(outcome);
        }
    }

    fn build_summary(&self) -> TwoPhaseSupervisorSummary {
        TwoPhaseSupervisorSummary {
            outcomes:    self.outcomes.clone(),
            stop_reason: self.stop_reason.clone(),
        }
    }

    fn evaluate_summary(
        &self,
        summary: TwoPhaseSupervisorSummary,
    ) -> Result<TwoPhaseSupervisorSummary> {
        if let Some(reason) = summary.stop_reason.clone() {
            return Err(anyhow!("Two-phase supervisor stopped: {reason}"));
        }
        Ok(summary)
    }

    fn maybe_send_weave_signals(&mut self) {
        if !matches!(self.stage, SupervisorStage::Weaving) {
            return;
        }
        if self.weave_timed_out {
            for handle in self.inflight.values_mut() {
                handle.close_weave();
            }
            return;
        }
        for (chapter, handle) in self.inflight.iter_mut() {
            let ready = self
                .harvest_reports
                .get(chapter)
                .map(|res| res.as_ref().ok().map(|r| r.ready_to_weave).unwrap_or(false))
                .unwrap_or(false);
            if ready {
                if self.weaving_active.len() < self.max_concurrent
                    && !self.weaving_active.contains(chapter)
                {
                    handle.start_weave();
                    self.weaving_active.insert(chapter.clone());
                }
            } else {
                handle.close_weave();
                self.weaving_active.remove(chapter);
            }
        }
    }
}

impl Message<StartTwoPhaseSupervision> for TwoPhaseSupervisor {
    type Reply = DelegatedReply<Result<TwoPhaseSupervisorSummary>>;

    async fn handle(
        &mut self,
        StartTwoPhaseSupervision {
            chapters,
            cli_opts,
            handles,
            dedup_worker,
        }: StartTwoPhaseSupervision,
        ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        if chapters.is_empty() {
            return ctx.spawn(async {
                bail!("--chapters matched no files; provide at least one chapter")
            });
        }
        if self.max_concurrent == 0 {
            return ctx.spawn(async { bail!("max-concurrent-chapters must be at least 1") });
        }
        if self.completion.is_some() {
            return ctx.spawn(async { bail!("Two-phase supervisor already running") });
        }
        if self.require_handles && handles.is_none() {
            return ctx.spawn(async { bail!("App handles are required to start the supervisor") });
        }
        if self.require_handles && dedup_worker.is_none() {
            return ctx.spawn(async { bail!("dedup/persist worker is required to start") });
        }

        self.pending = VecDeque::from(chapters.clone());
        self.inflight.clear();
        self.outcomes.clear();
        self.harvest_reports.clear();
        self.stop_reason = None;
        self.cli_opts = Some(cli_opts.clone());
        self.handles = handles;
        self.dedup_worker = dedup_worker;
        self.dedup_threshold = self
            .cli_opts
            .as_ref()
            .map(|c| c.dedup_auto_merge_threshold)
            .unwrap_or(0.0);
        self.phase_ctx = match self.build_phase_ctx().await {
            Some(Ok(ctx)) => Some(ctx),
            Some(Err(err)) => return ctx.spawn(async { Err(err) }),
            None => None,
        };
        self.total_chapters = chapters.len();
        self.stage = SupervisorStage::Harvesting;
        self.harvest_timeout = hours_to_duration(cli_opts.harvest_timeout_hours);
        self.weave_timeout = hours_to_duration(cli_opts.weave_timeout_hours);
        self.harvest_timed_out = false;
        self.weave_timed_out = false;
        self.cancel_harvest_timer();
        self.cancel_weave_timer();

        let (tx, rx) = oneshot::channel();
        self.completion = Some(tx);

        let actor = ctx.actor_ref().clone();
        self.spawn_harvest_timer(actor.clone());
        self.spawn_next(&actor);
        if self.pending.is_empty() && self.inflight.is_empty() {
            self.finish().await;
        }

        ctx.spawn(async move {
            rx.await
                .unwrap_or_else(|_| Err(anyhow!("Two-phase supervisor stopped unexpectedly")))
        })
    }
}

impl Message<StopAll> for TwoPhaseSupervisor {
    type Reply = Result<()>;

    async fn handle(
        &mut self,
        StopAll { reason }: StopAll,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        if self.stop_reason.is_none() {
            self.stop_reason = Some(reason.clone());
        }
        self.cancel_harvest_timer();
        self.cancel_weave_timer();
        self.pending.clear();
        self.stage = SupervisorStage::Stopped;
        let mut inflight = std::mem::take(&mut self.inflight);
        for (_, mut handle) in inflight.drain() {
            handle.stop(reason.clone()).await;
            if !handle.wait_with_timeout(Duration::from_secs(1)).await {
                handle.abort();
            }
        }
        self.finish().await;
        Ok(())
    }
}

pub struct ChildHarvestFinished {
    pub chapter: PathBuf,
    pub result:  Result<HarvestReport>,
}

pub struct ChildFinished {
    pub chapter: PathBuf,
    pub result:  Result<TwoPhaseSummary>,
}

impl Message<HarvestPhaseTimedOut> for TwoPhaseSupervisor {
    type Reply = ();

    async fn handle(
        &mut self,
        _msg: HarvestPhaseTimedOut,
        ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        if !matches!(self.stage, SupervisorStage::Harvesting) {
            return;
        }
        warn!("Global harvest timeout reached; stopping further harvest tasks");
        self.harvest_timed_out = true;
        self.cancel_harvest_timer();
        let reason = "harvest phase timed out before chapter started";
        while let Some(chapter) = self.pending.pop_front() {
            self.synthesize_skipped_chapter(chapter, reason);
        }
        for handle in self.inflight.values() {
            handle.harvest_timeout().await;
        }
        if self.harvest_reports.len() == self.total_chapters {
            self.stage = SupervisorStage::Deduplicating;
            self.run_dedup_barrier().await;
            self.stage = SupervisorStage::Weaving;
            let actor = ctx.actor_ref().clone();
            self.spawn_weave_timer(actor);
            self.maybe_send_weave_signals();
        }
        if self.pending.is_empty() && self.inflight.is_empty() {
            self.stage = SupervisorStage::Completed;
            self.finish().await;
        }
    }
}

impl Message<WeavePhaseTimedOut> for TwoPhaseSupervisor {
    type Reply = ();

    async fn handle(
        &mut self,
        _msg: WeavePhaseTimedOut,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        if !matches!(self.stage, SupervisorStage::Weaving) {
            return;
        }
        warn!("Global weave timeout reached; stopping further weave tasks");
        self.weave_timed_out = true;
        self.cancel_weave_timer();
        for handle in self.inflight.values() {
            handle.weave_timeout().await;
        }
        // Also cancel any remaining harvest-time tokens held by children.
        // This forces nested delegates to stop promptly.
        for handle in self.inflight.values() {
            handle.harvest_timeout().await;
        }
        self.weaving_active.clear();
    }
}

impl Message<ChildHarvestFinished> for TwoPhaseSupervisor {
    type Reply = ();

    async fn handle(
        &mut self,
        ChildHarvestFinished { chapter, result }: ChildHarvestFinished,
        ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        self.harvest_reports.insert(chapter.clone(), result);
        if matches!(self.stage, SupervisorStage::Harvesting)
            && self.harvest_reports.len() == self.total_chapters
        {
            self.stage = SupervisorStage::Deduplicating;
            self.run_dedup_barrier().await;
            self.stage = SupervisorStage::Weaving;
            let actor = ctx.actor_ref().clone();
            self.spawn_weave_timer(actor);
            self.maybe_send_weave_signals();
        }
        if self.stop_reason.is_none() {
            let actor = ctx.actor_ref().clone();
            self.spawn_next(&actor);
        }
    }
}

impl Message<ChildFinished> for TwoPhaseSupervisor {
    type Reply = ();

    async fn handle(
        &mut self,
        ChildFinished { chapter, result }: ChildFinished,
        ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        if let Some(handle) = self.inflight.remove(&chapter) {
            handle.wait().await;
        }
        self.weaving_active.remove(&chapter);
        if !self.harvest_reports.contains_key(&chapter) {
            self.harvest_reports
                .insert(chapter.clone(), Err(anyhow!("chapter finished without harvest report")));
        }
        match result {
            Ok(summary) => {
                if summary.is_failure() {
                    self.outcomes.push(ChildOutcome::Failure {
                        chapter: chapter.clone(),
                        error:   Arc::new(anyhow!(
                            "chapter reported failures (harvest_failures={}, weave_failures={}, \
                             harvest_only={}, stop={:?})",
                            summary.harvest_failures(),
                            summary.weave_failures(),
                            summary.harvest_only,
                            summary.stop_reason
                        )),
                        summary: Some(summary),
                    });
                } else {
                    self.outcomes.push(ChildOutcome::Success {
                        chapter: chapter.clone(),
                        summary,
                    });
                }
            }
            Err(err) => {
                self.outcomes.push(ChildOutcome::Failure {
                    chapter: chapter.clone(),
                    error:   Arc::new(err),
                    summary: None,
                });
            }
        }

        if matches!(self.stage, SupervisorStage::Harvesting)
            && self.harvest_reports.len() == self.total_chapters
        {
            self.stage = SupervisorStage::Deduplicating;
            self.run_dedup_barrier().await;
            self.stage = SupervisorStage::Weaving;
            let actor = ctx.actor_ref().clone();
            self.spawn_weave_timer(actor);
            self.maybe_send_weave_signals();
        }
        if matches!(self.stage, SupervisorStage::Weaving) {
            self.maybe_send_weave_signals();
        }
        if self.pending.is_empty() && self.inflight.is_empty() {
            self.stage = SupervisorStage::Completed;
            self.finish().await;
        }
    }
}

pub mod test_support {
    pub use super::{
        ChildFinished, ChildHandle, ChildHarvestFinished, ChildLauncher, ChildOutcome,
        TwoPhaseSupervisor, TwoPhaseSupervisor as SupervisorForTests, TwoPhaseSupervisorSummary,
    };

    impl TwoPhaseSupervisor {
        #[allow(dead_code)]
        pub fn with_launcher_for_tests(max_concurrent: usize, launch_child: ChildLauncher) -> Self {
            Self::with_test_launcher(max_concurrent, launch_child)
        }
    }
}

fn hours_to_duration(hours: Option<f64>) -> Option<Duration> {
    match hours {
        Some(value) if value.is_finite() && value > 0.0 => {
            let seconds = value * 3_600.0;
            if seconds > u64::MAX as f64 {
                None
            } else {
                Some(Duration::from_secs_f64(seconds))
            }
        }
        _ => None,
    }
}
