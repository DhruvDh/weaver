mod orchestrator;
mod prompts;
mod supervisor;
mod types;
mod worker;

use std::path::PathBuf;

use anyhow::{Result, bail};
use futures::future::BoxFuture;
use kameo::actor::Spawn;
#[allow(unused_imports)]
pub use orchestrator::{StartTwoPhase, StopTwoPhase, TwoPhaseOrchestrator};
#[allow(unused_imports)]
pub use supervisor::{
    ChildOutcome, StartTwoPhaseSupervision, StopAll, TwoPhaseSupervisor, TwoPhaseSupervisorSummary,
};
use tracing::{info, warn};
#[allow(unused_imports)]
pub use types::{
    ALL_HARVEST_NICHES, ALL_WEAVE_NICHES, HarvestNiche, HarvestReport, NicheOutcome, NicheStatus,
    PhaseKind, PhaseOutcome, PhaseStage, PhaseStatus, PhaseStatusRequest, StopReason,
    TwoPhaseSummary, WeaveNiche,
};
#[allow(unused_imports)]
pub use worker::DedupPersistWorker;

use crate::app::{AppHandles, AppHook, Cli, RuntimeOptions, run_app};

pub async fn run_two_phase_construction(cli: Cli, chapters: Vec<PathBuf>) -> Result<()> {
    if chapters.is_empty() {
        bail!("--chapters matched no files; provide at least one chapter");
    }
    if cli.max_concurrent_chapters == 0 {
        bail!("--max-concurrent-chapters must be at least 1");
    }

    let mut modified_cli = cli.clone();
    modified_cli.skip_demo = true;
    modified_cli.skip_dedup_on_insert = true;

    let supervisor_ref =
        TwoPhaseSupervisor::spawn(TwoPhaseSupervisor::new(modified_cli.max_concurrent_chapters));
    let dedup_worker = DedupPersistWorker::spawn(DedupPersistWorker);
    let hook_cli = modified_cli.clone();
    let hook_chapters = chapters.clone();
    let hook: AppHook =
        std::sync::Arc::new(move |handles: AppHandles| -> BoxFuture<'static, Result<()>> {
            let supervisor = supervisor_ref.clone();
            let chapters = hook_chapters.clone();
            let cli_for_start = hook_cli.clone();
            let dedup_worker = dedup_worker.clone();
            Box::pin(async move {
                let summary: TwoPhaseSupervisorSummary = supervisor
                    .ask(StartTwoPhaseSupervision {
                        chapters,
                        cli_opts: cli_for_start,
                        handles: Some(handles),
                        dedup_worker: Some(dedup_worker.clone()),
                    })
                    .await?;
                info!(
                    completed = summary.successes(),
                    failed = summary.failures(),
                    "Two-phase supervisor completed"
                );
                for outcome in summary.outcomes.iter() {
                    match outcome {
                        supervisor::ChildOutcome::Failure {
                            chapter,
                            error,
                            summary,
                        } => {
                            warn!(
                                chapter = %chapter.display(),
                                error = %error,
                                harvest_failures = summary.as_ref().map(|s| s.harvest_failures()),
                                weave_failures = summary.as_ref().map(|s| s.weave_failures()),
                                "Chapter run failed"
                            );
                        }
                        supervisor::ChildOutcome::Success { chapter, summary } => {
                            info!(
                                chapter = %chapter.display(),
                                tagged_nodes = summary.tagged_nodes,
                                harvest_failures = summary.harvest_failures(),
                                weave_failures = summary.weave_failures(),
                                harvest_only = summary.harvest_only,
                                stop_reason = ?summary.stop_reason,
                                "Chapter run succeeded"
                            );
                        }
                    }
                }
                Ok(())
            })
        });
    let runtime = RuntimeOptions {
        on_started: Some(hook),
        ..RuntimeOptions::default()
    };

    run_app(modified_cli, runtime).await
}

#[allow(unused_imports)]
pub mod test_support {
    pub use super::supervisor::test_support::*;
}
