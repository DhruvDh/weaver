use std::{path::PathBuf, sync::Arc};

use kameo::actor::Spawn;
use tokio::{
    sync::Mutex,
    time::{Duration, sleep, timeout},
};
use tokio_util::sync::CancellationToken;
use weaver::app::{
    Cli,
    two_phase::{
        HarvestNiche, HarvestReport, NicheOutcome, NicheStatus, PhaseOutcome,
        StartTwoPhaseSupervision, StopAll, StopReason, TwoPhaseSummary, TwoPhaseSupervisorSummary,
        WeaveNiche,
        test_support::{ChildFinished, ChildHandle, ChildLauncher, TwoPhaseSupervisor},
    },
};

fn test_cli() -> Cli {
    Cli {
        rerun_file:                  None,
        graph_snapshot_path:         PathBuf::from("graph_snapshot.json"),
        graph_autosave_secs:         300,
        graph_course_commit:         None,
        graph_strict_quality:        false,
        graph_prune_requires_s:      None,
        skip_demo:                   true,
        graph_validation_timeout_ms: 2_000,
        dedup_interval_secs:         3_600,
        dedup_auto_merge_threshold:  0.95,
        skip_dedup_on_insert:        true,
        interactive:                 false,
        interactive_writable:        false,
        analyst:                     false,
        harvest_timeout_hours:       None,
        weave_timeout_hours:         None,
        max_concurrent_chapters:     2,
        chapters_pattern:            None,
        chapters_dir:                None,
        workspace:                   PathBuf::from("."),
    }
}

fn completed_harvest_outcome() -> PhaseOutcome<HarvestNiche> {
    PhaseOutcome::new(
        vec![NicheOutcome {
            niche:      HarvestNiche::FactualConceptual,
            attempts:   1,
            status:     NicheStatus::Done,
            last_error: None,
        }],
        false,
    )
}

fn completed_weave_outcome() -> PhaseOutcome<WeaveNiche> {
    PhaseOutcome::new(
        vec![NicheOutcome {
            niche:      WeaveNiche::Requires,
            attempts:   1,
            status:     NicheStatus::Done,
            last_error: None,
        }],
        false,
    )
}

#[tokio::test]
async fn supervisor_respects_concurrency_cap() {
    let active = Arc::new(Mutex::new(0usize));
    let peak = Arc::new(Mutex::new(0usize));
    let launcher: ChildLauncher = {
        let active = Arc::clone(&active);
        let peak = Arc::clone(&peak);
        Arc::new(
            move |chapter: PathBuf,
                  _cli_opts: Cli,
                  _handles: Option<weaver::app::AppHandles>,
                  harvest_tx: tokio::sync::oneshot::Sender<HarvestReport>,
                  harvest_rx: tokio::sync::oneshot::Receiver<HarvestReport>,
                  weave_tx: tokio::sync::oneshot::Sender<()>,
                  weave_rx: tokio::sync::oneshot::Receiver<()>,
                  supervisor: kameo::actor::ActorRef<TwoPhaseSupervisor>| {
                let active = Arc::clone(&active);
                let peak = Arc::clone(&peak);
                let chapter_for_msg = chapter.clone();
                let join = tokio::spawn(async move {
                    {
                        let mut count = active.lock().await;
                        *count = count.saturating_add(1);
                        let mut max_seen = peak.lock().await;
                        *max_seen = (*max_seen).max(*count);
                    }
                    sleep(Duration::from_millis(25)).await;
                    {
                        let mut count = active.lock().await;
                        *count = count.saturating_sub(1);
                    }
                    let harvest = HarvestReport {
                        chapter:        chapter_for_msg.clone(),
                        chapter_tag:    "source:test".into(),
                        tagged_nodes:   1,
                        harvest:        completed_harvest_outcome(),
                        stop_reason:    None,
                        ready_to_weave: true,
                    };
                    let _ = harvest_tx.send(harvest);
                    let _ = weave_rx.await;
                    let summary = TwoPhaseSummary {
                        chapter:      chapter_for_msg.clone(),
                        chapter_tag:  "source:test".into(),
                        tagged_nodes: 1,
                        harvest:      completed_harvest_outcome(),
                        weave:        completed_weave_outcome(),
                        harvest_only: false,
                        stop_reason:  None,
                    };
                    let _ = supervisor
                        .tell(ChildFinished {
                            chapter: chapter_for_msg,
                            result:  Ok(summary),
                        })
                        .await;
                });
                let stop =
                    Box::new(|_reason: StopReason| -> futures::future::BoxFuture<'static, ()> {
                        Box::pin(async {})
                    });
                ChildHandle {
                    join,
                    stop,
                    harvest_rx: Some(harvest_rx),
                    weave_gate: Some(weave_tx),
                    orchestrator: None,
                }
            },
        )
    };

    let supervisor =
        TwoPhaseSupervisor::spawn(TwoPhaseSupervisor::with_launcher_for_tests(2, launcher));
    let chapters = vec![
        PathBuf::from("source/a/toctree.ptx"),
        PathBuf::from("source/b/toctree.ptx"),
        PathBuf::from("source/c/toctree.ptx"),
    ];
    let summary_result: Result<TwoPhaseSupervisorSummary, _> = supervisor
        .ask(StartTwoPhaseSupervision {
            chapters,
            cli_opts: test_cli(),
            handles: None,
            dedup_worker: None,
        })
        .await;
    let summary = summary_result.expect("supervisor run");
    assert_eq!(summary.successes(), 3);
    assert_eq!(summary.failures(), 0);
    let peak_value = *peak.lock().await;
    assert_eq!(peak_value, 2);
}

#[tokio::test]
async fn supervisor_stop_all_cancels_children() {
    let launcher: ChildLauncher = Arc::new(
        move |chapter: PathBuf,
              _cli_opts: Cli,
              _handles: Option<weaver::app::AppHandles>,
              harvest_tx: tokio::sync::oneshot::Sender<HarvestReport>,
              harvest_rx: tokio::sync::oneshot::Receiver<HarvestReport>,
              weave_tx: tokio::sync::oneshot::Sender<()>,
              _weave_rx: tokio::sync::oneshot::Receiver<()>,
              supervisor: kameo::actor::ActorRef<TwoPhaseSupervisor>| {
            let token = CancellationToken::new();
            let child_token = token.clone();
            let join = tokio::spawn(async move {
                let harvest = HarvestReport {
                    chapter:        chapter.clone(),
                    chapter_tag:    "source:test".into(),
                    tagged_nodes:   1,
                    harvest:        completed_harvest_outcome(),
                    stop_reason:    None,
                    ready_to_weave: true,
                };
                let _ = harvest_tx.send(harvest);
                tokio::select! {
                    _ = sleep(Duration::from_secs(5)) => {
                        let summary = TwoPhaseSummary {
                            chapter:      chapter.clone(),
                            chapter_tag:  "source:test".into(),
                            tagged_nodes: 1,
                            harvest:      completed_harvest_outcome(),
                            weave:        completed_weave_outcome(),
                            harvest_only: false,
                            stop_reason:  None,
                        };
                        let _ = supervisor.tell(ChildFinished { chapter, result: Ok(summary) }).await;
                    }
                    _ = child_token.cancelled() => {
                        let _ = supervisor
                            .tell(ChildFinished {
                                chapter,
                                result: Err(anyhow::anyhow!("stopped")),
                            })
                            .await;
                    }
                }
            });
            let stop =
                Box::new(move |_reason: StopReason| -> futures::future::BoxFuture<'static, ()> {
                    let token = token.clone();
                    Box::pin(async move {
                        token.cancel();
                    })
                });
            ChildHandle {
                join,
                stop,
                harvest_rx: Some(harvest_rx),
                weave_gate: Some(weave_tx),
                orchestrator: None,
            }
        },
    );

    let supervisor =
        TwoPhaseSupervisor::spawn(TwoPhaseSupervisor::with_launcher_for_tests(2, launcher));
    let chapters = vec![
        PathBuf::from("source/a/toctree.ptx"),
        PathBuf::from("source/b/toctree.ptx"),
    ];
    let run = supervisor.ask(StartTwoPhaseSupervision {
        chapters,
        cli_opts: test_cli(),
        handles: None,
        dedup_worker: None,
    });
    let stopper = supervisor.clone();
    tokio::spawn(async move {
        sleep(Duration::from_millis(50)).await;
        let _ = stopper
            .tell(StopAll {
                reason: StopReason::Explicit("test stop".to_string()),
            })
            .await;
    });
    let result = timeout(Duration::from_secs(2), run)
        .await
        .expect("supervisor to finish after stop");
    if result.is_ok() {
        panic!("expected cancellation error");
    }
}
