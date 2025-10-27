use std::time::{Duration, Instant};

use anyhow::Result;
use kameo::prelude::*;
use tokio::task::JoinSet;
use tracing::info;
use weaver::file_reader::{FileReader, TestChat};

const ROUNDS: usize = 3;
const ACTOR_CONFIGS: &[usize] = &[32, 48, 64, 96, 128];

#[derive(Debug)]
struct JobResult {
    job_id:      usize,
    actor_index: usize,
    reply:       String,
}

#[derive(Debug)]
struct RunSummary {
    actor_count: usize,
    rounds:      usize,
    total_jobs:  usize,
    duration:    Duration,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::new("%Y-%m-%d %H:%M:%S%.3f".into()))
        .init();

    info!("Starting load test suite");

    let mut summaries = Vec::with_capacity(ACTOR_CONFIGS.len());

    for &actor_count in ACTOR_CONFIGS {
        let summary = run_scenario(actor_count).await?;
        info!(
            actor_count = summary.actor_count,
            rounds = summary.rounds,
            total_jobs = summary.total_jobs,
            duration_secs = summary.duration.as_secs_f64(),
            "Scenario complete"
        );
        summaries.push(summary);
    }

    println!("## FileReader Load Test Summary\n");
    println!("| Actors | Rounds | Total Jobs | Duration (s) |");
    println!("| ------ | ------ | ---------- | ------------ |");
    for summary in summaries {
        println!(
            "| {} | {} | {} | {:.2} |",
            summary.actor_count,
            summary.rounds,
            summary.total_jobs,
            summary.duration.as_secs_f64()
        );
    }

    Ok(())
}

async fn run_scenario(actor_count: usize) -> Result<RunSummary> {
    info!(actor_count, rounds = ROUNDS, "Spawning FileReader actors for scenario");

    let mut actors = Vec::with_capacity(actor_count);
    for idx in 0..actor_count {
        let actor = FileReader::spawn(FileReader::from_env()?);
        info!(actor_index = idx, actor_count, "Spawned FileReader actor");
        actors.push(actor);
    }

    let total_jobs = actor_count * ROUNDS;
    info!(actor_count, rounds = ROUNDS, total_jobs, "Starting queued load test");

    let mut joinset: JoinSet<Result<JobResult>> = JoinSet::new();
    let mut next_job_id = 0usize;
    let mut inflight = 0usize;

    let start = Instant::now();

    while inflight < actor_count && next_job_id < total_jobs {
        spawn_job(&mut joinset, &actors, next_job_id, actor_count)?;
        next_job_id += 1;
        inflight += 1;
    }

    while inflight > 0 {
        if let Some(res) = joinset.join_next().await {
            let job = res??;
            inflight -= 1;

            let round = job.job_id / actor_count;
            info!(
                scenario_actor_count = actor_count,
                job_id = job.job_id,
                round,
                actor_index = job.actor_index,
                chars = job.reply.chars().count(),
                "Completed chat request"
            );

            if next_job_id < total_jobs {
                spawn_job(&mut joinset, &actors, next_job_id, actor_count)?;
                next_job_id += 1;
                inflight += 1;
            }
        } else {
            break;
        }
    }

    let duration = start.elapsed();

    Ok(RunSummary {
        actor_count,
        rounds: ROUNDS,
        total_jobs,
        duration,
    })
}

fn spawn_job(
    joinset: &mut JoinSet<Result<JobResult>>,
    actors: &[ActorRef<FileReader>],
    job_id: usize,
    jobs_per_round: usize,
) -> Result<()> {
    let actor_index = job_id % actors.len();
    let actor_ref = actors[actor_index].clone();
    let prompt = format!(
        "Job {job_id} (round {}) routed to actor {actor_index}. Provide a detailed readiness \
         brief for Weaver agents.",
        job_id / jobs_per_round
    );

    joinset.spawn(async move {
        let reply = actor_ref.ask(TestChat { prompt }).await?;
        Ok(JobResult {
            job_id,
            actor_index,
            reply,
        })
    });

    Ok(())
}
