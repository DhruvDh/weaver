use anyhow::{Result, bail};
use kameo::prelude::*;
use tokio_util::sync::CancellationToken;
use tracing::info;

use super::types::PhaseCtx;
use crate::{
    agents::deduplication::RunDeduplication,
    graph::manager::{AuditInvariants, PersistSnapshot},
};

#[derive(Actor, Default, Clone)]
pub struct DedupPersistWorker;

#[derive(Clone)]
pub struct RunDedupPersist {
    pub ctx:          PhaseCtx,
    pub threshold:    f64,
    pub cancellation: Option<CancellationToken>,
}

#[derive(Clone)]
pub struct PersistOnlyRequest {
    pub ctx:          PhaseCtx,
    pub cancellation: Option<CancellationToken>,
}

impl Message<RunDedupPersist> for DedupPersistWorker {
    type Reply = Result<()>;

    async fn handle(
        &mut self,
        RunDedupPersist {
            ctx,
            threshold,
            cancellation,
        }: RunDedupPersist,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        run_dedup_and_persist(&ctx, threshold, &cancellation).await
    }
}

impl Message<PersistOnlyRequest> for DedupPersistWorker {
    type Reply = Result<()>;

    async fn handle(
        &mut self,
        PersistOnlyRequest { ctx, cancellation }: PersistOnlyRequest,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        run_audit_and_persist(&ctx, &cancellation).await
    }
}

fn is_cancelled(token: &Option<CancellationToken>) -> bool {
    token.as_ref().map(|t| t.is_cancelled()).unwrap_or(false)
}

async fn run_dedup_and_persist(
    ctx: &PhaseCtx,
    threshold: f64,
    cancellation: &Option<CancellationToken>,
) -> Result<()> {
    if is_cancelled(cancellation) {
        bail!("Two-phase run cancelled before deduplication");
    }

    info!("Deduplication barrier starting");
    let dedup_report = ctx
        .dedup_agent
        .ask(RunDeduplication {
            auto_merge_threshold: threshold,
            dry_run:              false,
        })
        .await?;
    if is_cancelled(cancellation) {
        bail!("Two-phase run cancelled during deduplication");
    }
    info!(
        clusters = dedup_report.clusters_analyzed,
        auto_merged = dedup_report.auto_merged.len(),
        pending_review = dedup_report.pending_review.len(),
        "Deduplication complete"
    );

    run_audit_and_persist(ctx, cancellation).await
}

async fn run_audit_and_persist(
    ctx: &PhaseCtx,
    cancellation: &Option<CancellationToken>,
) -> Result<()> {
    if is_cancelled(cancellation) {
        bail!("Two-phase run cancelled before audit");
    }
    ctx.graph.ask(AuditInvariants).await?;
    if is_cancelled(cancellation) {
        bail!("Two-phase run cancelled during audit");
    }
    ctx.graph.ask(PersistSnapshot).await?;
    if is_cancelled(cancellation) {
        bail!("Two-phase run cancelled during persist");
    }
    Ok(())
}
