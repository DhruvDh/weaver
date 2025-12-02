use std::{
    collections::HashMap,
    convert::Infallible,
    path::PathBuf,
    time::{Instant, SystemTime},
};

use kameo::prelude::*;
use rerun::{
    GraphEdges, GraphNodes, RecordingStream, RecordingStreamBuilder, TimeCell,
    archetypes::{Arrows2D, Scalars, TextLog},
    sink::{FileSink, GrpcSink, LogSink},
};
use tokio::task;
use tracing::debug;

#[derive(Clone, Debug)]
pub enum RerunTarget {
    Grpc { name: String },
    File { name: String, path: PathBuf },
}

/// Send scalar time-series samples to one or more Rerun recording streams.
pub struct RerunSink {
    targets:      Vec<RerunTarget>,
    recs:         Vec<RecordingStream>,
    ready:        bool,
    last_attempt: Option<Instant>,
}

impl RerunSink {
    pub fn new(targets: Vec<RerunTarget>) -> Self {
        Self {
            targets,
            recs: Vec::new(),
            ready: false,
            last_attempt: None,
        }
    }

    fn ensure_streams(&mut self) {
        if self.ready && !self.recs.is_empty() {
            return;
        }
        let now = Instant::now();
        if let Some(prev) = self.last_attempt
            && now.saturating_duration_since(prev) < std::time::Duration::from_secs(5)
        {
            return;
        }
        self.last_attempt = Some(now);

        let mut sinks_by_name: HashMap<String, Vec<Box<dyn LogSink>>> = HashMap::new();
        for target in &self.targets {
            match target {
                RerunTarget::Grpc { name } => sinks_by_name
                    .entry(name.clone())
                    .or_default()
                    .push(Box::new(GrpcSink::default())),
                RerunTarget::File { name, path } => match FileSink::new(path) {
                    Ok(sink) => sinks_by_name
                        .entry(name.clone())
                        .or_default()
                        .push(Box::new(sink)),
                    Err(err) => debug!(error = %err, path = %path.display(), "rerun save failed"),
                },
            }
        }

        let mut recs = Vec::new();
        for (name, sinks) in sinks_by_name {
            if sinks.is_empty() {
                continue;
            }
            match RecordingStreamBuilder::new(name.clone()).set_sinks(sinks) {
                Ok(rec) => recs.push(rec),
                Err(err) => debug!(error = %err, name = %name, "rerun init failed"),
            }
        }

        if !recs.is_empty() {
            self.recs = recs;
            self.ready = true;
        }
    }
}

#[derive(Clone)]
pub struct LogScalar {
    pub path:    String,
    pub value:   f64,
    pub time_ns: Option<i64>,
}

#[derive(Clone)]
pub struct LogText {
    pub path:    String,
    pub value:   String,
    pub time_ns: Option<i64>,
}

#[derive(Clone)]
pub struct LogGraphFrame {
    pub entity_path:       String,
    pub edge_overlay_path: String,
    pub nodes:             GraphNodes,
    pub edges:             GraphEdges,
    pub arrows:            Option<Arrows2D>,
    pub time_ns:           Option<i64>,
}

impl Actor for RerunSink {
    type Args = Vec<RerunTarget>;
    type Error = Infallible;

    async fn on_start(
        targets: Vec<RerunTarget>,
        _actor_ref: ActorRef<Self>,
    ) -> Result<Self, Self::Error> {
        Ok(Self::new(targets))
    }
}

impl Message<LogScalar> for RerunSink {
    type Reply = ();

    async fn handle(
        &mut self,
        msg: LogScalar,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        self.ensure_streams();
        if self.recs.is_empty() {
            return;
        }
        let time_cell = msg
            .time_ns
            .map(TimeCell::from_timestamp_nanos_since_epoch)
            .or_else(now_timecell)
            .unwrap_or_else(|| TimeCell::from_timestamp_nanos_since_epoch(0));
        let recs = self.recs.clone();
        let path = msg.path;
        let value = msg.value;
        task::spawn(async move {
            for rec in recs {
                rec.set_time("time", time_cell);
                if let Err(err) = rec.log(path.clone(), &Scalars::new([value])) {
                    debug!(error = ?err, "rerun log failed");
                }
            }
        });
    }
}

fn now_timecell() -> Option<TimeCell> {
    let dur = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?;
    let nanos: i128 = dur.as_nanos() as i128;
    i64::try_from(nanos)
        .ok()
        .map(TimeCell::from_timestamp_nanos_since_epoch)
}

impl Message<LogText> for RerunSink {
    type Reply = ();

    async fn handle(&mut self, msg: LogText, _ctx: &mut Context<Self, Self::Reply>) -> Self::Reply {
        self.ensure_streams();
        if self.recs.is_empty() {
            return;
        }
        let time_cell = msg
            .time_ns
            .map(TimeCell::from_timestamp_nanos_since_epoch)
            .or_else(now_timecell)
            .unwrap_or_else(|| TimeCell::from_timestamp_nanos_since_epoch(0));
        let recs = self.recs.clone();
        let path = msg.path;
        let value = msg.value;
        task::spawn(async move {
            for rec in recs {
                rec.set_time("time", time_cell);
                if let Err(err) = rec.log(path.clone(), &TextLog::new(value.clone())) {
                    debug!(error = ?err, "rerun log failed");
                }
            }
        });
    }
}

impl Message<LogGraphFrame> for RerunSink {
    type Reply = ();

    async fn handle(
        &mut self,
        msg: LogGraphFrame,
        _ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        self.ensure_streams();
        if self.recs.is_empty() {
            return;
        }
        let time_cell = msg
            .time_ns
            .map(TimeCell::from_timestamp_nanos_since_epoch)
            .or_else(now_timecell)
            .unwrap_or_else(|| TimeCell::from_timestamp_nanos_since_epoch(0));
        let recs = self.recs.clone();
        let LogGraphFrame {
            entity_path,
            edge_overlay_path,
            nodes,
            edges,
            arrows,
            ..
        } = msg;

        task::spawn(async move {
            for rec in recs {
                rec.set_time("time", time_cell);
                if let Err(err) =
                    rec.log(entity_path.clone(), &[&nodes as &dyn rerun::AsComponents, &edges])
                {
                    debug!(error = ?err, "rerun log graph frame failed");
                }
                if let Some(arrows) = arrows.as_ref()
                    && let Err(err) = rec.log(edge_overlay_path.clone(), arrows)
                {
                    debug!(error = ?err, "rerun log edge overlay failed");
                }
            }
        });
    }
}
