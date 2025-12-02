use std::{
    collections::HashMap,
    convert::Infallible,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use kameo::prelude::*;
use rerun::{Color, GraphEdges, GraphNodes, archetypes::Arrows2D, components::GraphType};
use tokio::{task, time::sleep};
use tracing::debug;

use crate::{
    constants::viz::NODE_RADIUS_SCALE,
    graph::{
        CurriculumGraph, EdgeKind, NodeKind,
        audit::{MutationEvent, MutationEventSink},
        commands::{GetGraphVersion, GetGraphWithVersion},
        manager::GraphManager,
        model::NodePayload,
    },
    rerun_sink::{LogGraphFrame, RerunSink},
    schema::types::KnowledgeType,
};

#[derive(Clone)]
pub struct GraphVizSink {
    viz: ActorRef<GraphVisualizer>,
}

impl GraphVizSink {
    pub fn new(viz: ActorRef<GraphVisualizer>) -> Self {
        Self { viz }
    }
}

impl MutationEventSink for GraphVizSink {
    fn record(&self, event: &MutationEvent) {
        let viz = self.viz.clone();
        let evt = event.clone();
        task::spawn(async move {
            let _ = viz
                .tell(GraphVizEvent {
                    graph_version: evt.graph_version,
                    timestamp_ns:  evt.timestamp_ms.saturating_mul(1_000_000).try_into().ok(),
                })
                .await;
        });
    }
}

#[derive(Clone)]
pub struct GraphVizEvent {
    pub graph_version: u64,
    pub timestamp_ns:  Option<i64>,
}

#[derive(Clone)]
pub struct PrimeRender;

#[derive(Clone)]
struct RenderNow;

#[derive(Clone, Copy)]
struct Palette;

impl Palette {
    fn node_color(kind: &NodeKind) -> Color {
        match kind {
            NodeKind::Knowledge(k) => match k.knowledge_type {
                KnowledgeType::Factual => Color::from_rgb(55, 126, 184),
                KnowledgeType::Conceptual => Color::from_rgb(77, 175, 74),
                KnowledgeType::Procedural => Color::from_rgb(255, 127, 0),
                KnowledgeType::Metacognitive => Color::from_rgb(152, 78, 163),
                KnowledgeType::LearningOutcome => Color::from_rgb(228, 26, 28),
                KnowledgeType::AssessmentItem => Color::from_rgb(166, 86, 40),
            },
            NodeKind::TeachingStep(_) => Color::from_rgb(102, 194, 165),
        }
    }
}

fn node_radius(kind: &NodeKind) -> f32 {
    let base = match kind {
        NodeKind::Knowledge(k) => match k.knowledge_type {
            KnowledgeType::LearningOutcome => 9.5,
            KnowledgeType::AssessmentItem => 9.0,
            KnowledgeType::Procedural => 7.0,
            KnowledgeType::Conceptual => 7.0,
            KnowledgeType::Metacognitive => 7.0,
            KnowledgeType::Factual => 6.0,
        },
        NodeKind::TeachingStep(_) => 5.0,
    };

    base * NODE_RADIUS_SCALE
}

#[derive(Default, Clone, Copy)]
struct NodeStats {
    req_in:      usize,
    req_out:     usize,
    sup_in:      usize,
    sup_out:     usize,
    assess_in:   usize,
    assess_out:  usize,
    anchors_in:  usize,
    anchors_out: usize,
    pre_in:      usize,
    pre_out:     usize,
}

fn two_stats_mut(stats: &mut [NodeStats], a: usize, b: usize) -> (&mut NodeStats, &mut NodeStats) {
    let (low, high, swap) = if a < b { (a, b, false) } else { (b, a, true) };
    let (left, right) = stats.split_at_mut(high);
    let first = &mut left[low];
    let second = &mut right[0];
    if swap {
        (second, first)
    } else {
        (first, second)
    }
}

fn summarize_node(node: &NodePayload, stats: &NodeStats) -> String {
    let mut parts = Vec::new();
    match &node.kind {
        NodeKind::Knowledge(k) => {
            parts.push(format!("{:?}", k.knowledge_type).to_lowercase());
            parts.push(k.title.clone());
            if !k.statement.trim().is_empty() {
                parts.push(truncate(k.statement.as_str(), 200));
            }
            if !k.rubric_criteria.is_empty() {
                parts.push(format!("rubric: {}", truncate(&k.rubric_criteria.join("; "), 140)));
            }
            parts.push(format!("confidence: {:.0}%", k.confidence * 100.0));
        }
        NodeKind::TeachingStep(t) => {
            parts.push("teaching_step".into());
            parts.push(t.title.clone());
            parts.push(format!("purpose: {:?}", t.purpose).to_lowercase());
            if !t.statement.trim().is_empty() {
                parts.push(truncate(t.statement.as_str(), 200));
            }
        }
    }
    parts.push(format!(
        "deg req in/out {} / {} · sup {} / {} · assess {} / {} · anchor {} / {} · pre {} / {}",
        stats.req_in,
        stats.req_out,
        stats.sup_in,
        stats.sup_out,
        stats.assess_in,
        stats.assess_out,
        stats.anchors_in,
        stats.anchors_out,
        stats.pre_in,
        stats.pre_out
    ));
    if !node.tags.is_empty() {
        parts.push(format!("tags: {}", node.tags.join(", ")));
    }
    if let Some(src) = first_source_ref(node) {
        parts.push(format!("src: {}", src));
    }
    parts.join("\n")
}

fn first_source_ref(node: &NodePayload) -> Option<String> {
    let refs = match &node.kind {
        NodeKind::Knowledge(k) => &k.source_refs,
        NodeKind::TeachingStep(t) => &t.source_refs,
    };
    refs.first().map(|r| {
        let range = if r.start_line == r.end_line {
            format!("L{}", r.start_line)
        } else {
            format!("L{}–L{}", r.start_line, r.end_line)
        };
        format!("{}#{} @{}", r.path, range, r.revision)
    })
}

fn truncate(s: &str, max: usize) -> String {
    let trimmed = s.trim();
    if trimmed.len() <= max {
        return trimmed.to_string();
    }
    let mut end = max;
    while end > 0 && !trimmed.is_char_boundary(end) {
        end -= 1;
    }
    let mut out = trimmed[..end].to_string();
    out.push('…');
    out
}

#[derive(Clone, Debug)]
pub struct GraphVizConfig {
    pub entity_path: String,
    pub edge_path:   String,
    pub debounce_ms: u64,
}

impl Default for GraphVizConfig {
    fn default() -> Self {
        Self {
            entity_path: "graph/live".to_string(),
            edge_path:   "graph/live/edges".to_string(),
            debounce_ms: 150,
        }
    }
}

pub struct GraphVisualizer {
    graph:            ActorRef<GraphManager>,
    rerun:            ActorRef<RerunSink>,
    config:           GraphVizConfig,
    pending_version:  Option<u64>,
    pending_time_ns:  Option<i64>,
    render_scheduled: bool,
    last_rendered:    u64,
}

impl GraphVisualizer {
    pub fn new(
        graph: ActorRef<GraphManager>,
        rerun: ActorRef<RerunSink>,
        config: GraphVizConfig,
    ) -> Self {
        Self {
            graph,
            rerun,
            config,
            pending_version: None,
            pending_time_ns: None,
            render_scheduled: false,
            last_rendered: 0,
        }
    }

    fn schedule_render(&mut self, ctx: &mut Context<Self, ()>) {
        if self.render_scheduled {
            return;
        }
        self.render_scheduled = true;
        let actor = ctx.actor_ref().clone();
        let delay = Duration::from_millis(self.config.debounce_ms);
        let _ = ctx.spawn(async move {
            sleep(delay).await;
            let _ = actor.tell(RenderNow).await;
        });
    }

    fn merge_pending(&mut self, version: u64, timestamp_ns: Option<i64>) {
        self.pending_version = Some(self.pending_version.map_or(version, |v| v.max(version)));
        if let Some(ts) = timestamp_ns {
            self.pending_time_ns = match self.pending_time_ns {
                Some(prev) => Some(prev.max(ts)),
                None => Some(ts),
            };
        }
    }
}

impl Actor for GraphVisualizer {
    type Args = GraphVisualizer;
    type Error = Infallible;

    async fn on_start(state: Self::Args, _actor_ref: ActorRef<Self>) -> Result<Self, Self::Error> {
        Ok(state)
    }
}

impl Message<PrimeRender> for GraphVisualizer {
    type Reply = ();

    async fn handle(
        &mut self,
        _msg: PrimeRender,
        ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        let version = self
            .graph
            .ask(GetGraphVersion)
            .await
            .unwrap_or(self.last_rendered);
        self.merge_pending(version, None);
        self.schedule_render(ctx);
    }
}

impl Message<GraphVizEvent> for GraphVisualizer {
    type Reply = ();

    async fn handle(
        &mut self,
        GraphVizEvent {
            graph_version,
            timestamp_ns,
        }: GraphVizEvent,
        ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        if graph_version <= self.last_rendered
            && self.pending_version.is_some_and(|v| graph_version <= v)
        {
            return;
        }
        self.merge_pending(graph_version, timestamp_ns);
        self.schedule_render(ctx);
    }
}

impl Message<RenderNow> for GraphVisualizer {
    type Reply = ();

    async fn handle(
        &mut self,
        _msg: RenderNow,
        ctx: &mut Context<Self, Self::Reply>,
    ) -> Self::Reply {
        self.render_scheduled = false;
        let Some(_target_version) = self.pending_version.take() else {
            return;
        };

        let carried_time_ns = self.pending_time_ns.take();

        let (graph, version) = match self.graph.ask(GetGraphWithVersion).await {
            Ok(res) => res,
            Err(err) => {
                debug!(error = ?err, "graph fetch for visualization failed");
                return;
            }
        };

        let time_ns = carried_time_ns.or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .ok()
                .and_then(|d| i64::try_from(d.as_nanos()).ok())
        });

        if let Some(frame) = build_frame(graph.as_ref(), &self.config, time_ns) {
            let msg = LogGraphFrame {
                entity_path:       self.config.entity_path.clone(),
                edge_overlay_path: self.config.edge_path.clone(),
                nodes:             frame.nodes,
                edges:             frame.edges,
                arrows:            frame.arrows,
                time_ns:           frame.time_ns,
            };
            let sink = self.rerun.clone();
            let _ = ctx.spawn(async move {
                let _ = sink.tell(msg).await;
            });
            self.last_rendered = version;
        }

        if let Some(pending) = self.pending_version.take()
            && pending > self.last_rendered
        {
            let carry = self.pending_time_ns.take().or(carried_time_ns);
            self.merge_pending(pending, carry);
            self.schedule_render(ctx);
        }
    }
}

struct Frame {
    nodes:   GraphNodes,
    edges:   GraphEdges,
    arrows:  Option<Arrows2D>,
    time_ns: Option<i64>,
}

fn build_frame(
    graph: &CurriculumGraph,
    _config: &GraphVizConfig,
    time_ns: Option<i64>,
) -> Option<Frame> {
    if graph.node_count() == 0 {
        return None;
    }

    let mut nodes: Vec<(usize, &NodePayload)> = graph
        .node_indices()
        .map(|id| (id.index(), &graph[id]))
        .collect();
    nodes.sort_by(|(_, a), (_, b)| a.slug.cmp(&b.slug));

    let mut index_by_id = HashMap::new();
    for (ix, (node_id, _payload)) in nodes.iter().enumerate() {
        index_by_id.insert(*node_id, ix);
    }

    let node_ids: Vec<String> = nodes.iter().map(|(_, n)| n.slug.clone()).collect();
    let radii: Vec<f32> = nodes.iter().map(|(_, n)| node_radius(&n.kind)).collect();
    let colors: Vec<Color> = nodes
        .iter()
        .map(|(_, n)| Palette::node_color(&n.kind))
        .collect();

    let mut edge_pairs = Vec::new();
    let mut stats = vec![NodeStats::default(); nodes.len()];

    for edge_id in graph.edge_indices() {
        let Some((from, to)) = graph.edge_endpoints(edge_id) else {
            continue;
        };
        let from_ix = match index_by_id.get(&from.index()) {
            Some(ix) => *ix,
            None => continue,
        };
        let to_ix = match index_by_id.get(&to.index()) {
            Some(ix) => *ix,
            None => continue,
        };

        let payload = &graph[edge_id];
        edge_pairs.push((graph[from].slug.clone(), graph[to].slug.clone()));

        if from_ix == to_ix {
            let stat = &mut stats[from_ix];
            match &payload.kind {
                EdgeKind::Requires(_) => {
                    stat.req_out += 1;
                    stat.req_in += 1;
                }
                EdgeKind::Supports(_) => {
                    stat.sup_out += 1;
                    stat.sup_in += 1;
                }
                EdgeKind::Assesses(_) => {
                    stat.assess_out += 1;
                    stat.assess_in += 1;
                }
                EdgeKind::Precedes(_) => {
                    stat.pre_out += 1;
                    stat.pre_in += 1;
                }
                EdgeKind::Anchors(_) => {
                    stat.anchors_out += 1;
                    stat.anchors_in += 1;
                }
            }
        } else {
            let (stat_from, stat_to) = two_stats_mut(&mut stats, from_ix, to_ix);
            match &payload.kind {
                EdgeKind::Requires(_) => {
                    stat_from.req_out += 1;
                    stat_to.req_in += 1;
                }
                EdgeKind::Supports(_) => {
                    stat_from.sup_out += 1;
                    stat_to.sup_in += 1;
                }
                EdgeKind::Assesses(_) => {
                    stat_from.assess_out += 1;
                    stat_to.assess_in += 1;
                }
                EdgeKind::Precedes(_) => {
                    stat_from.pre_out += 1;
                    stat_to.pre_in += 1;
                }
                EdgeKind::Anchors(_) => {
                    stat_from.anchors_out += 1;
                    stat_to.anchors_in += 1;
                }
            }
        }
    }

    let node_labels: Vec<String> = nodes
        .iter()
        .enumerate()
        .map(|(ix, (_, n))| summarize_node(n, &stats[ix]))
        .collect();

    let nodes = GraphNodes::new(node_ids)
        .with_labels(node_labels)
        .with_colors(colors)
        .with_radii(radii);
    let edges = GraphEdges::new(edge_pairs).with_graph_type(GraphType::Directed);

    Some(Frame {
        nodes,
        edges,
        arrows: None,
        time_ns,
    })
}
