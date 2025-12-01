use std::sync::Arc;

use anyhow::anyhow;
use kameo::prelude::ActorRef;
use petgraph::visit::EdgeRef;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{
    analysis_cache::{AnalysisCache, AnalysisCacheKey, AnalysisKind},
    common::map_send_err_inf,
};
use crate::{
    analysis,
    graph::{CurriculumGraph, NodeId},
    tools::llm::{
        ToolExecutionError, ToolPrototype,
        common::{LearningOutcome, Slug, resolve_typed},
        graph_tools::common,
    },
};

mod alignment;
mod discourse;
mod gaps;

// Shared empty args
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NoArgs {}

pub(super) async fn load_graph_with_version(
    graph: &ActorRef<crate::graph::manager::GraphManager>,
    cache: &AnalysisCache,
) -> Result<(Arc<CurriculumGraph>, u64), ToolExecutionError> {
    let (g, version): (Arc<CurriculumGraph>, u64) = graph
        .ask(crate::graph::commands::GetGraphWithVersion)
        .await
        .map_err(map_send_err_inf)?;
    cache.prune_for_version(version);
    Ok((g, version))
}

pub(super) async fn load_lo_with_graph(
    graph: &ActorRef<crate::graph::manager::GraphManager>,
    cache: &AnalysisCache,
    lo_slug: &str,
    tool: &'static str,
) -> Result<(Arc<CurriculumGraph>, u64, NodeId), ToolExecutionError> {
    let lo = resolve_typed::<LearningOutcome>(graph, Slug::new(lo_slug.to_string()), tool).await?;
    let (g, version) = load_graph_with_version(graph, cache).await?;
    Ok((g, version, lo))
}

pub(super) async fn load_lo_with_graph_only(
    graph: &ActorRef<crate::graph::manager::GraphManager>,
    lo_slug: &str,
    tool: &'static str,
) -> Result<(Arc<CurriculumGraph>, NodeId), ToolExecutionError> {
    let lo = resolve_typed::<LearningOutcome>(graph, Slug::new(lo_slug.to_string()), tool).await?;
    let g: Arc<CurriculumGraph> = graph
        .ask(crate::graph::commands::GetGraph)
        .await
        .map_err(map_send_err_inf)?;
    Ok((g, lo))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedAssessmentReach {
    pub assessment_slug:                String,
    pub reachable_from_first_principle: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedCoverage {
    pub covered:                     Vec<String>,
    pub missing:                     Vec<String>,
    pub unused_observation_features: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedLoBundle {
    pub lo_slug:        String,
    pub assessments:    Vec<CachedAssessmentReach>,
    pub coverage:       CachedCoverage,
    pub target_anchors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedExampleGap {
    pub slug:        String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedSupportEdge {
    pub from: String,
    pub to:   String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedFadeabilityIssue {
    pub assessment_slug: String,
    pub support_edges:   Vec<CachedSupportEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedPracticeGap {
    pub slug: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedGapBundle {
    pub example_gaps:  Vec<CachedExampleGap>,
    pub fadeability:   Vec<CachedFadeabilityIssue>,
    pub practice_gaps: Vec<CachedPracticeGap>,
}

pub fn decode_cached<T: for<'de> Deserialize<'de>>(value: &Value) -> Result<T, ToolExecutionError> {
    serde_json::from_value(value.clone())
        .map_err(|err| ToolExecutionError::Internal(anyhow!("cache decode failed: {err}")))
}

pub fn cache_lo_bundle(
    cache: &AnalysisCache,
    graph_version: u64,
    graph: &Arc<CurriculumGraph>,
    lo: NodeId,
    lo_slug: &str,
) -> Arc<super::analysis_cache::AnalysisCacheValue> {
    let cache_key = AnalysisCacheKey {
        graph_version,
        kind: AnalysisKind::LoBundle {
            lo_slug: lo_slug.to_string(),
        },
    };

    cache.get_or_insert_with(cache_key, || {
        let fps = analysis::first_principles(graph);
        let reach = analysis::lo_reachability(graph, lo, &fps);
        let coverage = analysis::coverage_report(graph, lo);

        let assessments = reach
            .assessments
            .iter()
            .map(|a| CachedAssessmentReach {
                assessment_slug:                graph[a.assessment].slug.clone(),
                reachable_from_first_principle: a.reachable_from_first_principle,
            })
            .collect();

        let target_anchors: Vec<_> = graph
            .edges_directed(lo, petgraph::Direction::Incoming)
            .filter_map(|e| match &e.weight().kind {
                crate::graph::EdgeKind::Anchors(attrs)
                    if matches!(attrs.impact, crate::graph::AnchorImpact::Target) =>
                {
                    Some(graph[e.source()].slug.clone())
                }
                _ => None,
            })
            .collect();

        let bundle = CachedLoBundle {
            lo_slug: lo_slug.to_string(),
            assessments,
            coverage: CachedCoverage {
                covered:                     coverage.covered_criteria,
                missing:                     coverage.missing_criteria,
                unused_observation_features: coverage.unused_observation_features,
            },
            target_anchors,
        };

        serde_json::to_value(bundle).expect("serialize lo bundle")
    })
}

pub fn cache_gap_bundle(
    cache: &AnalysisCache,
    graph_version: u64,
    graph: &Arc<CurriculumGraph>,
) -> Arc<super::analysis_cache::AnalysisCacheValue> {
    let cache_key = AnalysisCacheKey {
        graph_version,
        kind: AnalysisKind::GapBundle,
    };

    cache.get_or_insert_with(cache_key, || {
        let example_gaps = analysis::example_gaps(graph)
            .into_iter()
            .map(|gap| CachedExampleGap {
                slug:        graph[gap.node].slug.clone(),
                description: gap.description,
            })
            .collect();

        let fadeability = analysis::fadeability_issues(graph)
            .into_iter()
            .map(|i| {
                let supports: Vec<_> = i
                    .support_edges
                    .into_iter()
                    .filter_map(|e| graph.edge_endpoints(e))
                    .map(|(u, v)| CachedSupportEdge {
                        from: graph[u].slug.clone(),
                        to:   graph[v].slug.clone(),
                    })
                    .collect();
                CachedFadeabilityIssue {
                    assessment_slug: graph[i.assessment].slug.clone(),
                    support_edges:   supports,
                }
            })
            .collect();

        let practice_gaps = analysis::procedural_practice_gaps(graph)
            .into_iter()
            .map(|g| CachedPracticeGap {
                slug: graph[g.node].slug.clone(),
            })
            .collect();

        let bundle = CachedGapBundle {
            example_gaps,
            fadeability,
            practice_gaps,
        };

        serde_json::to_value(bundle).expect("serialize gap bundle")
    })
}

pub(super) fn tool_prototypes() -> Vec<ToolPrototype> {
    let mut tools = alignment::tool_prototypes();
    tools.extend(gaps::tool_prototypes());
    tools.extend(discourse::tool_prototypes());
    tools
}
