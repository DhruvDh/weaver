use std::{fs, path::PathBuf};

use petgraph::Direction;
use weaver::graph::{CaseTag, EdgeKind, IntrinsicLoad, NodeKind, persist::GraphSnapshot};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let input = args
        .next()
        .expect("usage: backfill_validation <input_snapshot.json> [output.json]");
    let output = args.next();

    let data = fs::read_to_string(&input)?;
    let mut snapshot: GraphSnapshot = serde_json::from_str(&data)?;
    let course_commit = snapshot.course_commit.clone();

    let mut filled_case_tags = 0usize;
    let mut filled_coverage_tags = 0usize;
    let mut filled_rationales = 0usize;
    let mut revision_mismatches = Vec::new();

    let mut graph = snapshot.graph;

    let support_edges: Vec<_> = graph.edge_indices().collect();
    for edge in support_edges {
        let needs_coverage = graph
            .edge_endpoints(edge)
            .and_then(|(_, target)| graph.node_weight(target))
            .is_some_and(|node| {
                matches!(
                    node.kind,
                    NodeKind::Knowledge(ref k)
                        if matches!(k.intrinsic_load, Some(IntrinsicLoad::High))
                )
            });

        if let EdgeKind::Supports(ref mut attrs) = graph[edge].kind {
            if attrs.case_tag.is_none() {
                attrs.case_tag = Some(CaseTag::Typical);
                filled_case_tags += 1;
            }
            if needs_coverage && attrs.coverage_tags.is_empty() {
                attrs.coverage_tags.push("backfill".into());
                filled_coverage_tags += 1;
            }
        }
    }

    let nodes: Vec<_> = graph.node_indices().collect();
    for node in nodes {
        let slug = graph[node].slug.clone();
        let has_anchor = graph
            .edges_directed(node, Direction::Outgoing)
            .any(|e| matches!(&e.weight().kind, EdgeKind::Anchors(_)));

        if let Some(NodeKind::TeachingStep(ts)) = graph.node_weight_mut(node).map(|n| &mut n.kind) {
            let has_rationale = ts
                .rationale
                .as_ref()
                .map(|r| !r.trim().is_empty())
                .unwrap_or(false);
            if !has_anchor && !has_rationale {
                ts.rationale = Some("backfill: add rationale".into());
                filled_rationales += 1;
            }
            for span in &ts.source_refs {
                if span.revision != course_commit {
                    revision_mismatches.push((slug.clone(), span.revision.clone()));
                }
            }
        }

        if let Some(NodeKind::Knowledge(k)) = graph.node_weight(node).map(|n| &n.kind) {
            for span in &k.source_refs {
                if span.revision != course_commit {
                    revision_mismatches.push((slug.clone(), span.revision.clone()));
                }
            }
        }
    }

    snapshot.graph = graph;
    snapshot.course_commit = course_commit.clone();

    let out_path = output
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(&input));
    let serialized = serde_json::to_string_pretty(&snapshot)?;
    fs::write(&out_path, serialized)?;

    if !revision_mismatches.is_empty() {
        eprintln!("Revision mismatches (expected {course_commit}):");
        for (slug, rev) in revision_mismatches {
            eprintln!("- {slug}: {rev}");
        }
    }

    eprintln!(
        "Backfill complete: case_tag={}, coverage_tags={}, rationales={}",
        filled_case_tags, filled_coverage_tags, filled_rationales
    );

    Ok(())
}
