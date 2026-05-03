use std::{
    fs,
    path::{Component, Path, PathBuf},
};

use anyhow::{Context, bail};
use petgraph::Direction;
use weaver::{
    graph::{
        CaseTag, EdgeKind, IntrinsicLoad, NodeKind,
        persist::{GraphSnapshot, save_graph},
    },
    schema::types::SourceRef,
};

struct Options {
    input:              PathBuf,
    output:             Option<PathBuf>,
    source_root:        Option<PathBuf>,
    repair_source_refs: bool,
}

#[derive(Default)]
struct SourceRepairStats {
    paths_fixed:    usize,
    ranges_clamped: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let options = parse_args()?;
    let source_root = match options.source_root {
        Some(root) => Some(
            root.canonicalize()
                .with_context(|| format!("canonicalize source root {}", root.display()))?,
        ),
        None => None,
    };
    if options.repair_source_refs && source_root.is_none() {
        bail!("--repair-source-refs requires --source-root <dir>");
    }

    let data = fs::read_to_string(&options.input)
        .with_context(|| format!("read snapshot {}", options.input.display()))?;
    let mut snapshot: GraphSnapshot = serde_json::from_str(&data)?;
    let course_commit = snapshot.course_commit.clone();

    let mut filled_case_tags = 0usize;
    let mut filled_coverage_tags = 0usize;
    let mut filled_rationales = 0usize;
    let mut revision_mismatches = Vec::new();
    let mut source_stats = SourceRepairStats::default();

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
            repair_spans(
                &mut ts.source_refs,
                source_root.as_deref(),
                options.repair_source_refs,
                &format!("teaching_step `{slug}` source_ref"),
                &mut source_stats,
            )?;
            for span in &ts.source_refs {
                if span.revision != course_commit {
                    revision_mismatches.push((slug.clone(), span.revision.clone()));
                }
            }
        }

        if let Some(NodeKind::Knowledge(k)) = graph.node_weight_mut(node).map(|n| &mut n.kind) {
            repair_spans(
                &mut k.source_refs,
                source_root.as_deref(),
                options.repair_source_refs,
                &format!("knowledge `{slug}` source_ref"),
                &mut source_stats,
            )?;
            for span in &k.source_refs {
                if span.revision != course_commit {
                    revision_mismatches.push((slug.clone(), span.revision.clone()));
                }
            }
        }
    }

    let edges: Vec<_> = graph.edge_indices().collect();
    for edge in edges {
        let label = graph
            .edge_endpoints(edge)
            .map(|(from, to)| format!("{} -> {} evidence_ref", graph[from].slug, graph[to].slug))
            .unwrap_or_else(|| "detached edge evidence_ref".to_string());
        match &mut graph[edge].kind {
            EdgeKind::Requires(attrs) => {
                repair_spans(
                    &mut attrs.evidence_refs,
                    source_root.as_deref(),
                    options.repair_source_refs,
                    &label,
                    &mut source_stats,
                )?;
                for span in &attrs.evidence_refs {
                    if span.revision != course_commit {
                        revision_mismatches.push((label.clone(), span.revision.clone()));
                    }
                }
            }
            EdgeKind::Supports(attrs) => {
                repair_spans(
                    &mut attrs.evidence_refs,
                    source_root.as_deref(),
                    options.repair_source_refs,
                    &label,
                    &mut source_stats,
                )?;
                for span in &attrs.evidence_refs {
                    if span.revision != course_commit {
                        revision_mismatches.push((label.clone(), span.revision.clone()));
                    }
                }
            }
            _ => {}
        }
    }

    snapshot.graph = graph;
    snapshot.course_commit = course_commit.clone();

    let out_path = options.output.unwrap_or(options.input);
    save_graph(&snapshot.graph, &out_path, &snapshot.course_commit, snapshot.graph_version).await?;

    if !revision_mismatches.is_empty() {
        eprintln!("Revision mismatches (expected {course_commit}):");
        for (slug, rev) in revision_mismatches {
            eprintln!("- {slug}: {rev}");
        }
    }

    eprintln!(
        "Backfill complete: case_tag={}, coverage_tags={}, rationales={}, source_paths={}, \
         source_ranges={}",
        filled_case_tags,
        filled_coverage_tags,
        filled_rationales,
        source_stats.paths_fixed,
        source_stats.ranges_clamped
    );

    Ok(())
}

fn parse_args() -> anyhow::Result<Options> {
    let mut positionals = Vec::new();
    let mut source_root = None;
    let mut repair_source_refs = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--source-root" => {
                let root = args
                    .next()
                    .context("--source-root requires a directory argument")?;
                source_root = Some(PathBuf::from(root));
            }
            "--repair-source-refs" => repair_source_refs = true,
            "-h" | "--help" => {
                bail!(
                    "usage: backfill_validation <input_snapshot.json> [output.json] \
                     [--source-root <dir>] [--repair-source-refs]"
                );
            }
            flag if flag.starts_with('-') => bail!("unknown flag `{flag}`"),
            positional => positionals.push(PathBuf::from(positional)),
        }
    }

    if positionals.is_empty() || positionals.len() > 2 {
        bail!(
            "usage: backfill_validation <input_snapshot.json> [output.json] [--source-root <dir>] \
             [--repair-source-refs]"
        );
    }

    Ok(Options {
        input: positionals.remove(0),
        output: positionals.pop(),
        source_root,
        repair_source_refs,
    })
}

fn repair_spans(
    spans: &mut [SourceRef],
    source_root: Option<&Path>,
    repair_source_refs: bool,
    label: &str,
    stats: &mut SourceRepairStats,
) -> anyhow::Result<()> {
    let Some(root) = source_root else {
        return Ok(());
    };

    for span in spans {
        repair_span(root, span, repair_source_refs, label, stats)?;
    }
    Ok(())
}

fn repair_span(
    source_root: &Path,
    span: &mut SourceRef,
    repair_source_refs: bool,
    label: &str,
    stats: &mut SourceRepairStats,
) -> anyhow::Result<()> {
    let original_path = span.path.clone();
    let repaired_path = if repair_source_refs {
        normalize_source_path(source_root, &span.path)?
    } else {
        span.path.clone()
    };
    if repaired_path != original_path {
        span.path = repaired_path;
        stats.paths_fixed += 1;
    }

    let canonical = validate_source_path(source_root, &span.path)
        .with_context(|| format!("{label} path `{}` is invalid", span.path))?;
    let line_count = fs::read_to_string(&canonical)
        .with_context(|| format!("read source file {}", canonical.display()))?
        .lines()
        .count() as u32;
    if line_count == 0 {
        bail!("{label} path `{}` has no addressable source lines", span.path);
    }

    let before = (span.start_line, span.end_line);
    clamp_line_range(span, line_count);
    if before != (span.start_line, span.end_line) {
        stats.ranges_clamped += 1;
    }
    Ok(())
}

fn normalize_source_path(source_root: &Path, raw: &str) -> anyhow::Result<String> {
    if validate_source_path(source_root, raw).is_ok() {
        return Ok(raw.to_string());
    }

    if let Some(stripped) = raw.strip_suffix('?')
        && validate_source_path(source_root, stripped).is_ok()
    {
        return Ok(stripped.to_string());
    }

    if let Some(repaired) = resolve_dash_variant(source_root, raw)? {
        return Ok(repaired);
    }

    Ok(raw.to_string())
}

fn resolve_dash_variant(source_root: &Path, raw: &str) -> anyhow::Result<Option<String>> {
    let path = Path::new(raw);
    if path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir))
    {
        return Ok(None);
    }

    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return Ok(None);
    };
    let variants = dash_variants(file_name);
    if variants.is_empty() {
        return Ok(None);
    }
    let parent = path.parent().unwrap_or_else(|| Path::new(""));
    let search_dir = source_root.join(parent);
    let entries = match fs::read_dir(&search_dir) {
        Ok(entries) => entries,
        Err(_) => return Ok(None),
    };

    let mut matches = Vec::new();
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        if name
            .to_str()
            .is_some_and(|entry_name| variants.iter().any(|variant| variant == entry_name))
            && entry.file_type()?.is_file()
        {
            matches.push(parent.join(PathBuf::from(name)));
        }
    }

    if matches.len() == 1 {
        let rendered = matches
            .remove(0)
            .to_str()
            .context("source path contains invalid UTF-8")?
            .to_string();
        return Ok(Some(rendered));
    }

    Ok(None)
}

fn dash_variants(file_name: &str) -> Vec<String> {
    let underscore_positions: Vec<_> = file_name
        .char_indices()
        .filter_map(|(idx, ch)| (ch == '_').then_some(idx))
        .collect();
    if underscore_positions.is_empty() {
        return Vec::new();
    }

    let bytes = file_name.as_bytes();
    let mut variants = Vec::new();
    for mask in 1usize..(1usize << underscore_positions.len()) {
        let mut variant = Vec::with_capacity(bytes.len());
        for (idx, byte) in bytes.iter().copied().enumerate() {
            if let Some(pos_idx) = underscore_positions.iter().position(|pos| *pos == idx)
                && (mask & (1usize << pos_idx)) != 0
            {
                variant.push(b'-');
            } else {
                variant.push(byte);
            }
        }
        if let Ok(rendered) = String::from_utf8(variant) {
            variants.push(rendered);
        }
    }
    variants
}

fn validate_source_path(source_root: &Path, raw: &str) -> anyhow::Result<PathBuf> {
    let path = Path::new(raw);
    if path.is_absolute() {
        bail!("must be relative to the source root");
    }
    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        bail!("must not contain `..` path components");
    }

    let candidate = source_root.join(path);
    let canonical = candidate
        .canonicalize()
        .with_context(|| format!("cannot resolve file {}", candidate.display()))?;
    if !canonical.starts_with(source_root) {
        bail!("escapes source root {}", source_root.display());
    }
    if !canonical.is_file() {
        bail!("must reference a file");
    }
    Ok(canonical)
}

fn clamp_line_range(span: &mut SourceRef, line_count: u32) {
    let original_start = span.start_line;
    let original_end = span.end_line;

    if original_start > line_count && original_end > line_count {
        span.start_line = line_count;
        span.end_line = line_count;
        return;
    }

    span.start_line = span.start_line.clamp(1, line_count);
    span.end_line = span.end_line.clamp(1, line_count);
    if span.end_line < span.start_line {
        span.start_line = span.end_line;
    }
}
