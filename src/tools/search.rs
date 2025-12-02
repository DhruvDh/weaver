use std::{
    collections::HashSet,
    io::ErrorKind,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow};
use grep_regex::RegexMatcher;
use grep_searcher::{BinaryDetection, SearcherBuilder, sinks::Lossy};
use ignore::WalkBuilder;
use tokio::time::{Duration, timeout};
use tracing::{debug, trace};

use crate::constants::search::DEFAULT_BLOCKED_DIRS;

/// Directories skipped by default during recursive search to cut noise and
/// expensive traversals. Keep this list small and overridable via
/// `SearchOptions::allow`.
/// Result of a regex search within a file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchMatch {
    pub path:        PathBuf,
    pub line_number: usize,
    pub context:     String,
}

#[derive(Debug, Clone, Default)]
pub struct SearchOptions<'a> {
    /// Directory basenames to include even though they are excluded by default
    /// (e.g., `target`, `.git`, `node_modules`, `vendor`).
    pub allow:       &'a [String],
    /// Maximum matches to collect before truncating.
    pub max_matches: Option<usize>,
    /// Maximum bytes of matched context to collect before truncating.
    pub max_bytes:   Option<u64>,
    /// When true, short-circuit directory traversal once caps are hit.
    pub stop_early:  bool,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub matches:       Vec<SearchMatch>,
    pub truncated:     bool,
    pub total_matches: usize,
    pub total_bytes:   u64,
}

/// Recursively search for `pattern` beginning at `root` with configurable
/// limits and default ignores.
pub async fn search_recursive(
    root: impl AsRef<Path>,
    pattern: &str,
    opts: SearchOptions<'_>,
) -> Result<SearchResult> {
    let root = root.as_ref().to_path_buf();
    let pattern = pattern.to_owned();
    let allow: HashSet<String> = opts.allow.iter().cloned().collect();
    let stop_early = opts.stop_early;
    let max_matches = opts.max_matches.unwrap_or(usize::MAX);
    let max_bytes = opts.max_bytes.unwrap_or(u64::MAX);

    let search_future = tokio::task::spawn_blocking(move || -> Result<SearchResult> {
        let matcher = RegexMatcher::new_line_matcher(&pattern)
            .with_context(|| format!("failed to compile search pattern `{pattern}`"))?;
        let mut searcher = SearcherBuilder::new()
            .line_number(true)
            .binary_detection(BinaryDetection::quit(b'\x00'))
            .build();

        let mut results = Vec::new();
        let mut truncated = false;
        let mut total_matches = 0usize;
        let mut total_bytes = 0u64;
        let mut builder = WalkBuilder::new(&root);
        builder
            .standard_filters(false)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true)
            .ignore(true)
            .hidden(false)
            .parents(true)
            .filter_entry(move |entry: &ignore::DirEntry| {
                if entry.depth() == 0 {
                    return true;
                }
                let name = entry.file_name().to_string_lossy();
                if DEFAULT_BLOCKED_DIRS.iter().any(|blocked| name == *blocked)
                    && !allow.contains(name.as_ref())
                {
                    return false;
                }
                true
            });

        for entry in builder.build() {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            if !entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                continue;
            }
            let path = entry.into_path();
            let mut sink = Lossy(|lnum, line| {
                if total_matches >= max_matches || total_bytes >= max_bytes {
                    truncated = true;
                    return Ok(false);
                }
                total_matches = total_matches.saturating_add(1);
                total_bytes = total_bytes.saturating_add(line.len() as u64);
                if results.len() < max_matches {
                    results.push(SearchMatch {
                        path:        path.clone(),
                        line_number: lnum as usize,
                        context:     line.to_string(),
                    });
                }
                if total_matches >= max_matches || total_bytes >= max_bytes {
                    truncated = true;
                    return Ok(false);
                }
                Ok(true)
            });
            match searcher.search_path(&matcher, &path, &mut sink) {
                Ok(()) => {}
                Err(err) if err.kind() == ErrorKind::InvalidData => {
                    trace!(
                        target: "weaver.search",
                        path = %path.display(),
                        error = %err,
                        "skipping file with invalid data during search"
                    );
                    continue;
                }
                Err(err) => {
                    debug!(
                        target: "weaver.search",
                        path = %path.display(),
                        error = %err,
                        "search failed for path; continuing"
                    );
                    continue;
                }
            }
            if truncated && stop_early {
                break;
            }
        }

        Ok(SearchResult {
            matches: results,
            truncated,
            total_matches,
            total_bytes,
        })
    });
    let result =
        timeout(Duration::from_millis(crate::constants::SEARCH_TASK_TIMEOUT_MS), search_future)
            .await
            .map_err(|_| {
                anyhow!("search timed out after {} ms", crate::constants::SEARCH_TASK_TIMEOUT_MS)
            })?
            .context("blocking regex search task panicked")??;

    Ok(result)
}
