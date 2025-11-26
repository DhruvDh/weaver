use std::{
    io::ErrorKind,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use grep_regex::RegexMatcher;
use grep_searcher::{BinaryDetection, SearcherBuilder, sinks::Lossy};
use walkdir::WalkDir;

/// Result of a regex search within a file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchMatch {
    pub path:        PathBuf,
    pub line_number: usize,
    pub context:     String,
}

/// Recursively search for `pattern` beginning at `root`.
pub async fn search_recursive(root: impl AsRef<Path>, pattern: &str) -> Result<Vec<SearchMatch>> {
    let root = root.as_ref().to_path_buf();
    let pattern = pattern.to_owned();

    let matches = tokio::task::spawn_blocking(move || -> Result<Vec<SearchMatch>> {
        let matcher = RegexMatcher::new_line_matcher(&pattern)
            .with_context(|| format!("failed to compile search pattern `{pattern}`"))?;
        let mut searcher = SearcherBuilder::new()
            .line_number(true)
            .binary_detection(BinaryDetection::quit(b'\x00'))
            .build();

        let mut results = Vec::new();
        for entry in WalkDir::new(&root) {
            let entry = match entry {
                Ok(e) => e,
                Err(_err) => {
                    // Skip unreadable entries instead of aborting the search.
                    continue;
                }
            };
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.into_path();
            let mut sink = Lossy(|lnum, line| {
                results.push(SearchMatch {
                    path:        path.clone(),
                    line_number: lnum as usize,
                    context:     line.to_string(),
                });
                Ok(true)
            });
            match searcher.search_path(&matcher, &path, &mut sink) {
                Ok(()) => {}
                Err(err) if err.kind() == ErrorKind::InvalidData => continue,
                Err(_err) => {
                    // Skip files that failed to search; continue with others.
                    continue;
                }
            }
        }

        Ok(results)
    })
    .await
    .context("blocking regex search task panicked")??;

    Ok(matches)
}
