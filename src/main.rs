use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, bail};
use glob::glob;
use pathdiff::diff_paths;
use weaver::app::{Cli, RuntimeOptions, cli, run_app, run_two_phase_construction};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let args = cli().run();
    if !args.interactive {
        let workspace_abs = args
            .workspace
            .canonicalize()
            .unwrap_or_else(|_| args.workspace.clone());

        let chapters = collect_chapters(&args, &workspace_abs)?;

        run_two_phase_construction(args, chapters).await
    } else {
        run_app(args, RuntimeOptions::default()).await
    }
}

fn collect_chapters(args: &Cli, workspace_abs: &Path) -> Result<Vec<PathBuf>> {
    let mut chapters = Vec::new();
    let mut seen = HashSet::new();

    if let Some(pattern) = args.chapters_pattern.clone() {
        collect_from_pattern(&pattern, workspace_abs, &mut seen, &mut chapters)?;
        if chapters.is_empty() {
            bail!("no chapters found matching pattern `{pattern}`");
        }
    } else if let Some(dir) = args.chapters_dir.clone() {
        let dir_abs = if dir.is_absolute() {
            dir
        } else {
            let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let candidate = cwd.join(&dir);
            if candidate.is_dir() {
                candidate
            } else {
                workspace_abs.join(dir)
            }
        };
        if !dir_abs.is_dir() {
            bail!("--chapters-dir must be a directory");
        }
        let dir_str = dir_abs
            .to_str()
            .ok_or_else(|| anyhow!("chapters-dir contains invalid UTF-8"))?;
        let pattern = format!("{}/**/toctree.ptx", dir_str.trim_end_matches('/'));
        collect_from_pattern(&pattern, workspace_abs, &mut seen, &mut chapters)?;
        if chapters.is_empty() {
            bail!(
                "no chapter entrypoints found under `{}` matching toctree.ptx",
                dir_abs.display()
            );
        }
    } else {
        let default_pattern =
            format!("{}/source/*/toctree.ptx", workspace_abs.display()).replace("//", "/");
        collect_from_pattern(&default_pattern, workspace_abs, &mut seen, &mut chapters)?;
        if chapters.is_empty() {
            bail!(
                "no chapter entrypoints found via source/*/toctree.ptx under `{}`",
                workspace_abs.display()
            );
        }
    }

    chapters.sort();
    Ok(chapters)
}

fn collect_from_pattern(
    pattern: &str,
    workspace_abs: &Path,
    seen: &mut HashSet<PathBuf>,
    chapters: &mut Vec<PathBuf>,
) -> Result<()> {
    let resolved = if PathBuf::from(pattern).is_absolute() {
        PathBuf::from(pattern)
    } else {
        workspace_abs.join(pattern)
    };
    let pattern_str = resolved
        .to_str()
        .ok_or_else(|| anyhow!("invalid UTF-8 in chapters pattern"))?
        .to_string();
    for entry in glob(&pattern_str)? {
        let path = entry?;
        if !path.starts_with(workspace_abs) {
            bail!(
                "chapter `{}` is outside the workspace `{}`",
                path.display(),
                workspace_abs.display()
            );
        }
        let rel = diff_paths(&path, workspace_abs)
            .context("failed to render chapter path relative to workspace")?;
        if seen.insert(rel.clone()) {
            chapters.push(rel);
        }
    }
    Ok(())
}
