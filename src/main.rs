use std::{collections::HashSet, path::PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use glob::glob;
use pathdiff::diff_paths;
use weaver::app::{RuntimeOptions, cli, run_app, run_two_phase_construction};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let args = cli().run();
    if !args.interactive {
        let workspace_abs = args
            .workspace
            .canonicalize()
            .unwrap_or_else(|_| args.workspace.clone());

        let mut chapters = Vec::new();
        let mut seen = HashSet::new();

        if let Some(pattern) = args.chapters_pattern.clone() {
            let resolved = if PathBuf::from(&pattern).is_absolute() {
                PathBuf::from(&pattern)
            } else {
                workspace_abs.join(&pattern)
            };
            let pattern_str = resolved
                .to_str()
                .ok_or_else(|| anyhow!("invalid UTF-8 in chapters pattern"))?
                .to_string();
            for entry in glob(&pattern_str)? {
                let path = entry?;
                if !path.starts_with(&workspace_abs) {
                    bail!(
                        "chapter `{}` is outside the workspace `{}`",
                        path.display(),
                        workspace_abs.display()
                    );
                }
                let rel = diff_paths(&path, &workspace_abs)
                    .context("failed to render chapter path relative to workspace")?;
                if seen.insert(rel.clone()) {
                    chapters.push(rel);
                }
            }
            if chapters.is_empty() {
                bail!("no chapters found matching pattern `{}`", pattern);
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
            let pattern = format!("{}/**/*.ptx", dir_str.trim_end_matches('/'));
            for entry in glob(&pattern)? {
                let path = entry?;
                if !path.starts_with(&workspace_abs) {
                    bail!(
                        "chapter `{}` is outside the workspace `{}`",
                        path.display(),
                        workspace_abs.display()
                    );
                }
                let rel = diff_paths(&path, &workspace_abs)
                    .context("failed to render chapter path relative to workspace")?;
                if seen.insert(rel.clone()) {
                    chapters.push(rel);
                }
            }
            if chapters.is_empty() {
                bail!("no chapters found under `{}` matching *.ptx", dir_abs.display());
            }
        } else {
            bail!("--chapters or --chapters-dir is required when using --two-phase");
        }

        run_two_phase_construction(args, chapters).await
    } else {
        run_app(args, RuntimeOptions::default()).await
    }
}
