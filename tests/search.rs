use std::{fs, io::Write, path::PathBuf};

use uuid::Uuid;
use weaver::tools::search::{self, SearchOptions};

fn temp_workspace(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("weaver-search-{name}-{}", Uuid::new_v4()));
    fs::create_dir_all(&path).expect("create temp workspace");
    path
}

#[tokio::test(flavor = "multi_thread")]
async fn search_respects_gitignore() -> anyhow::Result<()> {
    let root = temp_workspace("gitignore");
    fs::create_dir_all(root.join(".git"))?;
    fs::write(root.join(".gitignore"), "ignored.txt\nignored_dir/\n")?;
    fs::write(root.join("ignored.txt"), "needle\n")?;
    fs::create_dir_all(root.join("ignored_dir"))?;
    fs::write(root.join("ignored_dir").join("file.txt"), "needle\n")?;
    fs::write(root.join("kept.txt"), "needle\n")?;

    let result = search::search_recursive(
        &root,
        "needle",
        SearchOptions {
            allow:       &[],
            max_matches: None,
            max_bytes:   None,
            stop_early:  false,
        },
    )
    .await?;

    let matched_files: Vec<_> = result
        .matches
        .iter()
        .filter_map(|m| {
            m.path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.to_string())
        })
        .collect();

    assert!(
        matched_files.contains(&"kept.txt".to_string()),
        "expected kept.txt to be returned"
    );
    assert!(
        matched_files.iter().all(|name| !name.contains("ignored")),
        "ignored entries should be filtered out"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn preview_respects_cap() -> anyhow::Result<()> {
    let root = temp_workspace("preview");
    let mut file = fs::File::create(root.join("spam.txt"))?;
    for _ in 0..100 {
        writeln!(file, "needle")?;
    }

    let result = search::search_recursive(
        &root,
        "needle",
        SearchOptions {
            allow:       &[],
            max_matches: Some(5),
            max_bytes:   Some(1_024),
            stop_early:  true,
        },
    )
    .await?;

    assert!(result.truncated, "preview should truncate after cap");
    assert_eq!(result.matches.len(), 5);
    assert_eq!(result.total_matches, 5);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn body_returns_full_results_within_limits() -> anyhow::Result<()> {
    let root = temp_workspace("body");
    fs::write(root.join("a.txt"), "alpha needle\nbeta\n")?;
    fs::write(root.join("b.txt"), "gamma\nneedle delta\n")?;

    let result = search::search_recursive(
        &root,
        "needle",
        SearchOptions {
            allow:       &[],
            max_matches: Some(50),
            max_bytes:   Some(10_000),
            stop_early:  true,
        },
    )
    .await?;

    assert!(!result.truncated);
    assert_eq!(result.matches.len(), 2);
    assert_eq!(result.total_matches, 2);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn search_blocks_noise_dirs_by_default() -> anyhow::Result<()> {
    let root = temp_workspace("blocked-default");

    // Signals we should keep.
    fs::create_dir_all(root.join("src"))?;
    fs::write(root.join("src/keep.txt"), "needle main\n")?;

    // Directories we expect to skip.
    fs::create_dir_all(root.join(".git"))?;
    fs::write(root.join(".git/ignored.txt"), "needle hidden\n")?;

    fs::create_dir_all(root.join("target"))?;
    fs::write(root.join("target/ignored.txt"), "needle build\n")?;

    fs::create_dir_all(root.join("vendor"))?;
    fs::write(root.join("vendor/ignored.txt"), "needle vendored\n")?;

    let result = search::search_recursive(
        &root,
        "needle",
        SearchOptions {
            allow:       &[],
            max_matches: None,
            max_bytes:   None,
            stop_early:  true,
        },
    )
    .await?;

    let rel_paths: Vec<String> = result
        .matches
        .iter()
        .filter_map(|m| m.path.strip_prefix(&root).ok())
        .map(|p| p.to_string_lossy().into_owned())
        .collect();

    assert!(rel_paths.iter().any(|p| p.contains("src/keep.txt")));
    assert!(
        rel_paths
            .iter()
            .all(|p| !p.contains(".git") && !p.contains("target") && !p.contains("vendor")),
        "noise directories should be ignored by default"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn search_can_opt_into_blocked_dirs() -> anyhow::Result<()> {
    let root = temp_workspace("blocked-allow");

    fs::create_dir_all(root.join("target"))?;
    fs::write(root.join("target/ignored.txt"), "needle build\n")?;
    fs::create_dir_all(root.join("vendor"))?;
    fs::write(root.join("vendor/ignored.txt"), "needle vendored\n")?;

    let result = search::search_recursive(
        &root,
        "needle",
        SearchOptions {
            allow:       &["target".to_string()],
            max_matches: None,
            max_bytes:   None,
            stop_early:  true,
        },
    )
    .await?;

    let rel_paths: Vec<String> = result
        .matches
        .iter()
        .filter_map(|m| m.path.strip_prefix(&root).ok())
        .map(|p| p.to_string_lossy().into_owned())
        .collect();

    assert!(rel_paths.iter().any(|p| p.contains("target/ignored.txt")));
    assert!(
        rel_paths.iter().all(|p| !p.contains("vendor/ignored.txt")),
        "vendor should stay blocked when not explicitly allowed"
    );

    Ok(())
}
