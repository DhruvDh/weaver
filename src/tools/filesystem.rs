use std::{
    fs::FileType,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use tokio::fs;

/// A simplified view of a directory entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirEntryInfo {
    pub name: String,
    pub path: PathBuf,
    pub kind: DirEntryKind,
    pub size: Option<u64>,
}

/// Basic classification of directory entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirEntryKind {
    File,
    Directory,
    Symlink,
    Other,
}

impl From<FileType> for DirEntryKind {
    fn from(ft: FileType) -> Self {
        if ft.is_file() {
            Self::File
        } else if ft.is_dir() {
            Self::Directory
        } else if ft.is_symlink() {
            Self::Symlink
        } else {
            Self::Other
        }
    }
}

impl DirEntryKind {
    pub fn as_str(self) -> &'static str {
        match self {
            DirEntryKind::File => "file",
            DirEntryKind::Directory => "directory",
            DirEntryKind::Symlink => "symlink",
            DirEntryKind::Other => "other",
        }
    }
}

/// List directory entries similarly to `ls`.
pub async fn list_dir(path: impl AsRef<Path>) -> Result<Vec<DirEntryInfo>> {
    let path = path.as_ref();
    let mut reader = fs::read_dir(path)
        .await
        .with_context(|| format!("failed to read directory {}", path.display()))?;
    let mut entries = Vec::new();

    while let Some(entry) = reader
        .next_entry()
        .await
        .with_context(|| format!("failed to iterate directory {}", path.display()))?
    {
        let entry_path = entry.path();
        let file_type = entry
            .file_type()
            .await
            .with_context(|| format!("failed to read file type for {}", entry_path.display()))?;
        let metadata = entry.metadata().await.ok();
        entries.push(DirEntryInfo {
            name: entry.file_name().to_string_lossy().into_owned(),
            path: entry_path,
            kind: DirEntryKind::from(file_type),
            size: metadata.as_ref().map(|m| m.len()),
        });
    }

    entries.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(entries)
}

/// Read an entire file into memory.
pub async fn read_file_full(path: impl AsRef<Path>) -> Result<String> {
    let bytes = fs::read(path.as_ref())
        .await
        .with_context(|| format!("failed to read file {}", path.as_ref().display()))?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Returned slice of a file based on line numbers.
#[derive(Debug, Clone)]
pub struct FileRange {
    pub path:       PathBuf,
    pub start_line: usize,
    pub end_line:   usize,
    pub text:       String,
}

/// Read a subset of lines (inclusive) from a file.
pub async fn read_file_range(
    path: impl AsRef<Path>,
    start_line: usize,
    end_line: usize,
) -> Result<FileRange> {
    if start_line == 0 {
        bail!("start_line must be >= 1");
    }
    if end_line < start_line {
        bail!("end_line must be >= start_line");
    }

    let path_buf = path.as_ref().to_path_buf();
    let bytes = fs::read(&path_buf)
        .await
        .with_context(|| format!("failed to read file {}", path_buf.display()))?;
    let content = String::from_utf8_lossy(&bytes);
    let total_lines = content.lines().count();

    if start_line > total_lines {
        bail!(
            "requested range {}-{} is outside the bounds of {}",
            start_line,
            end_line,
            path_buf.display()
        );
    }

    if end_line > total_lines {
        bail!(
            "requested end_line {} exceeds file line count {} for {}",
            end_line,
            total_lines,
            path_buf.display()
        );
    }

    let count = end_line - start_line + 1;
    let lines: Vec<&str> = content.lines().skip(start_line - 1).take(count).collect();

    if lines.is_empty() {
        bail!(
            "requested range {}-{} is outside the bounds of {}",
            start_line,
            end_line,
            path_buf.display()
        );
    }

    let extracted = lines.join("\n");
    let last_line = start_line + lines.len() - 1;

    Ok(FileRange {
        path: path_buf,
        start_line,
        end_line: last_line,
        text: extracted,
    })
}
