use std::{
    fs::FileType,
    path::{Path, PathBuf},
};

use anyhow::{Result, bail};
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
    let mut reader = fs::read_dir(path).await?;
    let mut entries = Vec::new();

    while let Some(entry) = reader.next_entry().await? {
        let file_type = entry.file_type().await?;
        let metadata = entry.metadata().await.ok();
        entries.push(DirEntryInfo {
            name: entry.file_name().to_string_lossy().into_owned(),
            path: entry.path(),
            kind: DirEntryKind::from(file_type),
            size: metadata.map(|m| m.len()),
        });
    }

    entries.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(entries)
}

/// Read an entire file into memory.
pub async fn read_file_full(path: impl AsRef<Path>) -> Result<String> {
    let bytes = fs::read(path).await?;
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
    let bytes = fs::read(&path_buf).await?;
    let content = String::from_utf8_lossy(&bytes);

    let mut extracted = String::new();
    let mut first_line = None;
    let mut last_line = None;

    for (idx, line) in content.lines().enumerate() {
        let line_no = idx + 1;
        if line_no < start_line {
            continue;
        }
        if line_no > end_line {
            break;
        }
        if first_line.is_none() {
            first_line = Some(line_no);
        }
        last_line = Some(line_no);
        extracted.push_str(line);
        extracted.push('\n');
    }

    if first_line.is_none() {
        bail!(
            "requested range {}-{} is outside the bounds of {}",
            start_line,
            end_line,
            path_buf.display()
        );
    }

    if extracted.ends_with('\n') {
        extracted.pop();
    }

    Ok(FileRange {
        path:       path_buf,
        start_line: first_line.unwrap(),
        end_line:   last_line.unwrap(),
        text:       extracted,
    })
}
