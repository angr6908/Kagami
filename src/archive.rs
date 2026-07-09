//! Media inside .zip/.cbz archives, browsed in place. Entries are listed from
//! the archive's central directory during the scan, and each image/video is
//! decompressed straight into memory when it is needed — nothing is ever
//! extracted to a temp file.

use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const ARCHIVE_EXTENSIONS: &[&str] = &["zip", "cbz"];
pub const READ_CHUNK: usize = 1 << 20;

pub fn has_ext(path: &Path, exts: &[&str]) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| exts.iter().any(|c| e.eq_ignore_ascii_case(c)))
        .unwrap_or(false)
}

pub fn is_archive_file(path: &Path) -> bool {
    has_ext(path, ARCHIVE_EXTENSIONS)
}

/// One browsable media item: a plain file on disk, or one entry inside an
/// archive. The whole app browses `Item`s; only the byte-fetching code cares
/// which variant it is.
#[derive(Clone)]
pub enum Item {
    File(PathBuf),
    /// `archive` is shared across all entries of the same archive.
    Archived { archive: Arc<PathBuf>, entry: String },
}

impl Item {
    /// The name whose extension decides how the item is handled (image/video).
    pub fn media_name(&self) -> &Path {
        match self {
            Item::File(p) => p,
            Item::Archived { entry, .. } => Path::new(entry),
        }
    }

    /// The file on disk backing this item — the archive itself for archived
    /// entries. This is what hydration touches and what the Trash would take.
    pub fn disk_path(&self) -> &Path {
        match self {
            Item::File(p) => p,
            Item::Archived { archive, .. } => archive,
        }
    }

    pub fn is_archived(&self) -> bool {
        matches!(self, Item::Archived { .. })
    }

    /// Title string relative to the opened root: `sub/img.jpg`, or
    /// `sub/comics.cbz/page01.jpg` for an archived entry.
    pub fn display(&self, root: &Path) -> String {
        let disk = self.disk_path();
        let disk = disk.strip_prefix(root).unwrap_or(disk).to_string_lossy();
        match self {
            Item::File(_) => disk.into_owned(),
            Item::Archived { entry, .. } => format!("{disk}/{entry}"),
        }
    }

    /// Lowercased string the browsing list is natural-sorted by; archive
    /// entries slot in under their archive's own path.
    pub fn sort_key(&self) -> String {
        match self {
            Item::File(p) => p.to_string_lossy().to_ascii_lowercase(),
            Item::Archived { archive, entry } => {
                format!("{}/{entry}", archive.to_string_lossy()).to_ascii_lowercase()
            }
        }
    }

    /// How to hand this item to the video player without a temp file. Stored
    /// (uncompressed) archive entries — the common case for video, which zip
    /// tools rarely re-compress — play straight out of the archive file at a
    /// byte range, costing no RAM; compressed entries have to be decompressed
    /// fully into memory, the only seekable option without extraction.
    pub fn video_source(&self) -> Option<VideoSource> {
        match self {
            Item::File(p) => Some(VideoSource::Path(p.clone())),
            Item::Archived { archive, entry } => {
                let file = std::fs::File::open(archive.as_path()).ok()?;
                let mut zip = zip::ZipArchive::new(BufReader::new(file)).ok()?;
                let entry = zip.by_name(entry).ok()?;
                if entry.compression() == zip::CompressionMethod::Stored
                    && !entry.encrypted()
                    && let Some(start) = entry.data_start()
                {
                    return Some(VideoSource::FileRange {
                        path: archive.as_path().to_path_buf(),
                        start,
                        len: entry.size(),
                    });
                }
                let size = entry.size() as usize;
                read_chunked(entry, size, &|| false).map(VideoSource::Bytes)
            }
        }
    }

    /// The item's full bytes, decompressing an archived entry into memory.
    /// `cancelled` is polled between chunks so stale background reads bail
    /// out; None means either cancellation or an I/O/archive error.
    pub fn read(&self, cancelled: &dyn Fn() -> bool) -> Option<Vec<u8>> {
        match self {
            Item::File(p) => read_chunked(std::fs::File::open(p).ok()?, 0, cancelled),
            Item::Archived { archive, entry } => {
                let file = std::fs::File::open(archive.as_path()).ok()?;
                let mut zip = zip::ZipArchive::new(BufReader::new(file)).ok()?;
                let entry = zip.by_name(entry).ok()?;
                let size = entry.size() as usize;
                read_chunked(entry, size, cancelled)
            }
        }
    }
}

/// See [`Item::video_source`].
pub enum VideoSource {
    Path(PathBuf),
    FileRange { path: PathBuf, start: u64, len: u64 },
    Bytes(Vec<u8>),
}

/// List an archive's media entries (per `is_media`), skipping directories and
/// metadata junk. Unreadable/encrypted archives yield an empty list.
pub fn scan_archive(path: &Path, is_media: impl Fn(&Path) -> bool) -> Vec<Item> {
    let Ok(file) = std::fs::File::open(path) else {
        return Vec::new();
    };
    let Ok(zip) = zip::ZipArchive::new(BufReader::new(file)) else {
        return Vec::new();
    };
    let archive = Arc::new(path.to_path_buf());
    zip.file_names()
        .filter(|name| !name.ends_with('/') && !is_junk(name) && is_media(Path::new(name)))
        .map(|name| Item::Archived { archive: archive.clone(), entry: name.to_string() })
        .collect()
}

/// macOS zips are littered with `__MACOSX/` resource forks and `._*`/`.DS_Store`
/// dotfiles; none of them are viewable media even when the extension says so.
fn is_junk(name: &str) -> bool {
    name.split('/').any(|part| part.starts_with('.') || part == "__MACOSX")
}

fn read_chunked(mut r: impl Read, size_hint: usize, cancelled: &dyn Fn() -> bool) -> Option<Vec<u8>> {
    let mut data = Vec::with_capacity(size_hint);
    let mut buf = vec![0u8; READ_CHUNK];
    loop {
        if cancelled() {
            return None;
        }
        match r.read(&mut buf) {
            Ok(0) => return Some(data),
            Ok(n) => data.extend_from_slice(&buf[..n]),
            Err(_) => return None,
        }
    }
}
