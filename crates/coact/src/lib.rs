//! Co-activation analysis and expert layout search for MoE GGUF models.

pub mod analyze;
pub mod cluster;
pub mod cost;
pub mod layout;
pub mod project;
pub mod reweight;
pub mod trace;

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;

/// SHA-256 of a file, streamed so a multi-gigabyte model never lands in memory.
pub fn file_sha256(path: impl AsRef<Path>) -> Result<String> {
    let path = path.as_ref();
    let mut f = std::fs::File::open(path).with_context(|| format!("hashing {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 22];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Derives the sidecar path that records which layout is currently applied to a model.
pub fn sidecar_path(model: &Path) -> std::path::PathBuf {
    model.with_extension("layout.json")
}
