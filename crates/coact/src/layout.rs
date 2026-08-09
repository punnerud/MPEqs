//! On-disk description of an expert layout.
//!
//! A layout is only meaningful relative to a specific model file, so the file's SHA-256 in
//! its *shipped* state is part of the document. `ggufperm` refuses to act on a mismatch —
//! applying a permutation derived from one model to another would silently produce a model
//! that still loads and still generates text, just wrong text.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

use crate::cost::Permutation;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layout {
    /// File name of the model this layout belongs to.
    pub model: String,
    /// SHA-256 of the model as originally downloaded, before any permutation.
    pub original_sha256: String,
    /// How the ordering was produced: `identity`, `random:<seed>`, `frequency`, `mincut`, …
    pub method: String,
    pub n_expert: u32,
    /// Layer index -> `perm[slot] = original expert id`.
    pub layers: BTreeMap<u32, Vec<u16>>,
}

impl Layout {
    pub fn identity(model: &str, sha: &str, n_expert: u32, layer_indices: &[u32]) -> Self {
        let perm: Vec<u16> = (0..n_expert as u16).collect();
        Layout {
            model: model.to_string(),
            original_sha256: sha.to_string(),
            method: "identity".into(),
            n_expert,
            layers: layer_indices.iter().map(|&l| (l, perm.clone())).collect(),
        }
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let s =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let l: Layout =
            serde_json::from_str(&s).with_context(|| format!("parsing {}", path.display()))?;
        l.validate()?;
        Ok(l)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        let path = path.as_ref();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).ok();
        }
        std::fs::write(path, serde_json::to_string_pretty(self)?)
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        for (layer, perm) in &self.layers {
            if perm.len() != self.n_expert as usize {
                bail!(
                    "layer {layer}: permutation has {} entries, expected {}",
                    perm.len(),
                    self.n_expert
                );
            }
            Permutation::from_perm(perm.clone()).with_context(|| format!("layer {layer}"))?;
        }
        Ok(())
    }

    pub fn permutation(&self, layer: u32) -> Result<Permutation> {
        let perm = self
            .layers
            .get(&layer)
            .ok_or_else(|| anyhow::anyhow!("layout has no entry for layer {layer}"))?;
        Permutation::from_perm(perm.clone())
    }

    pub fn is_identity(&self) -> bool {
        self.layers
            .values()
            .all(|p| p.iter().enumerate().all(|(i, &e)| i as u16 == e))
    }
}
