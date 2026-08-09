//! Lossless, in-place reordering of MoE experts inside a GGUF file.
//!
//! Permuting the expert axis of the fused `ffn_*_exps` tensors changes which physical slot
//! holds which expert. Permuting the same axis of the router weight `ffn_gate_inp` renames
//! the experts to match. Together the two are a relabelling: the computation is identical,
//! so the model's logits are bit-for-bit unchanged. That exactness is the point — it is the
//! oracle that compression-based claims never get.
//!
//! Nothing outside the tensor data section moves: offsets, sizes and the header are
//! untouched, so the file stays a valid GGUF that stock llama.cpp loads.

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use coact::cost::Permutation;
use coact::layout::Layout;
use coact::{file_sha256, sidecar_path};
use gguf::{moe::MoeModel, Gguf, TensorInfo};
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(about = "Losslessly reorder MoE experts inside a GGUF file, in place")]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Show which layout is currently applied.
    Status {
        #[arg(short, long)]
        model: PathBuf,
    },
    /// Record the model's pristine SHA-256 so `revert` can prove it restored the original.
    Init {
        #[arg(short, long)]
        model: PathBuf,
    },
    /// Apply a layout, transforming from whatever is currently applied.
    Apply {
        #[arg(short, long)]
        model: PathBuf,
        #[arg(short, long)]
        layout: PathBuf,
        /// Report what would move without writing anything.
        #[arg(long)]
        dry_run: bool,
    },
    /// Restore the shipped expert order and verify the file hashes back to the original.
    Revert {
        #[arg(short, long)]
        model: PathBuf,
    },
    /// Check that the file still parses and the sidecar is consistent with it.
    Verify {
        #[arg(short, long)]
        model: PathBuf,
        /// Also re-hash the file (slow, but the only real proof for an identity layout).
        #[arg(long)]
        hash: bool,
    },
}

fn main() -> Result<()> {
    match Args::parse().cmd {
        Cmd::Status { model } => status(&model),
        Cmd::Init { model } => init(&model),
        Cmd::Apply {
            model,
            layout,
            dry_run,
        } => apply(&model, &layout, dry_run),
        Cmd::Revert { model } => revert(&model),
        Cmd::Verify { model, hash } => verify(&model, hash),
    }
}

/// The layout currently on disk, defaulting to the shipped order.
fn current(model: &Path, moe: &MoeModel) -> Result<Layout> {
    let sc = sidecar_path(model);
    if sc.exists() {
        let l = Layout::load(&sc)?;
        if l.n_expert != moe.n_expert {
            bail!(
                "sidecar {} describes {} experts but the model has {}",
                sc.display(),
                l.n_expert,
                moe.n_expert
            );
        }
        Ok(l)
    } else {
        bail!(
            "no sidecar at {} — run `ggufperm init` first so the original SHA-256 is on record",
            sc.display()
        );
    }
}

fn init(model: &Path) -> Result<()> {
    let g = Gguf::open(model)?;
    let moe = MoeModel::from_gguf(&g)?;
    let sc = sidecar_path(model);
    if sc.exists() {
        bail!(
            "{} already exists; refusing to overwrite a known-good baseline",
            sc.display()
        );
    }
    eprintln!("hashing {} …", model.display());
    let sha = file_sha256(model)?;
    let indices: Vec<u32> = moe.layers.iter().map(|l| l.index).collect();
    let name = model
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    let l = Layout::identity(&name, &sha, moe.n_expert, &indices);
    l.save(&sc)?;
    println!("original sha256  {sha}");
    println!("sidecar          {}", sc.display());
    Ok(())
}

fn status(model: &Path) -> Result<()> {
    let g = Gguf::open(model)?;
    let moe = MoeModel::from_gguf(&g)?;
    let cur = current(model, &moe)?;
    println!("model            {}", model.display());
    println!("applied layout   {}", cur.method);
    println!("identity         {}", cur.is_identity());
    println!("original sha256  {}", cur.original_sha256);
    println!("MoE layers       {}", cur.layers.len());
    Ok(())
}

/// Tensors of one layer whose last axis is the expert axis, plus their per-expert stride.
fn permutable_tensors(moe: &MoeModel, layer: u32) -> Result<Vec<(&TensorInfo, u64)>> {
    let l = moe.layer(layer)?;
    l.tensors
        .iter()
        .map(|t| t.last_axis_stride().map(|s| (t, s)))
        .collect()
}

/// Rewrites one tensor so that slot `s` holds what slot `d[s]` held.
///
/// Cycle-following keeps peak memory at two blocks regardless of tensor size, which for a
/// 4 GB model with 1.7 MB experts is the difference between a 3.4 MB working set and a
/// second copy of the file.
fn permute_tensor(f: &File, offset: u64, stride: u64, d: &[u16]) -> Result<u64> {
    let n = d.len();
    let mut visited = vec![false; n];
    let mut saved = vec![0u8; stride as usize];
    let mut block = vec![0u8; stride as usize];
    let mut moved = 0u64;

    for s in 0..n {
        if visited[s] {
            continue;
        }
        if d[s] as usize == s {
            visited[s] = true;
            continue;
        }
        f.read_exact_at(&mut saved, offset + s as u64 * stride)
            .with_context(|| format!("reading slot {s}"))?;
        let mut cur = s;
        loop {
            visited[cur] = true;
            let src = d[cur] as usize;
            if src == s {
                break;
            }
            f.read_exact_at(&mut block, offset + src as u64 * stride)
                .with_context(|| format!("reading slot {src}"))?;
            f.write_all_at(&block, offset + cur as u64 * stride)
                .with_context(|| format!("writing slot {cur}"))?;
            moved += stride;
            cur = src;
        }
        f.write_all_at(&saved, offset + cur as u64 * stride)
            .with_context(|| format!("writing slot {cur}"))?;
        moved += stride;
    }
    Ok(moved)
}

fn apply(model: &Path, layout_path: &Path, dry_run: bool) -> Result<()> {
    let g = Gguf::open(model)?;
    let moe = MoeModel::from_gguf(&g)?;
    let cur = current(model, &moe)?;
    let target = Layout::load(layout_path)?;

    if target.n_expert != moe.n_expert {
        bail!(
            "layout is for {} experts, model has {}",
            target.n_expert,
            moe.n_expert
        );
    }
    if !target.original_sha256.is_empty()
        && !cur.original_sha256.is_empty()
        && target.original_sha256 != cur.original_sha256
    {
        bail!(
            "layout was built for a model with sha256 {} but this file's original was {}",
            target.original_sha256,
            cur.original_sha256
        );
    }
    for l in &moe.layers {
        if !target.layers.contains_key(&l.index) {
            bail!("layout is missing MoE layer {}", l.index);
        }
    }

    let f = OpenOptions::new()
        .read(true)
        .write(!dry_run)
        .open(model)
        .with_context(|| format!("opening {} for writing", model.display()))?;

    let mut total_moved = 0u64;
    let mut changed_layers = 0usize;
    for l in &moe.layers {
        let c = cur.permutation(l.index)?;
        let t = target.permutation(l.index)?;
        // Slot s must end up holding expert t[s]; that expert currently sits at c.slot_of.
        let d: Vec<u16> = (0..moe.n_expert as usize)
            .map(|s| c.slot_of(t.expert_at(s)) as u16)
            .collect();
        if d.iter().enumerate().all(|(i, &v)| i as u16 == v) {
            continue;
        }
        changed_layers += 1;
        for (tensor, stride) in permutable_tensors(&moe, l.index)? {
            if dry_run {
                total_moved += stride * moe.n_expert as u64;
                continue;
            }
            total_moved += permute_tensor(&f, tensor.offset, stride, &d)
                .with_context(|| format!("permuting {}", tensor.name))?;
        }
        eprint!("\rlayer {} / {}", l.index + 1, moe.layers.len());
    }
    eprintln!();

    if dry_run {
        println!(
            "dry run: {changed_layers} layers would change, up to {:.2} GB rewritten",
            total_moved as f64 / 1e9
        );
        return Ok(());
    }

    f.sync_all()?;
    drop(f);

    let mut applied = target.clone();
    applied.original_sha256 = cur.original_sha256.clone();
    applied.save(sidecar_path(model))?;

    println!(
        "applied '{}': {changed_layers} layers changed, {:.2} GB rewritten",
        target.method,
        total_moved as f64 / 1e9
    );

    // A reordered file that no longer parses would be caught at load time anyway, but
    // catching it here means the sidecar and the file are never out of step.
    let reread = Gguf::open(model)?;
    MoeModel::from_gguf(&reread)?;
    println!(
        "re-parsed OK: {} tensors, {} bytes",
        reread.tensors.len(),
        reread.file_size
    );
    Ok(())
}

fn revert(model: &Path) -> Result<()> {
    let g = Gguf::open(model)?;
    let moe = MoeModel::from_gguf(&g)?;
    let cur = current(model, &moe)?;
    if cur.original_sha256.is_empty() {
        bail!("sidecar has no original sha256; cannot prove a revert is correct");
    }

    let indices: Vec<u32> = moe.layers.iter().map(|l| l.index).collect();
    let ident = Layout::identity(&cur.model, &cur.original_sha256, moe.n_expert, &indices);
    let tmp = std::env::temp_dir().join("ggufperm-identity.json");
    ident.save(&tmp)?;
    apply(model, &tmp, false)?;
    std::fs::remove_file(&tmp).ok();

    eprintln!("re-hashing {} …", model.display());
    let sha = file_sha256(model)?;
    if sha != cur.original_sha256 {
        bail!(
            "REVERT FAILED: file hashes to {sha} but the original was {}",
            cur.original_sha256
        );
    }
    println!("reverted and verified: sha256 {sha} matches the original");
    Ok(())
}

fn verify(model: &Path, hash: bool) -> Result<()> {
    let g = Gguf::open(model)?;
    let moe = MoeModel::from_gguf(&g)?;
    let cur = current(model, &moe)?;
    cur.validate()?;

    for l in &moe.layers {
        let p: Permutation = cur.permutation(l.index)?;
        if p.len() != moe.n_expert as usize {
            bail!("layer {} permutation has the wrong length", l.index);
        }
        for (t, stride) in permutable_tensors(&moe, l.index)? {
            if stride * moe.n_expert as u64 != t.nbytes {
                bail!(
                    "tensor {} does not divide evenly into {} experts",
                    t.name,
                    moe.n_expert
                );
            }
        }
    }
    println!(
        "structure OK: {} MoE layers, {} experts",
        moe.layers.len(),
        moe.n_expert
    );

    if hash {
        let sha = file_sha256(model)?;
        let matches = sha == cur.original_sha256;
        println!("sha256           {sha}");
        println!("matches original {matches}");
        if cur.is_identity() && !matches {
            bail!("sidecar claims the shipped order but the bytes differ from the original");
        }
        if !cur.is_identity() && matches {
            bail!(
                "sidecar claims layout '{}' but the bytes are the original",
                cur.method
            );
        }
    }
    Ok(())
}
