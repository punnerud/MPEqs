//! Capture the real top-k router decisions of a MoE GGUF model.
//!
//! Output is a compact binary trace consumed by `coact` and `fetchbench`.

use anyhow::{bail, Context, Result};
use clap::Parser;
use gguf::{moe::MoeModel, Gguf};
use std::ffi::CString;
use std::path::PathBuf;

#[allow(non_snake_case)]
extern "C" {
    fn moetrace_run(
        model_path: *const std::os::raw::c_char,
        text_path: *const std::os::raw::c_char,
        out_path: *const std::os::raw::c_char,
        n_gpu_layers: i32,
        n_ctx: i32,
        n_ubatch: i32,
        n_threads: i32,
        n_layer: i32,
        n_expert: i32,
        top_k: i32,
        max_tokens: i32,
        chunk_tokens: i32,
        want_contrib: i32,
        vec_path: *const std::os::raw::c_char,
        vec_budget: i32,
    ) -> i64;
}

#[derive(Parser)]
#[command(about = "Capture MoE router decisions (ffn_moe_topk) from a GGUF model")]
struct Args {
    /// Path to the GGUF model.
    #[arg(short, long)]
    model: PathBuf,
    /// UTF-8 text file to run through the model.
    #[arg(short, long)]
    text: PathBuf,
    /// Output trace path.
    #[arg(short, long)]
    out: PathBuf,
    /// Layers to offload to the GPU. Use 0 if the Metal backend swallows the callback.
    #[arg(long, default_value_t = 999)]
    ngl: i32,
    #[arg(long, default_value_t = 4096)]
    n_ctx: i32,
    #[arg(long, default_value_t = 512)]
    ubatch: i32,
    #[arg(long, default_value_t = 5)]
    threads: i32,
    /// Stop after this many tokens (0 = whole file).
    #[arg(long, default_value_t = 0)]
    max_tokens: i32,
    /// Reset the KV cache every N tokens so routing reflects ordinary-length contexts.
    #[arg(long, default_value_t = 1024)]
    chunk: i32,
    /// Also record the L2 norm of each expert's output, so contribution can be separated
    /// from the router's gate weight. Copies the full MoE output off the device per layer,
    /// so use it with a few thousand tokens, not the whole corpus.
    #[arg(long)]
    contrib: bool,
    /// Also stream the raw per-expert output vectors here, as f16. Implies --contrib.
    /// Lets truncation and reweighting schemes be evaluated exactly offline rather than
    /// inferred from norms. ~64 KiB per token per layer, so pair it with --vec-tokens.
    #[arg(long)]
    vecs: Option<PathBuf>,
    /// Stop writing vectors after this many tokens (0 = no limit).
    #[arg(long, default_value_t = 512)]
    vec_tokens: i32,
}

fn cstr(p: &std::path::Path) -> Result<CString> {
    CString::new(p.as_os_str().to_string_lossy().as_bytes().to_vec())
        .with_context(|| format!("path {} contains a NUL byte", p.display()))
}

fn main() -> Result<()> {
    let args = Args::parse();

    let g = Gguf::open(&args.model)?;
    let moe = MoeModel::from_gguf(&g)?;
    let n_layer = g.arch_u64("block_count")? as i32;

    eprintln!(
        "moetrace: {} — {} MoE layers of {}, {} experts, top-{}",
        moe.arch,
        moe.layers.len(),
        n_layer,
        moe.n_expert,
        moe.n_expert_used
    );

    let (m, t, o) = (cstr(&args.model)?, cstr(&args.text)?, cstr(&args.out)?);
    let vec_c = args.vecs.as_deref().map(cstr).transpose()?;
    // The vector stream and the record stream are written from the same loop, so they are
    // only aligned 1:1 if both stop at the same token. Tie them together rather than leaving
    // a silent offset for the analysis to trip over.
    let effective_max_tokens = if args.vecs.is_some() {
        if args.max_tokens > 0 {
            args.max_tokens.min(args.vec_tokens)
        } else {
            args.vec_tokens
        }
    } else {
        args.max_tokens
    };
    if args.vecs.is_some() && effective_max_tokens != args.max_tokens {
        eprintln!("moetrace: capping --max-tokens to {effective_max_tokens} to match --vec-tokens");
    }
    let n = unsafe {
        moetrace_run(
            m.as_ptr(),
            t.as_ptr(),
            o.as_ptr(),
            args.ngl,
            args.n_ctx,
            args.ubatch,
            args.threads,
            n_layer,
            moe.n_expert as i32,
            moe.n_expert_used as i32,
            effective_max_tokens,
            args.chunk,
            (args.contrib || args.vecs.is_some()) as i32,
            vec_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            effective_max_tokens,
        )
    };

    if n < 0 {
        bail!("moetrace_run failed with code {n}");
    }
    if n == 0 {
        bail!(
            "captured 0 records — the eval callback never saw an '{}' tensor. Retry with --ngl 0.",
            "ffn_moe_topk"
        );
    }
    eprintln!(
        "moetrace: wrote {n} records ({} tokens x {} MoE layers) to {}",
        n as usize / moe.layers.len().max(1),
        moe.layers.len(),
        args.out.display()
    );
    Ok(())
}
