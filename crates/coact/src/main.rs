//! Build candidate expert layouts from a router trace and score them on held-out tokens.

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use gguf::{moe::MoeModel, Gguf};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use clap::ValueEnum;
use coact::cluster::{self, CoGraph, EdgeWeight};
use coact::cost::{evaluate, CostModel, LayerGeometry, LayerStats, Permutation};
use coact::layout::Layout;
use coact::trace::Trace;
use coact::{file_sha256, sidecar_path};

#[derive(Parser)]
#[command(about = "Co-activation analysis and expert layout search")]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print the MoE geometry of a model: expert sizes, per-token bytes and range count.
    Stats {
        #[arg(short, long)]
        model: PathBuf,
    },
    /// Search for layouts and score them against the shipped order.
    Build {
        #[arg(short, long)]
        model: PathBuf,
        #[arg(short, long)]
        trace: PathBuf,
        #[arg(long, default_value = "data/layouts")]
        outdir: PathBuf,
        #[arg(long, default_value = "data/layout-report.json")]
        report: PathBuf,
        /// Measured cost model from `fetchbench calibrate`.
        #[arg(long)]
        cost: Option<PathBuf>,
        /// Every Nth token is held out from the search and used for scoring.
        #[arg(long, default_value_t = 5)]
        holdout_every: usize,
        /// Greedy local-search sweeps after spectral bisection.
        #[arg(long, default_value_t = 3)]
        sweeps: usize,
        /// Cap on training tokens used by the local search (0 = all). Scoring always uses
        /// the full holdout regardless.
        #[arg(long, default_value_t = 20000)]
        search_tokens: usize,
        /// Number of random baselines; without these a gain is unattributable.
        #[arg(long, default_value_t = 3)]
        random_seeds: u64,
        /// Co-activation edge weight: plain co-selection counts, or the product of the two
        /// router gate probabilities (needs a version-2 trace).
        #[arg(long, value_enum, default_value_t = EdgeWeightArg::Count)]
        edge_weight: EdgeWeightArg,
        /// Sharpen the co-activation graph by `w' = w · lift^alpha` before the search, the
        /// analysis-side stand-in for reinforcing routes that worked. Scored on the unchanged
        /// holdout, so a gain means it denoised and a loss means it distorted.
        #[arg(long, default_value_t = 0.0)]
        lift_power: f64,
        /// Skip hashing the model (hours of CI time saved, safety lost).
        #[arg(long)]
        no_hash: bool,
    },
    /// Ceiling on any router rerank: gate order versus contribution order.
    Headroom {
        #[arg(short, long)]
        trace: PathBuf,
        #[arg(long, default_value = "data/headroom.json")]
        out: PathBuf,
    },
    /// Split the damage from fetching fewer experts into scale error and information loss.
    Reweight {
        #[arg(short, long)]
        trace: PathBuf,
        /// Vector stream from `moetrace --vecs`.
        #[arg(short, long)]
        vecs: PathBuf,
        #[arg(long, default_value = "data/reweight.json")]
        out: PathBuf,
    },
    /// Project per-token fetch cost for a model too large to hold, from its GGUF header.
    Project {
        /// Header bytes fetched with an HTTP range request (a few MB is plenty).
        #[arg(long, num_args = 1..)]
        header: Vec<PathBuf>,
        /// True size of each full file, in bytes, in the same order as --header.
        #[arg(long, num_args = 1.., value_delimiter = ',')]
        size: Vec<u64>,
        #[arg(long)]
        cost: Option<PathBuf>,
        /// Fraction of fetches a layout removes. Default is the value measured on OLMoE.
        #[arg(long, default_value_t = 0.144)]
        fetch_reduction: f64,
        #[arg(long, default_value = "data/projection.json")]
        out: PathBuf,
    },
    /// Do the experts vote together or correct each other? Needs `moetrace --vecs`.
    Ensemble {
        #[arg(short, long)]
        trace: PathBuf,
        #[arg(short, long)]
        vecs: PathBuf,
        #[arg(long, default_value = "data/ensemble.json")]
        out: PathBuf,
    },
    /// Where the contribution actually sits, and whether layers predict each other.
    Analyze {
        #[arg(short, long)]
        trace: PathBuf,
        #[arg(long, default_value = "data/analysis.json")]
        out: PathBuf,
    },
    /// Check that a permuted model routes to the same experts as the shipped one.
    ///
    /// This is a stronger correctness oracle than comparing logits. Layer 0's routing
    /// depends only on the embedding and the router weights, so if the permutation is
    /// implemented correctly it must agree there exactly — no floating-point excuse
    /// available. Deeper layers are allowed to drift, and how fast they drift measures how
    /// much float non-associativity the reordering injects.
    Compare {
        /// Trace captured from the model in its shipped order.
        #[arg(long)]
        base: PathBuf,
        /// Trace captured after applying `--layout`.
        #[arg(long)]
        permuted: PathBuf,
        #[arg(long)]
        layout: PathBuf,
        #[arg(long, default_value = "data/routing-agreement.json")]
        out: PathBuf,
    },
}

#[derive(Serialize)]
struct LayerAgreement {
    layer: u32,
    n_tokens: usize,
    /// Fraction of tokens selecting the same set of experts.
    set_match_pct: f64,
    /// Fraction selecting the same experts in the same top-k rank order.
    rank_match_pct: f64,
}

#[derive(Serialize)]
struct AgreementReport {
    layout: String,
    n_layers: usize,
    first_moe_layer_rank_match_pct: f64,
    overall_set_match_pct: f64,
    overall_rank_match_pct: f64,
    per_layer: Vec<LayerAgreement>,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, ValueEnum)]
enum EdgeWeightArg {
    Count,
    Gate,
}

impl From<EdgeWeightArg> for EdgeWeight {
    fn from(v: EdgeWeightArg) -> Self {
        match v {
            EdgeWeightArg::Count => EdgeWeight::Count,
            EdgeWeightArg::Gate => EdgeWeight::Gate,
        }
    }
}

/// Average router gate weight at each top-k rank, and the cumulative mass up to it.
///
/// This is the headroom for anything that trades quality for locality: if rank 8 carries
/// 2 % of the mass, a cache-aware router that declines to fetch it is giving up little. It
/// is reported but not acted on — every layout in this repo is selection-preserving.
#[derive(Serialize)]
struct GateMass {
    rank: usize,
    mean_weight: f64,
    cumulative_pct: f64,
}

#[derive(Serialize)]
struct MethodReport {
    method: String,
    /// Scored on the holdout split.
    fetches_per_token: f64,
    bytes_per_token: f64,
    cost_ns_per_token: f64,
    /// Relative to the shipped `identity` order; positive means faster.
    cost_reduction_pct: f64,
    per_layer: BTreeMap<u32, LayerStats>,
}

/// How much the best layout wins at a hypothetical per-fetch cost.
///
/// The measured `C_fetch` on local NVMe is a property of *this* transport, not of the idea.
/// The same layout feeds a page-fault path, a PCIe transfer, or network storage, where a
/// request costs one to three orders of magnitude more. Reporting the whole curve is the
/// difference between "no gain here" and "no gain anywhere", and only the first is true.
#[derive(Serialize)]
struct SweepPoint {
    c_fetch_us: f64,
    /// Experts worth reading through to merge two runs, at this cost.
    max_bridged_gap: u64,
    /// Best non-identity method at this point, and its reduction against the shipped order.
    best_method: String,
    best_reduction_pct: f64,
    /// Same for the random baseline, so the noise floor is always visible.
    random_reduction_pct: f64,
}

#[derive(Serialize)]
struct Report {
    model: String,
    arch: String,
    n_expert: u32,
    n_expert_used: u32,
    n_moe_layers: usize,
    bytes_per_expert: u64,
    max_bridged_gap: u64,
    cost_model: CostModel,
    train_tokens: usize,
    holdout_tokens: usize,
    methods: Vec<MethodReport>,
    cfetch_sweep: Vec<SweepPoint>,
    edge_weight: String,
    gate_mass_by_rank: Vec<GateMass>,
}

fn geometry(moe: &MoeModel, layer: u32) -> Result<LayerGeometry> {
    let l = moe.layer(layer)?;
    Ok(LayerGeometry {
        n_tensors: l.weight_tensors().count(),
        bytes_per_expert: l.bytes_per_expert()?,
    })
}

fn main() -> Result<()> {
    match Args::parse().cmd {
        Cmd::Stats { model } => stats(&model),
        Cmd::Build {
            model,
            trace,
            outdir,
            report,
            cost,
            holdout_every,
            sweeps,
            search_tokens,
            random_seeds,
            edge_weight,
            lift_power,
            no_hash,
        } => build(BuildArgs {
            model,
            trace,
            outdir,
            report,
            cost,
            holdout_every,
            sweeps,
            search_tokens,
            random_seeds,
            edge_weight,
            lift_power,
            no_hash,
        }),
        Cmd::Headroom { trace, out } => headroom(&trace, &out),
        Cmd::Reweight { trace, vecs, out } => reweight(&trace, &vecs, &out),
        Cmd::Ensemble { trace, vecs, out } => ensemble(&trace, &vecs, &out),
        Cmd::Project {
            header,
            size,
            cost,
            fetch_reduction,
            out,
        } => project(&header, &size, cost.as_deref(), fetch_reduction, &out),
        Cmd::Analyze { trace, out } => analyze(&trace, &out),
        Cmd::Compare {
            base,
            permuted,
            layout,
            out,
        } => compare(&base, &permuted, &layout, &out),
    }
}

fn reweight(trace: &Path, vecs: &Path, out: &Path) -> Result<()> {
    let (rows, n, per_layer) = coact::reweight::evaluate(trace, vecs)?;
    println!("relative L2 error of the FFN output against the untruncated sum, over {n} records:");
    println!(
        "{:>6} {:>11} {:>13} {:>12} {:>15} {:>17}",
        "keep", "truncate", "best scalar", "oracle", "best weights", "oracle+weights"
    );
    for r in &rows {
        println!(
            "{:>6} {:>10.2}% {:>12.2}% {:>11.2}% {:>14.2}% {:>16.2}%",
            r.keep,
            r.truncate_pct,
            r.best_scalar_pct,
            r.oracle_pct,
            r.best_weights_pct,
            r.oracle_best_weights_pct
        );
    }
    println!("\n'best weights' solves for the optimal per-expert weights over the kept set — the");
    println!("split and the combine tuned together. No learned combiner can beat it, because it");
    println!("is fitted against the very output it is trying to reproduce. That makes it the");
    println!("ceiling for a meta-model that overrides both selection and merging.");

    // Per-layer sensitivity, and whether a non-uniform budget beats a uniform top-k.
    let top_k = rows.len();
    let n_layers = per_layer.len();
    println!("\nper-layer truncation error at each keep depth (%):");
    print!("{:>6}", "layer");
    for k in 1..=top_k {
        print!(" {k:>6}");
    }
    println!();
    for (l, errs) in &per_layer {
        print!("{l:>6}");
        for e in errs {
            print!(" {:>5.1}%", 100.0 * e);
        }
        println!();
    }

    println!("\nspending the same expert budget non-uniformly across layers:");
    println!(
        "{:>12} {:>14} {:>16} {:>10}",
        "budget", "uniform err", "allocated err", "gain"
    );
    let mut plans = Vec::new();
    for keep in 2..top_k {
        let plan = coact::reweight::allocate(&per_layer, keep * n_layers);
        println!(
            "{:>12} {:>13.2}% {:>15.2}% {:>9.2}pp",
            format!("{keep}/layer"),
            plan.uniform_mean_error_pct,
            plan.allocated_mean_error_pct,
            plan.uniform_mean_error_pct - plan.allocated_mean_error_pct
        );
        plans.push(plan);
    }
    println!(
        "\nA gain near zero means the layers are equally sensitive and uniform top-k is already"
    );
    println!("the right allocation — there is no cheap layer to take experts away from.");

    #[derive(Serialize)]
    struct ReweightReport {
        policies: Vec<coact::reweight::PolicyError>,
        budget_plans: Vec<coact::reweight::BudgetPlan>,
    }
    if let Some(d) = out.parent() {
        std::fs::create_dir_all(d).ok();
    }
    std::fs::write(
        out,
        serde_json::to_string_pretty(&ReweightReport {
            policies: rows,
            budget_plans: plans,
        })?,
    )?;
    println!("\nwrote {}", out.display());
    Ok(())
}

fn project(
    headers: &[PathBuf],
    sizes: &[u64],
    cost: Option<&Path>,
    fetch_reduction: f64,
    out: &Path,
) -> Result<()> {
    if headers.len() != sizes.len() {
        bail!("{} headers but {} sizes", headers.len(), sizes.len());
    }
    let cm: CostModel = match cost {
        Some(p) => serde_json::from_str(&std::fs::read_to_string(p)?)?,
        None => {
            eprintln!(
                "warning: no --cost given; using constants from another machine. Run \
                 `fetchbench calibrate` before believing any timing derived from this."
            );
            CostModel::assumed_apple_nvme()
        }
    };

    let mut rows = Vec::new();
    for (h, &sz) in headers.iter().zip(sizes) {
        let bytes = std::fs::read(h)?;
        let name = h
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        match Gguf::from_header(&bytes, sz, &name) {
            Ok(g) => match coact::project::project(&g, &name, &cm, fetch_reduction) {
                Ok(p) => rows.push(p),
                Err(e) => eprintln!("{name}: {e}"),
            },
            Err(e) => eprintln!("{name}: {e}"),
        }
    }
    if rows.is_empty() {
        bail!("no headers could be parsed");
    }

    println!(
        "device: C_fetch = {:.2} us, C_byte = {:.4} ns/B. Layout assumed to remove {:.1}% of fetches.\n",
        cm.c_fetch_ns / 1000.0,
        cm.c_byte_ns,
        100.0 * fetch_reduction
    );
    println!(
        "{:<30} {:>7} {:>8} {:>7} {:>8} {:>10} {:>10} {:>9} {:>9}",
        "model", "GB", "ranges", "MiB/t", "rng/MiB", "overhead", "ms/t base", "speedup", "perfect"
    );
    for r in &rows {
        println!(
            "{:<34} {:>7.1} {:>8} {:>7.0} {:>9.1}% {:>10.1} {:>10.1} {:>8.3}x {:>8.3}x{}",
            if r.model.len() > 34 {
                &r.model[..34]
            } else {
                &r.model
            },
            r.file_gb,
            r.ranges_per_token,
            r.mib_per_token,
            r.fetch_overhead_share_pct,
            r.ms_per_token_shipped,
            r.ms_per_token_optimised,
            r.speedup,
            r.perfect_speedup,
            if r.extrapolated_from_shard { "  *" } else { "" }
        );
    }
    if rows.iter().any(|r| r.extrapolated_from_shard) {
        println!(
            "\n* header covers one shard only; per-token figures scaled by block_count / layers seen"
        );
    }
    println!("\n'overhead' is the share of per-token fetch cost that is per-request rather than");
    println!("transfer. It is the entire budget layout can compete for. 'perfect' is the ceiling:");
    println!("one contiguous run per layer, which no real routing distribution will reach.");

    if let Some(d) = out.parent() {
        std::fs::create_dir_all(d).ok();
    }
    std::fs::write(out, serde_json::to_string_pretty(&rows)?)?;
    println!("\nwrote {}", out.display());
    Ok(())
}

fn ensemble(trace: &Path, vecs: &Path, out: &Path) -> Result<()> {
    let e = coact::reweight::ensemble(trace, vecs)?;
    println!(
        "expert agreement inside one MoE layer, over {} records:\n",
        e.n_records
    );
    println!(
        "{:>6} {:>18} {:>20} {:>16}",
        "rank", "cos with total", "cos with the rest", "share of total"
    );
    for r in &e.per_rank {
        println!(
            "{:>6} {:>18.4} {:>20.4} {:>15.2}%",
            r.rank, r.cosine_with_total, r.mean_pairwise_cosine, r.projection_share_pct
        );
    }
    println!(
        "\nalignment ratio ||sum|| / sum||.|| = {:.4}  (orthogonal would be {:.4}, a unanimous \
         vote 1.0)",
        e.alignment_ratio, e.orthogonal_reference
    );
    if e.alignment_ratio > 0.75 {
        println!("The layer behaves like a vote: the experts largely agree and reinforce.");
    } else if e.alignment_ratio > e.orthogonal_reference * 1.15 {
        println!("Partly a vote, partly complementary features.");
    } else {
        println!(
            "Not a vote. The experts are near-orthogonal or cancelling, so each one carries \
             its own direction and none of them is redundant."
        );
    }
    if let Some(d) = out.parent() {
        std::fs::create_dir_all(d).ok();
    }
    std::fs::write(out, serde_json::to_string_pretty(&e)?)?;
    println!("\nwrote {}", out.display());
    Ok(())
}

fn headroom(trace: &PathBuf, out: &PathBuf) -> Result<()> {
    let tr = Trace::load(trace)?;
    let (rows, per_layer) = coact::analyze::headroom(&tr);
    if rows.is_empty() {
        bail!("this trace has no output norms — recapture with `moetrace --contrib`");
    }

    println!("contribution captured when only the first `keep` experts are fetched:");
    println!(
        "{:>6} {:>10} {:>14} {:>14} {:>12} {:>12}",
        "keep", "fetch cut", "by gate", "by oracle", "headroom", "overlap"
    );
    let k = tr.top_k;
    for r in &rows {
        println!(
            "{:>6} {:>9.0}% {:>13.2}% {:>13.2}% {:>11.2}pp {:>11.1}%",
            r.keep,
            100.0 * (k - r.keep) as f64 / k as f64,
            r.gate_capture_pct,
            r.oracle_capture_pct,
            r.headroom_pct,
            r.selection_agreement_pct
        );
    }

    let worst = rows
        .iter()
        .max_by(|a, b| a.headroom_pct.partial_cmp(&b.headroom_pct).unwrap())
        .unwrap();
    println!(
        "\nlargest headroom: {:.2} percentage points at keep={} — that is the ceiling for any",
        worst.headroom_pct, worst.keep
    );
    println!("reranker, including a retrained router, since it assumes perfect knowledge.");

    let mean_sp = per_layer.iter().map(|r| r.spearman).sum::<f64>() / per_layer.len() as f64;
    let mean_am = per_layer
        .iter()
        .map(|r| r.argmax_agreement_pct)
        .sum::<f64>()
        / per_layer.len() as f64;
    println!("\ngate order vs contribution order:");
    println!(
        "{:>6} {:>12} {:>18}",
        "layer", "spearman", "top-1 agreement"
    );
    for r in &per_layer {
        println!(
            "{:>6} {:>12.4} {:>17.2}%",
            r.layer, r.spearman, r.argmax_agreement_pct
        );
    }
    println!("\nmean spearman {mean_sp:.4}, mean top-1 agreement {mean_am:.2}%");

    #[derive(Serialize)]
    struct HeadroomReport {
        top_k: usize,
        by_keep: Vec<coact::analyze::Headroom>,
        per_layer: Vec<coact::analyze::RankAgreement>,
        mean_spearman: f64,
        mean_argmax_agreement_pct: f64,
    }
    let report = HeadroomReport {
        top_k: k,
        by_keep: rows,
        per_layer,
        mean_spearman: mean_sp,
        mean_argmax_agreement_pct: mean_am,
    };
    if let Some(d) = out.parent() {
        std::fs::create_dir_all(d).ok();
    }
    std::fs::write(out, serde_json::to_string_pretty(&report)?)?;
    println!("\nwrote {}", out.display());
    Ok(())
}

fn analyze(trace: &PathBuf, out: &PathBuf) -> Result<()> {
    let tr = Trace::load(trace)?;

    let contrib = coact::analyze::rank_contributions(&tr);
    if contrib.is_empty() {
        println!("no output norms in this trace — recapture with `moetrace --contrib`");
    } else {
        println!("contribution by top-k rank (gate weight x output norm, pooled over layers):");
        println!(
            "{:>6} {:>13} {:>13} {:>14} {:>13}",
            "rank", "gate weight", "output norm", "contribution", "cumulative"
        );
        for c in &contrib {
            println!(
                "{:>6} {:>13.4} {:>13.3} {:>13.2}% {:>12.2}%",
                c.rank,
                c.mean_gate_weight,
                c.mean_output_norm,
                c.contribution_pct,
                c.cumulative_pct
            );
        }
        let last = contrib.last().unwrap();
        println!(
            "\nthe lowest-ranked expert carries {:.2}% of the routed contribution",
            last.contribution_pct
        );
    }

    let cl = coact::analyze::clusters(&tr);
    if !cl.is_empty() {
        println!("\nco-activation lift (observed pairs / independence):");
        println!(
            "{:>6} {:>9} {:>9} {:>9} {:>9} {:>12} {:>14}",
            "layer", "p50", "p90", "p99", "max", ">=2x pairs", "top-n mean"
        );
        for c in &cl {
            println!(
                "{:>6} {:>9.3} {:>9.3} {:>9.3} {:>9.2} {:>11.2}% {:>14.3}",
                c.layer,
                c.lift_p50,
                c.lift_p90,
                c.lift_p99,
                c.lift_max,
                c.pairs_above_2x_pct,
                c.top_pairs_mean_lift
            );
        }
        let m = cl.iter().map(|c| c.top_pairs_mean_lift).sum::<f64>() / cl.len() as f64;
        println!(
            "\nA linear layout can make about n pairs adjacent, so the last column is what is"
        );
        println!("actually exploitable: mean lift {m:.2} over the strongest n pairs. Lift near 1");
        println!("would mean the router selects independently and no grouping exists to find.");
    }

    let xl = coact::analyze::cross_layer(&tr);
    if !xl.is_empty() {
        println!("\ncross-layer predictability of the top-1 expert:");
        println!(
            "{:>10} {:>10} {:>12} {:>12} {:>14}",
            "layers", "H (bits)", "H|prev", "MI (bits)", "best guess"
        );
        for c in &xl {
            println!(
                "{:>4} -> {:<3} {:>10.3} {:>12.3} {:>12.4} {:>13.2}%",
                c.from_layer,
                c.to_layer,
                c.entropy_bits,
                c.conditional_entropy_bits,
                c.mutual_information_bits,
                c.top1_predictability_pct
            );
        }
        let mean_mi = xl.iter().map(|c| c.mutual_information_bits).sum::<f64>() / xl.len() as f64;
        let mean_h = xl.iter().map(|c| c.entropy_bits).sum::<f64>() / xl.len() as f64;
        println!(
            "\nmean MI {mean_mi:.4} bits against mean entropy {mean_h:.3} bits: knowing the \
             previous layer removes {:.1}% of the uncertainty",
            100.0 * mean_mi / mean_h
        );
    }

    #[derive(Serialize)]
    struct AnalysisReport {
        rank_contributions: Vec<coact::analyze::RankContribution>,
        cross_layer: Vec<coact::analyze::CrossLayer>,
    }
    let report = AnalysisReport {
        rank_contributions: contrib,
        cross_layer: xl,
    };
    if let Some(d) = out.parent() {
        std::fs::create_dir_all(d).ok();
    }
    std::fs::write(out, serde_json::to_string_pretty(&report)?)?;
    println!("\nwrote {}", out.display());
    Ok(())
}

fn compare(base: &PathBuf, permuted: &PathBuf, layout: &PathBuf, out: &PathBuf) -> Result<()> {
    let a = Trace::load(base)?;
    let b = Trace::load(permuted)?;
    let lay = Layout::load(layout)?;

    if a.top_k != b.top_k || a.n_expert != b.n_expert {
        bail!("traces disagree on topology");
    }

    let mut per_layer = Vec::new();
    let (mut tot, mut tot_set, mut tot_rank) = (0usize, 0usize, 0usize);
    for (la, lb) in a.layers.iter().zip(&b.layers) {
        if la.layer != lb.layer {
            bail!("traces have different layer sets");
        }
        let n = la.n_tokens().min(lb.n_tokens());
        if n == 0 {
            continue;
        }
        let perm = lay.permutation(la.layer as u32)?;
        let (mut set_ok, mut rank_ok) = (0usize, 0usize);
        let mut sa = Vec::with_capacity(a.top_k);
        let mut sb = Vec::with_capacity(a.top_k);
        for t in 0..n {
            let orig = la.token(t);
            // The permuted run reports slots; map each back to the expert it now holds.
            let mapped: Vec<u16> = lb
                .token(t)
                .iter()
                .map(|&s| perm.expert_at(s as usize))
                .collect();
            if mapped == orig {
                rank_ok += 1;
                set_ok += 1;
                continue;
            }
            sa.clear();
            sa.extend_from_slice(orig);
            sa.sort_unstable();
            sb.clone_from(&mapped);
            sb.sort_unstable();
            if sa == sb {
                set_ok += 1;
            }
        }
        tot += n;
        tot_set += set_ok;
        tot_rank += rank_ok;
        per_layer.push(LayerAgreement {
            layer: la.layer as u32,
            n_tokens: n,
            set_match_pct: 100.0 * set_ok as f64 / n as f64,
            rank_match_pct: 100.0 * rank_ok as f64 / n as f64,
        });
    }
    if per_layer.is_empty() {
        bail!("no comparable layers");
    }

    println!(
        "{:>6} {:>10} {:>12} {:>12}",
        "layer", "tokens", "set match", "rank match"
    );
    for l in &per_layer {
        println!(
            "{:>6} {:>10} {:>11.3}% {:>11.3}%",
            l.layer, l.n_tokens, l.set_match_pct, l.rank_match_pct
        );
    }
    let first = per_layer[0].rank_match_pct;
    println!(
        "\noverall: set {:.4}%  exact-rank {:.4}%  over {tot} token-layers",
        100.0 * tot_set as f64 / tot as f64,
        100.0 * tot_rank as f64 / tot as f64
    );
    println!("first MoE layer exact-rank: {first:.4}%");
    if first < 100.0 {
        println!(
            "\nFAIL: the first MoE layer must match exactly. Its routing depends only on the \
             embedding and the router weights, so a mismatch means the permutation is wrong, \
             not that floats drifted."
        );
    } else {
        println!(
            "\nPASS: the permutation is exact. Deeper-layer drift is float non-associativity: \
             llama.cpp's router softmax normalises over experts in storage order, so reordering \
             them changes the last bits of every gate weight."
        );
    }

    let report = AgreementReport {
        layout: lay.method.clone(),
        n_layers: per_layer.len(),
        first_moe_layer_rank_match_pct: first,
        overall_set_match_pct: 100.0 * tot_set as f64 / tot as f64,
        overall_rank_match_pct: 100.0 * tot_rank as f64 / tot as f64,
        per_layer,
    };
    if let Some(d) = out.parent() {
        std::fs::create_dir_all(d).ok();
    }
    std::fs::write(out, serde_json::to_string_pretty(&report)?)?;
    println!("wrote {}", out.display());
    if first < 100.0 {
        bail!("permutation is not routing-exact at the first MoE layer");
    }
    Ok(())
}

fn stats(model: &PathBuf) -> Result<()> {
    let g = Gguf::open(model)?;
    let moe = MoeModel::from_gguf(&g)?;

    println!("model            {}", model.display());
    println!("arch             {}", moe.arch);
    println!("file size        {:.2} GB", g.file_size as f64 / 1e9);
    println!("MoE layers       {}", moe.layers.len());
    println!("experts / layer  {}", moe.n_expert);
    println!("experts / token  {}", moe.n_expert_used);

    let l0 = &moe.layers[0];
    println!("\nlayer {} expert tensors:", l0.index);
    for t in l0.weight_tensors() {
        println!(
            "  {:<32} {:>6} {:?}  {:>13} B/expert",
            t.name,
            t.ty.name,
            t.dims,
            t.last_axis_stride()?
        );
    }
    println!("  bytes / expert   {}", l0.bytes_per_expert()?);

    let eb = moe.expert_bytes()?;
    println!(
        "\nexpert weights   {:.2} GB ({:.1} % of file)",
        eb as f64 / 1e9,
        100.0 * eb as f64 / g.file_size as f64
    );
    println!(
        "per decode token {:.1} MiB in {} disjoint ranges",
        moe.bytes_per_token()? as f64 / (1 << 20) as f64,
        moe.ranges_per_token()
    );
    Ok(())
}

struct BuildArgs {
    model: PathBuf,
    trace: PathBuf,
    outdir: PathBuf,
    report: PathBuf,
    cost: Option<PathBuf>,
    holdout_every: usize,
    sweeps: usize,
    search_tokens: usize,
    random_seeds: u64,
    edge_weight: EdgeWeightArg,
    lift_power: f64,
    no_hash: bool,
}

fn build(a: BuildArgs) -> Result<()> {
    let g = Gguf::open(&a.model)?;
    let moe = MoeModel::from_gguf(&g)?;
    let tr = Trace::load(&a.trace)?;

    if tr.n_expert != moe.n_expert || tr.top_k != moe.n_expert_used as usize {
        bail!(
            "trace describes {} experts top-{} but the model has {} top-{}",
            tr.n_expert,
            tr.top_k,
            moe.n_expert,
            moe.n_expert_used
        );
    }

    let cm = match &a.cost {
        Some(p) => serde_json::from_str(&std::fs::read_to_string(p)?)?,
        None => {
            eprintln!(
                "warning: no --cost given. Falling back to constants measured on a different \
                 machine; every number below is unfounded until `fetchbench calibrate` has run."
            );
            CostModel::assumed_apple_nvme()
        }
    };

    // The model may not be in its shipped state; the sidecar records what is applied.
    let original_sha = if a.no_hash {
        String::new()
    } else {
        let sc = sidecar_path(&a.model);
        if sc.exists() {
            Layout::load(&sc)?.original_sha256
        } else {
            eprintln!("hashing {} …", a.model.display());
            file_sha256(&a.model)?
        }
    };

    let geom0 = geometry(&moe, moe.layers[0].index)?;
    eprintln!(
        "cost model: C_fetch = {:.1} us, C_byte = {:.4} ns/B -> bridge gaps up to {} experts",
        cm.c_fetch_ns / 1000.0,
        cm.c_byte_ns,
        cm.max_gap(geom0.bytes_per_expert, geom0.n_tensors)
    );

    // Split once per layer, then run every method against the same split.
    let mut splits = BTreeMap::new();
    for lt in &tr.layers {
        splits.insert(lt.layer as u32, lt.split(a.holdout_every));
    }

    let mut methods: Vec<(String, BTreeMap<u32, Permutation>)> = Vec::new();
    methods.push((
        "identity".into(),
        moe.layers
            .iter()
            .map(|l| (l.index, Permutation::identity(moe.n_expert as usize)))
            .collect(),
    ));
    for seed in 1..=a.random_seeds {
        methods.push((
            format!("random:{seed}"),
            moe.layers
                .iter()
                .map(|l| {
                    let p =
                        cluster::random_order(moe.n_expert as usize, seed * 1000 + l.index as u64);
                    (l.index, Permutation::from_perm(p).unwrap())
                })
                .collect(),
        ));
    }

    let mut freq = BTreeMap::new();
    let mut mincut = BTreeMap::new();
    let mut chain = BTreeMap::new();
    let mut greedy = BTreeMap::new();
    for l in &moe.layers {
        let (train, _) = splits
            .get(&l.index)
            .ok_or_else(|| anyhow::anyhow!("trace has no records for MoE layer {}", l.index))?;
        let mut graph = CoGraph::build_with(train, moe.n_expert as usize, a.edge_weight.into());
        graph.sharpen(a.lift_power);
        let geom = geometry(&moe, l.index)?;
        let search = train.subsample(a.search_tokens);

        freq.insert(
            l.index,
            Permutation::from_perm(cluster::frequency_order(&graph))?,
        );
        let sp = Permutation::from_perm(cluster::spectral_order(&graph))?;
        mincut.insert(l.index, sp.clone());
        let ch = Permutation::from_perm(cluster::chain_order(&graph))?;
        chain.insert(l.index, ch.clone());

        // Refine from whichever construction is already ahead on the training split.
        let seed = l.index as u64 + 1;
        let start = if evaluate(search.tokens(), &ch, geom, &cm).cost_ns
            < evaluate(search.tokens(), &sp, geom, &cm).cost_ns
        {
            ch
        } else {
            sp
        };
        let (refined, _) = cluster::greedy_refine(start, &search, geom, &cm, a.sweeps, seed);
        greedy.insert(l.index, refined);
        eprint!("\rlayer {} / {} searched", l.index + 1, moe.layers.len());
    }
    eprintln!();
    methods.push(("frequency".into(), freq));
    methods.push(("mincut".into(), mincut));
    methods.push(("chain".into(), chain));
    methods.push(("best+greedy".into(), greedy));

    std::fs::create_dir_all(&a.outdir)?;
    let model_name = a
        .model
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();

    let mut reports = Vec::new();
    let mut identity_cost = 0.0f64;
    for (name, perms) in &methods {
        let mut per_layer = BTreeMap::new();
        let mut total = LayerStats::default();
        for l in &moe.layers {
            let (_, holdout) = &splits[&l.index];
            let geom = geometry(&moe, l.index)?;
            let st = evaluate(holdout.tokens(), &perms[&l.index], geom, &cm);
            total.add(&st);
            per_layer.insert(l.index, st);
        }
        if name == "identity" {
            identity_cost = total.cost_per_token_ns();
        }
        reports.push(MethodReport {
            method: name.clone(),
            fetches_per_token: total.fetches_per_token(),
            bytes_per_token: total.bytes_per_token(),
            cost_ns_per_token: total.cost_per_token_ns(),
            cost_reduction_pct: 0.0,
            per_layer,
        });

        let layout = Layout {
            model: model_name.clone(),
            original_sha256: original_sha.clone(),
            method: name.clone(),
            n_expert: moe.n_expert,
            layers: perms
                .iter()
                .map(|(&l, p)| (l, p.as_slice().to_vec()))
                .collect(),
        };
        layout.save(a.outdir.join(format!("{}.json", name.replace(':', "-"))))?;
    }

    for r in &mut reports {
        r.cost_reduction_pct = if identity_cost > 0.0 {
            100.0 * (identity_cost - r.cost_ns_per_token) / identity_cost
        } else {
            0.0
        };
    }

    let (train_tokens, holdout_tokens) = splits
        .values()
        .next()
        .map(|(a, b)| (a.n_tokens(), b.n_tokens()))
        .unwrap_or((0, 0));

    println!(
        "\n{:<16} {:>12} {:>14} {:>14} {:>10}",
        "method", "fetches/tok", "MiB/tok", "us/tok", "vs ident"
    );
    for r in &reports {
        println!(
            "{:<16} {:>12.1} {:>14.1} {:>14.1} {:>9.2}%",
            r.method,
            r.fetches_per_token,
            r.bytes_per_token / (1 << 20) as f64,
            r.cost_ns_per_token / 1000.0,
            r.cost_reduction_pct
        );
    }

    // Router gate mass by rank, pooled over layers. Purely diagnostic.
    let mut rank_sum = vec![0.0f64; tr.top_k];
    let mut rank_n = 0u64;
    for lt in &tr.layers {
        if !lt.has_weights() {
            continue;
        }
        for t in 0..lt.n_tokens() {
            for (j, &w) in lt.token_weights(t).iter().enumerate() {
                rank_sum[j] += w as f64;
            }
            rank_n += 1;
        }
    }
    let mut gate_mass = Vec::new();
    if rank_n > 0 {
        let total: f64 = rank_sum.iter().sum();
        let mut cum = 0.0;
        for (j, &s) in rank_sum.iter().enumerate() {
            cum += s;
            gate_mass.push(GateMass {
                rank: j + 1,
                mean_weight: s / rank_n as f64,
                cumulative_pct: 100.0 * cum / total,
            });
        }
        println!("\nrouter gate mass by top-k rank (pooled over layers):");
        println!("{:>6} {:>14} {:>14}", "rank", "mean weight", "cumulative");
        for g in &gate_mass {
            println!(
                "{:>6} {:>14.4} {:>13.2}%",
                g.rank, g.mean_weight, g.cumulative_pct
            );
        }
    }

    // Sweep the per-fetch cost from local NVMe (~1 us) out to network storage (~1 ms).
    let mut sweep = Vec::new();
    for &c_fetch_us in &[
        0.5f64, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 1000.0,
    ] {
        // A hypothetical device, so the provenance is Assumed by construction: this sweep
        // answers "what would this layout be worth if a request cost X", not "what does it
        // cost here".
        let scm = CostModel {
            c_fetch_ns: c_fetch_us * 1000.0,
            c_byte_ns: cm.c_byte_ns,
            provenance: coact::cost::Provenance::Assumed,
        };
        let mut costs: Vec<(String, f64)> = Vec::new();
        for (name, perms) in &methods {
            let mut total = LayerStats::default();
            for l in &moe.layers {
                let (_, holdout) = &splits[&l.index];
                total.add(&evaluate(
                    holdout.tokens(),
                    &perms[&l.index],
                    geometry(&moe, l.index)?,
                    &scm,
                ));
            }
            costs.push((name.clone(), total.cost_per_token_ns()));
        }
        let base = costs
            .iter()
            .find(|c| c.0 == "identity")
            .map(|c| c.1)
            .unwrap_or(0.0);
        let pct = |c: f64| {
            if base > 0.0 {
                100.0 * (base - c) / base
            } else {
                0.0
            }
        };
        let best = costs
            .iter()
            .filter(|c| c.0 != "identity" && !c.0.starts_with("random"))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .cloned()
            .unwrap_or_else(|| ("none".into(), base));
        let rand_best = costs
            .iter()
            .filter(|c| c.0.starts_with("random"))
            .map(|c| c.1)
            .fold(f64::INFINITY, f64::min);
        sweep.push(SweepPoint {
            c_fetch_us,
            max_bridged_gap: scm.max_gap(geom0.bytes_per_expert, geom0.n_tensors),
            best_method: best.0,
            best_reduction_pct: pct(best.1),
            random_reduction_pct: if rand_best.is_finite() {
                pct(rand_best)
            } else {
                0.0
            },
        });
    }

    println!(
        "\nper-fetch cost sweep (C_byte fixed at the measured {:.4} ns/B):",
        cm.c_byte_ns
    );
    println!(
        "{:>12} {:>10} {:>16} {:>12} {:>12}",
        "C_fetch us", "gap", "best method", "best %", "random %"
    );
    for s in &sweep {
        println!(
            "{:>12.1} {:>10} {:>16} {:>11.2}% {:>11.2}%",
            s.c_fetch_us,
            s.max_bridged_gap,
            s.best_method,
            s.best_reduction_pct,
            s.random_reduction_pct
        );
    }

    let report = Report {
        model: model_name,
        arch: moe.arch.clone(),
        n_expert: moe.n_expert,
        n_expert_used: moe.n_expert_used,
        n_moe_layers: moe.layers.len(),
        bytes_per_expert: geom0.bytes_per_expert,
        max_bridged_gap: cm.max_gap(geom0.bytes_per_expert, geom0.n_tensors),
        cost_model: cm,
        train_tokens,
        holdout_tokens,
        methods: reports,
        cfetch_sweep: sweep,
        edge_weight: format!("{:?}", a.edge_weight).to_lowercase(),
        gate_mass_by_rank: gate_mass,
    };
    if let Some(d) = a.report.parent() {
        std::fs::create_dir_all(d).ok();
    }
    std::fs::write(&a.report, serde_json::to_string_pretty(&report)?)?;
    eprintln!(
        "\nwrote {} and layouts in {}",
        a.report.display(),
        a.outdir.display()
    );
    Ok(())
}
