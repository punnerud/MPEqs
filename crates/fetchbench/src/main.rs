//! Cold-cache measurement of what an expert layout actually costs to fetch.

mod cache;
mod nocache;

use anyhow::{bail, Result};
use cache::{runs, split_at_limit, working_set, CacheRun, Lru, Policy};
use clap::{Parser, Subcommand};
use coact::cost::{CostModel, LayerGeometry, Permutation};
use coact::layout::Layout;
use coact::trace::Trace;
use gguf::{moe::MoeModel, Gguf};
use nocache::{AlignedBuf, NoCacheFile};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

const BUF_BYTES: usize = 64 << 20;

#[derive(Parser)]
#[command(about = "Cold-cache (F_NOCACHE) fetch benchmark for MoE expert layouts")]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Measure this device's per-fetch and per-byte cost.
    Calibrate {
        #[arg(short, long)]
        model: PathBuf,
        #[arg(short, long, default_value = "data/costmodel.json")]
        out: PathBuf,
        #[arg(long, default_value_t = 40)]
        iters: usize,
        /// Smallest read size included in the affine fit, in KiB.
        #[arg(long, default_value_t = 4)]
        fit_min_kib: u64,
        /// Largest read size included in the affine fit, in KiB. Above ~8 MiB this SSD
        /// falls off a cliff that is not part of the affine regime.
        #[arg(long, default_value_t = 8192)]
        fit_max_kib: u64,
        /// Do not evict the page cache first. Only correct if you just did it yourself.
        #[arg(long)]
        no_drop_cache: bool,
    },
    /// Replay a router trace against one or more layouts and time the reads.
    Replay {
        #[arg(short, long)]
        model: PathBuf,
        #[arg(short, long)]
        trace: PathBuf,
        /// Layout JSON files. The first is treated as the reference.
        #[arg(short, long, num_args = 1..)]
        layout: Vec<PathBuf>,
        #[arg(long)]
        cost: Option<PathBuf>,
        /// Tokens to replay. Each one reads roughly `bytes_per_token` from the SSD.
        #[arg(long, default_value_t = 48)]
        tokens: usize,
        /// Batch size: experts are unioned across this many consecutive tokens.
        #[arg(long, default_value_t = 1)]
        batch: usize,
        #[arg(long, default_value = "data/fetchbench.json")]
        out: PathBuf,
        /// Repeat the whole sweep N times and report the median per layout.
        #[arg(long, default_value_t = 3)]
        reps: usize,
        /// Concurrent readers. NVMe delivers far more bandwidth at queue depth > 1, and a
        /// clustered layout is what makes the requests big enough to be worth parallelising.
        #[arg(long, default_value_t = 1)]
        threads: usize,
        /// Override the cost model's bridging gap: read through this many unwanted experts
        /// to turn two fetches into one. The affine cost model says 0, but it only knows a
        /// single bytes-per-nanosecond and this device is 3x faster at 8 MiB than at 1 MiB.
        /// Sweeping this is how the real trade is found.
        #[arg(long)]
        max_gap: Option<u64>,
    },
    /// Replay the trace through an LRU expert cache with real uncached reads on miss.
    ///
    /// Sweeps cache size, and optionally the cost-aware routing policy: a cold expert whose
    /// gate weight is small may not be worth a disk round trip. `--lambda` prices gate mass
    /// against nanoseconds; sweeping it traces the quality-versus-IO frontier.
    Cache {
        #[arg(short, long)]
        model: PathBuf,
        #[arg(short, long)]
        trace: PathBuf,
        #[arg(short, long, num_args = 1..)]
        layout: Vec<PathBuf>,
        #[arg(long)]
        cost: Option<PathBuf>,
        /// Cache budgets to sweep, in MiB. 0 means "no cache", the fetchbench baseline.
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "0,256,512,1024,2048,4096"
        )]
        cache_mib: Vec<u64>,
        #[arg(long, default_value_t = 512)]
        tokens: usize,
        /// Price of one unit of gate mass, in nanoseconds. 0 disables skipping entirely,
        /// which is the lossless policy.
        #[arg(long, value_delimiter = ',', default_value = "0")]
        lambda: Vec<f64>,
        /// Largest single read, in MiB. Above this the SSD's throughput halves.
        #[arg(long, default_value_t = 8)]
        max_read_mib: u64,
        /// Replacement policies to compare: lru, static, random, hybrid.
        #[arg(long, value_delimiter = ',', default_value = "lru,static,random")]
        policy: Vec<String>,
        /// Rank the pinned set on this trace instead of the one being replayed. Within a
        /// single trace the ranking already uses only the first half, so it is not fitted to
        /// what it scores — but a trace of mixed registers hides a register shift, because
        /// both halves contain every genre. Pointing this at prose and replaying Norwegian is
        /// the question a deployment actually asks.
        #[arg(long)]
        pin_trace: Option<String>,

        /// For `hybrid`: share of the budget given to pinned hot experts, the rest running
        /// random replacement.
        #[arg(long, default_value_t = 0.5)]
        pin_fraction: f64,
        /// For `decayed`: accesses between halvings of every slab's score.
        #[arg(long, default_value_t = 20000)]
        halflife: u64,
        /// Concurrent readers for the misses within one layer. Layers stay sequential: the
        /// router for layer L+1 needs L's output, so its experts are not knowable yet.
        #[arg(long, default_value_t = 1)]
        threads: usize,
        /// Repeat and report the median, so adjacent policies can be told apart.
        #[arg(long, default_value_t = 1)]
        reps: usize,
        #[arg(long, default_value = "data/cache.json")]
        out: PathBuf,
    },
}

fn main() -> Result<()> {
    match Args::parse().cmd {
        Cmd::Calibrate {
            model,
            out,
            iters,
            fit_min_kib,
            fit_max_kib,
            no_drop_cache,
        } => calibrate(
            &model,
            &out,
            iters,
            fit_min_kib << 10,
            fit_max_kib << 10,
            no_drop_cache,
        ),
        Cmd::Replay {
            model,
            trace,
            layout,
            cost,
            tokens,
            batch,
            out,
            reps,
            threads,
            max_gap,
        } => replay(ReplayArgs {
            model,
            trace,
            layouts: layout,
            cost,
            tokens,
            batch,
            out,
            reps,
            threads,
            max_gap,
        }),
        Cmd::Cache {
            model,
            trace,
            layout,
            cost,
            cache_mib,
            tokens,
            lambda,
            max_read_mib,
            policy,
            pin_fraction,
            pin_trace,
            halflife,
            threads,
            reps,
            out,
        } => run_cache(CacheArgs {
            model,
            trace,
            layouts: layout,
            cost,
            cache_mib,
            tokens,
            lambda,
            max_read_bytes: max_read_mib << 20,
            policies: policy
                .iter()
                .map(|p| p.parse::<Policy>())
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(anyhow::Error::msg)?,
            pin_fraction,
            pin_trace,
            halflife,
            threads,
            reps,
            out,
        }),
    }
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn calibrate(
    model: &Path,
    out: &Path,
    iters: usize,
    fit_min: u64,
    fit_max: u64,
    skip_drop: bool,
) -> Result<()> {
    let f = NoCacheFile::open(model)?;
    let mut buf = AlignedBuf::new(BUF_BYTES);

    // Sizes span the range a layout search actually produces: one expert tensor slice at
    // the low end, a fully-merged run of eight at the high end.
    let sizes: [u64; 9] = [
        4 << 10,
        64 << 10,
        256 << 10,
        1 << 20,
        2 << 20,
        4 << 20,
        8 << 20,
        16 << 20,
        32 << 20,
    ];

    // A cheap deterministic stride keeps offsets spread across the whole file without
    // pulling in an RNG dependency.
    let mut off = 0u64;
    let mut next_off = |sz: u64| -> u64 {
        off = off
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        off % (f.len - sz)
    };

    // Calibrating on a file whose pages are still resident is the single worst mistake this
    // tool can make, because every later measurement is scaled by the result. It happened:
    // a contaminated run reported C_fetch = 6.66 us and 12.8 GB/s, against the true cold
    // values of 230 us and 3.5 GB/s — a 35x error that silently disabled the contamination
    // guard, since the guard compares against this very number.
    if !skip_drop {
        eprintln!("evicting the page cache first (pass --no-drop-cache to skip)…");
        let _ = std::process::Command::new("./scripts/drop-cache.sh")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }

    // Discard a warm-up sweep: the first uncached reads after an idle period measure the
    // SSD controller waking up rather than steady state.
    for &sz in &sizes {
        for _ in 0..3 {
            let o = next_off(sz);
            f.read_range(o, sz, &mut buf)?;
        }
    }

    println!(
        "{:>10} {:>12} {:>12} {:>6}",
        "size", "median us", "GB/s", "fit"
    );
    let mut points = Vec::new();
    for &sz in &sizes {
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let o = next_off(sz);
            let t0 = Instant::now();
            f.read_range(o, sz, &mut buf)?;
            times.push(t0.elapsed().as_nanos() as f64);
        }
        let m = median(&mut times);
        let in_fit = sz >= fit_min && sz <= fit_max;
        println!(
            "{:>9}K {:>12.1} {:>12.2} {:>6}",
            sz / 1024,
            m / 1000.0,
            sz as f64 / m,
            if in_fit { "yes" } else { "-" }
        );
        if in_fit {
            points.push((sz as f64, m));
        }
    }
    if points.len() < 3 {
        bail!(
            "fit range {fit_min}..{fit_max} kept only {} points",
            points.len()
        );
    }

    // Least squares on t = a + b*size.
    let n = points.len() as f64;
    let sx: f64 = points.iter().map(|p| p.0).sum();
    let sy: f64 = points.iter().map(|p| p.1).sum();
    let sxx: f64 = points.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = points.iter().map(|p| p.0 * p.1).sum();
    let b = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let a = (sy - b * sx) / n;
    if b <= 0.0 {
        bail!("regression produced a non-positive per-byte cost ({b}); rerun on an idle machine");
    }
    let mean_y = sy / n;
    let ss_tot: f64 = points.iter().map(|p| (p.1 - mean_y).powi(2)).sum();
    let ss_res: f64 = points.iter().map(|p| (p.1 - (a + b * p.0)).powi(2)).sum();
    let r2 = 1.0 - ss_res / ss_tot;

    let cm = CostModel {
        c_fetch_ns: a.max(0.0),
        c_byte_ns: b,
        provenance: coact::cost::Provenance::Measured,
    };
    println!(
        "\nC_fetch = {:.2} us   C_byte = {:.4} ns/B ({:.2} GB/s asymptotic)   R2 = {:.5}",
        cm.c_fetch_ns / 1000.0,
        cm.c_byte_ns,
        1.0 / cm.c_byte_ns,
        r2
    );
    if r2 < 0.99 {
        eprintln!("warning: poor affine fit (R2 = {r2:.4}); narrow --fit-max-kib");
    }
    // The ratio below is what decides whether layout can matter at all: it is the number of
    // bytes whose transfer costs as much as one extra request.
    println!(
        "break-even: one extra fetch costs as much as {:.0} KiB of transfer",
        cm.c_fetch_ns / cm.c_byte_ns / 1024.0
    );
    if let Some(d) = out.parent() {
        std::fs::create_dir_all(d).ok();
    }
    std::fs::write(out, serde_json::to_string_pretty(&cm)?)?;
    println!("wrote {}", out.display());
    Ok(())
}

struct ReplayArgs {
    model: PathBuf,
    trace: PathBuf,
    layouts: Vec<PathBuf>,
    cost: Option<PathBuf>,
    tokens: usize,
    batch: usize,
    out: PathBuf,
    reps: usize,
    threads: usize,
    max_gap: Option<u64>,
}

#[derive(Serialize)]
struct LayoutResult {
    method: String,
    n_fetches: u64,
    bytes: u64,
    median_ms: f64,
    all_ms: Vec<f64>,
    ms_per_token: f64,
    fetches_per_token: f64,
    mib_per_token: f64,
    effective_gb_s: f64,
    speedup_vs_reference: f64,
}

#[derive(Serialize)]
struct ReplayReport {
    model: String,
    tokens: usize,
    batch: usize,
    reps: usize,
    cost_model: CostModel,
    max_bridged_gap: u64,
    /// True when throughput exceeded the device ceiling, i.e. the reads hit the page cache.
    cache_contaminated: bool,
    results: Vec<LayoutResult>,
}

/// One physical read: an absolute byte range in the model file.
type Range = (u64, u64);

/// Per-layer expert geometry: the (offset, per-expert stride) of each fused weight tensor,
/// and the total bytes one expert occupies across them.
type LayerGeom = (Vec<(u64, u64)>, u64);

/// Builds the read list one batch of tokens produces under a given layout.
/// Everything `build_ranges` needs that is not the layout being evaluated.
struct RangePlan<'a> {
    moe: &'a MoeModel,
    tr: &'a Trace,
    geoms: &'a BTreeMap<u32, LayerGeometry>,
    cm: &'a CostModel,
    tokens: usize,
    batch: usize,
    gap_override: Option<u64>,
}

fn build_ranges(p: &RangePlan, perms: &BTreeMap<u32, Permutation>) -> Result<Vec<Range>> {
    let RangePlan {
        moe,
        tr,
        geoms,
        cm,
        tokens,
        batch,
        gap_override,
    } = *p;
    let mut ranges = Vec::new();
    let n_batches = tokens.div_ceil(batch.max(1));

    for lt in &tr.layers {
        let layer = lt.layer as u32;
        let Some(p) = perms.get(&layer) else { continue };
        let geom = geoms[&layer];
        let max_gap =
            gap_override.unwrap_or_else(|| cm.max_gap(geom.bytes_per_expert, geom.n_tensors));
        let ml = moe.layer(layer)?;
        let tensors: Vec<(u64, u64)> = ml
            .weight_tensors()
            .map(|t| t.last_axis_stride().map(|s| (t.offset, s)))
            .collect::<Result<_>>()?;

        for b in 0..n_batches {
            let mut slots: Vec<u32> = Vec::new();
            for i in 0..batch {
                let idx = b * batch + i;
                if idx >= lt.n_tokens() || idx >= tokens {
                    break;
                }
                for &e in lt.token(idx) {
                    slots.push(p.slot_of(e) as u32);
                }
            }
            if slots.is_empty() {
                continue;
            }
            slots.sort_unstable();
            slots.dedup();

            // Group into runs, bridging gaps the cost model says are worth bridging.
            let mut start = slots[0];
            let mut prev = slots[0];
            let emit = |from: u32, to: u32, ranges: &mut Vec<Range>| {
                for &(base, stride) in &tensors {
                    ranges.push((base + from as u64 * stride, (to - from + 1) as u64 * stride));
                }
            };
            for &s in &slots[1..] {
                if (s - prev) as u64 - 1 > max_gap {
                    emit(start, prev, &mut ranges);
                    start = s;
                }
                prev = s;
            }
            emit(start, prev, &mut ranges);
        }
    }
    Ok(ranges)
}

/// Issues every range, spread over `threads` concurrent readers.
///
/// Each worker owns its own aligned buffer; they share the descriptor, which is safe because
/// `pread` carries its own offset.
fn read_all(f: &std::sync::Arc<NoCacheFile>, ranges: &[Range], threads: usize) -> Result<u64> {
    if threads <= 1 {
        let mut buf = AlignedBuf::new(BUF_BYTES);
        let mut total = 0;
        for &(off, len) in ranges {
            total += f.read_range(off, len, &mut buf)?;
        }
        return Ok(total);
    }
    let total = std::sync::atomic::AtomicU64::new(0);
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let mut buf = AlignedBuf::new(BUF_BYTES / 4);
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(&(off, len)) = ranges.get(i) else {
                        break;
                    };
                    match f.read_range(off, len, &mut buf) {
                        Ok(n) => {
                            total.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
                        }
                        Err(e) => {
                            eprintln!("read failed: {e}");
                            break;
                        }
                    }
                }
            });
        }
    });
    Ok(total.load(std::sync::atomic::Ordering::Relaxed))
}

fn replay(a: ReplayArgs) -> Result<()> {
    if a.layouts.is_empty() {
        bail!("need at least one --layout");
    }
    let g = Gguf::open(&a.model)?;
    let moe = MoeModel::from_gguf(&g)?;
    let tr = Trace::load(&a.trace)?;
    let cm: CostModel = match &a.cost {
        Some(p) => serde_json::from_str(&std::fs::read_to_string(p)?)?,
        None => {
            eprintln!(
                "warning: no --cost given; using constants from another machine. Run \
                 `fetchbench calibrate` before believing any timing derived from this."
            );
            CostModel::assumed_apple_nvme()
        }
    };

    let mut geoms = BTreeMap::new();
    for l in &moe.layers {
        geoms.insert(
            l.index,
            LayerGeometry {
                n_tensors: l.weight_tensors().count(),
                bytes_per_expert: l.bytes_per_expert()?,
            },
        );
    }

    let f = std::sync::Arc::new(NoCacheFile::open(&a.model)?);

    let mut plans = Vec::new();
    for path in &a.layouts {
        let lay = Layout::load(path)?;
        let mut perms = BTreeMap::new();
        for l in &moe.layers {
            perms.insert(l.index, lay.permutation(l.index)?);
        }
        let plan = RangePlan {
            moe: &moe,
            tr: &tr,
            geoms: &geoms,
            cm: &cm,
            tokens: a.tokens,
            batch: a.batch,
            gap_override: a.max_gap,
        };
        let ranges = build_ranges(&plan, &perms)?;
        plans.push((lay.method.clone(), ranges));
    }

    println!(
        "replaying {} tokens (batch {}) over {} layouts, {} reps, uncached\n",
        a.tokens,
        a.batch,
        plans.len(),
        a.reps
    );

    let mut timings: Vec<Vec<f64>> = vec![Vec::new(); plans.len()];
    // Interleave reps across layouts so thermal drift hits every layout equally rather
    // than penalising whichever one happens to run last.
    for rep in 0..a.reps {
        for (i, (method, ranges)) in plans.iter().enumerate() {
            let t0 = Instant::now();
            read_all(&f, ranges, a.threads)?;
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            timings[i].push(ms);
            eprint!(
                "\rrep {}/{}  {:<16} {:>8.1} ms",
                rep + 1,
                a.reps,
                method,
                ms
            );
        }
    }
    eprintln!("\n");

    // F_NOCACHE stops this descriptor from *populating* the cache; it does not stop the
    // kernel from serving pages that are already there. Right after ggufperm rewrites the
    // file, the whole model is resident and the replay silently measures RAM. Exceeding the
    // calibrated ceiling is the tell.
    let ceiling_gb_s = 1.0 / cm.c_byte_ns;
    let mut contaminated = false;

    let mut results = Vec::new();
    let mut reference_ms = 0.0f64;
    for (i, (method, ranges)) in plans.iter().enumerate() {
        let bytes: u64 = ranges.iter().map(|r| r.1).sum();
        let mut t = timings[i].clone();
        let med = median(&mut t);
        if i == 0 {
            reference_ms = med;
        }
        let gb_s = bytes as f64 / (med / 1000.0) / 1e9;
        if gb_s > ceiling_gb_s * 1.05 {
            contaminated = true;
        }
        results.push(LayoutResult {
            method: method.clone(),
            n_fetches: ranges.len() as u64,
            bytes,
            median_ms: med,
            all_ms: timings[i].clone(),
            ms_per_token: med / a.tokens as f64,
            fetches_per_token: ranges.len() as f64 / a.tokens as f64,
            mib_per_token: bytes as f64 / a.tokens as f64 / (1 << 20) as f64,
            effective_gb_s: gb_s,
            speedup_vs_reference: if med > 0.0 { reference_ms / med } else { 0.0 },
        });
    }

    println!(
        "{:<16} {:>12} {:>12} {:>11} {:>10} {:>9}",
        "layout", "fetches/tok", "MiB/tok", "median ms", "GB/s", "speedup"
    );
    for r in &results {
        println!(
            "{:<16} {:>12.1} {:>12.1} {:>11.1} {:>10.2} {:>8.3}x",
            r.method,
            r.fetches_per_token,
            r.mib_per_token,
            r.median_ms,
            r.effective_gb_s,
            r.speedup_vs_reference
        );
    }

    if contaminated {
        eprintln!(
            "\nWARNING: measured throughput exceeds the calibrated ceiling of {ceiling_gb_s:.2} GB/s.\n\
             Pages of this file are still resident, most likely because ggufperm just rewrote\n\
             it. These timings are not cold-cache numbers. Run `sudo purge` and repeat."
        );
    }

    let geom0 = geoms[&moe.layers[0].index];
    let report = ReplayReport {
        model: a
            .model
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
        tokens: a.tokens,
        batch: a.batch,
        reps: a.reps,
        cost_model: cm,
        max_bridged_gap: cm.max_gap(geom0.bytes_per_expert, geom0.n_tensors),
        cache_contaminated: contaminated,
        results,
    };
    if let Some(d) = a.out.parent() {
        std::fs::create_dir_all(d).ok();
    }
    std::fs::write(&a.out, serde_json::to_string_pretty(&report)?)?;
    println!("\nwrote {}", a.out.display());
    Ok(())
}

struct CacheArgs {
    model: PathBuf,
    trace: PathBuf,
    layouts: Vec<PathBuf>,
    cost: Option<PathBuf>,
    cache_mib: Vec<u64>,
    tokens: usize,
    lambda: Vec<f64>,
    max_read_bytes: u64,
    policies: Vec<Policy>,
    pin_fraction: f64,
    pin_trace: Option<String>,
    halflife: u64,
    threads: usize,
    reps: usize,
    out: PathBuf,
}

#[derive(Serialize)]
struct CacheResult {
    layout: String,
    policy: String,
    cache_mib: u64,
    lambda_ns_per_mass: f64,
    hit_rate_pct: f64,
    fetches_per_token: f64,
    mib_per_token: f64,
    /// Gate mass the policy declined to fetch, as a percentage of all routed mass.
    quality_cost_pct: f64,
    skipped_experts_per_token: f64,
    millis: f64,
    ms_per_token: f64,
}

#[derive(Serialize)]
struct WorkingSet {
    layer: u32,
    n_expert: u32,
    mean_distinct_512: f64,
    max_distinct_512: f64,
    mean_distinct_64: f64,
}

#[derive(Serialize)]
struct CacheReport {
    model: String,
    tokens: usize,
    max_read_bytes: u64,
    cost_model: CostModel,
    expert_bytes_total: u64,
    working_set: Vec<WorkingSet>,
    results: Vec<CacheResult>,
}

/// Fills a cache with the globally hottest experts before the replay starts.
///
/// Ranked on the first half of the trace and scored on the rest, so the policy is not fitted
/// to what it is measured on.
fn preload(
    lru: &mut Lru,
    policy: Policy,
    perms: &BTreeMap<u32, Permutation>,
    // The trace the ranking is read from, and how much of it to use: half when it is the
    // trace being replayed (so the policy is not fitted to its own score), all of it when it
    // is a separate one.
    src: (&Trace, f64),
    geom: &BTreeMap<u32, LayerGeom>,
    mib: u64,
    pin_fraction: f64,
) {
    // Static pinning preloads the globally hottest experts. Ranked on the first half of the
    // replayed trace by default, so the policy is not fitted to what it is scored on — or on
    // the whole of a separate trace when one is given, which is the stronger test: a mixed
    // corpus has every register in both halves, so only a different corpus can show whether
    // the hot set is a property of the model or of the text it was measured on.
    if policy == Policy::StaticPinned || policy == Policy::Hybrid {
        let pin_budget = if policy == Policy::Hybrid {
            ((mib << 20) as f64 * pin_fraction) as u64
        } else {
            u64::MAX
        };
        let mut pinned_bytes = 0u64;
        let mut freq: Vec<((u32, u32), u64)> = Vec::new();
        let mut counts: BTreeMap<(u32, u32), u64> = BTreeMap::new();
        let (src, frac) = src;
        for lt in &src.layers {
            let layer = lt.layer as u32;
            let Some(p) = perms.get(&layer) else { continue };
            let half = ((lt.n_tokens() as f64 * frac) as usize).max(1);
            for t in 0..half {
                for &e in lt.token(t) {
                    *counts.entry((layer, p.slot_of(e) as u32)).or_default() += 1;
                }
            }
        }
        freq.extend(counts);
        freq.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        for ((layer, slot), _) in freq {
            let (_, per_expert) = &geom[&layer];
            if pinned_bytes + *per_expert > pin_budget || !lru.pin((layer, slot), *per_expert) {
                break;
            }
            pinned_bytes += *per_expert;
        }
    }
}

fn run_cache(a: CacheArgs) -> Result<()> {
    let g = Gguf::open(&a.model)?;
    let moe = MoeModel::from_gguf(&g)?;
    let tr = Trace::load(&a.trace)?;
    let pin_tr = match &a.pin_trace {
        Some(p) => {
            let t = Trace::load(p)?;
            println!(
                "pinned set ranked on {p} ({} layers), replayed on {}",
                t.layers.len(),
                a.trace.display()
            );
            Some(t)
        }
        None => None,
    };
    // Half of the replayed trace, or all of a separate one.
    let pin_src: (&Trace, f64) = match &pin_tr {
        Some(t) => (t, 1.0),
        None => (&tr, 0.5),
    };
    let cm: CostModel = match &a.cost {
        Some(p) => serde_json::from_str(&std::fs::read_to_string(p)?)?,
        None => {
            eprintln!(
                "warning: no --cost given; using constants from another machine. Run \
                 `fetchbench calibrate` before believing any timing derived from this."
            );
            CostModel::assumed_apple_nvme()
        }
    };

    // Working set first: if a document touches nearly every expert there is nothing to cache
    // and the rest of this command is measuring noise.
    let mut ws = Vec::new();
    for lt in &tr.layers {
        let toks: Vec<&[u16]> = (0..lt.n_tokens().min(a.tokens.max(512)))
            .map(|t| lt.token(t))
            .collect();
        let (m512, x512) = working_set(&toks, 512);
        let (m64, _) = working_set(&toks, 64);
        ws.push(WorkingSet {
            layer: lt.layer as u32,
            n_expert: tr.n_expert,
            mean_distinct_512: m512,
            max_distinct_512: x512,
            mean_distinct_64: m64,
        });
    }
    println!(
        "working set per layer (of {} experts): {:>6} {:>10} {:>10}",
        tr.n_expert, "layer", "64 tok", "512 tok"
    );
    for w in &ws {
        println!(
            "{:>44} {:>10.1} {:>10.1}",
            w.layer, w.mean_distinct_64, w.mean_distinct_512
        );
    }
    let mean512 = ws.iter().map(|w| w.mean_distinct_512).sum::<f64>() / ws.len() as f64;
    println!(
        "\nmean over layers: a 512-token document touches {:.1} of {} experts ({:.0}%)\n",
        mean512,
        tr.n_expert,
        100.0 * mean512 / tr.n_expert as f64
    );

    // Geometry per layer: the file offsets and per-expert stride of each fused tensor.
    let mut geom: BTreeMap<u32, LayerGeom> = BTreeMap::new();
    let mut expert_bytes_total = 0u64;
    for l in &moe.layers {
        let tensors: Vec<(u64, u64)> = l
            .weight_tensors()
            .map(|t| t.last_axis_stride().map(|s| (t.offset, s)))
            .collect::<Result<_>>()?;
        let per_expert = l.bytes_per_expert()?;
        expert_bytes_total += per_expert * tr.n_expert as u64;
        geom.insert(l.index, (tensors, per_expert));
    }

    let f = std::sync::Arc::new(NoCacheFile::open(&a.model)?);
    let mut results = Vec::new();

    for layout_path in &a.layouts {
        let lay = Layout::load(layout_path)?;
        let mut perms = BTreeMap::new();
        for l in &moe.layers {
            perms.insert(l.index, lay.permutation(l.index)?);
        }

        for &mib in &a.cache_mib {
            for &lambda in &a.lambda {
                for &policy in &a.policies {
                    let mut lru = Lru::new(mib << 20);
                    if policy == Policy::Decayed {
                        lru = lru.with_halflife(a.halflife);
                    }

                    preload(
                        &mut lru,
                        policy,
                        &perms,
                        pin_src,
                        &geom,
                        mib,
                        a.pin_fraction,
                    );

                    // Repeat the whole replay; the cache state is rebuilt each time, so the
                    // hit counts are identical and only the timing varies.
                    let mut rep_ms: Vec<f64> = Vec::with_capacity(a.reps.max(1));
                    let mut run = CacheRun::default();
                    for rep in 0..a.reps.max(1) {
                        if rep > 0 {
                            lru = Lru::new(mib << 20);
                            if policy == Policy::Decayed {
                                lru = lru.with_halflife(a.halflife);
                            }
                            preload(
                                &mut lru,
                                policy,
                                &perms,
                                pin_src,
                                &geom,
                                mib,
                                a.pin_fraction,
                            );
                            run = CacheRun::default();
                        }
                        let t0 = Instant::now();

                        for tok in 0..a.tokens {
                            let mut any = false;
                            for lt in &tr.layers {
                                if tok >= lt.n_tokens() {
                                    continue;
                                }
                                let layer = lt.layer as u32;
                                let Some(p) = perms.get(&layer) else { continue };
                                let (tensors, per_expert) = &geom[&layer];
                                any = true;

                                let ids = lt.token(tok);
                                let w = lt.token_weights(tok);

                                // Decide what to fetch. A resident expert is always used. A cold one
                                // is used when its gate mass is worth the round trip at this lambda.
                                let fetch_price = tensors.len() as f64 * cm.c_fetch_ns
                                    + *per_expert as f64 * cm.c_byte_ns;
                                let mut want: Vec<u32> = Vec::with_capacity(ids.len());
                                for (j, &e) in ids.iter().enumerate() {
                                    let slot = p.slot_of(e) as u32;
                                    let mass = if w.is_empty() { 1.0 } else { w[j] as f64 };
                                    run.total_mass += mass;
                                    if policy == Policy::Decayed {
                                        lru.observe((layer, slot));
                                    }
                                    if lru.touch((layer, slot)) {
                                        continue;
                                    }
                                    if policy == Policy::StaticPinned {
                                        // Nothing is admitted after preload; a miss is always a fetch.
                                        want.push(slot);
                                        continue;
                                    }
                                    if lambda > 0.0 && mass * lambda < fetch_price {
                                        run.skipped_mass += mass;
                                        run.skipped_experts += 1;
                                        continue;
                                    }
                                    want.push(slot);
                                }
                                if want.is_empty() {
                                    continue;
                                }
                                want.sort_unstable();
                                want.dedup();

                                // Fetch whole runs. Everything the run covers is admitted, so a
                                // clustered layout pulls in neighbours for free. All misses within
                                // one layer are independent and go out concurrently; layers cannot,
                                // because the next router needs this layer's output first.
                                let mut pending: Vec<Range> = Vec::new();
                                for (from, to) in runs(&want, 0) {
                                    for (a0, b0) in
                                        split_at_limit(from, to, *per_expert, a.max_read_bytes)
                                    {
                                        for &(base, stride) in tensors {
                                            let off = base + a0 as u64 * stride;
                                            let len = (b0 - a0 + 1) as u64 * stride;
                                            pending.push((off, len));
                                            run.fetches += 1;
                                        }
                                        match policy {
                                            Policy::StaticPinned => {}
                                            Policy::Decayed => {
                                                for slot in a0..=b0 {
                                                    lru.insert_by_score((layer, slot), *per_expert);
                                                }
                                            }
                                            _ => {
                                                for slot in a0..=b0 {
                                                    lru.insert_with(
                                                        (layer, slot),
                                                        *per_expert,
                                                        policy == Policy::Random,
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                                run.bytes_from_disk += read_all(&f, &pending, a.threads)?;
                            }
                            if any {
                                run.tokens += 1;
                            }
                        }

                        run.millis = t0.elapsed().as_secs_f64() * 1000.0;
                        run.hits = lru.hits;
                        run.misses = lru.misses;
                        rep_ms.push(run.millis);
                    }
                    rep_ms.sort_by(|x, y| x.partial_cmp(y).unwrap());
                    run.millis = rep_ms[rep_ms.len() / 2];

                    let r = CacheResult {
                        layout: lay.method.clone(),
                        policy: format!("{policy:?}").to_lowercase(),
                        cache_mib: mib,
                        lambda_ns_per_mass: lambda,
                        hit_rate_pct: 100.0 * run.hit_rate(),
                        fetches_per_token: run.fetches as f64 / run.tokens.max(1) as f64,
                        mib_per_token: run.mib_per_token(),
                        quality_cost_pct: run.quality_cost_pct(),
                        skipped_experts_per_token: run.skipped_experts as f64
                            / run.tokens.max(1) as f64,
                        millis: run.millis,
                        ms_per_token: run.millis / run.tokens.max(1) as f64,
                    };
                    eprint!(
                        "\r{:<10} {:<14} {:>6} MiB  hit {:>5.1}%  {:>7.1} MiB/tok      ",
                        r.policy, r.layout, r.cache_mib, r.hit_rate_pct, r.mib_per_token
                    );
                    results.push(r);
                }
            }
        }
    }
    eprintln!("\n");

    println!(
        "{:<10} {:<12} {:>9} {:>8} {:>12} {:>11}",
        "policy", "layout", "cache MiB", "hit %", "MiB/token", "ms/token"
    );
    for r in &results {
        println!(
            "{:<10} {:<12} {:>9} {:>7.1}% {:>12.1} {:>11.2}",
            r.policy, r.layout, r.cache_mib, r.hit_rate_pct, r.mib_per_token, r.ms_per_token
        );
    }
    println!("\nUnder uniform access a cache holding fraction f of the data should hit about f of");
    println!("the time. Any policy far below its residency fraction is being actively harmful.");

    let report = CacheReport {
        model: a
            .model
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
        tokens: a.tokens,
        max_read_bytes: a.max_read_bytes,
        cost_model: cm,
        expert_bytes_total,
        working_set: ws,
        results,
    };
    if let Some(d) = a.out.parent() {
        std::fs::create_dir_all(d).ok();
    }
    std::fs::write(&a.out, serde_json::to_string_pretty(&report)?)?;
    println!("\nwrote {}", a.out.display());
    Ok(())
}
