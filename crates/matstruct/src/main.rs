//! Does a matrix have the structure matcodec's compression depends on?
//!
//! MPEE's matcodec compresses a distance matrix 6.4× on real road data and only ~1.8× on
//! structureless points. The difference is entirely whether cross-region blocks are additive
//! rank-1 through a gateway. This binary measures that, on two very different matrices, before
//! any codec is written:
//!
//! - **embeddings** — angular distance between text chunks. A true metric, N can reach 10⁶,
//!   and topic clusters bridged by shared concepts are a plausible analogue of gateways.
//! - **experts** — the MoE co-activation graph from `data/trace.bin`, as a negative control.
//!   Co-activation counts are similarities, not distances, so the triangle inequality has no
//!   reason to hold and the rank-1 base has no reason to fit. If the probe reports the same
//!   structure here as on embeddings, the probe is measuring something other than it thinks.

mod geodesic;
mod graph;
mod hublabel;
mod knn;
mod landmark;
mod matrix;
mod metric;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use matrix::Matrix;
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(about = "Measure whether a matrix has gateway structure worth compressing")]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Build an N×N angular-distance matrix from an embedding file.
    FromEmbeddings {
        /// Raw `[n, dim]` f32 written by `scripts/embed-corpus.sh`.
        #[arg(short, long)]
        embeddings: PathBuf,
        #[arg(long)]
        dim: usize,
        #[arg(short, long, default_value = "data/matrix-embeddings.matx")]
        out: PathBuf,
        /// Cap on points, so a probe can be run before committing to the full corpus.
        #[arg(long, default_value_t = 0)]
        limit: usize,
    },
    /// Build the control matrix from the MoE co-activation graph.
    FromTrace {
        #[arg(short, long)]
        trace: PathBuf,
        /// Which MoE layer to take. Layers are near-independent, so one is representative.
        #[arg(long, default_value_t = 8)]
        layer: u16,
        #[arg(short, long, default_value = "data/matrix-experts.matx")]
        out: PathBuf,
        /// Reinforcement strength. `w' = w · lift^alpha` before distances are computed, so
        /// pairs that already co-activate more than chance get strengthened further and the
        /// rest weaken — rich-get-richer on the hops that carry traffic.
        ///
        /// This stands in for reinforcing the routes that produced correct answers, without
        /// needing correctness labels: it tests whether *any* concentration of the graph
        /// creates the narrow cuts that a gateway codec needs. It reshapes the analysis graph
        /// only; the model is untouched.
        #[arg(long, default_value_t = 0.0)]
        reinforce: f64,
    },
    /// Turn a distance matrix into a kNN graph and re-measure it as shortest paths.
    ///
    /// This is the "roads, not straight lines" transform: in high dimensions the raw metric
    /// concentrates and has no bottlenecks, but travelling only along near-neighbour hops
    /// forces cross-cluster traffic through whichever few points bridge the clusters.
    Geodesic {
        #[arg(short, long)]
        matrix: PathBuf,
        /// Neighbours per point. Too few disconnects the graph; too many restores the
        /// straight-line metric and the bottlenecks disappear again.
        #[arg(short, long, default_value_t = 12)]
        k: usize,
        #[arg(short, long, default_value = "data/matrix-geodesic.matx")]
        out: PathBuf,
    },
    /// The synthesis: streaming kNN, geodesic distances, betweenness-chosen hubs.
    ///
    /// Every part of this thread pointed at hub *selection* rather than label structure —
    /// betweenness on the geodesic reached 39.08 % of cells exact where facility location on
    /// the chord reached 0.91 %. This runs both selections on the same graph metric so the
    /// comparison is like for like, and it streams throughout so it works above n = 8 000.
    Geohub {
        #[arg(short, long)]
        embeddings: PathBuf,
        #[arg(long)]
        dim: usize,
        /// Neighbours per node in the graph the geodesic is measured through.
        #[arg(long, default_value_t = 16)]
        graph_k: usize,
        #[arg(short = 'H', long, default_value_t = 64)]
        hubs: usize,
        #[arg(short, long, default_value_t = 16)]
        k: usize,
        #[arg(long, default_value_t = 200)]
        rows: usize,
        #[arg(long, default_value_t = 10)]
        recall_k: usize,
        #[arg(long, default_value_t = 0)]
        limit: usize,
        #[arg(short, long, default_value = "data/geohub.json")]
        out: PathBuf,
    },
    /// Two-sided hub labels: each point keeps its own way out and its own way in.
    ///
    /// The counterpart to `landmark`, which uses one global relay set. Scored for search —
    /// recall against a brute-force ground truth — rather than for compression.
    Hublabel {
        #[arg(short, long)]
        embeddings: Option<PathBuf>,
        #[arg(long, default_value_t = 0)]
        dim: usize,
        #[arg(short, long)]
        matrix: Option<PathBuf>,
        #[arg(short = 'H', long, default_value_t = 128)]
        hubs: usize,
        #[arg(short, long, default_value_t = 16)]
        k: usize,
        /// Min-plus squarings of the hub table. 0 is MPEE's two-hub model; 1 gives three hops.
        #[arg(long, default_value_t = 0)]
        squarings: u32,
        #[arg(long, default_value_t = 1000.0)]
        scale: f32,
        #[arg(long, default_value_t = 400)]
        rows: usize,
        #[arg(long, default_value_t = 10)]
        recall_k: usize,
        #[arg(long, default_value_t = 0)]
        limit: usize,
        #[arg(short, long, default_value = "data/hublabel.json")]
        out: PathBuf,
    },
    /// Exact k nearest neighbours, streamed — never forms the n x n similarity matrix.
    ///
    /// Writes a binary the Python probes read instead of doing `sim = x @ x.t()`, which is
    /// 10 GB at n = 50 000 and the reason every graph result so far is on a 4 000-point
    /// subsample.
    Knn {
        #[arg(short, long)]
        embeddings: PathBuf,
        #[arg(long)]
        dim: usize,
        #[arg(short, long, default_value_t = 16)]
        k: usize,
        /// Rows and columns per block. Peak scratch is block^2 floats.
        #[arg(long, default_value_t = 512)]
        block: usize,
        #[arg(long, default_value_t = 0)]
        limit: usize,
        /// Cross-check every neighbour list against the dense path, and report the graph the
        /// CSR produces. Only feasible while n^2 still fits, which is the point of checking.
        #[arg(long)]
        verify: bool,
        #[arg(short, long, default_value = "data/knn.bin")]
        out: PathBuf,
    },
    /// Build the landmark road network from embeddings and measure what it explains.
    ///
    /// The counterpart to `probe`: that one scores matcodec's per-cluster rank-1 base, this one
    /// scores the min-plus landmark base. Streams from the embedding file, never materialises
    /// n x n, so it is the only path usable above n ~ 20 000.
    Landmark {
        #[arg(short, long)]
        embeddings: Option<PathBuf>,
        #[arg(long, default_value_t = 0)]
        dim: usize,
        /// Score a prebuilt .matx instead of embeddings — the only way to test the geodesic
        /// variant, since that transform needs the dense matrix anyway.
        #[arg(short, long)]
        matrix: Option<PathBuf>,
        /// Number of landmarks. matcodec sweeps 8, 16, 32, 64.
        #[arg(short = 'L', long, default_value_t = 32)]
        landmarks: usize,
        /// Candidate cap for the greedy search. 0 scans every point, as matcodec does.
        #[arg(long, default_value_t = 0)]
        candidates: usize,
        /// Integer units per radian. 1000 gives milliradians, so pi is 3142.
        #[arg(long, default_value_t = 1000.0)]
        scale: f32,
        /// Rows sampled for the exactness measurement; the full sweep is O(n^2).
        #[arg(long, default_value_t = 2000)]
        rows: usize,
        /// Residual, in units, still counted as reproduced by the base.
        #[arg(long, default_value_t = 2)]
        tol_units: i32,
        #[arg(long, default_value_t = 0)]
        limit: usize,
        #[arg(short, long, default_value = "data/matstruct-landmark.json")]
        out: PathBuf,
    },
    /// Run every structural probe and report a go/no-go.
    Probe {
        #[arg(short, long)]
        matrix: PathBuf,
        #[arg(long, default_value_t = 8)]
        clusters: usize,
        #[arg(long, default_value_t = 1_000_000)]
        triples: u64,
        /// Grid the matrix is quantised to before residuals are counted, in matrix units.
        #[arg(long)]
        quantum: Option<f64>,
        /// Residual magnitude, in quanta, still treated as reproduced by the base.
        #[arg(long, default_value_t = 1)]
        tol_units: i64,
        #[arg(short, long, default_value = "data/matstruct.json")]
        out: PathBuf,
    },
}

fn main() -> Result<()> {
    match Args::parse().cmd {
        Cmd::FromEmbeddings {
            embeddings,
            dim,
            out,
            limit,
        } => from_embeddings(&embeddings, dim, &out, limit),
        Cmd::FromTrace {
            trace,
            layer,
            out,
            reinforce,
        } => from_trace(&trace, layer, &out, reinforce),
        Cmd::Geodesic { matrix, k, out } => geodesic_cmd(&matrix, k, &out),
        Cmd::Geohub {
            embeddings,
            dim,
            graph_k,
            hubs,
            k,
            rows,
            recall_k,
            limit,
            out,
        } => {
            let (emb, n) = read_embeddings(&embeddings, dim, limit)?;
            let esrc = landmark::EmbeddingSource::new(emb, dim, 1000.0)?;
            eprintln!("{n} points — streaming kNN at k={graph_k}…");
            let (off, edges) = knn::knn_blocked(&esrc, graph_k, 1024).to_csr();
            let g = graph::GraphSource::new(off, edges, n, 1000.0);
            let reach = g.reachable_pct(16, 3);
            let (order, conc) = g.betweenness(120, 50, 5);
            eprintln!(
                "graph: {reach:.1} % reachable, gateway concentration {conc:.1}x — labelling…"
            );

            let opts = hublabel::HubOpts {
                hubs,
                k,
                ..Default::default()
            };
            let bw = hublabel::measure(
                &g,
                &hublabel::build_with_hubs(&g, opts, order[..hubs.min(n)].to_vec())?,
                rows,
                recall_k,
            );
            let fl = hublabel::measure(&g, &hublabel::build(&g, opts)?, rows, recall_k);

            println!("{:>28} {:>12} {:>12}", "", "betweenness", "facility loc");
            println!(
                "{:>28} {:>11.2}% {:>11.2}%",
                "cells exact", bw.exact_cell_pct, fl.exact_cell_pct
            );
            println!(
                "{:>28} {:>12.3} {:>12.3}",
                format!("recall@{recall_k}"),
                bw.recall_at_k,
                fl.recall_at_k
            );
            println!(
                "{:>28} {:>12.3} {:>12.3}",
                "residual / distance", bw.residual_ratio, fl.residual_ratio
            );
            println!(
                "{:>28} {:>11.1}K {:>11.1}K",
                "resident",
                bw.resident_bytes as f64 / 1024.0,
                fl.resident_bytes as f64 / 1024.0
            );
            let rep = serde_json::json!({
                "n": n, "graph_k": graph_k, "hubs": hubs, "k": k,
                "reachable_pct": reach, "gateway_concentration": conc,
                "betweenness": bw, "facility_location": fl,
            });
            std::fs::write(&out, serde_json::to_string_pretty(&rep)?)?;
            println!("wrote {}", out.display());
            Ok(())
        }
        Cmd::Hublabel {
            embeddings,
            dim,
            matrix: matrix_in,
            hubs,
            k,
            squarings,
            scale,
            rows,
            recall_k,
            limit,
            out,
        } => {
            let opts = hublabel::HubOpts {
                hubs,
                k,
                ..Default::default()
            };
            let mut rep = match (&embeddings, &matrix_in) {
                (Some(p), None) => {
                    let (emb, n) = read_embeddings(p, dim, limit)?;
                    let src = landmark::EmbeddingSource::new(emb, dim, scale)?;
                    eprintln!("{n} points, {dim}-dim, H={hubs} k={k} sq={squarings}…");
                    let mut idx = hublabel::build(&src, opts)?;
                    for _ in 0..squarings {
                        idx.square_dhh();
                    }
                    hublabel::measure(&src, &idx, rows, recall_k)
                }
                (None, Some(p)) => {
                    let m = matrix::Matrix::load(p)?;
                    let n = m.n;
                    let d = m.d.iter().map(|v| (v * scale).round() as i32).collect();
                    let src = landmark::DenseSource { n, d };
                    eprintln!("{n}x{n} matrix, H={hubs} k={k} sq={squarings}…");
                    let mut idx = hublabel::build(&src, opts)?;
                    for _ in 0..squarings {
                        idx.square_dhh();
                    }
                    hublabel::measure(&src, &idx, rows, recall_k)
                }
                _ => bail!("pass exactly one of --embeddings (with --dim) or --matrix"),
            };
            rep.squarings = squarings;
            println!(
                "{:>26} {:>12}",
                "hubs / k",
                format!("{} / {}", rep.hubs, rep.k)
            );
            println!(
                "{:>26} {:>11.1} KiB",
                "resident",
                rep.resident_bytes as f64 / 1024.0
            );
            println!("{:>26} {:>11.2}%", "cells exact", rep.exact_cell_pct);
            println!("{:>26} {:>11.3}", "residual / distance", rep.residual_ratio);
            println!(
                "{:>26} {:>11.3}",
                format!("recall@{recall_k}"),
                rep.recall_at_k
            );
            println!(
                "{:>26} {:>12}  (worst {} units)",
                "violations", rep.violations, rep.max_violation
            );
            std::fs::write(&out, serde_json::to_string_pretty(&rep)?)?;
            println!("wrote {}", out.display());
            Ok(())
        }
        Cmd::Knn {
            embeddings,
            dim,
            k,
            block,
            limit,
            verify,
            out,
        } => {
            let (emb, n) = read_embeddings(&embeddings, dim, limit)?;
            let src = landmark::EmbeddingSource::new(emb, dim, 1000.0)?;
            eprintln!("{n} points, {dim}-dim, k={k}, block={block} — streaming…");
            let nb = knn::knn_blocked(&src, k, block);
            // Header then two parallel arrays, so Python can mmap or read it in one go.
            let mut buf = Vec::with_capacity(16 + nb.ids.len() * 8);
            buf.extend_from_slice(b"KNN1");
            buf.extend_from_slice(&(nb.n as u32).to_le_bytes());
            buf.extend_from_slice(&(nb.k as u32).to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            for v in &nb.ids {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            for v in &nb.dist {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            if verify {
                // The dense path, built the way every other kNN in this repo builds it, and
                // compared list against list. A streaming kernel that silently disagrees would
                // look exactly like a result, which is the failure mode this project keeps
                // hitting.
                let m = matrix::angular_from_embeddings(
                    &(0..n)
                        .flat_map(|i| src.vec(i).to_vec())
                        .collect::<Vec<f32>>(),
                    n,
                    dim,
                );
                let dense = geodesic::build_knn(&m, k);
                let (off, edges) = nb.to_csr();
                let mut checked = 0usize;
                for i in 0..n {
                    let mut a: Vec<u32> =
                        edges[off[i]..off[i + 1]].iter().map(|&(j, _)| j).collect();
                    let mut b: Vec<u32> = dense.edges[dense.offsets[i]..dense.offsets[i + 1]]
                        .iter()
                        .map(|&(j, _)| j)
                        .collect();
                    a.sort_unstable();
                    b.sort_unstable();
                    if a != b {
                        bail!("streaming and dense kNN disagree at row {i}");
                    }
                    checked += 1;
                }
                println!("verified {checked} neighbour lists against the dense path — identical");
            }
            std::fs::write(&out, &buf)?;
            let peak = (block * block * 4 + nb.n * nb.k * 8) as f64 / 2f64.powi(20);
            println!(
                "wrote {} ({} x {}, {:.1} MiB); peak scratch {:.1} MiB against {:.1} MiB for a \
                 dense n x n",
                out.display(),
                nb.n,
                nb.k,
                buf.len() as f64 / 2f64.powi(20),
                peak,
                (n * n * 4) as f64 / 2f64.powi(20)
            );
            Ok(())
        }
        Cmd::Landmark {
            embeddings,
            dim,
            matrix: matrix_in,
            landmarks,
            candidates,
            scale,
            rows,
            tol_units,
            limit,
            out,
        } => landmark_cmd(LandmarkArgs {
            embeddings,
            dim,
            matrix: matrix_in,
            landmarks,
            candidates,
            scale,
            rows,
            tol_units,
            limit,
            out,
        }),
        Cmd::Probe {
            matrix,
            clusters,
            triples,
            quantum,
            tol_units,
            out,
        } => probe(&matrix, clusters, triples, quantum, tol_units, &out),
    }
}

struct LandmarkArgs {
    embeddings: Option<PathBuf>,
    dim: usize,
    matrix: Option<PathBuf>,
    landmarks: usize,
    candidates: usize,
    scale: f32,
    rows: usize,
    tol_units: i32,
    limit: usize,
    out: PathBuf,
}

fn read_embeddings(path: &Path, dim: usize, limit: usize) -> Result<(Vec<f32>, usize)> {
    let raw = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if dim == 0 || raw.len() % (dim * 4) != 0 {
        bail!(
            "{} is {} bytes, not a whole number of {dim}-dimensional f32 rows",
            path.display(),
            raw.len()
        );
    }
    let mut n = raw.len() / (dim * 4);
    if limit > 0 && limit < n {
        n = limit;
    }
    let emb = raw[..n * dim * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok((emb, n))
}

fn landmark_cmd(a: LandmarkArgs) -> Result<()> {
    // Two sources, one scoring path. The matrix branch quantises with the same scale so the
    // "exact block" criterion means the same thing in both, and a geodesic matrix can be
    // compared against raw angles without the units drifting between them.
    let mut rep = match (&a.embeddings, &a.matrix) {
        (Some(path), None) => {
            let (emb, n) = read_embeddings(path, a.dim, a.limit)?;
            let src = landmark::EmbeddingSource::new(emb, a.dim, a.scale)?;
            eprintln!(
                "{n} points, {}-dim, picking {} landmarks…",
                a.dim, a.landmarks
            );
            let lm =
                landmark::pick_landmarks(&src, a.landmarks, a.candidates, 0xA5A5_5A5A_DEAD_BEEF);
            let idx = landmark::LandmarkIndex::build(&src, lm)?;
            landmark::measure(&src, &idx, a.rows, a.tol_units)
        }
        (None, Some(path)) => {
            let m = matrix::Matrix::load(path)?;
            let n = m.n;
            let d = m.d.iter().map(|v| (v * a.scale).round() as i32).collect();
            let src = landmark::DenseSource { n, d };
            eprintln!("{n}×{n} matrix, picking {} landmarks…", a.landmarks);
            let lm =
                landmark::pick_landmarks(&src, a.landmarks, a.candidates, 0xA5A5_5A5A_DEAD_BEEF);
            let idx = landmark::LandmarkIndex::build(&src, lm)?;
            landmark::measure(&src, &idx, a.rows, a.tol_units)
        }
        _ => bail!("pass exactly one of --embeddings (with --dim) or --matrix"),
    };
    rep.quantum_units_per_radian = a.scale;

    println!("{:>28} {:>12}", "landmarks", rep.landmarks);
    println!(
        "{:>28} {:>11.1} KiB",
        "resident index",
        rep.resident_bytes as f64 / 1024.0
    );
    println!("{:>28} {:>12}", "rows sampled", rep.rows_sampled);
    println!(
        "{:>28} {:>11.2}%",
        "blocks exact (no decode)", rep.exact_block_pct
    );
    println!(
        "{:>28} {:>11.2}%",
        format!("blocks within {} units", rep.tol_units),
        rep.within_tol_block_pct
    );
    println!("{:>28} {:>11.2}%", "cells exact", rep.exact_cell_pct);
    println!("{:>28} {:>11.3}", "residual / distance", rep.residual_ratio);
    println!(
        "{:>28} {:>12}  (worst {} units)",
        "triangle violations", rep.triangle_violations, rep.max_violation
    );
    println!("{:>28} {:>11.1}", "mean bound width", rep.mean_bound_width);
    if rep.max_violation > 2 {
        println!("\nWARNING: violations exceed rounding, so the source is not a metric and");
        println!("cell_bounds is not valid on it.");
    } else if rep.triangle_violations > 0 {
        println!(
            "\n(violations are all within {} units — quantisation, not a broken metric)",
            rep.max_violation
        );
    }
    println!(
        "\n{}",
        if rep.exact_block_pct >= 5.0 {
            "GO — a real share of blocks is answerable from the index alone."
        } else {
            "STOP — the landmark base explains almost nothing that a constant would not."
        }
    );
    std::fs::write(&a.out, serde_json::to_string_pretty(&rep)?)?;
    println!("wrote {}", a.out.display());
    Ok(())
}

fn from_embeddings(path: &Path, dim: usize, out: &Path, limit: usize) -> Result<()> {
    let raw = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if raw.len() % (dim * 4) != 0 {
        bail!(
            "{} is {} bytes, not a whole number of {dim}-dimensional f32 rows",
            path.display(),
            raw.len()
        );
    }
    let mut n = raw.len() / (dim * 4);
    if limit > 0 && limit < n {
        n = limit;
    }
    let emb: Vec<f32> = raw[..n * dim * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    eprintln!("building {n}×{n} angular distances from {dim}-dim embeddings…");
    let m = matrix::angular_from_embeddings(&emb, n, dim);
    m.save(out)?;
    println!(
        "wrote {} ({n}×{n}, {:.1} MB)",
        out.display(),
        (n * n * 4) as f64 / 1e6
    );
    Ok(())
}

fn from_trace(trace: &Path, layer: u16, out: &Path, reinforce: f64) -> Result<()> {
    let tr = coact::trace::Trace::load(trace)?;
    let lt = tr
        .layers
        .iter()
        .find(|l| l.layer == layer)
        .ok_or_else(|| anyhow::anyhow!("trace has no layer {layer}"))?;
    let n = tr.n_expert as usize;

    // Co-activation counts, and the marginals needed to turn them into lift.
    let mut w = vec![0.0f64; n * n];
    let mut freq = vec![0.0f64; n];
    #[allow(unused_assignments)]
    for t in 0..lt.n_tokens() {
        let tok = lt.token(t);
        for (a, &i) in tok.iter().enumerate() {
            freq[i as usize] += 1.0;
            for &j in &tok[a + 1..] {
                w[i as usize * n + j as usize] += 1.0;
                w[j as usize * n + i as usize] += 1.0;
            }
        }
    }
    let nt = lt.n_tokens() as f64;

    // Reinforcement: scale each pair's weight by its own lift, raised to alpha. Pairs already
    // above chance gain, pairs below lose, and the marginals are then recomputed from the
    // reshaped weights so lift is measured against the new distribution rather than the old.
    if reinforce > 0.0 {
        let mut scaled = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let expected = freq[i] * freq[j] / nt;
                if expected <= 0.0 {
                    continue;
                }
                let lift = w[i * n + j] / expected;
                if lift > 0.0 {
                    scaled[i * n + j] = w[i * n + j] * lift.powf(reinforce);
                }
            }
        }
        w = scaled;
        for (i, f) in freq.iter_mut().enumerate() {
            *f = (0..n).map(|j| w[i * n + j]).sum::<f64>() / 2.0;
        }
        eprintln!("reinforced with alpha = {reinforce}");
    }
    let nt = freq.iter().sum::<f64>() / (lt.top_k as f64 / 2.0).max(1.0);

    // Similarity to distance: d = -ln(lift). Experts that co-occur far more than chance are
    // "close". Whether the result obeys the triangle inequality is exactly what the probe
    // asks — there is no reason from the construction that it should.
    let mut m = Matrix::zeros(n);
    let mut max_finite = 0.0f32;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let expected = freq[i] * freq[j] / nt;
            let lift = if expected > 0.0 {
                w[i * n + j] / expected
            } else {
                0.0
            };
            let d = if lift > 0.0 {
                (-lift.ln()) as f32
            } else {
                f32::INFINITY
            };
            if d.is_finite() && d > max_finite {
                max_finite = d;
            }
            m.set(i, j, d);
        }
    }
    // Pairs that never co-occur would otherwise be infinitely far; cap them just above the
    // largest real distance so the matrix stays finite without inventing a smaller value.
    let cap = max_finite * 1.1;
    for v in m.d.iter_mut() {
        if !v.is_finite() {
            *v = cap;
        }
    }
    // Shift so the smallest off-diagonal distance is zero, matching a distance convention.
    let lo = (0..n)
        .flat_map(|i| (0..n).filter(move |&j| i != j).map(move |j| (i, j)))
        .map(|(i, j)| m.get(i, j))
        .fold(f32::INFINITY, f32::min);
    if lo.is_finite() && lo < 0.0 {
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    m.set(i, j, m.get(i, j) - lo);
                }
            }
        }
    }
    m.symmetrise();
    m.save(out)?;
    println!(
        "wrote {} ({n}×{n} from layer {layer}, {} tokens)",
        out.display(),
        lt.n_tokens()
    );
    Ok(())
}

fn geodesic_cmd(path: &Path, k: usize, out: &Path) -> Result<()> {
    let mut m = Matrix::load(path)?;
    m.symmetrise();
    eprintln!(
        "building a {k}-nearest-neighbour graph over {} points…",
        m.n
    );
    let g = geodesic::build_knn(&m, k);
    let edges = g.edges.len() / 2;
    eprintln!("{edges} undirected edges; running Dijkstra from every source…");
    let (geo, disconnected) = geodesic::all_pairs(&g);
    if disconnected {
        eprintln!(
            "warning: the graph is not connected at k={k}. Unreachable pairs were capped just \
             above the largest real distance, which is a choice, not a measurement — raise k."
        );
    }
    geo.save(out)?;
    println!("wrote {} ({}×{})", out.display(), geo.n, geo.n);
    Ok(())
}

#[derive(Serialize)]
struct ProbeReport {
    matrix: String,
    n: usize,
    quantum: f64,
    tol_units: i64,
    spread: metric::SpreadReport,
    triangle: metric::TriangleReport,
    clusters: metric::ClusterReport,
    rank1: metric::Rank1Report,
    /// Predicted compression, against matcodec's own measured reference points.
    predicted_ratio: f64,
    verdict: String,
}

fn probe(
    path: &Path,
    k: usize,
    triples: u64,
    quantum: Option<f64>,
    tol_units: i64,
    out: &Path,
) -> Result<()> {
    let mut m = Matrix::load(path)?;
    m.symmetrise();
    let n = m.n;

    // Default quantum: 1/1000 of the mean off-diagonal distance, so the integer grid is fine
    // enough that quantisation is not what the residual is measuring.
    let mean: f64 = {
        let mut s = 0.0f64;
        let mut c = 0u64;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    s += m.get(i, j) as f64;
                    c += 1;
                }
            }
        }
        s / c.max(1) as f64
    };
    let quantum = quantum.unwrap_or((mean / 1000.0).max(1e-9));

    println!(
        "matrix {} — {n}×{n}, mean distance {mean:.4}, quantum {quantum:.3e}\n",
        path.display()
    );

    let sp = metric::spread(&m);
    println!("0. distance spread");
    println!(
        "   mean {:.4}, sd {:.4}, CV {:.3}; p01 {:.4} .. p99 {:.4}, dynamic range {:.2}x",
        sp.mean, sp.stddev, sp.coefficient_of_variation, sp.p01, sp.p99, sp.dynamic_range
    );
    println!(
        "   -> {}",
        if sp.coefficient_of_variation > 0.4 {
            "well spread — far-apart regions exist, which is what a gateway bridges."
        } else if sp.coefficient_of_variation > 0.2 {
            "moderately spread. Some regions are genuinely far apart, but the cuts between \
             them may still be too wide for a rank-1 base to beat a constant."
        } else {
            "concentrated — everything is roughly equidistant, so there are no far-apart \
             regions for a gateway to connect."
        }
    );

    println!("\n1. triangle inequality");
    let tri = metric::triangle(&m, triples, 0.01, 12345);
    println!(
        "   {} of {} sampled triples violate it ({:.4} %), worst by {:.1} % of the direct edge",
        tri.violations,
        tri.triples,
        tri.violation_pct,
        100.0 * tri.max_relative
    );
    println!(
        "   -> {}",
        if tri.is_metric {
            "metric. The rank-1 base and the O(L) bounds are meaningful."
        } else {
            "NOT a metric. Triangle-based bounds do not apply; half of matcodec is unusable."
        }
    );

    println!("\n2. clusterability (k-medoids, k = {k})");
    let (assign, _medoids) = metric::kmedoids(&m, k, 30, 7);
    let cq = metric::cluster_quality(&m, &assign, k, 512.min(n));
    println!(
        "   intra/inter {:.3}, silhouette {:.3}, cluster sizes {}..{}",
        cq.intra_over_inter, cq.mean_silhouette, cq.smallest_cluster, cq.largest_cluster
    );

    println!("\n3. rank-1 gateway fit on cross-cluster blocks");
    let r1 = metric::rank1_fit(&m, &assign, k, quantum, tol_units);
    println!(
        "   {} blocks, residual RMS {:.1} % of block RMS",
        r1.blocks,
        100.0 * r1.mean_residual_ratio
    );
    println!(
        "   {:.2} % of cells reproduced within tolerance, {:.2} % of blocks entirely so",
        r1.cells_within_tol_pct, r1.blocks_fully_within_tol_pct
    );
    println!(
        "   null model (block mean only) leaves {:.1} %; rank-1 explains {:.1} % of what is left",
        100.0 * r1.mean_only_residual_ratio,
        r1.rank1_gain_over_mean_pct
    );

    println!("\n4. residual entropy");
    println!(
        "   zigzag-varint {:.3}× raw, deflated {:.3}× raw  ->  predicted compression {:.2}×",
        r1.varint_ratio,
        r1.deflate_ratio,
        1.0 / r1.deflate_ratio.max(1e-9)
    );

    let predicted = 1.0 / r1.deflate_ratio.max(1e-9);
    // The verdict keys on matcodec's *mechanism*, not on the deflate ratio. A quantised
    // distance matrix compresses several-fold under plain entropy coding whatever its shape,
    // so reporting that as success would be measuring zlib and calling it a gateway model.
    // What matcodec uniquely provides is triangle bounds and blocks the resident index answers
    // without decompressing, so those are what decide.
    let verdict = if !tri.is_metric {
        "STOP — not a metric. matcodec's bounds machinery cannot be used here at all."
    } else if r1.rank1_gain_over_mean_pct < 20.0 {
        "STOP — the rank-1 gateway base does not beat a per-block constant. Any compression \
         here is generic entropy coding, not matcodec's mechanism."
    } else if r1.blocks_fully_within_tol_pct < 5.0 {
        "MARGINAL — the base fits, but almost no block is exact, so nothing can be read \
         without decoding."
    } else {
        "GO — gateway structure is present and worth a codec."
    };
    println!(
        "\nmatcodec reference: 6.4× real road matrix, ~10× single gateway, ~1.8× structureless"
    );
    println!("{verdict}");

    let rep = ProbeReport {
        matrix: path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
        n,
        quantum,
        tol_units,
        spread: sp,
        triangle: tri,
        clusters: cq,
        rank1: r1,
        predicted_ratio: predicted,
        verdict: verdict.to_string(),
    };
    if let Some(d) = out.parent() {
        std::fs::create_dir_all(d).ok();
    }
    std::fs::write(out, serde_json::to_string_pretty(&rep)?)?;
    println!("\nwrote {}", out.display());
    Ok(())
}
