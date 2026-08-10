# LLMdb — making a memory-starved MoE fetch faster

> **Correction, 2026-08-08.** The router trace these results were computed from was an
> artefact: `ffn_moe_topk` is a view of llama.cpp's argsort node, so a flat backend read
> returned each token's entire 64-wide expert ranking instead of its top-k. That made expert
> access uniform *by construction*, which is the premise most of the conclusions below rest on.
> On a corrected trace LRU hits **78.6 %** where the artefact reported 4.5 %, and real routing
> has **47x** frequency skew rather than 1.00. Every trace-derived number here is suspended
> pending a re-run; see the last entry in [SUMMARY.md](SUMMARY.md) for exactly which ones and
> which stand. `make check-numbers` no longer asserts the suspended ones.


Take a MoE model that does not fit in RAM. Trace how it actually routes. Reorder the experts
on disk so a token's are adjacent. The reordering is exact, reversible, done in place, and
needs no fork of llama.cpp.

Measured cold on an M3 Pro, at zero quality cost: **7.8 % on OLMoE-1B-7B** and **14.0 % on
Qwen3.6-35B-A3B**. Parallel reads are worth much more — up to 2.7× from queue depth alone —
and the two partly overlap. Against a naive serial reader on the shipped file, queue depth 4
plus the layout is **2.38×** on Qwen3.6-35B-A3B.

Whether it is worth anything for *your* model is predicted by one number, readable from the
GGUF header in an 8 MB range request: **requests per mebibyte of per-token traffic**. Above 3,
expect low double digits; below 0.5, the model is bandwidth-bound and this will not help.
Model size does not predict it — the 480 B-class models score lowest, because they use few
large experts.

**End to end with a hard memory budget, output unchanged: 3.1× on OLMoE-1B-7B (2 GiB) and
3.92× on Qwen3.6-35B-A3B (4 GiB)** — queue depth, then cache policy, then layout. Adopt them
in that order; only the last one touches the model file.

The largest single win is not the layout at all: **replacing LRU with static frequency pinning
is worth 2.24×** at a fixed memory budget. Load-balanced routing is near-uniform, which is
the textbook LRU pathology — it hits 4.5 % where holding 54 % of the data should hit 54 %.

A second study asks whether [MPEE](https://github.com/punnerud/mpee)'s matcodec — which
compresses a distance matrix 6.4× and answers many cells *without decompressing* — transfers to
an LLM stack. Measured: no, and for a precise reason. Expert co-activation is not a metric;
embedding space is a perfect metric with nothing in it (384 dimensions, distance CV 0.076).
Treating embeddings as a road network instead of a straight-line metric manufactures 3.7× more
structure, but a kNN graph over a concentrated cloud is an expander — no narrow cuts, and
narrow cuts are what a gateway codec needs. See [`design/DESIGN-MPEE-TRANSFER.md`](design/DESIGN-MPEE-TRANSFER.md).

The repository also contains the record of everything that *did not* work, which is most of
it. Fetching fewer experts, reranking them, recombining them, caching them and prefetching
them were all measured and all fail, for two reasons that are properties of how MoE models are
trained. [`RESULTS.md`](RESULTS.md) has the numbers, [`design/DESIGN.md`](design/DESIGN.md)
the reasoning, [`SUMMARY.md`](SUMMARY.md) the chronology including the mistakes.

## Quick start

```sh
make model      # download the GGUF and record its pristine SHA-256
make baseline   # llama-bench + reference logits, on the shipped file
make pipeline   # trace -> cost model -> layout search -> cold-cache measurement
make apply      # rewrite the file in place with the winning layout
make verify     # generations and argmax must be unchanged
make revert     # restore the shipped order, proven by SHA-256
```

`make study` runs every experiment, including the negative ones. `make help` lists all
targets. `make all` is the quality gate: `cargo fmt --check`, `clippy -D warnings`, tests.

## The headline findings

| lever | what it needs | measured | outcome |
|---|---|---|---|
| Prefetch | layer L predicts L+1 | 0.66 of 5.96 bits | dead |
| Caching with LRU | recency predicts reuse | 4.5 % hit at 54 % residency | actively harmful |
| **Static expert pinning** | any frequency skew | 55.4 % hit at 54 % residency | **2.24×** |
| Truncation | some experts contribute little | rank 8 = 4.4 % | costly |
| Reranking | gate order ≠ contribution order | Spearman 0.86 | ≤ 2.4 pp |
| Recombination | kept experts substitute for dropped | pairwise cosine 0.04 | ≤ 3.0 pp |
| Per-layer budget | layers differ in sensitivity | 3.23 pp at 7/layer | 3–5 %, needs fork |
| **Queue depth** | device has spare parallelism | 2.46 → 3.90 GB/s | **1.6×** |
| **Layout** | co-selected experts can be adjacent | 14.4 % fewer fetches | **7.8 %, exact** |

Load balancing makes expert access near-uniform, which kills everything that depends on skew.
The experts learn near-orthogonal functions, which kills everything that depends on
redundancy. Layout survives because it changes nothing about the computation.

## Crates

| Crate | Purpose |
|---|---|
| `gguf` | GGUF v3 reader. Per-type block geometry, expert byte ranges, header-only parsing so a 400 GB model can be analysed from an 8 MB range request. |
| `moetrace` | C shim over `libllama`. Captures router decisions, gate weights, per-expert output norms, and optionally the full output vectors. |
| `coact` | Co-activation graph and layout search, plus `headroom`, `reweight`, `ensemble`, `analyze`, `compare`, `project`. |
| `fetchbench` | `F_NOCACHE` replay, device calibration, queue-depth sweep, LRU expert cache with a hard memory budget. |
| `ggufperm` | The in-place lossless permutation, with SHA-256-verified revert. |
| `ballast` | Holds incompressible RAM resident, for memory-pressure experiments. |
| `experiments/sparsemem` | The controlled test of the whole project's explanation: same memory net trained with and without load balancing, measuring how much of it has to come off disk. |
| `matstruct` | Does a matrix have the gateway structure MPEE's matcodec compresses? Triangle inequality, clusterability, rank-1 fit against a null model, residual entropy — plus a kNN-geodesic transform that manufactures structure the raw metric lacks. |

## Requirements

- llama.cpp with headers (`brew install llama.cpp`); `pkg-config llama` must resolve.
- Rust, Python 3, `hf` for the download.
- macOS for `fetchbench`: uncached reads use `fcntl(F_NOCACHE)`. Everything else is portable.
- ~5.5 GB of disk. There is never a second copy of the model.

## Safety

`ggufperm` mutates a multi-gigabyte download in place. Before it will touch anything,
`ggufperm init` records the file's pristine SHA-256 in a sidecar, and `ggufperm revert`
refuses to report success unless the file hashes back to exactly that. `--dry-run` reports
what would move without writing.

## MPEqs — the question-solver study (loopmem)

The second half of the repository is a phase-by-phase study of how far a small local
LLM gets when everything that CAN be mechanical IS mechanical: the model only reads and
plans, while an exact record executes reversible bricks (rRETL), routes units through
type space, solves equations by both-sides operations, stores solved roads and
translations as memory, maps problems onto a library of generic exact solvers, and gates
every delivery mechanically.

### What it is measured to do

| band | the model alone | through MPEqs |
|---|---|---|
| Grade-school word problems (GSM8K, n=30) | **28/30** | 21/30 |
| Hard exact arithmetic (held out, n=20) | 0/20 | **17/20** |
| Exact probability and unit conversion (n=20) | 1/20 | **17/20** |
| Statistics, calendars, percentage chains (n=18) | 8/18 | **18/18** |
| Nine classes mixed, retrieved exemplars (n=27) | 7/27 | **25/27**, none wrong |
| The same 27 driven by a **1B** model | 1/27 | **12/27** (27/27 specs valid) |
| Shape, sets, formulas — easy / hard (n=15 each) | 15/15 / 7/15 | 15/15 / **14/15** |
| Competition mathematics (AIME, n=15) | 2/10 solo | 1/15 mapped (5/15 hand-mapped) |
| Sequences, matrices, partitions, logs (n=16) | 3/16 | **15/16** |
| Bases, approximation, words, primes (n=16) | 5/16 | **16/16** |
| Seventeen classes at 37 solvers (n=34) | 10/34 | **33/34** |
| Fresh 24, full policy live (n=24) | 3/24 | **23/24**, zero wrong |
| **Every two-armed battery together** | **95/309** | **261/309** |

The middle row is the point: twelve-step exact fraction folds, big-integer powers, counts
over hundreds of thousands, divisor structure of large factorials — the model answers all
twenty and gets none right, never once abstaining, while mapping them onto exact machines
gets seventeen. The top row is the honest counterweight: where a model can already do the
sums, routing through machinery only adds a place to slip.

### How it works

1. **Solvers** (`solvers.py`, `solvers2.py`) — 37 parameterised exact machines: a generic
   search over ranges/divisors/factorial-divisors with 22 predicates, multi-variable
   search whose conditions are AST-vetted arithmetic expressions, a fold, exact linear
   systems, polynomials, CRT, modular arithmetic, coordinate geometry, series,
   combinatorics, and the word-problem staples. Everything in `Fraction`; every refusal
   named; a linear innermost variable is solved rather than scanned.
2. **Mapping** (`solvemap.py`, `mapmemory.py`) — the model emits a typed spec, never code.
   Embeddings propose the solver from a bank of exemplar mappings; the typed shape
   disposes; the slots may overrule a wrong solver name.
3. **Gates** (`specgate.py`, `gateablate.py`) — value echo (every literal must appear in
   the problem) and two-machine agreement. Measured on both sides: they make delivery
   lie-free where the risk is mapping, and they do not transfer to the arithmetic band.

The scaling recipe, each part measured before being combined: an exact machine that
refuses by name, a schema line that states its units and leads with its distinction
(the model reads it literally), and an exemplar that is **retrieved** rather than fixed
— a catalogue cannot grow inside a prompt. With generic examples a nine-class mixed
battery scores 20/27 and two classes vanish entirely; with retrieval, 25/27 and nothing
wrong.

### Reading it

- [`summary2-part1.txt`](summary2-part1.txt) (phases 0–25),
  [`summary2-part2.txt`](summary2-part2.txt) (26–49),
  [`summary2-part3.txt`](summary2-part3.txt) (50–91),
  [`summary2.txt`](summary2.txt) (92 onward) — the full chronology, negative results and
  autopsied mistakes included.
- [`experiments/loopmem/`](experiments/loopmem/) — one self-contained script per phase.
- `make all` — the quality gate; `scripts/check-numbers.py` re-verifies **570 pinned
  numbers** from the phase results in `data/custom/*.json`.

---

**MPEqs** stands for **Morten Punnerud-Engelstad Question Solver**.
