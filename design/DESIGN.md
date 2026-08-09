# Why a memory-starved MoE is slow, and which of the obvious fixes actually work

## The question

Put a several-hundred-gigabyte MoE model on an SSD, on a machine with a fraction of that in
RAM. Learn from how the model is actually used. Make it meaningfully faster.

Two families of answer suggest themselves, and only one of them survives measurement.

**Fetch fewer experts.** The router spreads each token across eight of sixty-four. If most of
that is slack — an artefact of the load-balancing loss rather than something the task needs —
then routing to four would halve the disk traffic. This is the intuitive answer and it is
where most of this repository's effort went. It does not work, and the reasons are specific
and measurable.

**Fetch the same experts more cheaply.** Reorder them on disk so a token's eight land next to
each other, issue the reads in parallel, and hold the hottest ones in whatever memory there is.
All three work, all three are exactly lossless, and none needs a fork of llama.cpp. Ranked by
what they are worth: **cache policy ≈ 2.05×**, **queue depth up to 2.7×**, **layout 7.8–14.0 %**.
The first was nearly missed, because the obvious policy is LRU and LRU is the one that fails.

`RESULTS.md` has the numbers. This document is the reasoning and the design.

## The shape of the problem

`coact stats` on `OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf`:

```
MoE layers 16    experts/layer 64    experts/token 8
expert weights 3.90 GB (92.6 % of file)
per decode token 465.0 MiB in 384 disjoint ranges
```

Three facts set everything up.

**One expert is not one range.** llama.cpp fuses each layer's experts into three tensors —
`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` — with the expert axis last. Expert *e* of
layer *L* is three separate slabs, roughly 100 MB apart. Eight experts, three tensors, sixteen
layers: 384 ranges per token.

**Quantisation is not uniform.** Q4_K_M keeps `ffn_down_exps` at Q6_K in half the layers and
Q4_K in the rest, so bytes-per-expert differs by 13 % between layers. Every calculation here
is per-layer.

**Cold reads are expensive.** Measured on this SSD after evicting the page cache:
`C_fetch = 230.74 µs`, `C_byte = 0.2856 ns/B`. A cold 4 KiB read takes 220 µs. Per-fetch
overhead is 39 % of the per-token cost — which is the entire budget any layout can compete
for, and the reason layout is worth anything at all.

## The lossless lever: expert layout

Permuting the last axis of the `*_exps` tensors moves experts between physical slots.
Permuting the last axis of the router weight `ffn_gate_inp` renames them to match. In exact
arithmetic the pair is a relabelling: the model computes the identical function. Because it
swaps equal-sized blocks, `ggufperm` does it **in place** by cycle-following with two block
buffers — peak extra disk, zero.

Which order? Build a co-activation graph from a real router trace, where `w(i,j)` counts
tokens that selected both, and lay it out so tightly-coupled experts are adjacent. Two
constructions are implemented: recursive spectral bisection with Kernighan–Lin refinement
(minimising cuts) and a greedy nearest-neighbour chain over normalised affinity (minimising
adjacency directly). The chain wins, which makes sense — the cut is a proxy and adjacency is
the objective.

Result: **14.4 % fewer fetches, 7.8 % less cold fetch time, zero quality cost.** Against a
random-layout noise floor of 1.002–1.015×, and with `frequency` (hot experts first) sitting on
zero, so the gain is attributable to the joint structure and not to anything simpler.

### On exactness

The project set out expecting bit-identical logits — an oracle of the kind that compression
claims never get, since those are irreducibly statistical. Half of that survived.

What survived: the **first MoE layer routes identically for all 4096 traced tokens**. Its
routing depends only on the embedding and the router weights, so a mismatch would mean the
permutation is wrong; there is no floating-point excuse available. Generations are
character-identical and argmax never moves across twelve prompts in nine languages and
notations.

What did not: logit bytes differ by up to 6 × 10⁻⁴ relative. llama.cpp computes gate weights
as `SOFT_MAX` over all 64 router logits, and that sum runs **in storage order**. Reordering
experts perturbs every gate weight in the last bits, and sixteen layers compound it. A control
rules out ordinary noise — the same file at 4, 5 and 8 threads gives bit-identical logits.

So the oracle exists, at the routing level rather than the bit level, and the gap is a
property of `ggml_soft_max` rather than of layout. A storage-order-independent reduction there
would close it.

## Why fetching fewer experts fails

Each of these was measured, and each failed for a different-looking reason that turns out to
be the same reason twice.

**Truncation is not free.** `llama-perplexity --kl-divergence` against the unmodified model's
own logits: k=7 costs KLD 0.057 and 3.5 % perplexity for a 12.5 % fetch reduction; k=4 costs
KLD 0.549 and 42 % perplexity for 50 %. The lossless layout removes 14.4 % of fetches at KLD
exactly zero, so **it beats the cheapest truncation on both axes at once**.

**Reranking has a 2.4-point ceiling.** The router orders by gate probability; what matters is
contribution, `w · ‖out‖`. Mean Spearman between the two orders is 0.859, and an oracle with
perfect knowledge captures at most 2.38 percentage points more contribution than the router's
own order. That bounds any learned reranker, and it is why the router-retraining branch was
abandoned after an hour of analysis rather than a week of building.

**No reweighting recovers a dropped expert.** With the raw output vectors captured, the
least-squares-optimal *per-expert* weights over the kept subset — the split and the combine
tuned jointly, with an oracle for both — move the error at keep=4 from 33.35 % to 30.32 %.
Three percentage points. That is the ceiling for any meta-model that overrides both selection
and merging, because it is fitted against the very output it is trying to reproduce.

The naive fix for `norm_topk_prob = false`, rescaling by `w_total / w_kept`, makes things
*worse*: 51.80 % at keep=4, because the optimal scalar is 1.049 and the heuristic wants 1.37.

**Caching works, but only once LRU is thrown out.** A 512-token document touches every expert
in every layer, and with LRU the hit rate is 4.5 % at 54 % residency — worse than random
replacement's 27.4 %, and far below the ~54 % that residency alone should deliver. Uniform
access is the textbook LRU pathology. Static frequency pinning reaches **55.4 %** and is
**2.05× faster than LRU** at the same budget. That is the largest single win measured here, and
it is a policy change, not an algorithm.

**Prefetching has nothing to predict.** Mutual information between consecutive layers' top-1
expert is 0.66 bits against 5.96 bits of entropy. Best single-guess accuracy: 9 %.

## The common cause

Two training decisions explain every failure above.

**Load balancing makes expert access near-uniform.** That is what the auxiliary loss is for —
without it the router collapses and most experts never receive gradient. The side effect is
that there is very little skew to exploit: not in the temporal distribution (so prefetching
fails), and not in the importance distribution (so truncation and reranking fail).

The marginal distribution is the interesting exception. It is *nearly* uniform, not uniform,
and the residual skew is enough that pinning the hottest experts beats their share of memory —
55.4 % of accesses from 54 % of the bytes. Near-uniformity does not kill caching; it kills
*adaptive* caching, because when recency carries no signal an eviction policy that acts on it
throws information away.

**The experts learn near-orthogonal functions.** Mean pairwise cosine between the eight
outputs of one token: 0.034–0.044. Alignment ratio `‖Σ w v‖ / Σ w‖v‖` is 0.477, against 0.354
for perfectly orthogonal and 1.0 for a unanimous vote. The layer is not an ensemble that
votes; it is eight specialists pointing in different directions. Orthogonal components cannot
be reconstructed from one another, so nothing that depends on redundancy — recombination,
merging, distilling into fewer — has anything to work with.

Layout survives because it **changes nothing about the computation**. It needs no skew, no
redundancy, no predictability — only that the eight chosen together can be placed side by side,
a property of the joint distribution that load balancing does not flatten. Static pinning
survives for the mirror-image reason: it needs nothing but the marginal distribution, and asks
no question that near-uniformity makes unanswerable.

## What else is worth doing in the memory-starved case

**Cache policy first, and it is not LRU.** See above: 2.05× for a policy change.

**Queue depth second.** Cold bandwidth goes 2.46 → 3.90 GB/s from queue depth 1 to 4, and
saturates there. That is a 1.6× — larger than layout. But the two compete: parallelism buys
down the same per-request latency layout was saving, so the layout gain shrinks from 3.3 % at
QD1 to 0.6 % at QD4. Use both; do not expect them to add.

**Bigger reads do not pay, for this model.** Bridging gaps to merge fetches raises bandwidth
1.6× while raising bytes 6.5×. It would pay for a model with small experts — at 820 KB per
expert the break-even gap is 2, not 0 — so it is a property of the model, and `fetchbench
replay --max-gap` exists to find out.

**Per-layer expert budgets.** Layers differ in sensitivity: truncation error at keep=7 ranges
from 8.8 % to 13.7 %. Spending the same budget greedily instead of uniformly cuts the error
from 10.88 % to 7.65 % — worth 3–5 % fewer fetches at fixed quality. This is the one
routing-side lever that survived, and it needs a fork, since `expert_used_count` is a single
global value in GGUF.

## Pipeline

| Stage | Crate | What it does |
|---|---|---|
| Trace | `moetrace` | Links `libllama` through a C shim; `cb_eval` captures router decisions, gate weights, and optionally each expert's full output vector |
| Model | `gguf` | GGUF v3 reader; per-type block geometry; expert byte ranges; header-only parsing for remote models |
| Search | `coact build` | Co-activation graph → spectral bisection, affinity chain, greedy refinement |
| Analysis | `coact` | `headroom`, `reweight`, `ensemble`, `analyze`, `project` |
| Measure | `fetchbench` | `F_NOCACHE` replay, device calibration, queue depth, LRU cache simulation |
| Rewrite | `ggufperm` | In-place lossless permutation with SHA-256-verified revert |

### Getting a real trace

`llama-debug --tensor-filter` exists but truncates every dimension to three elements — it
would report 6 of 8 expert ids. `llama-imatrix` records per-expert activation *counts*, which
are marginals; a co-activation graph needs joints. So `moetrace` links `libllama` directly.

C rather than hand-written Rust FFI, because `llama_context_params` and `ggml_tensor` are read
by value and a struct that drifts from the installed headers is silent memory corruption
rather than a link error.

One trap: llama.cpp prunes the final layer to the tokens whose logits are requested, so with
the default the last MoE layer is traced for one token per ubatch. `moetrace` sets
`batch.logits[i] = 1` for every token.

### Measurement hygiene, learned the hard way

Both of these produced wrong numbers before they were caught, and both are now enforced in
code rather than in discipline.

**`F_NOCACHE` does not evict.** It stops a descriptor from populating the cache; the kernel
still serves pages that are already resident. Right after `ggufperm` rewrites the file, replay
reports 17 GB/s. `fetchbench` now flags any run whose throughput exceeds the calibrated
ceiling.

**Calibrating on a warm file poisons everything downstream.** A contaminated calibration
reported `C_fetch = 6.66 µs` and 12.8 GB/s against the true 230.74 µs and 3.50 GB/s — a 35×
error that also disabled the guard above, since the guard compares against that very number.
`fetchbench calibrate` now evicts the cache itself before measuring.

**Holdout splits are not ceremony.** `best+greedy`, local search on the measured objective,
beat `chain` on the training split and lost on holdout.

**One prompt is not a measurement.** At k=4 the model still answers "The capital of France is
Paris." It also has 42 % worse perplexity.

## Where this does not apply

**Batching.** With batch 16 the union of selected experts covers essentially all 64, both
layouts collapse to one contiguous run per layer, and the gain is exactly zero. This is a
single-user, local-inference technique, not a serving technique.

**Machines that fit the model.** Warm, `llama-bench` shows no difference at all — correctly,
since nothing is being fetched. The warm run is the regression test, not the result.

**Very large models, partly.** Projected from GGUF headers with the measured device constants:
a 186 GB model runs at about 2.1 tokens/s here, bandwidth-bound, with 17.5 % of the cost being
per-request overhead. Layout competes for that 17.5 %; queue depth competes for more.
