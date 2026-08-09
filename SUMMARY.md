# Running log

Append-only. Newest entry at the bottom. `RESULTS.md` holds the settled numbers; this is
what we learned and in what order, including the wrong turns.

**Goal.** A general method: take a pretrained MoE LLM, learn from how it is actually used,
dampen the spread-out expert activation, and turn that into faster disk fetching — so the
model runs quicker on a machine that cannot hold it in RAM.

---

## 2026-08-08 07:00–08:45 — Phase 1: lossless layout

**Built.** Rust workspace: `gguf` (GGUF v3 reader with per-type block geometry), `moetrace`
(C shim over `libllama`, `cb_eval` capture of router decisions), `coact` (co-activation graph,
spectral bisection + Kernighan–Lin, greedy affinity chain), `fetchbench` (`F_NOCACHE` replay
and device calibration), `ggufperm` (in-place lossless permutation with SHA-256 revert),
`ballast`.

**Measured.** OLMoE-1B-7B Q4_K_M: 465 MiB touched per decode token in 384 disjoint ranges,
because llama.cpp fuses experts into three tensors and one expert is three slabs ~100 MB
apart. `chain` layout cuts fetches 14.4 % and cold fetch time 7.8 %, against a random noise
floor of 1.002–1.015×. Nothing warm. Nothing at batch 16.

**Learned the hard way.**

1. *Q4_K_M is not uniformly quantised.* `ffn_down_exps` is Q6_K in 8 of 16 layers, Q4_K in the
   rest. A global bytes-per-expert constant is wrong by 13 %. Everything is per-layer now.
2. *llama.cpp prunes the last layer* to the tokens whose logits are requested. The first trace
   captured layer 15 for exactly one token per ubatch. Fix: `batch.logits[i] = 1` for all.
3. *`F_NOCACHE` does not evict.* It stops a descriptor from populating the cache but the
   kernel still serves resident pages. Right after `ggufperm` rewrites the file, replay
   reports 17 GB/s — above the device ceiling. Cost one round of wrong numbers. `fetchbench`
   now refuses to report silently and flags any run above the calibrated bandwidth.
4. *The SSD is not affine.* 8 MiB reads sustain 12.75 GB/s, 16 MiB only 6.73. A cost model
   with only `C_fetch` and `C_byte` will merge straight past that cliff.
5. *Local NVMe is the worst case for this technique.* `C_fetch` is 6.66 µs against 36 ms of
   transfer per token. The per-fetch-cost sweep shows the same layout buying 6.9 % at 100 µs
   and 12.5 % at 1 ms — the regimes where MoE weights cross PCIe or a network.
6. *Local search overfits.* `best+greedy` beat `chain` on the training split and lost on
   holdout. The split earned its keep.

**The exactness claim, corrected.** The permutation is exact in exact arithmetic and the
implementation is provably right: the first MoE layer routes identically for all 4096 traced
tokens, and its routing depends only on the embedding and the router weights, so there is no
floating-point excuse available. But the logit bytes differ by up to 6 × 10⁻⁴ relative,
because llama.cpp computes gate weights as `SOFT_MAX` over all 64 router logits and that sum
runs *in storage order*. A control rules out ordinary noise: the same file at 4, 5 and 8
threads gives bit-identical logits. So the oracle lives at the routing level, not the bit
level — and closing the gap is a property of `ggml_soft_max`, not of layout.

**Activation analysis.** Contribution `gate × ‖output‖`: rank 1 = 28.6 %, rank 8 = 4.4 %.
Top-8 of 64 captures only 43 % of the softmax mass. Cross-layer mutual information between
consecutive top-1 experts: 0.66 bits against 5.96 bits of entropy — knowing the previous layer
removes 11 % of the uncertainty. **Every prefetch scheme in the MoE-offload literature bets on
that number being large. On this model it is not.** Which is the strongest argument for the
layout approach: it needs no prediction at all.

---

## 2026-08-08 08:45 — Phase 2 opens: is the flat router slack or information?

Hypothesis under test: the router is flat because load-balancing loss made it so during
training, not because the task needs eight experts. If that is slack, fetching fewer is a
bigger win than moving them around.

**First evidence, and a warning about anecdotes.** `--override-kv
olmoe.expert_used_count=int:K` changes top-k at load with no code change. On one prompt:

```
k=8: "The capital of France is Paris."      k=4: "The capital of France is Paris."
k=2: "...a city in France, it is not a country."   k=1: "the the the  the   .   ] ]"
```

That looked like a free 2×. The rigorous oracle says otherwise: `llama-perplexity
--kl-divergence` at k=4 gives mean KLD 0.60 and PPL ratio 1.40. "Paris" is simply an easy
token. **Recorded as a lesson: one prompt is not a measurement.**

**Reranking ceiling: 2.4 percentage points.** `coact headroom` compares what the router's own
order captures against what an oracle with perfect knowledge of every expert's contribution
would capture:

| keep | by gate | by oracle | headroom |
|---|---|---|---|
| 2 | 46.95 % | 49.33 % | **2.38 pp** |
| 4 | 72.81 % | 74.36 % | 1.54 pp |
| 6 | 88.99 % | 89.89 % | 0.90 pp |

Mean Spearman between gate order and contribution order: 0.859. **This kills the router
retraining plan** in its rerank form — an hour of analysis that saved a week of building. Any
learned reranker is bounded by a number that assumes perfect knowledge, and that number is
2.4 points.

Caveat that keeps the door open: this bounds reranking *within* the eight experts the router
already selected. We never observe the other 56, so it does not bound a router retrained to
select a different eight.

**Metal segfaults on odd k.** `kernel_mul_mm_id_map0_ne20_3` has no compiled variant, so k=3
crashes. The whole KL sweep runs on `-ngl 0`, base logits included, so the divergence measures
routing and not the backend.

---

## 2026-08-08 08:45–09:00 — The quality-versus-k frontier, measured properly

`llama-perplexity --kl-divergence`, 24 chunks of 512 tokens, CPU backend throughout:

| k | fetches | PPL ratio | mean KLD | median KLD | same top-1 |
|---|---|---|---|---|---|
| 8 | 100 % | 1.005 | 0 | 0 | 100 % |
| 7 | 88 % | 1.035 | 0.057 | 0.018 | 90.2 % |
| 6 | 75 % | 1.084 | 0.139 | 0.045 | 84.7 % |
| 5 | 62 % | 1.191 | 0.279 | 0.101 | 80.0 % |
| 4 | 50 % | 1.424 | 0.549 | 0.221 | 72.3 % |
| 3 | 38 % | 1.942 | 1.008 | 0.501 | 64.0 % |
| 2 | 25 % | 4.394 | 2.051 | 1.366 | 50.1 % |
| 1 | 12 % | 143.96 | 5.915 | 5.506 | 19.1 % |

**This inverts the working assumption.** The lossless layout gives a 14.4 % fetch reduction
at KLD exactly 0. The cheapest possible truncation, k=7, gives 12.5 % at KLD 0.057 and 3.5 %
worse perplexity. *Layout beats truncation on both axes simultaneously.* Anything that drops
experts has to clear a bar that the exact method already sits above.

## 2026-08-08 09:00 — Is the damage scale or information? Neither renormalising nor reranking saves it

`--override-kv olmoe.expert_weights_norm=bool:true` and `expert_weights_scale` are silently
ignored — the OLMoE graph does not read those keys — so the question had to be settled
offline on the actual output vectors. `moetrace --vecs` streams them as f16 (256 MB for 512
tokens), and `coact reweight` evaluates every truncation policy exactly.

Relative L2 error of the FFN output against the untruncated sum:

| keep | truncate | renormalised | oracle | oracle + renorm |
|---|---|---|---|---|
| 2 | 57.60 % | 114.56 % | 53.63 % | 125.44 % |
| 4 | 33.35 % | **51.80 %** | 30.77 % | 52.79 % |
| 6 | 17.51 % | 22.54 % | 16.02 % | 22.13 % |
| 7 | 10.88 % | 12.57 % | 9.78 % | 11.91 % |

**Renormalising makes it strictly worse, at every depth.** OLMoE has `norm_topk_prob = false`,
so the natural guess was that dropping half the experts shrinks the FFN output by a quarter
and one scalar fixes it. The opposite is true: the expert outputs partially **cancel**, so the
kept experts already have close to the right magnitude. Scaling them up by `w_total / w_kept`
overshoots by 18 points at keep=4.

Three things follow, and they agree with each other:

1. The damage is **direction, not magnitude**. No reweighting recovers it.
2. Oracle selection buys 2.6 points at keep=4 — matching `coact headroom`'s independent
   estimate of 1.54 pp of contribution. Two different measurements, same ceiling.
3. The experts are **complementary and partly cancelling**, not "one real expert plus seven
   riders". That is what a load-balanced distributed representation looks like from the
   inside, and it is why router-only retraining cannot make k=4 cheap: the information is
   genuinely spread across the eight, and the frozen experts cannot be asked to carry more.

Route not taken, and why: retraining the router to *select a different eight* is not bounded
by any of this, because we never observe the 56 unselected experts. But making k=4 as good as
k=8 would require the four kept experts to supply what eight supplied, which frozen weights
cannot do. That is expert fine-tuning, i.e. training a smaller MoE — a different project.

**Where this leaves the goal.** Fetching fewer experts is not the lever. The lever is fetching
the same experts more cheaply: layout (done, 7.8 % cold) plus residency. Residency is next.

---

## 2026-08-08 09:05–09:25 — Residency is dead too, and the merge step explains why

**There is no working set.** `fetchbench cache` on the real trace: a 512-token document
touches **64 of 64 experts in every layer**. LRU hit rate against cache size:

| cache | 256 MiB | 512 MiB | 1 GiB | 2 GiB | 3 GiB |
|---|---|---|---|---|---|
| hit rate | 0.0 % | 0.7 % | 1.9 % | 6.9 % | 31.9 % |

The expert weights total 3.72 GB. The cache only starts working when it is about to hold the
whole thing — which is not caching, that is just having enough RAM. Identity and chain give
the same hit rate to within 0.1 pp, because hit rate depends on *which* experts repeat, and
layout does not change that.

(The wall-clock column from this run is not trustworthy: one sample per configuration, taken
right after a heavy vector capture. The hit rates and byte counts are deterministic and are.)

**Do the experts vote?** Worth checking, because if a layer is an ensemble where a majority
must agree, dropping members breaks the vote and the fix would be a correction at the merge.
`coact ensemble` over 8192 records:

| rank | cos with total | cos with the other seven | share of total |
|---|---|---|---|
| 1 | 0.640 | 0.042 | 38.4 % |
| 4 | 0.340 | 0.043 | 9.7 % |
| 8 | 0.173 | 0.034 | 2.2 % |

Alignment ratio `‖Σ w v‖ / Σ w‖v‖` = **0.477**, against 0.354 for perfectly orthogonal and
1.0 for a unanimous vote. Mean pairwise cosine between experts: **0.034–0.044**.

So it is not a vote. The eight experts are **near-orthogonal specialists**, each carrying its
own direction, only weakly correlated with each other. No expert is redundant and none is
subtracting a correction.

**The best possible scalar cannot save truncation.** The earlier renormalisation test used the
obvious heuristic, `w_total / w_kept`. That is a strawman, so this run adds the
least-squares-optimal scalar — the floor for any rescaling whatsoever:

| keep | truncate | naive renorm | **best scalar** | optimal α |
|---|---|---|---|---|
| 2 | 57.60 % | 114.56 % | **56.99 %** | 1.105 |
| 4 | 33.35 % | 51.80 % | **32.96 %** | 1.049 |
| 6 | 17.51 % | 22.54 % | **17.37 %** | 1.018 |

The optimal scalar at keep=4 is 1.049, not the 1.37 the heuristic wanted, and it buys 0.4
percentage points. **No reweighting scheme can recover what truncation removes.** With
near-orthogonal experts the kept partial sum already has almost the right length; what is
missing is the directions of the ones you dropped, and a scalar cannot produce a direction.

## The pattern behind every negative result

Five separate levers were tested and four are dead. They fail for the same reason:

| lever | needs | measured | verdict |
|---|---|---|---|
| Prefetch | layer L predicts layer L+1 | 0.66 of 5.96 bits | dead |
| Caching | a document reuses a subset | 64 of 64 experts | dead |
| Truncation | some experts contribute little | rank 8 = 4.4 % | costly |
| Reranking | gate order ≠ contribution order | Spearman 0.86 | ≤ 2.4 pp |
| **Layout** | co-selected experts can be adjacent | 14.4 % fewer fetches | **works, and is exact** |

The load-balancing loss makes expert access close to **uniform random**. That single property
kills everything that depends on skew — in the marginal distribution, in the temporal
distribution, or in the importance distribution. Layout survives because it is the only lever
that exploits the **joint** distribution: it does not care that every expert is equally likely,
only that the eight chosen together can be made adjacent. And it is the only one that is
lossless.

That reframes the goal. The way to speed up a memory-starved MoE is not to fetch fewer
experts — it is to fetch the same experts more cheaply.

---

## 2026-08-08 09:30–09:50 — The calibration was contaminated by 35x, and correcting it fixed the story

The gap sweep said bridging never pays and bandwidth was flat at 3.9 GB/s, which contradicted
the calibrated 12.75 GB/s at 8 MiB. One of the two had to be wrong. Recalibrating immediately
after `drop-cache`:

| | contaminated | **truly cold** |
|---|---|---|
| C_fetch | 6.66 µs | **230.74 µs** |
| C_byte | 0.0781 ns/B | 0.2856 ns/B |
| asymptotic | 12.80 GB/s | **3.50 GB/s** |
| break-even fetch | 83 KiB | **789 KiB** |

A 4 KiB cold read on this SSD takes 220 µs. The earlier calibration had been reading pages
that were still resident from the trace and permutation runs. Worse, the contamination guard
compares achieved bandwidth against *this* number, so an inflated ceiling silently disabled
the guard that was supposed to catch it. `fetchbench calibrate` now evicts the cache itself
before measuring.

**Correcting it made the model and the measurement agree.** With the real constants, per-fetch
overhead is **39 %** of OLMoE's per-token cost, not 8 %, and the cost model predicts `chain`
at **5.22 %** against the 7.8 % measured cold. The earlier per-fetch-cost sweep had already
predicted ~10 % at 230 µs — it was right, we just did not know we were at that point on the
curve.

**Read size and queue depth both buy the same latency, and neither is free.** Cold bandwidth
by request size at QD1: 1 MiB → 1.82, 8 MiB → 3.21, 32 MiB → 3.69 GB/s. Queue depth 4 with
1.4 MiB requests already reaches 3.90 GB/s, so parallelism gets there without touching the
layout — and in doing so it eats layout's advantage, from 3.3 % at QD1 down to 0.6 % at QD4.

Bridging gaps to make reads bigger loses at every setting, at both queue depths:

| gap | identity MiB/token | GB/s | ms/token |
|---|---|---|---|
| 0 | 465 | 2.44 | **2402** |
| 4 | 821 | 3.15 | 3279 |
| 16 | 2423 | 3.41 | 8933 |
| 63 | 3029 | 3.92 | 9727 |

Bandwidth rises 1.6× across that range; bytes rise 6.5×. For OLMoE's 3.9 MB experts the trade
never closes. It would close for a model with many small experts — at 820 KB per expert the
break-even gap is 2, not 0 — which is a property of the model, not of the idea.

## 2026-08-08 09:50 — Co-tuning the split and the combine: the last hypothesis, closed

Fair criticism of everything above: each test changed **one** thing. Truncate without adapting
the merge. Rescale without changing the selection. Rerank inside a fixed eight. A method that
only works when both are tuned together would have looked like a failure in every one of them.

So: solve for the least-squares-optimal *per-expert* weights over the kept set. That is the
split and the combine tuned jointly, with an oracle for both — no learned meta-model can beat
it, because it is fitted against the very output it is trying to reproduce.

| keep | truncate | best scalar | oracle select | **best weights** | **oracle + best weights** |
|---|---|---|---|---|---|
| 2 | 57.60 % | 56.99 % | 53.63 % | 56.89 % | **52.94 %** |
| 4 | 33.35 % | 32.96 % | 30.77 % | 32.81 % | **30.32 %** |
| 6 | 17.51 % | 17.37 % | 16.02 % | 17.27 % | **15.84 %** |

At keep=4, everything a perfect meta-model could ever do — choose the best four *and* weight
them optimally — moves the error from 33.35 % to 30.32 %. **Three percentage points.**

And the geometry already predicted it. The experts are near-orthogonal (pairwise cosine 0.04).
Orthogonal components are by construction not reconstructible from one another: no choice of
weights on the kept four can synthesise a direction that only lived in the dropped four. The
ensemble measurement and the recombination bound are the same fact seen twice.

## Why every lever except layout failed — one cause

| lever | what it needs | measured | outcome |
|---|---|---|---|
| Prefetch | layer L predicts L+1 | 0.66 of 5.96 bits | dead |
| Caching | a document reuses a subset | 64 of 64 experts | dead |
| Truncation | some experts contribute little | rank 8 = 4.4 % | costly |
| Reranking | gate order ≠ contribution order | Spearman 0.86 | ≤ 2.4 pp |
| Recombination | kept experts can stand in for dropped | cosine 0.04 | ≤ 3.0 pp |
| Bigger reads | bandwidth grows faster than bytes | 1.6× vs 6.5× | loses |
| **Layout** | co-selected experts can be adjacent | 14.4 % fewer fetches | **7.8 %, and exact** |

Two facts do all the work. **Load balancing makes expert access near-uniform**, which kills
everything that depends on skew — in the marginal distribution (caching), the temporal one
(prefetch), or the importance one (truncation, reranking). **The experts are near-orthogonal**,
which kills everything that depends on redundancy (recombination, merging, distillation into
fewer).

Layout survives because it is the only lever that changes *nothing about the computation*. It
does not need experts to be skewed, redundant, or predictable. It only needs the eight chosen
together to be placeable side by side — a property of the joint distribution, which load
balancing does not flatten.

**For the stated goal — a several-hundred-GB model on a machine that cannot hold it — that
means the honest levers are, in order: queue depth (2.44 → 3.90 GB/s, 1.6×), then layout
(a further 5–8 %), and nothing else that does not cost quality.** Projected with the corrected
device constants: DeepSeek-V3 at 186 GB runs at ~2.1 tokens/s on this SSD, bandwidth-bound,
with 17.5 % of that being per-request overhead that layout can compete for.

---

## 2026-08-08 09:50–10:00 — One thing that does work on the routing side: per-layer budgets

Uniform top-k spends the same number of fetches on every layer. The layers are not equally
sensitive, so that is the wrong allocation. Per-layer truncation error at keep=7 ranges from
8.8 % (layer 14) to 13.7 % (layer 8).

Greedy allocation of the same total budget, against uniform top-k:

| budget | uniform error | allocated error | gain |
|---|---|---|---|
| 2/layer | 57.60 % | 57.60 % | 0.00 pp |
| 4/layer | 33.35 % | 33.23 % | 0.11 pp |
| 6/layer | 17.51 % | 15.77 % | 1.74 pp |
| 7/layer | 10.88 % | **7.65 %** | **3.23 pp** |

Near the shallow end there is nothing to redistribute — every layer is equally desperate. Near
the top there is: at 7 experts per layer the allocation cuts the error by 30 % relative, which
is worth roughly 3–5 % fewer fetches at fixed quality.

That is the one routing-side lever that survived, and it is small. It also needs a fork to
use, since `expert_used_count` is a single global value in GGUF.

---

## 2026-08-08 10:00 — Layout's value is a property of the model, and it varies fivefold

`coact project` now runs on five architectures from range-requested headers alone — olmoe,
qwen3_5_moe, qwen3moe, gpt-oss and deepseek2 — with nothing special-cased. What it shows is
the most useful generalisation in the whole study:

| model | ranges/token | MiB/token | fetch overhead | perfect-clustering ceiling |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | 960 | 314 | **70.2 %** | **2.593×** |
| Qwen3-30B-A3B | 1152 | 1046 | 45.9 % | 1.671× |
| OLMoE-1B-7B | 384 | 465 | 39.0 % | ~1.6× |
| gpt-oss-20b | 576 | 1213 | 26.8 % | 1.251× |
| DeepSeek-V3 | 1372/shard | 4984 | 17.5 % | 1.179× |
| Qwen3-Coder-480B | 1458/shard | 8110 | 12.2 % | 1.119× |

The deciding ratio is requests per byte. Many small experts → most of the cost is per-request
overhead → layout has everything to compete for. Few large experts → bandwidth-bound → layout
has almost nothing. Qwen3.6-35B-A3B, with 256 experts of ~110 KB per layer, is nearly twice as
favourable as OLMoE; the 480 B-class models are half as favourable.

**So "does expert layout help?" has no model-independent answer, and `coact project` answers it
for a specific model in about eight megabytes of download.** That is probably the most reusable
thing built here: before tracing anything or moving a byte, you can tell whether the technique
can possibly pay for your model.

The absolute numbers stay sobering. Qwen3-Coder-480B reads ~8 GB per token uncached — 0.36
tokens/s on this SSD, which layout moves to 0.37. When the active parameter set exceeds what
the device can stream in a reasonable time, no layout fixes it; a smaller quantisation does.

**Two measurement bugs, both now enforced in code rather than discipline.** `F_NOCACHE` does
not evict resident pages (guard added: flag any run above the calibrated ceiling). Calibrating
on a warm file poisoned every downstream number by 35× *and* disabled that guard, since the
guard compares against the calibration (fix: `calibrate` drops the cache itself). Both cost a
round of confidently wrong results.

---

## 2026-08-08 10:00 — The search is not the bottleneck; the structure is

`chain` captures 17 % of the clustering that is theoretically available (a random layout
averages `k(1 − (k−1)/(n−1))` = 7.13 runs per layer, a perfect one 1, and `chain` reaches
6.10). The obvious question is whether a better optimiser gets more.

Added a relocation neighbourhood to the local search — remove an expert from one slot and
reinsert it at another, shifting the block between. That is the standard escape from
swap-local optima in minimum-linear-arrangement problems, and swaps alone are a poor
neighbourhood because they disturb two positions at once and cannot slide a cluster along the
axis.

**Correction, caught by the compiler two hours later.** The first run of this reported
`best+greedy` at 293.9 against `chain`'s 292.9 and concluded relocation overfits. It did not:
`clippy -D warnings` flagged `relocate` as never called. The function had been added but the
loop that uses it had not, so that run measured swaps only and the conclusion was about
nothing. A dead-code warning caught a wrong experimental result — worth remembering.

With relocation actually wired in, `best+greedy` reaches **292.8** against `chain`'s 292.9. It
helps, by a tenth of a fetch per token. The revised conclusion is the same in substance and
better founded: **the construction already captures essentially all of the available gain, and
two different local-search neighbourhoods add a rounding error.** The limit is how much
co-activation structure exists, not the optimiser.

## Attempted: the many-small-experts case

`coact project` identified Qwen3.6-35B-A3B as the best case available — 40 layers × 256
experts of ~110 KB, 70.2 % fetch overhead, a perfect-clustering ceiling of 2.593×, against
OLMoE's 39 % and ~1.6×. That is the model this technique should be tested on, and the reason
is structural: many small experts means the per-token cost is dominated by request count
rather than bytes.

Started the download (UD-IQ1_M, 10.05 GB) after freeing regenerable data. It sustains
1.8 MB/s unauthenticated, an 89-minute ETA, which does not leave room to trace a 40-layer
model, search 256-expert layouts and measure before the deadline. Left running; the toolchain
is already parameterised for it (`make pipeline MODEL=...`) and needs no code change.

What that experiment would test, precisely: whether the 17 % capture rate is a property of
OLMoE or of MoE routing in general. If Qwen3.6 also captures ~17 %, the layout gain there is
`0.17 × (2.593 − 1) ≈ 27 %` of the fetch overhead, or roughly 11 % end to end — the number
already in the projection table. If the sparser routing (8 of 256 rather than 8 of 64) leaves
more structure to find, it could be several times that.

---

## 2026-08-08 10:18–10:25 — The many-small-experts case, measured. It is worse, not better.

`coact project` predicted Qwen3.6-35B-A3B as the best case: 40 layers × 256 experts, 70 %
fetch overhead, a perfect-clustering ceiling of 2.59×. Downloaded UD-IQ1_M (10.05 GB) and ran
the pipeline. Measured geometry: 811 KB per expert, **960 ranges per token for only 249 MiB**
— exactly the request-dominated regime the projection identified.

The layout gain is **half** what OLMoE gives.

| | OLMoE (64 experts) | Qwen3.6 (256 experts) |
|---|---|---|
| runs/layer, random | 7.13 | 7.78 |
| runs/layer, `chain` | 6.10 | 6.88 |
| clustering captured | 17.0 % | **13.3 %** |
| fetch reduction | 14.4 % | **6.6 %** |
| modelled gain | 5.22 % | **4.04 %** |

The two effects work against each other. More, smaller experts raise the share of cost that is
per-request overhead — good for layout. But they also make co-activation *sparser*: eight
selections out of 256 produce far rarer repeated pairs than eight out of 64, so there is less
structure for any clustering to find. The capture rate falls from 17 % to 13.3 %, and the two
effects nearly cancel.

Qwen3.6's router is flatter still: top-8 of 256 captures **18.5 %** of the softmax mass
(against OLMoE's 43 %), with rank 1 at 0.0514 and rank 8 at 0.0123.

**Correction to the projection.** `coact project --fetch-reduction 0.144` used OLMoE's measured
reduction for every model. That is now known to be optimistic for expert-rich models: Qwen3.6's
real value is 0.066. The projected 1.112× for Qwen3.6 should read closer to 1.05×. The flag
exists precisely so the assumption is visible, and the runbook now says to measure it rather
than inherit it.

**The generalisation.** Layout's payoff is the product of two model properties that move in
opposite directions as expert count grows: the fetch-overhead share (rises) and the achievable
clustering (falls). Neither can be read off the architecture alone — the first comes from the
header in eight megabytes, the second needs a trace. Expect single-digit percent either way.

## On training a routing model against internal embeddings

The proposal — train a small always-resident model on the hidden states that drive routing,
have it override which experts are used, and adjust the merge to match — is exactly the design
bounded above, and the bound is measured rather than argued:

- Reranking inside the selected eight: **≤ 2.4 pp** (`coact headroom`, oracle).
- Choosing the best subset *and* solving for optimal per-expert merge weights: **33.35 % →
  30.32 %** at keep=4 (`coact reweight`, oracle for both).

Both ceilings assume perfect knowledge, so no learned model can exceed them.

The geometry says why, and it extends beyond the selected eight. Mean pairwise cosine between
expert outputs is 0.034–0.044. A near-orthogonal basis cannot synthesise a direction that lies
outside its span, so no subset of size *m* reproduces a sum over *k > m* experts, whichever
subset is chosen. Selecting a different eight does not escape this; it only changes which
orthogonal directions are missing.

What that leaves open, honestly: we only ever observe the eight experts llama.cpp actually
computes, so the orthogonality measurement covers co-selected pairs rather than the whole
256×256 pool. Closing that would need a fork that evaluates unselected experts — which is
worth doing only if someone doubts the geometry generalises.

---

## 2026-08-08 10:26–10:35 — Correction: the many-small-experts case is better after all

The previous entry called Qwen3.6 "worse, not better". That was measured on the wrong
variable. Fetch *count* fell less (6.6 % against OLMoE's 14.4 %), but fetch *time* fell more.
Cold replay, 48 tokens, 5 reps, queue depth 1:

| layout | fetches/token | median ms | GB/s | speedup |
|---|---|---|---|---|
| identity | 883.1 | 8918.9 | 1.52 | 1.000× |
| random:1 | 879.0 | 8728.8 | 1.56 | 1.022× |
| frequency | 882.4 | 8632.2 | 1.57 | 1.033× |
| mincut | 786.6 | 8426.4 | 1.66 | 1.058× |
| **chain** | **749.4** | **7820.6** | **1.73** | **1.140×** |

**14.0 % faster on Qwen3.6 against 7.8 % on OLMoE**, against a noise floor of 1.022–1.033×.
Qwen3.6 issues 960 requests of 270 KB each per token; at that size the device only reaches
1.5 GB/s, so each request is dominated by latency and removing one is worth far more. Fetch
count was the wrong intermediate to judge by.

### The predictor: requests per mebibyte

| model | GB | ranges/MiB | overhead | measured cold gain |
|---|---|---|---|---|
| Qwen3.6-35B-A3B IQ1_M | 10 | **3.86** | — | **1.140×** |
| Qwen3.6-35B-A3B Q2_K_XL | 12 | 3.06 | 70.2 % | — |
| Qwen3-30B-A3B Q4_K_M | 19 | 1.10 | 45.9 % | — |
| OLMoE-1B-7B Q4_K_M | 4 | 0.83 | 39.0 % | 1.078× |
| gpt-oss-20b MXFP4 | 12 | 0.47 | 26.8 % | — |
| DeepSeek-V3 IQ1_S | 186 | 0.28 | 17.5 % | — |
| Qwen3-Coder-480B Q2_K_XL | 180 | **0.18** | 12.2 % | — |

Monotonic, and both measured points sit on it. `coact project` now reports this column first.

**This answers the scaling question, and the answer is not the hoped-for one.** Layout does
*not* scale with model size. It scales with request density, and the largest models available
today have the *lowest* density — DeepSeek-V3 at 0.28 and Qwen3-Coder-480B at 0.18, against
Qwen3.6-35B-A3B's 3.86 — because they use few large experts rather than many small ones. A
600 GB model is bandwidth-bound and layout is worth 1–3 % there; a 10 GB model with 256 tiny
experts is latency-bound and layout is worth 14 %.

The lever is expert *granularity*, not model size. That is a property an architect chooses,
and it is visible from the GGUF header before anything is downloaded.

## Generality, proven on a second architecture

`ggufperm` applied to Qwen3.6-35B-A3B — `qwen35moe`, 40 layers, 256 experts, IQ2_XXS, a
quantisation type never permuted before, 8.39 GB rewritten in 33 seconds in place:

- First MoE layer routes **100.0000 %** identically, as on OLMoE. The permutation is exact.
- All six greedy generations character-identical.
- `revert` restored SHA-256 `0dc2488c…` byte for byte on a 10 GB file.

Deep-layer drift is far larger here: 67.1 % same expert set and 49.2 % same rank order, against
OLMoE's 98.1 % and 93.4 %. Two causes compound — 40 layers instead of 16, and a much flatter
router (top-8 of 256 carries 18.5 % of the softmax mass, with rank 1 at 0.051 and rank 8 at
0.012), so the top-8 boundary is crowded with near-ties that a last-bit perturbation flips.

That the generations are nevertheless identical is the point: the flips are between experts
whose gate weights are almost equal, so the output barely moves. But it is a warning — on an
expert-rich model the routing-exactness proof has to lean on layer 0, because the aggregate
agreement number looks alarming and is not.

---

## 2026-08-08 10:36–10:42 — Do the clusters exist at all? Yes, but they are weak

The layout idea rests on an assumption nobody had tested: that experts form groups which recur
— "these three usually arrive together, so fetch them as one read". That is measurable.
Compare each pair's observed co-selection against independence:
`lift(i,j) = observed / (freq_i · freq_j / n_tokens)`.

| layer | median | p90 | p99 | max | pairs ≥ 2× | top-n mean |
|---|---|---|---|---|---|---|
| 0 | 0.844 | 1.242 | 2.250 | 3.96 | 1.29 % | 2.146 |
| 7 | 0.828 | 1.359 | 2.680 | 6.58 | 2.48 % | 2.779 |
| 15 | 0.789 | 1.477 | 3.086 | 5.53 | **4.02 %** | 2.929 |

Three things fall out.

**The median lift is below 1.** Top-k selection is mildly anti-correlated by construction:
picking one expert means not picking another, so a typical pair co-occurs slightly *less* than
independence predicts.

**Real groups exist, in a small minority.** Only 1–4 % of pairs are co-selected at least twice
as often as chance, with maxima of 4–6.6×.

**What a linear layout can actually use is 2.5×.** Each expert has two neighbours on a line, so
only about `n` pairs can ever be made adjacent. Those strongest `n` pairs average 2.50× lift —
genuine structure, but a long way from "always together".

That number *is* the 13–17 % capture rate, seen from the other side. The ping-pong grouping is
real, which is why `chain` beats random by a clear margin; it is weak, which is why it beats it
by 14 % rather than 80 %.

**Deeper layers cluster more.** Pairs above 2× rise monotonically from 1.29 % at layer 0 to
4.02 % at layer 15, and the exploitable lift from 2.15 to 2.93. Early layers route almost
independently; later ones specialise. A per-layer effort budget — search hard where the
structure is — is the obvious follow-up, and it is cheap because the analysis already reports
where to spend.

---

## 2026-08-08 10:42 — What the weak clustering means

The observation that a tightly co-activated group of two or three experts is "really what we
would otherwise have called one expert" is the right frame, and the measurements say those
groups are **soft**, not hard.

Two independent numbers agree on it. Co-activation lift among the pairs a linear layout can
actually exploit is **2.50×** — real, but nowhere near the 10–30× a hard group would show.
Mean pairwise cosine between the expert *outputs* is **0.034–0.044** — they are not computing
variations of the same thing.

Soft grouping is a stronger design than hard grouping, and that is why fine granularity is
worth its overhead. If expert A always arrived with B, the pair could be merged into one
larger expert at training time and nothing would be lost — and the model would then be one of
the coarse-grained ones that sit at the bottom of the requests-per-MiB table. Because A pairs
with B in one context and C in another, N experts taken k at a time provide combinatorially
many effective experts from the same parameters. The overlap *is* the capacity.

For this project the consequence is direct and slightly deflating: the effective fetch unit is
the group, not the expert, but the groups are soft, so no fixed placement captures more than a
fraction of them. A layout puts each expert next to its two best partners; every other
partnership it has stays scattered. That is the mechanism behind the 13–17 % capture rate, and
it is a ceiling on placement as such — not on this particular search.

---

## 2026-08-08 10:41 — Queue depth on the model that matters: layout survives here

Same sweep as on OLMoE, run on Qwen3.6-35B-A3B, 32 tokens, 3 reps, cold before each:

| queue depth | identity | chain | GB/s | layout gain |
|---|---|---|---|---|
| 1 | 5021.9 ms | 4517.0 ms | 1.80 → 1.98 | **1.112×** |
| 2 | 2994.9 ms | 2763.8 ms | 3.01 → 3.24 | 1.084× |
| 4 | 2303.5 ms | 2113.1 ms | 3.92 → 4.24 | **1.090×** |
| 8 | 1885.9 ms | 1850.3 ms | 4.78 → 4.84 | 1.019× |

Two differences from OLMoE, both from the smaller request size. Parallelism scales further —
1.80 to 4.78 GB/s, a **2.7×**, against OLMoE's 1.6× — because 270 KB requests leave far more
latency for the device to overlap. And layout *survives* it: still 1.090× at queue depth 4,
where on OLMoE it had collapsed to 1.006×.

**Combined against a naive serial reader on the shipped file** — queue depth 1, identity, at
5021.9 ms — queue depth 4 with the chain layout gives **2.38×** (2113.1 ms) and queue depth 8
gives **2.71×** (1850.3 ms). That is the practical recommendation for a memory-starved
fine-grained MoE, and the largest result here by a wide margin.

(An earlier draft of this entry multiplied the 2.7× bandwidth scaling by the 1.09× layout gain
and reported 2.9×. Those are measured at different queue depths and do not compose; the
combined figure is the direct ratio of the two endpoints.)

## The cause is a training decision, and so is the cure

The diagnosis is right: expert access looks uniform because training made it so. Load
balancing exists to keep the router from collapsing, and it assumes an abundance regime where
every expert is resident anyway and spreading costs nothing. On a machine that cannot hold the
model, that assumption inverts — spreading is exactly what makes it slow.

Where the measurements complicate the proposed cure. The flatness is not only in *how often*
each expert is picked, which the auxiliary loss targets directly. It is also in *what the
experts learned to compute*: pairwise output cosine 0.034–0.044, near-orthogonal. That second
property is not a routing artefact, and no router can undo it. Keeping four of eight loses
33 % of the FFN output even with an oracle choosing the four and solving for the optimal merge
weights. The specialisation a sparse router would need is not present in the experts, so
retraining the router alone cannot create it.

But the same data points at a training-time change that would work, and it is the natural
conclusion of everything here. The exploitable co-activation lift is **2.50×**. An auxiliary
loss that rewards *cluster* locality — spread the load, but among a small number of co-located
groups rather than uniformly across all N — would raise that number directly. Everything
downstream scales with it: layout gain, cache hit rate, and the viability of swapping groups
of experts in and out. Load balancing already shapes the joint distribution as a side effect;
this would shape it on purpose.

That is a pretraining change, not an inference one, and it is outside what this repository can
test. What this repository does provide is the instrument: `coact analyze` reports the lift,
and `fetchbench` converts a change in lift into wall-clock, so the effect of such a loss would
be measurable rather than argued.

---

## 2026-08-08 10:49 — Caching confirmed dead on the second architecture too

Same measurement on Qwen3.6-35B-A3B, 192 tokens, against 8.36 GB of expert weights:

| cache | 512 MiB | 1 GiB | 2 GiB | 4 GiB | 6 GiB |
|---|---|---|---|---|---|
| hit rate | 0.0 % | 0.2 % | 0.7 % | 4.1 % | 19.2 % |

A 512-token document touches **256 of 256 experts** in every layer, exactly as OLMoE touched
64 of 64. The shape is identical, so the finding is a property of load-balanced MoE routing and
not a quirk of one model. A hard memory cap does not become useful until it approaches the full
expert footprint, which is not a cache — it is having enough RAM.

`chain` is faster than `identity` at every cache size (151.0 vs 155.5 ms/token uncached, 124.5
vs 135.3 at 6 GiB), which is the layout gain showing through independently of residency.

## Closing state

Both models are in their shipped byte order, SHA-256 verified. `make all` passes: fmt, clippy
with `-D warnings`, 22 tests. `make check-numbers` confirms every figure quoted in the docs
still matches the JSON it came from.

The two measurement errors that cost real time are now impossible to repeat silently:
`fetchbench calibrate` evicts the page cache itself, and `fetchbench replay` refuses to report
a throughput above the calibrated ceiling without warning. The one experimental error that a
compiler caught — a dead-code warning revealing that the relocation search had never run —
argues for keeping `-D warnings` on work like this, where a silently disabled code path
produces a plausible wrong result rather than a crash.

---

## 2026-08-08 10:53 — Caching was not dead. LRU was. This overturns an earlier conclusion.

The 0.7 % hit rate at 2 GiB should have been suspicious immediately. A cache holding 24 % of
the data ought to hit roughly 24 % of the time even with random replacement. Getting 0.7 %
means the policy was doing worse than chance — actively harmful, not merely useless.

It was. Load-balanced routing is near-uniform, which is the textbook LRU pathology: recency
carries no information, so every admission evicts something exactly as likely to be needed as
the newcomer. Comparing three policies at the same budget:

**OLMoE, 2 GiB against 3.72 GB of experts (54 % residency):**

| policy | hit rate | MiB/token | ms/token | vs LRU |
|---|---|---|---|---|
| LRU | 4.5 % | 445.0 | 187.55 | — |
| random replacement | 27.4 % | 338.4 | 145.15 | 1.29× |
| **static frequency pinning** | **55.4 %** | **209.8** | **91.60** | **2.05×** |

**Qwen3.6-35B-A3B, 4 GiB against 8.36 GB (48 % residency):**

| policy | hit rate | MiB/token | ms/token | vs LRU |
|---|---|---|---|---|
| LRU | 4.2 % | 241.8 | 145.76 | — |
| random replacement | 19.8 % | 202.4 | 120.70 | 1.21× |
| **static frequency pinning** | **51.7 %** | **122.6** | **77.56** | **1.88×** |

Static pinning preloads the globally hottest experts, ranked on the first half of the trace
and scored on the rest, and never evicts anything. Nothing adapts — that is the point.

Three things follow.

**The hit rate slightly exceeds the residency fraction** — 55.4 % at 54 %, and 25.8 % at 24 %
on Qwen3.6. So there *is* exploitable frequency skew, just not much, and a static policy
collects all of it while LRU collects none.

**Random replacement beats LRU by 5–6× in hit rate.** Whenever that is true, the adaptive
policy is not merely failing to help, it is destroying information.

**This is the largest single win in the study**, roughly 2×, and it is exactly the "hold some
experts, swap the rest dynamically" idea — working, once the swapping rule stops trying to be
clever. It also composes with layout: these runs already use the `chain` order, and layout
still shows through in the ms/token column.

**Correction to the record.** The earlier entries concluding "caching is dead, a document
touches 64 of 64 experts" were measuring LRU and generalising to caching. The working-set
observation stands — every expert *is* touched — but it does not imply that residency cannot
help. What matters is not whether an expert is touched again but *how often*, and a static
policy exploits that where LRU cannot.

---

## 2026-08-08 10:58 — The two levers are independent, and compose

Static pinning and layout act on different things — pinning removes traffic, layout makes what
remains cheaper to fetch — so the question is whether they overlap. OLMoE, 96 tokens:

| layout | cache | hit rate | MiB/token | ms/token |
|---|---|---|---|---|
| identity | none | 0.0 % | 466.3 | 201.48 |
| chain | none | 0.0 % | 466.2 | 198.20 |
| identity | 2 GiB pinned | 55.4 % | 209.9 | 93.70 |
| **chain** | **2 GiB pinned** | **55.4 %** | **209.8** | **91.85** |

Hit rate is identical for both layouts, as it must be: pinning ranks experts by frequency, and
placement does not change frequency. Layout keeps its edge on top of that, and the composed
result is **201.48 → 91.85 ms/token, 2.19×**, from a 2 GiB budget and a byte permutation.

Caveat on these particular timings: one sample per row, and the layout gap here (1.6–2.0 %)
is smaller than the 7.8 % from the dedicated five-repetition replay. The hit rates and byte
counts are exact; treat the millisecond column as indicative and the replay numbers as the
measurement.

## Where this ended up

Ranked by what they are worth on a machine that cannot hold the model:

| lever | worth | cost to adopt |
|---|---|---|
| Static expert pinning instead of LRU | **2.05×** | a policy change |
| Queue depth 4–8 instead of serial reads | **1.6–2.7×** | a thread pool |
| Co-activation layout | 7.8 % (OLMoE), 14.0 % (Qwen3.6) | one in-place rewrite, reversible |
| Everything on the routing side | ≤ 3 pp, and not lossless | not worth it |

The three that work share a property: none of them changes what the model computes. The ones
that failed all tried to, and ran into the same two training decisions — load balancing, which
flattens the access distribution, and near-orthogonal experts, which removes redundancy.

The single most useful diagnostic to come out of it is the cheapest: **requests per mebibyte of
per-token traffic**, readable from a GGUF header in an 8 MB range request, predicts whether any
of this can pay for a given model before anything is downloaded.

---

## 2026-08-08 11:00 — The memory-budget curve, and a rule of thumb

OLMoE, static pinning, `chain` layout, 96 tokens. Expert footprint 3.72 GB:

| budget | residency | hit rate | MiB/token | ms/token |
|---|---|---|---|---|
| 0 | 0 % | 0.0 % | 466.2 | 196.35 |
| 512 MiB | 13.8 % | 12.9 % | 402.1 | 170.78 |
| 1 GiB | 27.6 % | 27.3 % | 337.9 | 151.77 |
| 1.5 GiB | 41.4 % | 41.5 % | 274.0 | 130.21 |
| 2 GiB | 55.2 % | 55.4 % | 209.8 | 92.63 |
| 3 GiB | 82.8 % | 83.7 % | 81.4 | 46.41 |
| 3.62 GiB | 99.9 % | 99.7 % | 1.5 | 0.71 |

Hit rate tracks residency almost exactly, drifting slightly *above* it as the budget grows —
the frequency skew is real but small, and pinning collects more of it the more it can hold.

**Rule of thumb: fetch time is proportional to the fraction of the expert weights you cannot
hold.** `ms/token ≈ (1 − cache_fraction) × uncached_ms`. No knee, no threshold, no minimum
viable working set. Every gigabyte buys the same proportional speedup, which is the most
useful thing to know when sizing a machine for a model that does not fit — and it only holds
if the replacement policy is static, because LRU delivers essentially none of it.

---

## 2026-08-08 11:07 — All five policies. Adaptivity costs; it does not pay.

Two proposals worth testing against pure static pinning: a **hybrid** that pins the hot half of
the budget and runs replacement over the rest, and **ageing** — frequency scoring where every
slab's score halves periodically, so an expert that was hot and goes cold falls out. The second
is the principled version of "everything is really dynamic".

OLMoE, 2 GiB against 3.72 GB of experts (54 % residency), `chain` layout, 96 tokens:

| policy | hit rate | MiB/token | ms/token |
|---|---|---|---|
| LRU | 4.5 % | 445.0 | 203.58 |
| random replacement | 26.9 % | 340.8 | 145.35 |
| hybrid, half pinned | 28.4 % | 333.2 | 141.30 |
| decayed frequency (ageing) | 31.2 % | 320.9 | 137.06 |
| **static pinning** | **55.4 %** | **209.8** | **90.82** |

Ageing works as intended — it beats random and the hybrid, and it beats LRU by seven times —
but pure static still wins by a wide margin, and the hybrid is barely better than random.

The reason is that the access distribution is **stationary**. The globally hottest set,
measured once, is already the right answer, so every adaptive mechanism spends capacity
relearning it and evicts genuinely hot experts during local lulls. Splitting the budget is
worse still: the pinned half covers the highest-value bytes, and handing the other half to a
policy that cannot beat random throws it away.

**The honest boundary.** The corpus interleaves four registers every 4000 characters, so it
drifts fast and shallow. A real session — one user, one topic, an hour — drifts slowly and
deeply, and there ageing could win. `--policy decayed --halflife N` exists to test that on a
real workload trace; the machinery is built and the answer will be workload-specific. What is
not workload-specific is that **LRU is the wrong default**, by a factor of seven against the
weakest alternative measured.

---

## 2026-08-08 11:10 — Same ordering on Qwen3.6

4 GiB against 8.36 GB of experts (48 % residency), `chain` layout:

| policy | hit rate | MiB/token | ms/token |
|---|---|---|---|
| LRU | 4.2 % | 241.8 | 145.76 |
| decayed frequency | 15.4 % | 213.3 | 156.20 |
| random replacement | 17.6 % | 207.8 | 128.04 |
| hybrid, half pinned | 27.1 % | 184.2 | 132.80 |
| **static pinning** | **51.7 %** | **122.6** | **79.06** |

Static wins on both models by roughly the same margin, so the conclusion is not an OLMoE
artefact. Two smaller observations worth flagging rather than polishing away:

Ageing does *worse* than random here (15.4 % against 17.6 %), reversing the OLMoE ordering.
That is an artefact of the admission rule in this implementation — a newcomer is only admitted
if it already scores higher than the entry it would displace, so with 10 240 slabs competing
the cache freezes around whatever got in first. A proper LFU-DA admits and then ages. The
finding "static beats every adaptive policy tested" holds; "ageing beats random" does not
generalise, and the ordering between the weak policies is implementation-sensitive.

The hybrid has a better hit rate than random but a worse time on Qwen3.6, which is single-
sample timing noise on a 96-token run. Hit rates and byte counts are exact; the millisecond
column between adjacent policies is not.

---

## 2026-08-08 11:17 — The whole stack, measured together

`fetchbench cache` gained concurrent misses (layers stay sequential — the router for layer L+1
needs L's output — but a layer's misses are independent) and repetitions. OLMoE, 2 GiB budget,
64 tokens, median of 3:

| layout | policy | queue depth | hit rate | MiB/token | ms/token |
|---|---|---|---|---|---|
| identity | LRU | 1 | 4.7 % | 444.2 | 201.00 |
| identity | LRU | 4 | 4.7 % | 444.2 | 134.84 |
| chain | LRU | 4 | 4.7 % | 444.1 | 130.45 |
| chain | static | 1 | 55.4 % | 209.8 | 93.41 |
| **chain** | **static** | **4** | **55.4 %** | **209.8** | **62.80** |

**3.20× end to end**, on a memory budget of 2 GiB against a 3.72 GB expert footprint, with the
model's output unchanged. Decomposed multiplicatively:

| step | factor |
|---|---|
| queue depth 1 → 4 | 1.49× |
| identity → chain layout | 1.03× |
| LRU → static pinning | 2.08× |

The ordering matters for anyone implementing this: **parallelism and cache policy are almost
all of it, and both are cheaper to adopt than the layout.** The layout is the only part that
requires touching the model file, and it contributes 3 % once the other two are in place —
against 7.8 % measured alone. They overlap, as they must: all three buy down the same
per-request latency.

---

## 2026-08-08 11:25–11:40 — Phase 3: does MPEE's matcodec transfer to an LLM stack?

MPEE never materialises its N×N matrix — 50k×50k (~10 GB) streams through 500 MB — and its
**matcodec** stores such a matrix 6.4× smaller while answering many cells *without
decompressing*. The mechanism: road regions connect through a few gateways, so a cross-region
block is `D[a][b] = d(a,gw) + HWY + d(gw,b)`, additive rank-1, and the exact residual then has
almost no entropy. A resident landmark index answers zero-residual blocks in O(L) with no
inflate, and gives triangle bounds for solver pruning. On structureless points it degrades to
~1.8×, i.e. plain deflate.

The whole thing is conditional on that structure existing. `crates/matstruct` measures the
condition before any codec is written — the same discipline as `coact headroom`, which killed
router retraining in an hour.

### The probe, and the null model that matters

Four measurements: distance spread, the triangle inequality, k-medoids clusterability, and the
rank-1 fit with its residual entropy. Plus one guard that changed the conclusion: **the null
model.** A rank-1 base captures a block's mean for free, so residual-RMS-over-block-RMS looks
impressive whatever the data. What matters is whether rank-1 beats a per-block *constant*.

Validated against synthetic ground truth: a single-gateway world gives exactly zero residual
and 100 % of blocks reproducible; structureless noise leaves >20 %; a planted triangle
violation is caught; k-medoids recovers planted clusters; a ring's geodesic inflates the chord
into the arc.

### Result 1 — the expert graph is not a metric

Co-activation converted to distance by `d = -ln(lift)`, layer 8, 200k tokens:

| | value |
|---|---|
| triangle violations | **0.36 %** of triples, worst by 31 % of the direct edge |
| silhouette | 0.168 |
| rank-1 over the null | +11.0 % |
| blocks readable without decoding | **0.00 %** |
| deflate | 3.41× |

Not a metric, so `cell_bounds` — half of matcodec — cannot be used at all. The 3.41× is zlib on
a small quantised matrix, nothing to do with gateways.

### Result 2 — embeddings are a perfect metric with no structure

3893 corpus chunks, all-minilm, 384 dimensions, angular distance:

| | value |
|---|---|
| triangle violations | **0 of 499 610** |
| coefficient of variation | **0.076** |
| dynamic range (p99/p01) | **1.61×** |
| silhouette | 0.051 |
| rank-1 over the null | **−93.1 %** |
| blocks readable without decoding | 0.00 % |

Mean distance 1.5163 against π/2 = 1.5708: this is textbook **distance concentration**. In 384
dimensions every pair is at right angles to every other and nothing is far from anything. The
rank-1 base is not merely useless here, it is *worse than a constant* — it adds parameters to
model variation that does not exist.

### Result 3 — routing beats flying, and by a lot

The idea that turned this around: raw angular distance is the "fly straight there" metric, and
a road network is not that. You cannot fly; you follow roads, and that is precisely what
creates bottlenecks. So build the roads — connect each point to its k nearest neighbours and
measure shortest paths through the graph. Crossing between clusters then *has* to pass through
whichever points bridge them, and shortest-path distance satisfies the triangle inequality by
construction.

| geodesic k | connected | CV | dynamic range | silhouette | rank-1 vs null | deflate |
|---|---|---|---|---|---|---|
| raw (∞) | — | 0.076 | 1.61× | 0.051 | −93.1 % | 3.34× |
| **11** | yes | **0.280** | **5.85×** | **0.191** | **−25.5 %** | 3.28× |
| 12 | yes | 0.275 | 5.65× | 0.163 | −27.8 % | 3.31× |
| 16 | yes | 0.265 | 5.14× | 0.117 | −52.1 % | 3.02× |
| 24 | yes | 0.238 | 4.24× | 0.172 | −41.6 % | 3.05× |

**The transform manufactures about 3.7× more structure on every axis** — spread, dynamic range
and clusterability all roughly quadruple — and it cuts the rank-1 penalty from −93 % to −25 %.
Below k=11 the graph disconnects, and the trend above it is monotone back towards the raw
metric, as it must be: at k = n−1 the geodesic *is* the straight line.

A trap caught on the way: at k=6 the probe reported 46 % of blocks readable without decoding.
That was an artefact. The graph was disconnected, unreachable pairs were capped at a single
constant, and blocks made entirely of that constant are trivially "exact". Only connected
graphs count, and every connected one reports 0 %.

### Where it stops, and why

Even at its best the geodesic matrix does not reach matcodec's mechanism: rank-1 still loses to
a per-block constant, and no block is exactly reproducible.

The reason is structural. **A kNN graph over a concentrated cloud is an expander** — richly
connected everywhere, with no narrow cuts. Road networks have narrow cuts because geography
imposes them: a river admits three bridges, a mountain range one pass. Semantic space has no
geography, so however the graph is drawn there is always another way around, and no small set
of points is on most paths between two regions.

That gives a clean, transferable rule for when matcodec applies: **not "is it a metric" and not
"does it cluster", but "are there narrow cuts".** Both other properties can be manufactured —
the geodesic transform does exactly that — while narrow cuts cannot be manufactured without
inventing them.

Not pursued, and why: pruning the kNN graph towards a spanning tree would force bottlenecks and
make matcodec work beautifully. It would also mean the distances no longer approximate
anything, which is engineering the answer rather than measuring it.

### On "fire together, wire together"

The Hebbian reading is exact and worth stating. The co-activation graph *is* fire-together;
`chain` layout is wire-together, placing what fires together adjacent in the file. What phase 3
adds is that the wiring is computed as a **route** rather than a pairwise link — and the
measurement above says the routes exist and are informative (3.7× more structure), but the
network they form has no chokepoints to exploit.

## Closing state, phase 3

`make all` green: fmt, clippy `-D warnings`, **38 tests**. `make check-numbers` verifies 17
figures against the JSON they came from, now including the structural ones. Both models remain
in their shipped byte order, SHA-256 verified.

New: `crates/matstruct` (`from-embeddings`, `from-trace`, `geodesic`, `probe`),
`scripts/embed-corpus.sh`, `design/DESIGN-MPEE-TRANSFER.md`, and `make embed`,
`probe-embeddings`, `probe-geodesic`, `probe-experts`.

The pattern that has now paid off three times: **measure the condition a technique depends on
before building the technique.** `coact headroom` bounded router retraining at 2.4 pp in an
hour. `coact reweight` bounded any meta-model over split and combine at 3.0 pp. `matstruct
probe` showed matcodec's gateway model loses to a per-block constant before a single line of
codec was written. Three weeks of building avoided by three afternoons of measuring — and in
each case the negative result came with a mechanism, which is what makes it reusable rather
than merely discouraging.

---

## 2026-08-08 11:41 — The geodesic transform repairs the expert graph, and inverts the ranking

Shortest-path distance satisfies the triangle inequality *by construction*, so the transform
should repair the expert graph's headline failure. It does, and it does more than that:

| expert graph | raw | geodesic k=6 | k=10 | k=16 |
|---|---|---|---|---|
| triangle violations | 0.36 % | **0 %** | 0 % | 0 % |
| coefficient of variation | 0.260 | **0.389** | 0.386 | 0.370 |
| silhouette | 0.168 | **0.278** | 0.223 | 0.226 |
| **rank-1 over the null** | +11.0 % | **+23.4 %** | +19.5 % | +6.8 % |
| blocks readable without decoding | 0.00 % | 0.00 % | 0.00 % | 0.00 % |

**This is the best-structured matrix in the whole study** — highest spread, highest
clusterability, and the only configuration anywhere where the rank-1 gateway base beats a
per-block constant by a clear margin. Better than the embedding graph on every axis, which
inverts the ranking the earlier entries assumed.

The monotone fall with k (23.4 → 19.5 → 6.8) is the expected signature: more neighbours means
more ways around, which means shallower bottlenecks, converging on the raw metric.

What still does not appear is the property matcodec is actually named for: **0 % of blocks are
exactly reproducible**, so nothing can be answered without decompressing. The compression
mechanism starts working; the zero-decode index does not.

### Why the vector is right for one hop and wrong for four

The framing that makes this make sense: the angular distance is the *first leg*. It correctly
says who your neighbours are — that is what builds the kNN edges — but a three- or four-hop
route cannot be derived from it. The numbers are that statement measured: raw angular CV is
0.076, and the same points routed through their own neighbourhood graph reach 0.389. The
direct vector carries local truth and almost no global truth.

This also explains why the raw embedding matrix looked structureless and the geodesic one did
not. Concentration in high dimensions destroys the *global* signal in the direct metric while
leaving the *local* ordering intact, and the geodesic is precisely the operator that rebuilds
global structure out of surviving local structure.

### Honest limits on this last result

- Only 64 experts, so the blocks are small (8 clusters of ~8) and the rank-1 base has few cells
  to be wrong about. The embedding case had 3893 points and is the harder test.
- `d = -ln(lift)` is one choice among several for turning co-occurrence into distance; a
  different one could change the numbers.
- The k=16 embedding run reported 25 % blocks-exact where k=12 and k=24 both report 0 %. That
  is unexplained and is treated as noise from a single degenerate block rather than a result.

---

## 2026-08-08 11:42 — The unexplained 25 % was not noise. It was the mechanism, appearing once.

The previous entry dismissed one number as noise: the k=16 geodesic embedding matrix reported
25 % of blocks readable without decoding, where k=12 and k=24 both reported 0 %. Leaving an
unexplained number in the data is a liability, so it got checked.

25.00 % of 56 blocks is exactly 14, which is 2 × 7 — every block involving **one** cluster. The
first hypothesis was a degenerate singleton cluster, whose blocks are trivially rank-1 because
a single row *is* its own base. But the smallest cluster is 30 points, so that is not it.

Re-clustering the same matrix settles it:

| k-medoids | cluster sizes | blocks exact | rank-1 over null |
|---|---|---|---|
| 4 | 735..1392 | 0.00 % | −73.8 % |
| 6 | 418..1129 | 0.00 % | −72.8 % |
| 8 | **30**..1021 | **25.00 %** | −52.1 % |
| 12 | **30**..585 | **16.67 %** | −26.0 % |

Both configurations that show the effect are the ones that isolate a particular 30-point
group, and in both the exact blocks are exactly that one cluster's (14 of 56, then 22 of 132).
At k=4 and k=6 the group is absorbed and the effect disappears.

**So the corpus contains one genuine gateway-attached region.** Those thirty chunks reach the
rest of the corpus through a single route, their cross-blocks are exactly additive rank-1, and
matcodec would answer every one of their cells with no decompression at all. The mechanism is
real and it does occur in semantic space.

It occurs once, in 3893 points. That is why every aggregate says no, and it sharpens the
conclusion rather than softening it: **the question is not whether gateway structure exists in
an embedding graph, but what fraction of the matrix it covers.** Here it is about 0.8 % of the
points and a quarter of the blocks under a favourable partition — far too little to build a
codec around, and exactly the case matcodec already handles by degrading to deflate.

Correcting the record twice in one afternoon on the same number is worth noting as a lesson:
the first pass called it noise without checking, the second found a real singleton-cluster
artefact hypothesis that was also wrong, and only the third — re-clustering and counting which
blocks were exact — actually explained it. "Probably noise" is a hypothesis, not a finding, and
it should be labelled that way until someone spends the ten minutes.

---

## 2026-08-08 11:45 — Reinforcing the hops that mattered: the conclusion the data points at

The last idea, and the one everything measured today argues for: when an answer turns out
correct, strengthen the hops the route actually used — including a small link that mattered for
the whole.

That is not a separate proposal. It is the inverse of the study's central finding, and the two
fit together exactly:

| measured today | what reinforcement would do to it |
|---|---|
| Load balancing flattens the access distribution, so caching, prefetch and truncation all fail for want of skew | Reward-weighted use *creates* skew — the edges that carried good answers get used more |
| The kNN graph is an expander: no narrow cuts, so no gateways | Repeatedly reinforced bridges become preferred routes; alternatives atrophy; cuts narrow |
| Exactly one region of thirty points already shows perfect gateway structure | Reinforcement deepens the bridges that already exist rather than inventing them |
| `chain` captures only 17 % of the theoretically available clustering, because co-activation lift is 2.50× | Higher lift is exactly what a reinforced graph has, and every downstream number scales with it |

So the honest reading of this whole project is that its ceiling is set at *training* time, by an
objective that rewards spreading. Every inference-time lever ran into that: layout got 7.8–14 %,
static pinning 2.2×, and nothing on the routing side got past 3 percentage points. A signal that
rewards *concentration* — whether a load-balancing loss with a locality term, or online
reinforcement of the routes that produced correct answers — changes the input to all of them
rather than competing with them.

### The experiment, specified

It is measurable with what is already built, and cheap:

1. Capture routes with credit. Extend the trace to record, per token, which experts fired *and*
   whether the sequence's answer was correct — any task with a checkable answer will do.
2. Reweight the graph: `w'(i,j) = w(i,j) · (1 + α)` for pairs co-activated on correct
   sequences, unchanged otherwise.
3. Re-run `matstruct probe` and `coact build` on the reweighted graph and watch four numbers
   that are all already reported: co-activation lift (2.50× today), silhouette (0.278 today at
   best), rank-1 over the null (+23.4 % today at best), and the fraction of blocks readable
   without decoding (0 % today, everywhere).

If those rise with α, the mechanism is real and its size is measured rather than argued. If they
do not, reinforcement is reshuffling noise, and that is worth knowing before anyone changes a
training objective over it.

The honest caveat: reweighting an *analysis* graph is not the same as reinforcing a *model*.
Step 2 changes what the layout optimiser sees; it does not change what the model computes. A
real test needs the reinforcement inside training, which is outside what this repository can
run. What it can do is say in advance how much structure would have to appear before any of the
inference-time levers here get meaningfully better — and, from the sweep in
`data/layout-report.json`, that answer is already on file.

---

## 2026-08-08 11:46 — Reinforcement, and the artefact that had to be killed in code

Simulating the "strengthen the hops that carried a correct answer" idea without needing
correctness labels: reweight each pair by its own lift, `w' = w · lift^α`, so pairs already
above chance gain and the rest weaken. Rich-get-richer on the edges that carry traffic. Then
recompute the marginals, the distances, the geodesic, and the probe.

First results looked spectacular — at α=4, 25 % of blocks readable without decoding. **It was
the same artefact for the third time.** At the clusterings that showed it, the smallest cluster
was **1**: a block one row wide is reproduced exactly by the rank-1 base because `col0` *is* the
block. More singletons, higher number — 25 %, then 46.97 %, then 58.33 % as k rose.

That is now impossible to report by accident. `rank1_fit` counts degenerate blocks separately
and excludes them from the readable-without-decoding fraction, with a test that plants a
singleton in pure noise and asserts the number stays zero. Re-running the sweep with the guard,
α=4's 25 % becomes 0.00 %.

### The corrected sweep

| α | CV | dynamic range | silhouette | degenerate | blocks exact | rank-1 over null |
|---|---|---|---|---|---|---|
| 0 | 0.389 | 7.79× | 0.278 | 0 | 0.00 % | +23.4 % |
| 1 | 0.385 | 8.99× | 0.307 | 0 | 0.00 % | +17.7 % |
| **2** | 0.400 | 8.12× | **0.339** | **0** | 3.57 % | +28.8 % |
| 4 | 0.478 | 13.92× | 0.250 | 14 | 0.00 % | +42.8 % |
| 8 | 0.586 | 25.68× | 0.147 | 14 | 19.05 % | +46.8 % |

Spread rises monotonically with α and so does the rank-1 fit, but **silhouette peaks at α=2 and
then collapses**. Above that, reinforcement is not building regions, it is stranding individual
experts — which is what the degenerate-block count is reporting. α=2 is the last setting that
concentrates the graph without fragmenting it.

### What survives re-clustering — the test that killed the others

Compared at matched, degeneracy-free partitions:

| k-medoids | α=0 rank-1 over null | α=2 rank-1 over null |
|---|---|---|
| 4 | **−62.5 %** | **+18.0 %** |
| 6 | +7.9 % | +18.3 % |
| 8 | +23.4 % | +28.8 % |
| 12 | +39.0 % | +34.9 % |

The headline is not that α=2 is uniformly better — at k=12 it is slightly worse. It is that
**reinforcement makes the gateway fit robust to where the cluster boundaries are drawn.**
Without it the fit swings from −62.5 % to +39.0 % depending purely on the partition, which is
the signature of an artefact of clustering rather than a property of the graph. With it, the
fit sits in a +18 to +35 % band whatever the partition. The structure becomes intrinsic.

What reinforcement does *not* do is produce blocks readable without decoding: still 0–5 %,
against matcodec's road-network case where the majority of cross-region blocks are exact.

### Where that leaves the idea

Directionally confirmed and quantified. Concentrating the graph — which is what reinforcing
successful routes would do — measurably deepens exactly the structure every lever in this
project needed, and it does so monotonically in the strength of the concentration. It also has
a limit that is now measured rather than guessed: past α≈2 it isolates individual experts
instead of forming regions, and isolation is not the same thing as a gateway.

The three artefacts caught today all had the same shape: a number that looked like structure
but was an arithmetic identity of the partition. Each was believed briefly. The general lesson
is worth more than the specific guard: **when a metric depends on a clustering, vary the
clustering before believing the metric.**

---

## 2026-08-08 11:49 — Sharpening the graph does nothing to the layout, and the reason is a proof

Does reinforcing the graph produce a *better layout*? Sharpen with `w' = w · lift^α`, build the
layout, and score on the unchanged holdout — so a gain means the sharpening denoised, and a
loss means it optimised a belief the access pattern does not share.

| lift-power α | mincut fetches/token | chain fetches/token |
|---|---|---|
| 0 | 303.8 | **292.9** |
| 0.5 | 303.5 | 292.9 |
| 1 | 303.1 | 292.9 |
| 2 | 303.6 | 292.9 |
| 4 | 304.3 | 292.9 |

`chain` is **exactly** 292.9 at every α, to the digit. That is not a measurement, it is an
identity: `chain_order` scores neighbours by `aff(i,j) = w(i,j) / (freq_i · freq_j)`, which is
lift up to a constant. Sharpening multiplies each edge by `lift^α`, so `aff' ∝ aff^(1+α)` — a
monotone transform, and a greedy argmax over a monotone transform picks the same neighbour
every time.

**The best construction was already doing the reinforcement.** That also explains, after the
fact, why the affinity chain beats spectral min-cut by 11 fetches per token: `chain` normalises
by expected co-occurrence and `mincut` works on raw counts, so min-cut spends its budget
separating experts that are merely *frequent* rather than genuinely *associated*.

Feeding sharpened weights to min-cut does help, and by almost nothing: 303.8 → 303.1 at α=1,
then worse again at α≥2 as the sharpening starts amplifying rare pairs whose lift is high only
because their expected count is tiny.

So the reinforcement idea splits cleanly in two. As a *graph transform at analysis time* it is
already fully exploited and there is nothing left in it. As a *training signal* — reinforcing
routes that produced correct answers, so the model's actual co-activation becomes concentrated
— it is untouched, and the α sweep two entries above is the measurement of what it would be
worth: monotonically deeper structure up to the point where concentration turns into isolation.

---

## 2026-08-08 11:53 — The full stack on the model it suits best

Integration check after a day of changes, and the answer to the goal as stated. Both arms
cold, median of 3, model output unchanged.

**OLMoE-1B-7B, 2 GiB against 3.72 GB of experts:**

| | hit rate | MiB/token | ms/token |
|---|---|---|---|
| identity + LRU + serial | 4.7 % | 444.2 | 192.95 |
| chain + static pinning + QD4 | 55.4 % | 209.8 | **62.89** |

**3.07×** here, against 3.20× measured earlier in the day. The whole spread is in the baseline
arm (201.0 vs 193.0); the full-stack figure reproduces to 0.1 ms. Reported as **3.1×** rather
than picking whichever run flattered it, and the check-numbers guard on it carries a
deliberately wide tolerance so it fails on regressions rather than on noise.

**Qwen3.6-35B-A3B, 4 GiB against 8.36 GB of experts:**

| | hit rate | MiB/token | ms/token |
|---|---|---|---|
| identity + LRU + serial | 5.8 % | 238.0 | 153.04 |
| chain + static pinning + QD4 | 51.6 % | 122.7 | **39.03** |

**3.92×** — the best result in the study, on the model that best matches the target case. And
it is predicted rather than lucky: Qwen3.6 sits at 3.86 requests per mebibyte against OLMoE's
0.83, so it is latency-bound where OLMoE is closer to bandwidth-bound, and every one of the
three levers buys down per-request cost.

That closes the loop on the goal. A general recipe now exists, it is measured on two
architectures, it costs nothing in quality, and the single number that predicts whether it will
help a third model is readable from that model's GGUF header in an 8 MB range request.

---

## 2026-08-08 12:16 — Load balancing *is* the cause. Measured, not inferred.

Every phase of this project ended at the same explanation: load-balanced routing makes expert
access near-uniform, and without skew, caching, prefetching, truncation and reranking all fail.
That was an inference from two pretrained models. It had never been tested by switching the
auxiliary loss off.

`experiments/sparsemem/` does that. A product-key memory net — keys resident as the index,
values on disk as the payload — trained three times from the same seed for the same number of
steps, differing only in the auxiliary term:

| regime | auxiliary loss |
|---|---|
| `plain` | none |
| `balanced` | load balancing, as production MoE carries |
| `concentrated` | entropy penalty rewarding peaked per-example routing |

### The result: identical accuracy, 16× the memory

Four-register corpus, three balanced classes, 16384 slots, 64 MB of values on disk:

| regime | accuracy (k=32) | slots touched | Gini | top-1 % share |
|---|---|---|---|---|
| plain | 0.996 | **1.6 %** | 0.996 | 99.4 % |
| balanced | 0.994 | **26.1 %** | 0.959 | 80.9 % |
| concentrated | 0.994 | **0.8 %** | 0.996 | 100 % |

Translated into the pinned-cache terms the MoE work used:

| budget | resident | plain | balanced | concentrated |
|---|---|---|---|---|
| 1 MB | 1.6 % | **100.0 %** | 75–84 % | **100.0 %** |
| 4 MB | 6.2 % | 100.0 % | 84–90 % | 100.0 % |
| 8 MB | 12.5 % | 100.0 % | 93–94 % | 100.0 % |
| 16 MB | 25.0 % | 100.0 % | **95–99 %** | 100.0 % |

The load-balanced arm is given as a range: it drifts by several points between runs despite a
fixed seed, because MPS reductions are not deterministic. The other two arms sit on 100.0 % at
every budget and do not move at all, which is itself informative — a footprint that small is
not a matter of which slots happened to win.

**One megabyte serves every retrieval for the model trained without load balancing. The
load-balanced model needs sixteen to get to 99.5 %.** Same architecture, same seed, same step
count, and accuracy identical to the third decimal. The only difference is the auxiliary term.

That is the day's central claim, moved from inference to controlled experiment: **load
balancing is what makes a MoE expensive to fetch, and it costs nothing in accuracy to drop it
on a task this size.**

### The synthetic control, and what it added

The instrument was validated first on a task whose answer requires combining several latent
factors, so small k genuinely destroys information:

| regime | k=1 | k=32 | slots touched |
|---|---|---|---|
| plain | 0.877 | 0.891 | 5.0 % |
| balanced | 0.847 | 0.897 | 57.6 % |
| concentrated | 0.888 | 0.902 | 3.7 % |

Here the *frontier* separates too, not only the footprint: the load-balanced model is worse at
every retrieval depth below 32 and only catches up when given the full budget. It does not just
spread its weights around — it becomes dependent on fetching more of them.

Two design notes worth keeping. The first synthetic task gave every regime 1.000 at k=1 and a
flat, uninformative frontier, because one prototype per example meant a single slot sufficed; a
frontier measures nothing unless small k actually loses information. And the concentration
penalty at 1e-2 **collapsed** the router — 100 % of retrievals onto the busiest 1 % of slots,
accuracy 0.253 against 0.125 chance. That is precisely the failure load balancing exists to
prevent, so the weight is a knob with a real upper bound, found at 1e-3.

### What this does and does not license

It licenses: on a task where a small model has ample capacity, the auxiliary loss is pure cost
in fetch terms. It does not license the same claim at frontier scale, where load balancing is
doing real work keeping hundreds of experts trained and a collapsed router would waste most of
the parameters. The honest statement is that **the loss has a price that is now measured**, and
that a locality-aware variant — spread the load, but among a small number of co-located groups
— has a concrete target to beat: 16× less resident memory for the same accuracy.

### On streaming the N×N and refining on demand

Yes, and MPEE already has the shape of it. `cell_bounds` returns O(L) lower and upper bounds
from the resident landmark index without touching a compressed frame, so a solver prunes on
bounds and materialises only the cells where the bounds do not already decide the branch — what
`DESIGN-MPEE-SOLVER` §9.5 calls making the cost side demand-driven. The extreme-point analysis
falls out of the same structure: the cells that matter to an argmin are the ones whose bounds
straddle the incumbent, and those are a small minority when the bounds are tight.

Phase 3 measured the prerequisite for that here and it failed: on both the embedding graph and
the expert graph, **0 % of blocks are answerable from the index alone**, so every bound would
require a decompression and the demand-driven advantage disappears. The mechanism is sound; the
matrices in this stack do not have the structure it needs. The one place it appeared was a
single 30-point region out of 3893.

---

## 2026-08-08 12:34 — Compression is the measurement. Two sweeps, and a corrected method.

Three corrections to how the last experiment was being run, all of them right:

1. **Cranking the locality weight was forcing the behaviour, not measuring it.** The sweep went
   1 → 10 → 100 and moved contiguous runs from 31.8 to 26.6 of 32 while accuracy fell — a
   penalty strong enough to matter is strong enough to distort.
2. **Hold the model constant and vary the data instead.** Compression is what learning *is*, so
   the question is how much data a fixed memory absorbs before its footprint must grow.
3. **Data entropy is the control variable, which is why the synthetic task was weak.** A task a
   model can memorise has nothing to compress, and its footprint measures capacity rather than
   learning.

### Sweep 1: fixed model, more data → *smaller* footprint

16384 slots throughout, `plain` regime:

| training samples | accuracy | slots used | samples per slot |
|---|---|---|---|
| 200 | 0.986 | 459 | 0.44 |
| 500 | 0.989 | 573 | 0.87 |
| 1000 | 0.990 | 1065 | 0.94 |
| 2000 | 0.989 | 426 | 4.70 |
| 4300 | 0.995 | **262** | **16.40** |

**The footprint shrinks as the data grows.** Below about a thousand examples the model
memorises — roughly one slot per example, 0.44 to 0.94 samples per slot — and the footprint
tracks the dataset. Past that it can no longer memorise, is forced to find shared structure,
and collapses to 262 slots serving 4300 examples.

That transition is compression becoming visible, and it is exactly the property this project
needs: **the better a model has learned, the less of it has to come off disk.**

### Sweep 2: fixed data, smaller net → where does it break?

4300 samples throughout. Since the data cannot be grown further here, shrink the model instead:

| slots available | accuracy | slots used | used/available | samples per slot |
|---|---|---|---|---|
| 16384 | 0.994 | 197 | 1.2 % | 21.9 |
| 4096 | 0.997 | 197 | 4.8 % | 21.9 |
| 1024 | 0.997 | 150 | 14.6 % | 28.8 |
| 256 | 0.994 | 96 | 37.5 % | 44.8 |
| 64 | 0.996 | 62 | 96.9 % | 69.3 |
| **16** | **0.996** | **16** | 100 % | **268.8** |

**Accuracy does not break.** It holds between 0.994 and 0.997 from 16384 slots down to 16 —
4300 examples compressed into sixteen slots at 269 examples each, with no loss. The task's
intrinsic complexity is at most sixteen slots, and a 16384-slot layer discovers that on its
own, leaving 98.8 % of itself unused.

Which also says the experiment has hit its own ceiling: the four-register task is too easy to
locate the compression limit. Finding where accuracy actually breaks needs data with far more
entropy than four writing registers, and that is the honest boundary of what was measured here.

### What the two sweeps say together

The footprint a model needs on disk is set by **the ratio of data entropy to model capacity**,
not by the architecture and not by a locality penalty. Give a fixed model more than it can
memorise and it compresses; give it a task simpler than its capacity and it uses a sliver
regardless of size.

That reframes the whole project's result. OLMoE and Qwen3.6 need 25–55 % residency not because
MoE is inherently scattered but because they are trained with load balancing on language, which
has enormous entropy relative to any model built so far. The controlled experiment isolates
the first factor — dropping load balancing cut the required memory 16× at identical accuracy —
and these sweeps isolate the second.

### On "aha" as the thing that will not compress

The reading that follows from this: if compression is learning, then the residue — what the
model *cannot* fold into existing structure — is precisely what is worth spending capacity on.
Measurably, that is the slot allocated to a small number of examples while everything else
reaches 269 examples per slot. A training loop that watched for that signal would be doing
credit assignment on novelty rather than on error, and it has a concrete target here: at 4300
samples, 262 slots carry the load and the tail of rarely-used slots is where the incompressible
material sits. Not tested; the hook exists in `slot_use` and would be a few lines.

---

## 2026-08-08 12:38 — Closing: can the N×N solver drive the compression?

The proposal that ties the two halves together: stream the vector angles as matrices, run the
solver over them, and use that structure to force the model into smaller knowledge clusters
that compress further.

What today's measurements say about it, in order of how much they constrain it:

**The grouping pressure is not the binding constraint.** A locality loss — minimise the
variance of the routing distribution over slot *position*, which is the training-time version
of phase 1's layout permutation — was tried across three weights. Contiguous reads per
inference moved from 32.0 of 32 to 31.8, then 31.1, then 26.6, and accuracy fell at the weight
that finally bit. Meanwhile the model, with no locality pressure at all, already collapsed to
**1.2 % of its slots**. There is little for a grouping term to win when the footprint is
already that small.

**Data entropy is the binding constraint.** The two sweeps say the footprint is set by how much
the data exceeds what the model can memorise. Below a thousand examples the net memorises and
its footprint tracks the dataset; above it, the net compresses. That is not something a solver
over an N×N matrix can change — it is a property of the corpus.

**And the matrices in this stack lack the structure the solver needs.** Phase 3 measured it:
0 % of blocks answerable from the resident index on both the embedding graph and the expert
graph, because a kNN graph over a concentrated cloud is an expander with no narrow cuts. The
solver's pruning and the codec's zero-decode path both depend on those cuts.

So the honest shape of the idea: the N×N solver is the right instrument for *deciding where
things go* once groups exist, and phase 1 showed that is worth 7.8–14 % of fetch time. What it
cannot do is *create* the groups — that comes from the training signal and the data, and the
one intervention measured to change it by a large factor is dropping load balancing, which is
worth 16×.

The order that follows: fix the training objective first, then let the solver place what
results. Doing it the other way round optimises the arrangement of a structure that was never
formed.

---

## 2026-08-08 12:42 — The compression limit, now that the task has entropy

The last entry admitted a gap: the four-register task was too easy to locate where compression
breaks — accuracy held from 16384 slots down to 16. Relabelling the *same bytes* by which of
the 23 source files a chunk came from gives a 30-way task where all the Python chunks are
CPython stdlib and all the C chunks are curl, so the distinctions are subtle. Same corpus, far
more entropy.

4140 samples, 30 classes, chance 0.033:

| slots available | accuracy | slots used | used/available | samples per slot |
|---|---|---|---|---|
| 16384 | 0.638 | 164 | 1.0 % | 20.2 |
| 4096 | 0.628 | 172 | 4.2 % | 19.3 |
| 1024 | 0.639 | 288 | 28.1 % | 11.5 |
| 256 | 0.612 | 200 | 78.1 % | 16.6 |
| 64 | **0.601** | 64 | 100 % | 51.8 |
| 16 | **0.554** | 16 | 100 % | 207.0 |

**The limit is visible now.** Accuracy holds within four points down to 256 slots and then
breaks: −6 % at 64, −13 % at 16. And at full capacity the model still uses only 1.0 % of what
it has, so the task's intrinsic size is roughly 160–290 slots however much memory it is given.

Set against the register task, whose limit was below 16 slots, that is the entropy claim
measured on both sides with the same bytes on disk: **the floor a model can compress to is set
by the data, and relabelling the same corpus at higher entropy raised that floor by more than
an order of magnitude.**

For the disk goal this is the sizing rule. The resident footprint you need is not a property of
the architecture — it is the intrinsic complexity of what the model has to represent, and it is
measurable by shrinking the model until accuracy moves.

## On using the N×N over the weights: first hop by matmul, the rest by traversal

The architecture this points at: the weight product gives the *first hop* — which stored
embeddings the query directly matches — and from there the N×N structure over the stored
embeddings is walked along dense links to reach further relevant slots, instead of scoring
against everything. Retrieval becomes one matmul plus a graph walk.

There is an inversion here worth recording, because it flips a phase-3 result from bad news to
good. That phase found the kNN graph over embeddings to be an **expander**: richly connected,
no narrow cuts. For matcodec that was fatal — a gateway codec needs bottlenecks, and the
measurement said there are none. For *traversal* the same property is exactly what you want.
Expanders have short paths between any two points, which is why small-world graphs are the
basis of every practical approximate-nearest-neighbour index. The structure that makes the
embedding graph incompressible is the structure that makes it fast to walk.

So the two halves of the MPEE transfer split cleanly along that line, and the measurements say
which is which:

| MPEE mechanism | needs | measured on embeddings | verdict |
|---|---|---|---|
| matcodec's gateway codec | narrow cuts | expander, 0 % blocks exact | does not transfer |
| streaming solver's route traversal | short paths | expander, short paths | transfers |

Not built here — that is an index, and building one was explicitly out of scope for a phase
that set out to measure whether it could pay. What is now on record is that the measurement
which killed the codec is the same measurement that supports the traversal, which is not a
result anyone would have guessed from either half alone.

---

## 2026-08-08 12:43 — The inversion, measured: 6 hops covers the corpus

The claim in the previous entry was that the expander property which killed matcodec is the
same one that makes traversal cheap. Checked rather than asserted — BFS hop counts over the
kNN graph on 3893 stored embeddings:

| k | mean hops | median | p99 | max | reachable |
|---|---|---|---|---|---|
| 4 | 11.16 | 11 | 19 | 24 | 88.4 % |
| 8 | 6.84 | 7 | 12 | 15 | 95.5 % |
| **16** | **4.81** | **5** | 8 | 10 | **100 %** |
| 32 | 3.57 | 4 | 5 | **6** | 100 % |

At k=16 the median is five hops and every point is reachable. At k=32 the **maximum** over all
sampled pairs is six — the whole corpus lies within six hops of any starting point.

That is the "5–6 hops if all are efficient" figure, measured, and it confirms the inversion:

| MPEE mechanism | needs | measured | verdict |
|---|---|---|---|
| matcodec's gateway codec | narrow cuts | none — 0 % blocks exact | does not transfer |
| route traversal over the graph | short paths | diameter 6 at k=32 | transfers |

Same graph, same measurement, opposite conclusions for the two halves. Worth stating plainly
because neither half predicts the other: a structure with no bottlenecks is incompressible and
fast to search, and those are the same sentence.

It also sets the design the proposal was reaching for. The weight product gives the first hop —
which stored embeddings the query matches directly — and from there five or six link-following
steps reach anything, instead of scoring against all 3893. What that costs in *fetch* terms is
the open question this repository is equipped to answer but did not: six hops of a few slots
each, against reading the whole layer. The instrument exists (`experiments/hops/hopcount.py`
for the graph, `NoCacheFile` for the bytes); the measurement does not yet.


## Traversal over an on-disk index: priced, verified, and it loses below ~100k

The open item left in the previous entry was the cost of the graph walk — "six hops of a few
slots each, against reading the whole layer. The instrument exists; the measurement does not."
It exists now, and it changed shape once on the way.

**The framing was wrong first.** I was about to conclude that the resident index is the binding
constraint: N x 384 x 4 is 15 GB at N = 10M, which defeats the purpose on a 36 GB machine. That
assumed the index has to be resident. It does not — an on-disk index is a straightforward design
(and see the correction appended later in this file: it is *not* something MPEdb implements) — and once it is on disk the cost is not bytes *held* but fetch rounds *issued*, which is
the quantity `hopcount.py` already measures. The whole question moves from memory to I/O, where
the rest of this project already lives.

So the comparison is a full scan of the index against a walk that reads only the nodes it
visits, with each node stored DiskANN-style (vector and neighbour ids adjacent, so arriving
somewhere tells you both what it is and where to go next).

**Three findings, two of them negative.**

*Pure greedy descent is not the searcher.* At k=8 it reached the target in 8 % of queries, at
k=32 in 49 %. The hop counts say a path always exists, so that gap is the descent getting stuck
in local minima, not the index failing. A beam of 4 fixes it — 99.5 % reach at k=32 — at a
known cost in fetches, which is the trade this measurement exists to price. Reporting the
greedy number alone would have blamed the index for the searcher's mistake, so both are kept.

*Disk ordering does not help here, and the reason is the same one as everywhere else.* A node's
44 neighbours form 24.0 contiguous runs in identity order and **28.2 after a BFS reorder** —
the ordering makes it worse. This is the expander property for the third time: it killed
matcodec (no narrow cuts to build a gateway on), it enabled short paths (`hopcount.py`), and it
now prevents a 1-D layout from coalescing a walk's reads, because a node's neighbours are not
near each other in any linear order when the graph has no local structure. The expert layout
work transfers as a *technique* and fails as an *outcome*, on this graph, for a stated reason.

*The walk loses until the index is large.* One fetch costs 806 KiB of sequential transfer on
this device, so a 206-fetch walk is only worth it once the index exceeds that many
fetch-equivalents:

| k | walk visits | reaches | scan | walk | break-even N |
|---|---|---|---|---|---|
| 8 | 342 | 87.0 % | 1.93 ms | 55.8 ms | 127 572 |
| 16 | 350 | 98.0 % | 1.97 ms | 50.2 ms | 111 593 |
| 32 | 377 | 99.5 % | 2.06 ms | 47.1 ms | 99 507 |

Below ~100 000 embeddings — about 160 MiB of index — reading the whole thing is cheaper than
being clever about it. That is a useful negative: it says when *not* to build the graph.

**Verified against the device, and the verification took three attempts.** `verify_fetch.py`
does the scan and the walk with real F_NOCACHE reads. The first run reported 27 GB/s for the
scan, because offset 0 is the GGUF header that every other tool here had just read. The second
reported 5.38 GB/s for scattered 1.7 KB reads, because five repetitions of the same offsets
mean repetition 1 pays for the disk and 2..5 are served from cache — F_NOCACHE stops a read
from *populating* the cache but not from being handed a page that is already there. Both are
the 35x calibration bug in another costume. With eviction, a warm-up discard, a fresh region
per repetition, and scan and walk partitioned into different halves of the file:

```
 scan  measured    1.46 ms   predicted    2.05 ms   (6.3 MiB at 4.52 GB/s)
 walk  measured   26.15 ms   predicted   47.25 ms   (1.1 MiB, 206 scattered reads)
 walk/scan  measured 17.91x   predicted 23.04x
```

Both sides come in below the cost model, in the same direction — the affine fit underestimates
large sequential reads, which calibration already noted (this device is 3x faster at 8 MiB than
at 1 MiB). So the ratio is the trustworthy part, and it agrees: the walk loses by about an
order of magnitude at this index size, and the break-even shifts down to roughly 77 000.

**What this settles.** The two halves of phase 3 now have a single consistent story. The
expander property is one fact with three consequences, two helpful and one not: no gateway
codec, short paths, no layout gain. And the traversal architecture is not free — it has an
index-size threshold, measured, below which the boring answer wins.

**Also fixed:** `data/costmodel.json` predated the `Provenance` field, so the *measured* cost
model was being deserialised as `Assumed`. Rather than hand-edit a data file to claim a
provenance — which is what the field exists to prevent — the calibration was re-run. Two
independent cold runs gave 230.74 us / 0.2856 ns/B and 227.80 us / 0.2760 ns/B: 1.3 % and 3.4 %
apart. The device is stable, the contaminated reading really was 35x off rather than unlucky,
and the guards now carry that measured spread as their tolerance.


## The trace was wrong, and it was wrong in the direction that made the story work

While adding `--pin-trace` to test whether the pinned expert set survives a register change,
the diagnostic printed a Gini coefficient of exactly 0.000 and a hottest-half share of exactly
50.0 %. Exact is not a measurement, so I checked the trace directly.

**Every group of 8 consecutive records was an exact partition of all 64 experts — 25 000 groups
out of 25 000.** Real routing collides; a partition does not. The same held on Qwen3.6:
500/500 groups of 32 partitioning all 256 experts.

**Cause.** `ffn_moe_topk` is not a tensor, it is a *view*. llama.cpp builds it as
`ggml_top_k(probs, n_expert_used)`, which is `ggml_argsort` followed by a view keeping the
first k of each row. So `ne[0] == k` — which is what `moetrace` asserted and got — but `nb[1]`
is the full argsort row stride, `n_expert * 4`. A flat
`ggml_backend_tensor_get(t, dst, 0, ggml_nbytes(t))` copies straight through the gaps and
returns each token's *entire 64-wide ranking*, which the writer then chopped into eight bogus
"tokens". The gate weights came from a different, genuinely contiguous tensor and were correct
throughout, which is why every row looked plausible: descending weights, sensible ids. Only
record 0 of each group of 8 was a real top-k.

The fix reads row by row at `nb[1]` and refuses to proceed if `nb[0]` is not 4 bytes.

**What the artefact did.** A full argsort touches every expert exactly once, so the access
pattern was uniform *by construction* — max/min exactly 1.00. That uniformity was this
project's central empirical premise. Real routing is nothing like it:

| | artefact | real routing |
|---|---|---|
| expert frequency max/min | 1.00 | **47.69** |
| share of traffic in the hottest half | 0.500 | 0.672 |
| experts shared by adjacent tokens | 0.12 | **2.44** (chance 1.00) |
| groups of 8 partitioning all 64 | 25000/25000 | 0/512 |

And the headline reverses. Same model, same budget, same disk, only the trace corrected:

| policy, 2 GiB | artefact | real routing |
|---|---|---|
| LRU | 4.5 % hit | **78.6 %** |
| static pinning | 55.4 % | 78.4 % |
| random | ~11 % | 74.8 % |
| MiB/token | 209.9 | **99.8** |
| ms/token | 12.56 | **5.99** |

So: **LRU is not pathological, it is the best of the three.** "Static pinning beats LRU 2.24x"
is dead, and so is the explanation attached to it — that load balancing flattens routing until
caching cannot work. Routing is skewed 47x and locally correlated at 2.4x chance. Caching works
because there *is* a hot set, which is the ordinary answer I had argued my way out of.

**How it survived so long.** Three ways, all of them my fault:

1. The uniformity was *predicted* before it was measured. Load-balancing losses are real and
   flat routing is a plausible consequence, so the artefact confirmed a hypothesis I already
   held, and I read every later oddity as further confirmation instead of as a symptom. LRU
   hitting 4.5 % at 54 % residency should have been treated as an impossible number, not an
   interesting one.
2. `make check-numbers` guards 40 numbers against *drift*, and every one of them was computed
   from the same trace. A consistency gate cannot detect a consistent error.
3. The one guard that could have caught it — the exact-partition check — did not exist, because
   nothing prompted me to ask whether the trace was a trace.

The contamination guard and the `Provenance` field exist because of an earlier 35x measurement
error, and I wrote at the time that the discipline transferred from MPEE "and mattered most".
It clearly did not transfer far enough: it guarded the *cost model* and never the *input*.

**What this invalidates, precisely.** Everything derived from the router trace:

- the cache-policy comparison and the 2.24x static-pinning claim
- the layout work: the co-activation graph, `chain`/spectral orderings, 14.4 % fewer fetches
- the composed 3.1x (OLMoE) and 3.92x (Qwen3.6), including the 1.49x / 2.08x / 1.03x split
- cross-layer mutual information (0.66 of 5.96 bits), headroom, reweighting, the ensemble study
- `matstruct probe-experts`, the negative control for phase 3
- the cross-register pinning result measured an hour ago

**What stands, because it never touched the trace:**

- losslessness of the permutation: layer-0 routing identity, byte-exact revert, identical
  generations (the permutation machinery is fine; only the *choice* of permutation was informed
  by a bad graph)
- the KL-divergence and truncation sweeps, which ran through `llama-perplexity`
- the cost model, its calibration, and the contamination guard
- phase 4's PyTorch experiments — a self-contained toy, and its causal claim about load
  balancing stands *as a statement about that toy*, no longer as an explanation of OLMoE
- the embedding-side phase 3 work: `probe-embeddings`, `probe-geodesic`, hop counts, and
  today's traversal costing

**Where this leaves the project.** Better off, and not only in the moral sense. Real routing has
47x frequency skew and 2.4x-chance co-activation between adjacent tokens — far *more* structure
than the artefact showed. Every technique that was declared dead for lack of skew (prefetch,
frequency pinning, cost-aware skipping) deserves re-measurement, and the layout solver now has a
graph with genuine locality to work on rather than noise. The 5.99 ms/token already measured on
the corrected trace is less than half the 12.56 ms the artefact reported.

The re-run is the next session's work. Nothing derived from the trace should be quoted until
then, and `check-numbers.py` now fails loudly on those entries rather than asserting them.


## First numbers on the corrected trace

Not the re-run — that is a session's work — but enough to show which way things move.

**Layout gets better, because the graph is now real.** Fetches per token, which is layout
arithmetic and so unaffected by page-cache state:

| trace | identity | chain | reduction |
|---|---|---|---|
| artefact | 396.0 | 339.0 | 14.4 % |
| corrected | 339.1 | 231.2 | **31.8 %** |

The artefact's co-activation graph was near-noise (adjacent tokens shared 0.12 experts against
a chance level of 1.00 — *below* chance, as a partition must be). The real graph has 2.44, and
the same solver finds more than twice the reduction in it. The timing half could not be
completed: the replay reported 19.6 GB/s against a 3.62 GB/s ceiling, the contamination guard
fired, and `sudo purge` is not available here. Fetch counts stand; milliseconds do not.

**The cache-policy story is budget-dependent, not a blanket win.** Hit rates are exact cache
simulation, so contamination does not touch them:

| budget | LRU | static pinning | random |
|---|---|---|---|
| 256 MiB | **0.0 %** | 8.8 % | 7.2 % |
| 512 MiB | **35.6 %** | 23.2 % | 22.2 % |
| 1024 MiB | **49.1 %** | 46.3 % | 44.1 % |
| 2048 MiB | 76.8 % | **77.6 %** | 74.1 % |

LRU thrashes only at 256 MiB — about 6 % residency, where the working set of one layer does not
fit and every insert evicts something needed. Above that it is competitive or better, and by
2 GiB the three policies are within 3.5 points. So the correct claim is not "static pinning
beats LRU 2.24x" but "below roughly 512 MiB LRU collapses and frequency pinning is the safe
choice; above it, the policy barely matters and the budget is what matters". That is a smaller
claim and a true one.

**What this does to the project's thesis.** The original argument was: load balancing flattens
routing, flat routing defeats caching, therefore the only lever left is layout plus queue depth.
The middle step was the artefact. Routing is skewed 47x and locally correlated, caching works,
and 2 GiB of cache alone takes 465 MiB/token down to 105 — a 4.4x reduction in bytes fetched
from a mechanism the earlier writeup had declared dead.

The levers ruled out on *routing* grounds — prefetch from cross-layer mutual information,
cost-aware skipping, frequency-based pinning — were all ruled out because the trace showed no
structure to exploit. Every one of them is open again.


## A model of our own, because the trace had no ground truth

The morning's bug had one root cause that no amount of care downstream could have caught: the
trace was an artefact and there was nothing to check it against. Every tool agreed with every
other tool because they all read the same wrong file. The fix is not more care, it is a case
where the right answer is known in advance — so: build the model.

**Validating the chain.** `experiments/customsmoe/planted.py` trains a real MoE (32 experts,
top-4) in which 8 latent input groups are pushed onto their own slice of 4 experts. The ground
truth is then arithmetic: an expert fires for one group in eight, the four experts of a group
fire together, so lift = (1/8)/(1/8)² = **8.0** inside a group and 0 across.

`coact analyze` on that trace reports max lift 8.51 and 12.18 % of pairs above 2x. An
independent reimplementation in `score.py` reports 8.51 and 12.18 %. The chain is validated
against something other than itself for the first time.

Two mistakes on the way there, both worth keeping:

- The first verdict said FAIL, because measured within-group lift was 5.69 against a predicted
  8.00. That conflates two different failures — an analysis that cannot see structure, and a
  model that never learned the structure it was asked for. Only the first is a defect here. The
  router realised 71.2 % of the planted lift; the chain sees all of what the router has.
- The first version of the agreement check returned PASS when it could not find the fields in
  `analysis.json` — which does not persist the lift table at all. A check that cannot fail is
  not a check, which is the whole lesson of the morning. It now runs the binary and parses what
  it prints.

The negative control passes too: the retired artefact trace is rejected by
`Trace::reject_argsort_artefact` before any analysis runs.

## Compression as the training signal

Planting the structure is circular as a claim about learning. So the next model is told nothing
about which expert should serve which input — only to make the access pattern **cheap to
describe**. Experts sit in fixed blocks the size of one fetch, and the auxiliary loss is the
expected number of distinct blocks a token touches. Fewer blocks is both a shorter description
and fewer disk reads; here they are the same number.

| regime | accuracy | blocks/token | max lift | pairs ≥2x |
|---|---|---|---|---|
| plain | 1.000 | 3.52 of 8 | 7.87 | 13.10 % |
| balanced (load balancing) | 1.000 | 3.47 | **2.45** | **0.60 %** |
| compressed | 1.000 | **2.85** | **13.33** | 12.04 % |

Load balancing does not cost accuracy and barely changes fetch count, but it **erases the
co-activation structure** — 0.60 % of pairs above 2x lift against 13.10 % without it. The
compression objective moves the other way: 18.9 % fewer fetches at identical accuracy, with
lift nearly doubled over plain. This is the causal claim the toy memory layer made in phase 4,
now made on routing itself, which is where it was always meant to apply.

## Exact arithmetic, and looping instead of stacking

Synthetic Gaussian groups have no right answer to be exactly right about, and their underlying
rule is not small. Arithmetic is better on both counts: `a op b` for a, b in 0..9 over +, − and
× is 300 problems with one exact answer each, generated by a rule small enough that a model
which has *learned* it should need almost no capacity. Fitting 300 pairs with 300 experts is
memorisation; fitting them while the access pattern collapses is not.

And one forward pass is not how arithmetic is done. The same block is applied T times, so later
hops can revisit — and re-fetch experts that are already resident, which is free.

| hops | compression | exact accuracy | block fetches | distinct | repeats |
|---|---|---|---|---|---|
| 1 | no | 1.000 | 3.34 | 3.34 | 0 % |
| 1 | yes | 1.000 | **1.02** | 1.02 | 0 % |
| 2 | yes | 1.000 | 2.21 | 1.49 | 32.3 % |
| 4 | no | 1.000 | 12.88 | 4.46 | **65.4 %** |
| 4 | yes | 1.000 | 5.67 | 2.44 | 56.9 % |

Every regime is exactly right on all 300 problems. With the compression objective a single hop
solves all of them from **1.02 blocks** — four experts. And looping pays for itself in the way
the architecture predicted: at 4 hops, 65 % of block fetches are blocks a previous hop already
brought in.

## Choosing the next pass from the last one

Re-running a dense router each hop is a loop, but it decides the next pass by recomputing
everything rather than by following where the last one landed. `graph_loop.py` makes hop 1 dense
and every later hop score **only the graph neighbours of the previous hop's picks**, with the
graph accumulated online from co-activation — `fire together, wire together` used as a routing
structure rather than a disk order.

| hops | routing | exact accuracy | experts scored/hop | graph degree |
|---|---|---|---|---|
| 2 | dense | 1.000 | 32.0 | 18.3 |
| 2 | graph | 1.000 | 22.0 | 23.3 |
| 4 | dense | 1.000 | 32.0 | 26.1 |
| 4 | graph | 1.000 | **19.5** | 30.9 |

Accuracy is untouched and 39 % fewer experts are scored per hop. But the honest reading is in
the last column: the learned graph has degree 27–31 out of 32, so it connects nearly everything
and the saving is small. This is the expander property once more, now on a graph the model
learned rather than one we built. A neighbourhood is only cheap if it is *small*, and nothing
in the training pressured it to be.

## Where compression overshoots

Collapsing to 1.02 blocks looks like a triumph and is partly a warning. Addition and
subtraction are one circuit with a sign flipped; multiplication is not. A model that learned
arithmetic should share experts between + and − and separate ×, which is a falsifiable
prediction about structure that does not go through accuracy at all:

| λ | exact accuracy | + vs − | + vs × | − vs × | experts used |
|---|---|---|---|---|---|
| 0.0 | 1.000 | 0.422 | **0.545** | 0.412 | 16 |
| 0.1 | 1.000 | 0.528 | 0.540 | 0.435 | 15 |
| 0.3 | 1.000 | 0.010 | 0.007 | **0.990** | 8 |
| 1.0 | 1.000 | 0.000 | 0.000 | **1.000** | 8 |

**The prediction fails, at every setting.** Untrained by compression the model shares + with ×
more than with −; pushed by compression it splits into {+} against {−, ×} and uses 8 of 32
experts. Accuracy stays at 1.000 throughout, which is exactly why accuracy could not have
revealed this.

Two readings, and the measurement does not yet separate them. The encoding may be doing the
work: results are shifted by 9, so + spans classes 9–27, − spans 0–18 and × spans 9–90, and −
is the only operation producing negatives. Grouping by output range rather than by algorithm
would look like this. The other reading is that the compression objective simply does not care
about algorithmic structure — it minimises blocks touched, and any partition that achieves that
will do.

Either way the practical conclusion holds and it is the one that was pushed back on: **more
experts is a goal, not an accident to be optimised away.** At λ ≥ 0.3 the objective spends
distinctions to buy fetches and the accounting never shows it, because the task is small enough
to survive. On a model where the task is not small, that trade would surface as capability loss
long after the fetch numbers looked excellent. A compression objective needs a floor — a term
that prices the *loss of distinctions*, not only the cost of reading them.

That term is the open problem this leaves. The instruments to measure it now exist and are
validated against ground truth, which is more than could be said this morning.


## The floor, and the trade that turned out not to be one

The previous entry left an open problem: a compression objective needs a term that prices the
*loss of distinctions*, not only the cost of reading them. `floor.py` adds one — the mean
pairwise total-variation distance between the operations' routing profiles, subtracted from the
loss. Total variation because it is bounded in [0, 1], so the two terms stay commensurable and
the trade can be read off rather than tuned into.

| λ (compress) | β (floor) | exact acc | blocks/problem | experts used | worst pair overlap |
|---|---|---|---|---|---|
| 0.0 | 0.0 | 1.000 | 3.34 | 16 | 0.545 |
| 1.0 | 0.0 | 1.000 | **1.00** | 8 | **1.000** |
| 1.0 | 0.1 | 1.000 | 1.11 | 8 | 0.938 |
| 1.0 | 0.3 | 1.000 | 1.32 | 8 | 0.998 |
| 1.0 | 1.0 | 1.000 | 2.22 | **15** | **0.320** |

Compression alone reaches one block per problem and a worst-pair overlap of **1.000** — two
operations routed identically, which is the collapse `op_structure.py` found, now visible as a
single number. Accuracy is 1.000 there too, which remains the point: the accounting never
objected.

The floor at β = 1 gives **2.22 blocks against the unregularised 3.34 — 33.5 % fewer fetches —
while worst-pair overlap falls to 0.320**, better separated than the model trained with no
auxiliary loss at all (0.545), using 15 of the 16 experts that model used.

So the expected trade is not there. Most of the compression survives, and the distinctions come
out *better* than baseline rather than merely preserved. The pure objective was not too
aggressive, it was **misspecified**: minimising blocks touched is satisfied equally well by a
partition that respects the task's structure and by one that destroys it, and with nothing to
break the tie it picked whichever the optimiser reached first. Naming the tie-break was the
whole fix.

Small β is worse than none: at 0.1 and 0.3 overlap stays near 1.0 while blocks creep up, so the
term pays a cost without buying separation until it is strong enough to change which partition
wins. A weak floor is the worst setting available, which is worth knowing before tuning one.

**The caveat, unresolved.** The floor is defined over a label — the operation — that we happen
to have. A real MoE has no such label, and the analogous quantity would have to come from
behaviour: distinctions the model *makes* rather than ones we can name. Mutual information
between input and expert selection is the obvious candidate and is not measured here. The
result stands as an existence proof that compression and distinction are compatible, not yet as
a method that transfers.


## Scaling the mathematics: does harder arithmetic recruit more experts?

The objection to the one-digit results was that 300 problems can survive almost any collapse.
So: two-digit operands over five operations, up to 49 800 problems, answers emitted as sign
plus four digits, and a problem counts as solved only when all five heads are right.

| operations | problems | λ | β | exact acc | experts used | blocks/problem | worst overlap |
|---|---|---|---|---|---|---|---|
| + | 10 000 | 0 | 0 | 1.000 | 8 | 5.54 | 0.000 |
| + | 10 000 | 1 | 0 | 1.000 | 5 | **1.00** | 0.000 |
| + − | 20 000 | 0 | 0 | 1.000 | 17 | 4.99 | 0.630 |
| + − | 20 000 | 1 | 1 | 0.999 | 20 | 1.61 | 0.776 |
| + − × | 30 000 | 0 | 0 | 0.816 | 58 | 6.08 | 0.591 |
| + − × | 30 000 | 1 | 0 | 0.716 | **52** | **1.00** | 0.989 |
| + − × // % | 49 800 | 1 | 0 | 0.738 | 41 | **1.00** | 0.994 |
| + − × // % | 49 800 | 1 | 1 | 0.721 | **63** | 2.46 | 0.881 |

**Experts recruited grow with difficulty**: 5 → 20 → 52 under compression alone, 5 → 20 → 44 →
63 with the floor. The one-digit collapse to 8 experts was a property of the task, not of the
objective — the objection was right and the fix was to make the mathematics harder.

The important column is the one next to it. **Blocks per problem stays at 1.00** while 41–52
experts are in use across the corpus. That is the shape this whole project is looking for: a
large expert population on disk, one fetch per token. Capacity scales with the task; the read
does not.

Accuracy falls to ~0.72–0.82 once multiplication enters, at 2500 steps. That is an undertrained
model, not a ceiling — single-operation and ± tasks are exactly solved — but it means the
three- and five-operation rows compare routing under equal budget rather than at convergence,
and they are read that way.

## Reuse across hops, and whether the needed experts are knowable in advance

Two claims the disk cares about, measured on 3-hop models.

**A loop should return to the same experts rather than recruit new ones.** If a longer
expression costs hops over resident blocks, it is cheap; if each hop pulls fresh experts, the
loop is a deep network in disguise and costs a fetch every time.

| operations | λ, β | hop reuse | blocks needed | probe recall | popularity baseline |
|---|---|---|---|---|---|
| 2 | 0, 0 | **0.811** | 5 | 0.729 | **0.786** |
| 2 | 1, 1 | 0.783 | 2 | 0.666 | 0.666 |
| 5 | 0, 0 | 0.691 | 6 | **0.718** | 0.596 |
| 5 | 1, 1 | **0.821** | 2 | 0.574 | 0.557 |

Hop reuse is 0.69–0.82: most of what a hop selects, the previous hop already had. The loop does
behave as hoped — it revisits rather than grows — and under compression *and* the floor it
revisits most of all (0.821 on the hardest task). Applying the same operator expert repeatedly
is what the model does when allowed to.

**Foreknowledge does not hold up.** A linear probe on the input predicts the blocks a problem
will touch at recall 0.574–0.729, against a baseline of always guessing the globally most
popular blocks at 0.557–0.786. It loses on the two-operation task, ties on one, and beats the
baseline meaningfully in exactly one cell (0.718 against 0.596). The honest reading is that the
needed blocks are *mostly* knowable, but almost all of that is popularity — which needs no
prediction, no probe, and no model, because it is static pinning under another name.

Two caveats before this is treated as settled. The probe is linear over a one-hot encoding,
which is the weakest predictor available; and the compressed models need only 2 blocks, so
there is little left to predict once popularity has spoken. A stronger probe on a task that
needs more blocks is the version of this experiment that could still change the answer.

## What this leaves for the knowledge-graph idea

The remaining proposal — that parentheses, ±, × should be distinct experts and that the *order*
of applying them could come from a knowledge graph, MPEE-style — is not tested here, because
every problem in this dataset is a single operation. There is no order to discover. Testing it
needs multi-term expressions where the answer depends on precedence, and the measurement would
be whether the hop sequence recovers the evaluation order. The instruments are in place: hop
reuse is already measured, `graph_loop.py` already routes hop n+1 from hop n's neighbourhood,
and `coact analyze` reads the resulting trace. The dataset is the missing piece.


## Expressions with brackets, and equations with an unknown

Two datasets the earlier arithmetic could not support, because every problem there was a single
operation and therefore had no order to discover.

**`exprs.py` — 18 000 expressions `a op1 b op2 c`, with and without a bracket.** Precedence
decides the order when there is no bracket; the bracket overrides it. Same three operands, same
two operators, one flag different, different answer.

| λ, β | exact acc | hop1 → 1st op | hop1 → 2nd op | majority | divergence, reordering | divergence, decorative |
|---|---|---|---|---|---|---|
| 0, 0 | 0.982 | 0.762 | 0.839 | 0.445 | 0.134 | 0.030 |
| 1, 0 | 0.895 | 0.617 | 0.576 | 0.445 | 0.043 | 0.004 |
| 1, 1 | 0.892 | **0.935** | 0.469 | 0.445 | **0.270** | **0.005** |

**`equations.py` — 6 498 equations, X unknown, a fraction line, four forms.** `a*X+b=c`,
`X/a+b=c`, `(X+a)/b=c`, `(X-a)*b=c`. The written order and the solution order run opposite ways:
you subtract before dividing, but multiply before subtracting, depending on the form. Solved
**exactly, 1.000, on all four forms** with no auxiliary loss at all.

| λ, β | solve acc | hop1 → step 1 | majority | hop1 → step 2 | order class |
|---|---|---|---|---|---|
| 0, 0 | **1.000** | 0.561 | 0.538 | 0.885 | 0.916 |
| 1, 1 | 0.978 | 1.000 | 0.538 | 1.000 | 1.000 |

**What is circular here and what is not.** The β term trains hop-1 routing to separate by which
step belongs at position 1, so `hop1 → step 1 = 1.000` is close to a tautology and is reported
as such. Two things are not circular:

- **The decorative-bracket control.** Brackets that do not change the evaluation order produce
  routing divergence 0.005; brackets that do produce 0.270 — a factor of 54. The objective
  mentions order, never the flag, so a model reacting to the bracket itself would have
  separated both. It separates only the ones that reorder.
- **Solving accuracy**, which the auxiliary losses do not touch.

**And the negative result underneath.** Without the floor, hop 1 predicts the first solution
step at 0.561 against a majority baseline of 0.538 — chance. The model solves every equation
perfectly while its first hop knows nothing about which step comes first. Procedure structure is
*not* a free consequence of solving the task; it appears only when asked for. That is worth
knowing before assuming a large MoE's hops correspond to reasoning steps.

## Measuring what was actually claimed: compression, subject to being right

Accuracy and fetches have been reported side by side throughout, and that hides the claim.
Compression is the thing being tested — a network has learned a task set when it reproduces it
from less than the set costs to write down — and accuracy is not a second axis but the
constraint compression has to satisfy. `compression_report.py` puts them in one quantity across
every network built today:

    reproduced = (problems solved exactly) x log2(answer space)
    stored     = (expert parameters actually routed to + router + heads) x 32 bits

**Over all 24 networks: 0.55 MiB reproduced from 29.35 MiB stored, a ratio of 0.019.** Not one
of them compresses. By the definition this project has been using, none of these models has
learned anything — they memorise with room to spare, and the task sets are small enough that
they can. Best single result is 0.083, on two-digit addition under the compression objective.

That reframes the day's results rather than overturning them. The compression objective does
move the ratio the right way almost everywhere: 0.055 → 0.083 on addition, 0.034 → 0.043 on
five operations, 0.013 → 0.022 on expressions. It is doing what it says.

**But it corrects one earlier claim.** The floor was reported as free — "the expected trade is
not there" — because it was measured in blocks read per problem, which is per-inference traffic.
On stored bits the trade is plainly there and it is large: 0.043 → 0.027 on five operations,
0.022 → 0.011 on expressions, 0.005 → 0.003 on equations. The floor recruits experts, recruited
experts are stored, and roughly half the compression goes to pay for the structure. The earlier
statement was true of the quantity it measured and wrong as a general claim, which is the
distinction I failed to draw.

Two levers now separate cleanly and pull against each other:

| | fetches per problem | stored bits | procedure structure |
|---|---|---|---|
| compression term λ | **down** | **down** | degrades |
| floor term β | up | **up ~2x** | **appears** |

**What would settle it.** A ratio above 1 needs either a much larger task set against a fixed
network, or a much smaller network against a fixed set. The second is a direct experiment and
has not been run at this scale: shrink the expert count until the ratio crosses 1 and see
whether exact accuracy survives. On the one-digit task that limit was found (16 slots) and on
the 30-way corpus task it broke at 64; on arithmetic, where the underlying rule really is tiny,
it should be findable and is the obvious next measurement.


## Shrinking to the crossing: it exists, and it is not where you want it

Direct test of the open question. Fixed task set — 49 800 two-digit problems over five
operations — and the network shrunk along both axes until reproduced bits exceed stored bits.
Budget to cross 1.0 at 32 bits per weight: 22 235 parameters, everything resident included.

| d | experts | used | exact acc | params | ratio @32b | ratio @16b | vs gzip |
|---|---|---|---|---|---|---|---|
| 96 | 64 | 26 | **0.821** | 254 442 | 0.072 | 0.143 | 0.022 |
| 64 | 32 | 23 | 0.774 | 102 058 | 0.169 | 0.337 | 0.056 |
| 48 | 16 | 14 | 0.758 | 37 386 | 0.451 | 0.901 | 0.152 |
| 32 | 16 | 12 | 0.650 | 15 722 | 0.919 | 1.838 | 0.362 |
| 32 | 8 | 8 | 0.717 | 11 370 | **1.401** | 2.803 | 0.500 |
| 24 | 8 | 8 | 0.481 | 7 002 | 1.527 | 3.054 | 0.813 |
| 16 | 4 | 4 | 0.437 | 2 570 | 3.782 | 7.563 | 2.214 |

**The ratio crosses 1.0 at d=32 with 8 experts — where exact accuracy is 0.717.** No
configuration reached 0.95, and none reached 0.99. Constrained to accuracy ≥ 0.80 the best
ratio available is **0.072**, which is the largest model in the sweep. On this task set
compression and correctness cannot be had at the same time; every step towards one is a step
away from the other, with no configuration in between.

Two baselines keep it honest, and both are lost. (The gzip figure was first written
down as 4.4x from memory rather than read off the run; `check-numbers` caught it at 10.9x,
which is the number below and makes the loss wider, not narrower.)

- **gzip -9 on the answer table** compresses it 10.9x. Every configuration with usable accuracy
  is well below that, so a general-purpose compressor beats the network at the network's own
  stated job.
- **A program that computes the answers** is 40 bytes — 6 225x better than the raw table. That
  is arithmetic's actual description length, and it is what "having learned the rule" would
  look like. Nothing here is within three orders of magnitude of it.

The trap this sweep had to close: at 400 steps it reported "CROSSED 1.0" at d=16 with **12.8 %
accuracy**. A ratio earned by being tiny and wrong is discarding, not compressing, and because
accuracy scales the numerator linearly while shrinking scales the denominator, the
unconstrained maximum always drifts to the smallest broken model. The accuracy floor is
reported alongside every ratio for that reason.

**What this settles and what it does not.** It settles that these networks do not compress this
task set: the honest summary of the whole day's arithmetic work is that the models memorise, and
when forced to be small they stop being right rather than becoming clever. It does not settle
whether the ratio is a property of the architecture or of the task set's size — 49 800 problems
is a small numerator, and the same network measured against the full enumerable space of a
larger task would cross comfortably *if it generalised*. Whether it generalises is exactly what
carries, borrows and negative operands would test, and that is the next dataset.


## Letting the network choose where to practise, and letting it run experiments

Three additions, on a task space too large to enumerate: three-digit signed operands over four
operations, about **16 million problems**, with held-out accuracy measured on problems the
training sampler is forbidden to draw (one residue class in 97). Reproduced bits may only be
claimed for problems never seen, so the compression ratio is now an estimate over the whole
space rather than a count of memorised answers.

That change alone settles one thing. Against 16 million problems the ratio is **64.0** at 47 %
accuracy — it crosses 1.0 almost regardless. `shrink.py` could not cross it at usable accuracy
because 49 800 problems is too small a numerator, not because the architecture cannot compress.
Which means the ratio stops being the interesting number the moment the task space is realistic,
and accuracy becomes the only question left.

### Self-curriculum: works as designed, does not pay

Each round the model is scored on held-out problems, and the buckets it is least certain about
get a larger share of the next round's sampling. Operations widen on a schedule. Against a
uniform arm at identical compute:

| | held-out acc | experts | ratio | 0 carries | 1 | 2 | 3 |
|---|---|---|---|---|---|---|---|
| uniform | **0.466** | 16 | **64.0** | 0.647 | 0.588 | 0.532 | 0.437 |
| uncertainty-driven | 0.458 | 26 | 40.9 | 0.637 | 0.585 | 0.563 | **0.499** |

It does exactly what it is built to do — **+0.062 on the hardest carry bucket, −0.010 on the
easiest** — and the average does not move. It also recruits 26 experts against 16, so the
compression ratio falls by a third. Redistribution, not improvement, and paid for.

Also measured, and it is the difficulty the rest of this section is about: **accuracy falls
monotonically with carries**, 0.647 → 0.588 → 0.532 → 0.437. Three-digit multiplication is not
learned at all (0.005) by anything tried here.

### Dreaming: reading the routing graph backwards

When new knowledge lands on particular experts, whatever else rode on those experts degrades.
Full replay avoids that and costs everything. The proposal is to ask the network which old
knowledge shares the experts about to be disturbed — the co-activation graph read in reverse —
and rehearse only that.

Base model on + and −, held-out accuracy 0.462. Then × is introduced, three arms at identical
compute:

| arm | old buckets rehearsed | old accuracy | retained | experts |
|---|---|---|---|---|
| new-only | 0 | 0.000 | 0 % | 13 |
| full-replay | 72 | 0.406 | **88 %** | 5 |
| dream | **18** | 0.143 | 31 % | 6 |

**Dream closes 35 % of the forgetting gap while rehearsing 25 % of the buckets** — better than
proportional, so the reverse lookup carries real signal, but far short of full replay. The
reason is visible in the selection itself: shared expert mass is 0.989 for the chosen buckets
against 0.941 for the rest. Almost everything shares experts. That is the expander property for
the fourth time today — the routing graph does not localise knowledge sharply enough to target
rehearsal precisely.

**A caveat that limits the whole experiment.** No arm learned multiplication (0.003–0.004). So
this measures forgetting caused by training on a task the model never acquired, not knowledge
displaced by knowledge. The forgetting is real and the weights genuinely moved, but the framing
it was built for is not satisfied.

### Experimenting: nudge a number, check the result, learn from the difference

The strongest result of the session, and the one that came from the objection that experimenting
is central to learning.

Alongside each problem the model is also asked what the answer would be if the first operand were
nudged by ±1, ±10 or ±100. The solver knows, so the model trains on the difference between what
it expected and what happened. A +1 that crosses a 9 boundary **is** a carry, presented as an
event rather than left implicit in a table of finished sums.

Staged like learning to drive — addition alone for the first half, subtraction added after.

| arm | held-out | 0 carries | 1 | 2 | 3 | experts |
|---|---|---|---|---|---|---|
| baseline | 0.530 | 0.578 | 0.573 | 0.512 | 0.411 | 5 |
| shuffled control | 0.512 | 0.596 | 0.532 | 0.502 | 0.377 | 6 |
| **probing** | **0.814** | 0.822 | 0.820 | 0.811 | **0.792** | 8 |

**+0.283 overall and +0.381 on three carries.** And the carry gradient nearly vanishes: the
baseline loses 0.167 accuracy going from no carries to three, probing loses 0.030. The thing
that made addition hard stops making it hard.

The control is what makes this a result rather than an anecdote. `shuffled` has the same
auxiliary head, the same number of parameters and the same number of gradient updates, but its
perturbation label comes from a *different* problem. It scores 0.512, no better than the
baseline's 0.530. The gain is in the **content of the experiment**, not in the extra signal —
which is the distinction that separates "self-experimentation works" from "more training works".

It costs three experts (5 → 8), so compression is essentially preserved.

### What is open

Multiplication remains unlearned by every method tried. It is also the operation that most
obviously needs decomposition and somewhere to put intermediate results — 432 × 7 as a sequence
of partial products that have to be held and summed. Nothing built today has anywhere to hold
them: state passes between hops only through the residual, which is a fixed-width vector shared
with everything else. An explicit scratchpad the experts read and write, carried across hops and
across agent switches, is the missing mechanism, and the fact that carries were fixed by making
the *event* visible suggests partial products would want the same treatment.


## A scratchpad for multiplication: three attempts, three negatives

The proposal was concrete: 432 x 7 is three partial products that have to be held while the
others are computed, and nothing built here has anywhere to hold them — state crosses hops only
through the residual, a fixed-width vector shared with everything else. So the experts got a
place to write: eight slots of 32 values, content-addressed, read at the start of each hop and
written at the end, per-example and discarded after.

**First it corrected me.** At three digits by one, *every* arm including the baseline reaches
**1.000** exact accuracy on held-out problems. I had written that "multiplication was never
learned by any method"; that was 2x2 and 3x3. 3x1 is learned outright, and a task all arms solve
cannot tell them apart. One real finding survived from that run: the scratchpad **halved expert
usage, 24 to 12**, at identical accuracy — a compression gain, not an accuracy one.

**At three digits by two it breaks, and nothing fixes it.**

| arm | exact | 10⁰ | 10¹ | 10² | 10³ | 10⁴ | experts |
|---|---|---|---|---|---|---|---|
| baseline | 0.051 | 1.000 | 0.319 | 0.145 | 0.958 | 0.996 | 10 |
| scratch | 0.049 | 1.000 | 0.324 | 0.143 | 0.934 | 0.995 | 8 |
| partials | 0.058 | 1.000 | 0.287 | 0.161 | 0.925 | 0.994 | 16 |
| scratch+partials | 0.056 | 1.000 | 0.330 | 0.142 | 0.934 | 0.995 | 7 |

Slot memory −0.002, partial-product supervision +0.007, both +0.005. All noise.

The digit breakdown is the informative part and it is sharp: **the units digit is perfect, the
top two digits are near-perfect, and everything collapses at 10¹ and 10²** — exactly where
shifted partial-product rows are summed and carries propagate along a chain.

**That diagnosis suggested a fix, and the fix also failed.** If the rows are not the hard part,
supervise the accumulation instead: after hop k, the running sum of the first k shifted rows.
It mirrors what made the perturbation objective work — a before-and-after pair rather than one
more target.

| arm | exact | 10⁰ | 10¹ | 10² | experts |
|---|---|---|---|---|---|
| baseline | 0.048 | 1.000 | 0.318 | 0.141 | 9 |
| running sum | 0.054 | 1.000 | 0.305 | 0.150 | 10 |
| scratch + running sum | 0.062 | 1.000 | 0.352 | 0.164 | 11 |

+0.006 and +0.014. Also noise. Three attempts, three negatives, and the hypothesis that
multiplication fails for want of somewhere to put intermediates is not supported.

What the evidence points to instead is that a fixed number of hops over a shared residual has no
mechanism for an iterative accumulation with a carry chain, and adding a memory the same hops
read and write does not create one — the memory is another route for the same information, not
a register file with a program over it. Addition's carry structure is local and one step deep,
which is why making the carry visible as an event fixed it. Multiplication's is neither.

## What actually made a model learn, across everything tried today

Six mechanisms were tested on the same architecture, same task family, with controls:

| mechanism | effect | verdict |
|---|---|---|
| perturbation: nudge an operand, train on the difference | **+0.283**, +0.381 at three carries | works, and the shuffled control rules out "more gradient" |
| scratchpad memory | +0.000 accuracy, **−50 % experts** | compresses, does not teach |
| partial-product supervision | +0.007 | nothing |
| running-sum supervision | +0.006 | nothing |
| uncertainty-driven curriculum | +0.062 on hard buckets, −0.008 average, −36 % compression | redistributes, and is paid for |
| reverse-routed rehearsal | 35 % of the forgetting gap on 25 % of the data | real but weak signal |

One thing separates the winner from the rest. The perturbation objective is the only mechanism
that gave the model a **paired intervention with an observable consequence** — this problem
against that problem, differing by one controlled change. Everything else gave it *more targets*:
richer labels on the same static examples. Partial products, running sums and a scratchpad are
all more supervision; the nudge is an experiment.

That is a usable rule for the question of how to build models that test and learn, and it is
falsifiable: **supervise differences the model can verify, not more of the answer.** The
shuffled control is what makes it more than a slogan — same head, same parameters, same number
of updates, perturbation label taken from a different problem, and the gain vanishes entirely
(0.512 against a 0.530 baseline). The effect is in the pairing, not in the extra signal.

What is not yet tested is whether the model can choose its own interventions. Today the nudges
were fixed (±1, ±10, ±100) and applied uniformly. The uncertainty machinery from
`curriculum.py` already knows where the model is unsure, and it failed at choosing *problems*;
whether it does better at choosing *experiments* is a different question and an open one.


## Retraction: the perturbation result was one seed, and it does not survive three

Two entries above I reported that nudging an operand and training on the difference gives
**+0.283** on held-out accuracy, that the shuffled control ruled out "more gradient", and I drew
a rule from it: *supervise differences the model can verify, not more of the answer.* That was
wrong, and the way it came apart is worth recording in full.

**First crack.** `fewshot.py` reproduced the same objective in a different harness — uniform
operand sampling, no staged curriculum — and the perturbation arm came out **worse** than the
baseline: 0.148 against 0.410 at 5 000 steps, 0.404 against 0.763 at 9 600. Opposite sign, same
mechanism.

**Second crack.** Removing only the staging inside the original harness, changing nothing else,
moved the *control* rather than the treatment. Shuffled went from 0.512 to **0.771** while
probing went 0.814 to 0.839, so the gap attributable to the nudge collapsed from +0.302 to
+0.068. The control that made the claim solid was itself unstable.

**Three seeds settle it.** Same code, seeds 0, 1, 2, 4 800 steps:

| setting | baseline | shuffled | probing |
|---|---|---|---|
| staged | 0.548 ± 0.069 | 0.465 ± 0.125 | **0.450 ± 0.161** |
| unstaged | 0.502 ± 0.133 | 0.610 ± 0.031 | 0.648 ± 0.049 |

Staged, the perturbation arm is *worse* than the baseline on average and spans **0.223 to
0.586** across seeds. Unstaged it is best, but only +0.038 above its own control, which is
inside the spread. The honest summary is that the perturbation objective has no established
effect, and the +0.283 was a favourable seed in a favourable schedule.

**What this does to the rest of today.** The standard deviations here — 0.03 to 0.16 — are
larger than most of the differences reported in this session, all of which came from single
runs at seed 0:

| result | reported | inside one-seed noise? |
|---|---|---|
| perturbation, +0.283 | single seed | **no — retracted** |
| uncertainty curriculum, −0.008 average, +0.062 on hard buckets | single seed | yes, cannot be distinguished |
| reverse-routed rehearsal closing 35 % of the gap | single seed | probably; unmeasured |
| partial-product supervision, +0.007 | single seed | yes |
| running-sum supervision, +0.014 | single seed | yes |
| scratchpad, ±0.000 accuracy | single seed | yes |

The negatives survive, because they were already reporting no effect and the variance only
makes that more certain. The one positive does not. Two results stand independently of seeds
because they are categorical rather than marginal: **3x1 multiplication is solved exactly
(1.000) by every arm**, and **3x2 collapses to ~0.05 for all of them**, with the failure
localised to the 10¹ and 10² digits where shifted rows are summed.

**The methodological failure, which is the real content of this entry.** This project built a
provenance field for the cost model, a contamination guard for the page cache, an
exact-partition guard for the router trace, and a 60-number regression gate — all after being
burned — and then compared learning mechanisms at n=1 for an entire session. Guarding the
inputs and re-deriving the conclusions is worthless if the comparison itself has no error bars.
`check-numbers` cheerfully verified +0.283 against the JSON that contained it; a consistency
gate cannot detect a consistent error, which is the second time today that sentence applies.

Nothing in `experiments/customsmoe/` should be compared across arms at a single seed again. The
seed sweep is in `data/custom/seeds.json` and the spreads there are the scale any future claim
has to clear.

## What still stands from the whole session

- The router trace was an argsort artefact; the fix is in `moetrace.c` and guarded by
  `Trace::reject_argsort_artefact`. Real routing is skewed 47x, not uniform, and LRU hits 78.6 %
  where the artefact reported 4.5 %. **21 numbers remain quarantined pending a re-run.**
- The cost model is measured, its provenance is recorded, and two independent cold calibrations
  agree to 1.3 % and 3.4 %.
- Walking an on-disk embedding index loses to scanning it below ~100 000 embeddings; verified
  against real uncached reads at 17.9x measured against 23.0x predicted.
- The kNN graph's expander property has one cause and four consequences, all measured: no
  gateway codec, short paths, no layout gain on index nodes, and no sharp locality for
  reverse-routed rehearsal.
- Networks on arithmetic do not compress: 0.55 MiB reproduced from 29.35 MiB stored across 24 of
  them, and the ratio only crosses 1.0 where accuracy has already fallen to 0.717. gzip on the
  answer table beats every usable configuration; a 40-byte program beats all of it by 6 225x.
- None of these networks learns from examples and generalises. At 4 096 distinct training
  problems every arm is below 0.08 on held-out problems; only unlimited fresh sampling reaches
  0.406. They need to see the space, not learn the rule.


## Correction: MPEdb does not store an embedding index, and I cited a document that does not exist

Two claims made earlier in this file and in `RESULTS.md`, `design/DESIGN-MPEE-TRANSFER.md` and
`experiments/hops/traversal_cost.py` were wrong, and they were wrong in the same way.

1. "MPEdb stores exactly this kind of index on disk." It does not. Searching all of
   `/Users/punnerud/Downloads/mpedb` for vector, embedding, columnar, zone-map, min/max, HNSW or
   IVF finds nothing — not in the 55 000 lines of Rust, not in the six design documents. MPEdb
   is an embedded multi-process SQL engine with one COW B+tree per table; its
   `DESIGN-MPEE-OPT.md` §1.6 explicitly *rejects* matrix compression because msync works at page
   granularity.
2. "`DESIGN-COLUMNAR` §1's per-block min/max, skipping blocks a predicate cannot satisfy." That
   document does not exist in the repository. I cited it with a section number.

The user raised the on-disk index as a possibility — "indeksene for embeddings kan ligge i MPEdb
på disk" — and I converted a proposal into a statement of fact about what a piece of software
does, then cited a non-existent design section in support of it. The second is the worse of the
two: a wrong attribution is a mistake, an invented citation is not.

**What this does and does not affect.** Nothing measured changes. `data/traversal-cost.json` and
`data/traversal-verify.json` were computed from embedding files, the measured cost model, and
real `F_NOCACHE` reads; MPEdb was never in the path. What was wrong was the *justification* for
treating the index as on-disk. The design still stands on its own — an index on disk is
perfectly ordinary — it simply is not something MPEdb implements today.

The guards in `check-numbers.py` could not have caught this. They compare documented numbers
against the JSON that produced them, and no number here was ever wrong. A claim about what other
software does has no data file behind it, which is precisely the category of error this project
has no instrument for.


## Both matcodec models have now been tested on embeddings. Both fail, for different reasons.

The previous conclusion — "matcodec does not transfer" — scored only the per-cluster rank-1
base, because that is the one `metric.rs` implements. matcodec keeps the better of *two* models
per matrix, and the second one, min-plus routing through landmarks, had never been tried here.
It is also the one that matches the proposal directly: find the extreme points, compute N×N only
among them, let everything else route through that skeleton.

`crates/matstruct/src/landmark.rs` ports it from `landeveier/mpee/crates/matcodec`.
`pick_landmarks` never touches n² — greedy facility location over a sample of pairs, O(L·n·S) —
and the resident index is one L×n table, 6.4 MB at n = 50 000 with L = 32.

**The port was validated before it was used.** Two unit tests, a positive and a negative
control: MPEE's own synthetic gateway world (regions joined by three roads each) must come out
mostly index-exact, and structureless points must not. Both pass. Without that the measurements
below would be unreadable, because a broken port and an unstructured dataset produce the same
number.

**One bug the validation caught, and it would have produced a false negative.** The first
implementation computed the angle as `acos(dot)`. `acos` has an infinite derivative at 1, so the
1e-7 slack left by f32 normalisation became a full quantisation unit — a point's distance to
*itself* came out as 1 rather than 0. Since "exact block" is integer equality of the residual,
that noise would have quietly destroyed exactly the quantity the whole experiment turns on, and
the result would have looked like a clean negative. Replaced with the half-angle form
`2·atan2(‖a−b‖, ‖a+b‖)`, which is well conditioned across the range, accumulated in f64.

### Result

| matrix | exact blocks | exact cells | residual / mean distance |
|---|---|---|---|
| corpus embeddings, raw | 0.62 % | 1.43 % | 0.624 |
| corpus embeddings, geodesic k=11 | 0.32 % | **24.91 %** | **0.096** |
| layer `add`-L0 / L1 / L2, raw | 0.00 / 0.17 / 0.33 % | 0.10 / 0.26 / 0.42 % | 0.425 / 0.378 / 0.365 |
| layer `mul`-L0 / L1 / L2, raw | 1.66 / 0.50 / 1.16 % | 2.46 / 1.31 / 2.04 % | 0.350 / 0.343 / 0.319 |
| layer `add`-L2, geodesic | 0.30 % | 13.35 % | 0.134 |
| layer `mul`-L2, geodesic | 0.74 % | 20.87 % | 0.099 |

The layers come from two models trained for this: `add` (three-digit signed + and −) reaches
0.730 held out, `mul` (three-digit by two-digit) reaches 0.046. The contrast was deliberate —
if the road network looked the same on a task the model had learned and one it had not, it would
not be measuring what the network knows.

**The base fits well. The block never does.** On geodesic distances the residual is under 10 %
of mean distance and a quarter of individual cells are reproduced exactly, which is a real fit.
But a block is answerable from the index only if *every* cell in it is exact, and that happens
under 1 % of the time everywhere. On a graph with no narrow cuts there is always one pair that
routes around the landmark, and one is enough to spoil the block. A good average fit with no
clean partition underneath it is precisely this signature.

The stop condition written into the plan has therefore fired, and phases 4 and 5 — the residual
as a novelty signal, and using it to select training examples — are dropped. There is no point
building a selection rule on a block criterion that fires 0.3 % of the time.

### Two things worth keeping

**The violation count was nearly a false alarm.** The geodesic matrix reported 158 799 triangle
violations, which for a shortest-path metric should be impossible. They are all exactly **1
unit** — quantisation of f32 path lengths to integers, nothing more. Reporting the count without
the magnitude would have been another unqualified number; `max_violation` is now part of the
report and the tool says which of the two it is looking at.

**The failing model looks more structured than the working one.** `mul`, at 0.046 accuracy, is
consistently more landmark-explainable than `add` at 0.730 — 1.16 % against 0.33 % exact blocks,
residual 0.319 against 0.365. The likely reason is that a model which learned little produces
less differentiated embeddings, which a skeleton fits more easily. Unverified, and a standing
caution: on this measure, more structure is not better representation.

### What was corrected on the way

Earlier today I wrote that MPEdb stores this kind of index on disk, and cited `DESIGN-COLUMNAR`
§1 for per-block min/max skipping. Neither is true; that document does not exist, and MPEdb has
no vector, embedding, columnar or zone-map machinery in code or design. The full correction is
in the entry above. Nothing measured depended on it — but an invented citation is worse than a
wrong attribution, and it belongs in the record next to the results it was used to justify.


## Three objections to the negative result, tested

The landmark result — under 1 % of blocks index-exact everywhere — invited three fair
objections. All three are now measured, and the result survives all three, which it might not
have.

### "The embeddings need training before they mean anything"

True of the models used, and the objection had teeth: `mul` had reached 0.046 held out, which is
barely learned at all. So measure the trajectory rather than one point — same architecture, same
held-out sample at every checkpoint, only the gradient steps varying.

| task | steps | accuracy | exact blocks | exact cells | residual / distance |
|---|---|---|---|---|---|
| add | 0 | 0.000 | 0.16 % | 0.57 % | 0.538 |
| add | 1 200 | 0.132 | 0.65 % | 1.11 % | 0.302 |
| add | 4 800 | 0.540 | 0.16 % | 0.60 % | 0.366 |
| add | **19 200** | **0.928** | 0.65 % | 1.12 % | 0.343 |
| mul | 0 | 0.000 | **1.79 %** | 3.06 % | 0.533 |
| mul | 4 800 | 0.042 | 1.30 % | 2.53 % | 0.356 |
| mul | 19 200 | 0.085 | 0.97 % | 2.11 % | 0.351 |

**Training does not create narrow cuts.** `add` goes from useless to 92.8 % correct — a
93-point swing — and its exact-block share does not move off the noise floor. And the *untrained*
`mul` network at step 0 has the highest exact-block share of anything measured in this project
(1.79 %), declining as it learns. That is the outcome named in advance as the least comfortable
one: an undifferentiated initial geometry fits a skeleton *better* than a shaped one does.

It also raised the accuracy the rest of this work should have been done at. The `add` model
dumped for the landmark measurement was trained to 9 600 steps and scored 0.730; 19 200 steps
gives 0.928. Earlier claims about what this architecture can reach on addition were undertrained,
not ceilings.

### "Maybe it is our 64-dimensional toy network, not embeddings"

The right control, and the cheapest of the three. Take the *same* held-out problems, write them
as text, and embed them with a real pretrained sentence embedder (all-minilm, 384 dimensions,
the one already used for the corpus). If a skeleton finds gateways in its view and not in ours,
the negative result is about the toy.

    blocks exact 0.49 %   cells exact 0.97 %   residual / distance 0.529

Same range as everything else. A trained embedder's view of these problems has no more gateway
structure than a 64-dimensional toy's. The result is about embedding geometry, not model size.

(The row count was verified against the input line count rather than assumed —
`llama-embedding` returned 271 rows for 250 lines earlier in this project, and a silent
misalignment would have made this table meaningless.)

### "Then use the embedding to group the training data"

Grouping does not need narrow cuts, so this survives the above and is worth its own measurement.
16-way spherical k-means over the pretrained embeddings, scored against facts we know about each
problem:

| grouped by minilm embedding | purity | majority baseline | lift |
|---|---|---|---|
| operation, + against − | 0.963 | 0.505 | **1.90x** |
| sign of the first operand | 0.786 | 0.500 | 1.57x |
| digits in the first operand | 0.901 | 0.901 | 1.00x |
| **carry count** | 0.406 | 0.354 | **1.15x** |

The embedding groups by **surface form** — which operator symbol is present, whether there is a
minus sign — and is close to blind to the one property that predicts difficulty. Carries are
where held-out accuracy falls from 0.647 to 0.437, and carries are the axis with 1.15x lift.

So a curriculum built on these clusters would sort problems by how they *look*, not by what is
hard about them, and there is no reason to expect it to beat uniform sampling. That is worth
knowing before spending three seeds per arm to find out — the uncertainty-driven curriculum
already failed at −0.008, and this would be a differently-motivated version of the same move.

### What the three together say

The negative result is now bounded rather than merely asserted. It is not a training-budget
artefact, not a small-model artefact, and the grouping fallback is blind to the property that
matters. What remains true is narrower and better supported than when it was one measurement:
**on these embeddings the landmark base is a good approximation with no clean partition
underneath it**, and the block criterion that would make it useful never fires.


## Grouping by carry count instead: it works, and it is the largest effect measured today

The embeddings grouped these problems by surface form and were blind to carries — 1.90x lift on
which operator is present, 1.15x on carry count, which is the axis accuracy actually depends on.
So group by carries directly, taking the count from the solver rather than asking the geometry
for it.

Four arms on three-digit signed addition and subtraction, identical architecture and compute,
scored on a held-out set drawn from the **natural** carry mix (0:17 %, 1:36 %, 2:34 %, 3:14 %)
so that oversampling the rare buckets has to pay for the common ones it displaces.

Six seeds for the two staged arms, three for the rest:

| arm | accuracy | vs uniform | z | pairwise seed wins |
|---|---|---|---|---|
| uniform | 0.408 ± 0.114 | — | — | — |
| balanced (equal mass per bucket) | 0.538 ± 0.013 | +0.057 | 1.6 | 6/9 |
| easy-first (0 → 3) | 0.565 ± 0.044 | **+0.157** | 3.1 | **35/36** |
| hard-first (3 → 0) | 0.633 ± 0.077 | **+0.226** | 4.0 | **34/36** |

**This is real**, and it is the only effect this project has established today that clears its
own noise floor. Thirty-five of thirty-six head-to-head seed comparisons is not a marginal call.

Per-bucket, the staged arms lift the hardest cases most: at three carries, uniform reaches 0.376
and easy-first reaches **0.746**, while giving up nothing on the easy end (0.424 → 0.612).

**Two things it is not.** It is not a "start easy" curriculum: `hard-first` beats `easy-first`
by 0.068, so the direction that matches the driving-lesson intuition is the weaker of the two.
And it is not really a schedule — `balanced`, which just reweights without any ordering, already
captures a third of the gain. What the three share is departing from the natural mix, which is
concentrated in buckets 1 and 2 and starves the 17 % and 14 % at the ends.

**A statistical correction, made mid-analysis.** The first verdict rule compared the difference
of means against the largest single-run standard deviation, and duly called +0.226 with 34 of 36
pairwise wins "inside the noise". The spread of individual runs is not the uncertainty in their
average; the right denominator is the standard error. The rule now reports the standard error,
the z, and a distribution-free pairwise-win count, and demands that the parametric and
non-parametric tests agree — and it refuses to give a verdict at all below five seeds. Loosening
a threshold to reach a wanted answer would have been the other way to fix this, and it is worth
recording which one was done.

**What the whole thread amounts to.** The landmark road network found nothing on embeddings, at
any training level, for a pretrained embedder as much as for our own. Clustering those same
embeddings sorted problems by appearance. But the partition that the geometry could not see —
carry count, one integer per problem, free from the solver — is worth +0.16 to +0.23 held-out
accuracy. The grouping idea was right. The embeddings were the wrong place to look for it.


## Can the grouping be found automatically? Six signals, three training levels, no.

Grouping by carry count is worth +0.157 to +0.226. The count came from the solver. The question
is whether anything the model can compute for itself recovers the same partition — and "without
ground truth" means something precise here: we have the *answers*, they are free on a solver
task; what we do not have is the *concept* of a carry.

Six candidates, bucketed four ways to match the four carry levels, at three training levels:

| steps | accuracy | best lift | between/within | signal means, carry 0 → 3 |
|---|---|---|---|---|
| 1 200 | 0.037 | 1.03x | 0.036 | residual 0.331 → 0.374 |
| 6 000 | 0.265 | 1.03x | 0.020 | residual 0.533 → 0.570 |
| 19 200 | **0.920** | 1.04x | 0.006 | residual 0.663 → **0.642** |

Layer clustering, pretrained-embedding clustering, predictive entropy, per-example loss,
gradient norm, and the MPEE landmark residual. Every one of them lands between 1.00x and 1.04x
at every training level. None reaches even the 1.15x the pretrained embedding managed, and the
true partition would score near 4x.

**The effect-size column is the finding, not the lift column.** The signal means *do* shift
monotonically with carries — loss at 1 200 steps runs 4.04, 4.03, 4.09, 4.22; the landmark
residual runs 0.331, 0.348, 0.357, 0.374. So difficulty really does track carries, exactly as
the accuracy curve said. But between-group variance over within-group variance is 0.0006 to
0.036: the spread *inside* one carry level is thirty to fifteen hundred times the gap *between*
levels. A per-example ranking cannot recover a partition of that shape, however good the
estimator, because the examples are not separated — only their averages are.

That distinction is why the measurement reports it. Purity alone would have said "look for a
better signal"; the variance ratio says there is nothing to sort by, and those call for opposite
next moves.

**Training does not fix it, and slightly hurts.** At 0.920 accuracy the effect sizes are the
*smallest* of the three stages, and the landmark residual's ordering reverses (0.663 → 0.642,
decreasing with carries where it previously increased). A converged model has low loss
everywhere, so what little separation existed is squeezed out.

### What this settles about the whole thread

The chain is now complete and every link is measured:

1. The landmark road network finds no exploitable structure in embeddings — at any training
   level, in our own network or a pretrained one, raw or geodesic. Under 1 % of blocks exact.
2. Clustering those embeddings sorts problems by surface form: 1.90x on which operator appears,
   1.15x on carries.
3. Grouping by carries directly is worth **+0.157 to +0.226**, at z 3.1–4.0 with 34–35 of 36
   pairwise seed wins.
4. No model-derived signal recovers that grouping, because carries move the mean of every such
   signal by far less than the examples vary within a level.

So the useful partition exists, is large, and is invisible to geometry, to uncertainty, to loss,
to gradients, and to the MPEE residual alike. It came from knowing what arithmetic *is*. That is
a specific and slightly uncomfortable conclusion: on this task the win came from domain
knowledge, and the automatic methods did not find a route to it.

**One caveat worth stating.** This is one property of one task. Carries are unusual in being a
sharp, discrete, low-cardinality feature that barely shows in the surface form — a nearly
adversarial case for representation-based discovery. A property that *is* reflected in how the
input looks would presumably be found; the pretrained embedder found the operator symbol at
1.90x without being asked. The result bounds what these signals can do, it does not show they
are useless.


## The control: a difficulty axis that IS visible in the input

The carry result came with a caveat — carries are sharp, discrete and nearly absent from the
surface form, which is close to an adversarial case for representation-based discovery. A
caveat that is never tested is an excuse, so: same architecture, same five signals, same
scoring, on an axis the input shows.

Operation type is the cleanest one available. The symbol is right there in the input, the
classes are balanced by construction rather than by resampling, and the difficulty genuinely
differs — at 19 200 steps the model reaches 0.836 on +, 0.820 on −, and **0.005** on ×.

Both axes are scored on the *same* problems from the *same* trained model, so nothing varies but
the property being looked for:

| signal | operation lift | op between/within | carry lift | carry between/within |
|---|---|---|---|---|
| entropy | **2.02x** | **59.20** | 1.00x | 0.0066 |
| loss | 2.01x | 24.06 | 1.01x | 0.0063 |
| layer-cluster | 1.66x | — | 1.00x | — |
| residual (MPEE) | 1.22x | 0.0655 | 1.01x | 0.0050 |
| gradnorm | 1.03x | 0.0007 | 1.01x | 0.0013 |

Perfect recovery is 3.00x on operation and 4.00x on carries.

**The caveat was right, and the size of the difference is the point.** Predictive entropy
separates the visible axis with an effect size of 59.2 and the invisible one with 0.0066 — a
factor of about nine thousand, in one run of one model against one metric. This is no longer an
argument about whether the estimator was good enough.

**But it refines the rule rather than just confirming it.** "Signals find visible properties" is
too generous. What found the operation was the model's **output uncertainty** — entropy at 2.02x
and loss at 2.01x. The geometry managed 1.66x by clustering and the MPEE landmark residual only
**1.22x**, and gradient norm found nothing at all (1.03x, effect size 0.0007) even here, where
the property is blatant and worth 0.8 accuracy.

So the ordering is: what the model is *unsure about* is legible; where it *sits in embedding
space* is much less so; how hard it is *pulling on the weights* is not legible at all. The
geometric route this whole thread pursued — landmarks, road networks, streaming residuals — is
the weakest of the three on the case designed to favour it, which is a more useful result than
the flat negative on carries.

### Where the thread ends

| question | answer |
|---|---|
| Does the landmark road network find structure in embeddings? | No — under 1 % exact blocks, at every training level, our model and a pretrained one |
| Does clustering embeddings group usefully? | By surface form, 1.90x on the operator; 1.15x on the axis that matters |
| Does the right grouping help? | Yes — **+0.157 to +0.226**, the largest verified effect in the project |
| Can that grouping be found automatically? | Not for carries. For a visible property, yes, and by uncertainty rather than by geometry |

The honest summary is that the payoff came from knowing what arithmetic is, and the automatic
methods reach only as far as the input makes the property visible. That is a real boundary, it is
now measured on both sides of itself, and it says where to point the next attempt: at what the
model cannot predict, not at where its activations land.


## 3LC's use case works. MPEE's does not. On the same embeddings.

The suggestion was to look at the kind of embeddings 3LC computes — a trained backbone's
features, mapped so similar items land together, used to curate a dataset by eye. The objection
implicit in it is fair: arithmetic may simply have no semantic cluster structure, and every
negative in this thread might be about the data rather than the method.

There is a set in the repository that settles it. `data/labelled/emb.f32` is 7 493 text chunks
in **39 known classes** — English prose, Python, C and Norwegian, split by source file —
embedded by the same pretrained sentence model. Unambiguous class structure, from a real
embedder, with labels to score against. As close to "cats against cars" as this project has.

| measurement on the same 7 493 embeddings | result |
|---|---|
| k-means recovery of the 39 classes | **8.21x lift** (purity 0.438 against a 0.053 majority) |
| the same after PCA to 32 dimensions | 7.72x |
| landmark gateway compression, exact blocks | **0.40 %** |
| residual as a share of mean distance | 0.604 |

**Clustering works. Gateway compression does not. On the same numbers.** Perfect class recovery
would be 18.7x, so 8.21x is a genuinely strong grouping — these embeddings really do put similar
things together, and a map drawn from them would be informative. That is 3LC's use case and it
is well supported.

What fails is the other thing, and now it is clear that the failure was never about the data
being structureless:

- **Clustering needs density variation** — regions where points sit closer to each other than to
  outsiders. That exists, at 8.21x.
- **Gateway compression needs narrow cuts** — a small set of points that *every* cross-region
  path must pass through. That does not exist, at 0.40 %.

Those are different properties, and a dataset can have the first in abundance while having none
of the second. This thread spent its length looking for the second in data that has the first,
which is why the geodesic transform kept looking encouraging (residual 0.096, a quarter of cells
exact) without ever producing an exact block.

Dimensionality reduction is not the missing piece either: PCA to 32 dimensions moves class
recovery by −0.5x and changes nothing about the block criterion. UMAP is not installed here, but
its essential move — a kNN graph and shortest paths through it — is precisely the geodesic
transform, which was measured and did not close the gap.

### The thread, complete

| what was asked | what was measured |
|---|---|
| Do embeddings have gateway structure a skeleton can exploit? | No — under 1 % exact blocks on every dataset tried, including one with 8.21x class structure |
| Do they cluster usefully? | **Yes**, 8.21x on real classes; 1.90x on the visible axis of the arithmetic task |
| Does the right grouping help training? | **Yes**, +0.157 to +0.226 — the project's largest verified effect |
| Can the useful grouping be found automatically? | Only when the input shows the property, and then by uncertainty (2.02x), not by geometry (1.22x) |

The two mechanisms MPEE and 3LC represent are not variants of one idea. One needs bottlenecks
and one needs neighbourhoods, embeddings supply the second and not the first, and no amount of
transform moved a dataset from one column to the other.


## Correction: there ARE waypoints. I was measuring the wrong criterion.

Every negative in this thread rested on one number — the share of blocks reproduced exactly by
the landmark index — and on the interpretation that a low value means "no narrow cuts, the graph
is an expander, every node is as good a waypoint as any other". The reframing that MPEE is
calculation between embeddings to find *hops* prompted the measurement I should have made first,
and it says the opposite.

**Betweenness on the kNN graph over the 39-class embeddings:**

| | |
|---|---|
| top 1 % of nodes carry | **31.3 %** of all traversals |
| a flat graph would give | 1.0 % |
| gateway concentration | **31.3x** |
| half of all traversals pass through | **148 of 4 000** nodes |

That is not an expander. Routes concentrate hard, exactly as they do on a road network.

**Why both things are true at once, and why it invalidates the earlier conclusion.** On a road
network the true distance *is* the routed distance, so a gateway base is exact by construction.
In embedding space the true distance is the chord — points do not have to travel through
anything — so the *metric* has no bottleneck even where the *graph* does. Every landmark
measurement in this thread paired facility-location selection with the angular metric, and on
that pairing no choice of landmark can ever make a block exact. The failure was in the pairing,
not in the data.

**The pairing that was never tried:** landmarks chosen by betweenness, scored against the
geodesic metric.

| | exact cells |
|---|---|
| facility-location landmarks, angular metric (all thread) | 0.91 % |
| **betweenness landmarks, geodesic metric** | **39.08 %** |

Forty-three times better. Audited before reporting, because a jump that size is more likely to
be a bug than a result: 158 008 pairs, **zero triangle violations** (the base never undercuts the
true geodesic, so the metric holds), and only 1.3 % of the exact matches have a landmark as an
endpoint, so it is not the trivial case. **Thirty-two points — 0.8 % of the set — reproduce 39 %
of all pairwise distances exactly.**

### What this retracts

Written repeatedly in this file and in `design/DESIGN-MPEE-TRANSFER.md`: "the embedding graph is
an expander, there are no narrow cuts, so no gateway codec can work". The first clause is false
as stated. There are narrow cuts in the routing structure; what is absent is any reason for the
*angular* metric to respect them. The correct statement is narrower and more useful:

> A gateway base is exact only when the metric being compressed is the one the graph induces.
> Compress geodesic distances and the gateways are real; compress chord distances and they are
> irrelevant, however concentrated the graph's traffic is.

That also explains the one result that never fitted the story. The geodesic transform kept
looking encouraging — residual falling from 0.624 to 0.096, cells exact rising from 1.43 % to
24.91 % — while blocks stayed under 1 %. It was half of the fix. The other half was choosing
landmarks by what the routes actually use rather than by facility location on the wrong metric.

### What it does not change

The block criterion is still the thing a codec needs, and 39 % of cells is not 39 % of blocks —
a block needs every cell exact. And nothing here revives the *training* results: grouping by
carry count is still worth +0.157 to +0.226, and no model-derived signal recovers it. Those
measurements never depended on the landmark model.

What is now open again, and was closed on a mistake, is whether the streaming index can serve
distance queries cheaply on geodesic embeddings. On this evidence it can answer two in five
exactly from 0.8 % of the data.


## Image embeddings from a YOLO backbone, and a cost function that is not geometry

### YOLO: the same shape as text

Imagenette's ten photo classes through a locally trained YOLO checkpoint, deepest backbone stage
global-average-pooled to 384 dimensions, 3 897 images. Scored exactly as the text sets were:

| | text, 39 classes | **images, 10 classes** |
|---|---|---|
| k-means class recovery | 8.21x | **5.58x** |
| exact blocks, angular metric | 0.40 % | **1.08 %** |
| gateway concentration (betweenness) | 31.3x | **19.6x** |
| exact cells, betweenness + geodesic | 39.08 % | **28.17 %** |

A perceptual backbone trained on photographs produces the same geometry as a sentence embedder,
in every respect measured: clustering works, the chord metric has no gateways, the graph does,
and pairing betweenness landmarks with the geodesic recovers a quarter to two-fifths of all
distances exactly. Whatever this structure is, it is not specific to text.

### The cost function is a lever, and a bigger one than the landmark choice

MPEE's road matrices cost travel time. Every measurement here assumed the analogue is angular
distance between embeddings — where a point *sits*. Weight strength is an equally available
cost in a routed network: the router's distribution over experts is how strongly an input
*engages* each part of the model.

Three costs, same trained model, same 4 000 problems:

| cost | carry lift | gateway concentration | mean hops | nodes on any route |
|---|---|---|---|---|
| embedding, final hidden state | 1.00x | 10.5x | 4.52 | 2 926 |
| **routing, soft distributions** | 1.01x | **31.7x** | 10.68 | 1 923 |
| selection, hard expert sets | 1.00x | ~~100.0x~~ | 2.02 | **4** |

**The routing cost triples the gateway concentration** over the embedding cost on the same data,
10.5x to 31.7x — a larger swing than any landmark-selection change. The cost function is the
lever, exactly as suggested, and geometry was never the only choice available.

**The 100x is discarded, and the reason is worth more than the number.** Hard expert selections
give only **four distinct signatures across 4 000 problems** — 3 996 duplicates — so the
"concentration" measures a four-point space. The tell was the value itself: 100.0x is precisely
what `top 1 % carries everything` produces at n = 4 000, and a round number arriving exactly at
its own ceiling is a shape, not a measurement.

That degeneracy is a finding in its own right. The block-compression term did its job far too
well: a model trained to touch few expert blocks routes 4 000 different arithmetic problems
through four patterns, so the hard routing carries essentially no information about the input.
It is the same over-collapse measured earlier at λ ≥ 0.3 — visible here as a geometry with four
points in it.

**And none of the three costs recovers carries** — 1.00x, 1.01x, 1.00x. The cost function moves
the routing structure by a factor of three and the useful partition not at all. Those remain two
separate problems, and only the first is responding to any of this.


## Embeddings encode category, not within-category structure — and that makes the null correct

The suggestion was that a trained embedding often carries little more than "this is a maths
problem", so the failure to recover carries may be the right answer rather than a missed one.
Testable, and it comes with a specific prediction attached: images would be one group, physics
formulas another but somewhat related.

Four domains through one embedder — arithmetic, physics formulas with values, English prose,
Norwegian — 800 chunks each, 3 200 points. (Python was generated too and **excluded**:
`llama-embedding` returned 818 rows for 800 lines at every chunk size tried, and a class that
cannot be aligned is dropped rather than guessed at. Embedding one domain per call is what kept
that from silently shifting every label after it.)

**Category is resolved almost perfectly.** Domain recovery purity **0.974** against a 0.250
majority — a lift of **3.90x of a possible 4.00x**. Four domains, four clusters, essentially no
confusion.

**Within a category there is nothing**, as measured all thread: 1.15x on carries, 1.00–1.04x for
every model-derived signal. The two numbers together are the whole story, and they say the null
result was correct rather than a failure of method. An embedding that tells you "this is
arithmetic" with 97 % purity and cannot tell 47 + 8 from 12 + 3 is behaving exactly as described.

**And the prediction about physics holds, to the digit.** Mean angular distance between domain
centroids:

| | arithmetic | physics | prose | norwegian |
|---|---|---|---|---|
| arithmetic | — | **1.253** | 1.446 | 1.443 |
| physics | 1.253 | — | 1.421 | 1.379 |
| prose | 1.446 | 1.421 | — | 1.281 |

Physics sits **0.168 closer to arithmetic than to prose**, and both legs through it (1.253 and
1.421) are shorter than the direct arithmetic-to-prose distance (1.446). It is not merely near
maths — it is geometrically *between* maths and prose, which is what "another group, but a bit
related" means if it means anything measurable.

**Multiple real domains give the strongest gateway structure yet found:**

| dataset | gateway concentration | exact cells, betweenness + geodesic |
|---|---|---|
| 39-class text, one register family | 31.3x | 39.08 % |
| YOLO image embeddings, 10 classes | 19.6x | 28.17 % |
| **four distinct domains** | 24.3x | **45.71 %** |

Nearly half of all pairwise distances reproduced exactly from 32 points out of 3 200. Genuine
domain boundaries make genuine bottlenecks — which is the same claim as before, now with the
boundaries supplied by the data rather than looked for inside a single domain.

### The thread in one line

Embeddings are a map of *what kind of thing* something is, they are close to uniform *within* a
kind, and every measurement in this session is consistent with that: strong clustering, strong
between-domain gateways, and nothing at all on a within-domain property like carrying a digit.
The place to look for within-domain structure is not the geometry.


## Embedding search can help find experts — and the corrected trace shows why

The closing point was that this form of embedding search should help *find experts*, even though
it cannot find carries. It is right, and the measurement that shows it also overturns another
artefact-era conclusion.

Expert routing is a **domain-level** property, which is exactly the level embeddings resolve at
0.974 purity. Per-register traces recaptured with the fixed `moetrace`:

| top-half expert overlap | prose | Norwegian | Python | C |
|---|---|---|---|---|
| **prose** | 100 % | 61.7 % | 42.6 % | **39.8 %** |
| **Norwegian** | 61.7 % | 100 % | 48.8 % | 50.6 % |
| **Python** | 42.6 % | 48.8 % | 100 % | **83.2 %** |
| **C** | 39.8 % | 50.6 % | 83.2 % | 100 % |

Chance for two random halves is 50 %. Code shares 83.2 % of its hot experts with other code;
prose shares 39.8 % with C, which is *below* chance — the registers actively avoid each other's
experts. Gini rises from 0.318 on English prose to 0.590 on C, and the hottest half of experts
carries 71.5 % to 88.1 % of traffic.

**On the artefact trace every one of those cells read 93.8 %, and the hottest half carried
exactly 50.0 %.** That is what produced the earlier conclusion that a pinned expert set
transfers across registers for free — 55.4 % hit either way — and it is now clearly wrong. It
was uniformity by construction, and the number was so flat it should have been suspicious on its
own.

### Why this reconnects to the original goal

The project began with: take an LLM, learn from how it is used, and fetch experts from disk fast
enough to run a model that does not fit in memory. The chain now closes:

1. Embeddings identify the domain with 0.974 purity and 3.90x of a possible 4.00x lift.
2. The domain determines a large share of the expert set — overlap ranges 39.8 % to 83.2 %, so
   knowing the register narrows the working set substantially.
3. That is a prefetch signal available **before the router runs**, from the input alone.

Prefetch was declared dead earlier in this project on the strength of cross-layer mutual
information of 0.66 of 5.96 bits — measured on the artefact trace, and therefore measuring
nothing. It is open again, and the route to it is the one suggested: search the embedding, get
the domain, fetch that domain's experts.

**And routing is a connection graph in its own right.** The table above is a measured similarity
between registers, derived from nothing but which experts fire: Python-to-C at 83.2 % is a link
the model itself asserts. That is the same "fire together, wire together" object the layout work
used, read at the level of domains rather than tokens — and unlike the embedding geometry, it
is built from what the weights actually do.

### Still outstanding

The 21 quarantined numbers remain quarantined. This entry re-measures one of them
(cross-register pinning) and confirms it was wrong; the rest — layout gains, the composed 3.1x
and 3.92x, cross-layer mutual information, headroom, reweighting, the ensemble study, and the
`matstruct probe-experts` control — still need a full re-run on corrected traces.


## The payoff: knowing the domain is worth 5.6x in fetch time

MPEE's shape is find the costs, solve, compress the solutions. With the corrected traces all
three parts are now measurable on the thing the project set out to do.

**The costs** are the register-to-register expert overlaps measured above, 39.8 % to 83.2 %.
**The solution** is a pinned expert set per domain. **The compression** is that domains which
overlap can share one set instead of holding two.

Static pinning at a 2 GiB budget, replaying one register with a set chosen from another:

| replay | pinned on | overlap | hit | ms/token |
|---|---|---|---|---|
| C | C | oracle | **89.4 %** | **2.68** |
| C | Python | 83 % | 85.4 % | 3.68 |
| C | prose | 40 % | 43.4 % | **14.98** |
| Norwegian | Norwegian | oracle | 89.9 % | 2.57 |
| Norwegian | prose | 62 % | 62.7 % | 9.85 |
| Norwegian | C | 51 % | 63.5 % | 9.48 |

**A 5.6x spread in fetch time** between pinning on the right domain and the wrong one, on
identical hardware, identical budget and identical layout. That is the whole case for embedding
search as an expert-finding mechanism: the embedding identifies the domain at 0.974 purity, and
the domain is worth 2.68 ms against 14.98.

**And the solutions compress.** C served from Python's pin set costs 3.68 ms against its own
2.68 — a 1.37x penalty for holding one set instead of two. Prose's set costs C 5.6x. So the
sharing structure is real and uneven: code shares with code nearly for free, and nothing shares
with prose. A cache serving mixed traffic should hold one set for the code family and a separate
one for prose, which is a partition the overlap matrix hands over directly.

**One result breaks the pattern and is left standing.** Norwegian pinned on **C** (51 % overlap)
beats Norwegian pinned on **prose** (62 % overlap) — 63.5 % against 62.7 %. Top-half set overlap
therefore does not predict hit rate monotonically. The likely reason is that a 2 GiB budget holds
roughly 54 % of the experts, so what decides a hit is frequency-weighted *mass* rather than set
membership, and two sets can intersect less while agreeing more about what is hot. That is a
measurable distinction and it has not been measured; the discrepancy is recorded rather than
smoothed away.

**Against the artefact.** Every cell of this table read 55.4 % and 12.56 ms on the old trace,
because uniform access makes the choice of pin set irrelevant by construction. The conclusion
drawn then — that a pinned set transfers across registers for free — was exactly backwards: it
transfers for free only between *related* domains, and costs 5.6x between unrelated ones.

### Where this leaves the original goal

Run a model that does not fit in memory, by fetching experts from disk fast enough. The chain is
now measured end to end on corrected data:

1. Embeddings identify the domain — 0.974 purity, 3.90x of a possible 4.00x.
2. Routing is strongly domain-dependent — Gini 0.318 to 0.590, overlaps 39.8 % to 83.2 %.
3. Domain-conditioned pinning is worth **5.6x in fetch time** over a mismatched set.
4. Related domains share a set for a 1.37x penalty, so the number of sets held is much smaller
   than the number of domains served.

What remains is the re-run of the 21 quarantined numbers on corrected traces, and the obvious
next measurement: predict the domain from the embedding at inference time and prefetch its set
before the router runs, which the numbers above say is worth most of a 5.6x.


## The 21 quarantined numbers, re-run on corrected traces

All three OLMoE traces were recaptured with the fixed `moetrace` — 25 000 tokens plain, 8 192
with contributions, 512 with raw expert vectors — plus a fresh Qwen3.6 trace, and every analysis
that fed a quarantined number was re-run against them.

**Eight came back unchanged. Eleven moved. Two inverted.**

| number | artefact | re-run | |
|---|---|---|---|
| first MoE layer routing match % | 100.0 | **100.0** | unchanged |
| gate/contribution spearman | 0.859 | **0.8624** | unchanged |
| rerank headroom keep=4 | 1.54 | **1.49** | unchanged |
| oracle + optimal weights keep=4 % | 30.32 | **30.319** | unchanged |
| expert alignment ratio | 0.477 | **0.477** | unchanged |
| chain cold speedup | 1.078 | **1.090** | changed — see the correction below |
| expert geodesic violations % | 0.0 | **0.0** | unchanged |
| expert geodesic readable blocks % | 0.0 | **0.0** | unchanged |
| reinforced a=4 readable blocks | 0.0 | **0.0** | unchanged |
| chain fetch reduction % | 14.4 | **36.02** | 2.5x better |
| chain modelled gain % | 5.22 | **13.27** | 2.5x better |
| static pinning hit, 2 GiB | 55.4 | **67.6** | better |
| decayed hit, 2 GiB | 31.2 | **76.5** | now the best policy |
| expert triangle violations % | 0.36 | **8.31** | 23x less metric |
| expert geodesic rank-1 gain % | 23.4 | **38.3** | better |
| OLMoE full stack ms/token | 62.8 | **41.99** | faster |
| **LRU hit, 2 GiB** | 4.5 | **74.7** | **inverted** |
| **static over LRU** | 2.24x faster | **0.81x — slower** | **inverted** |
| reinforced a=2 rank-1 gain % | 28.8 | **0.6** | collapsed |
| reinforced a=2 degenerate blocks | 0.0 | **0.0** | unchanged |
| Qwen full stack ms/token | 39.0 | 3.67 | not comparable, see below |

**The pattern is clean and it is the pattern you would predict.** Everything that measures
something *independent of the access pattern* survived: the permutation is still exact to
100.0000 %, gate order still correlates with contribution at 0.86, experts are still nearly
orthogonal at 0.477, optimal per-expert weights still recover only 30.3 %, and reranking still
has 1.49 points of headroom. Those conclusions never depended on the trace being right, and they
are unchanged to three significant figures.

Everything that measures *the access pattern itself* moved, and the two headline cache claims
inverted. LRU does not hit 4.5 %; it hits 74.7 %. Static pinning is not 2.24x faster than LRU;
it is 0.81x, i.e. **slower**. The best policy at 2 GiB is `decayed` — frequency with ageing — at
76.5 % and 48.08 ms/token, which was never the answer under the artefact.

**Two results are better than they were.** The layout solver now finds **36.0 %** fewer fetches
where the artefact allowed 14.4 %, because it finally has a co-activation graph with real
structure in it (lift 17.2 on the strongest pairs). And the geodesic-transformed expert graph
now clears matcodec's GO threshold at **+38.3 %** rank-1 gain over the null, against 23.4 %
before — though readable blocks remain at 0.00 %, so the codec still does not close.

**One collapsed.** Reinforcing the co-activation graph (`w' = w · lift^2`) was worth +28.8 % on
the artefact and is worth **+0.6 %** on real routing. The rich-get-richer sharpening was
amplifying a uniform graph into a structured-looking one; against a graph that already has
structure it adds nothing.

**One is recorded rather than claimed.** Qwen3.6 comes out at 3.67 ms/token against 39.0, but
the new trace is 2 000 tokens where the original was longer, and the hit rate is 97.5 % — the
working set is small enough that this is a different measurement, not an improvement. It is
guarded with that caveat attached rather than quoted as a 10x.

**Quarantine lifted.** `check-numbers.py` no longer carries a QUARANTINED block; all 21 are back
in the main gate with their re-measured values, and `make all` checks 132 numbers.


### Correction to the entry above: one of the nine was never re-run

`chain cold speedup` was listed as unchanged at 1.078x. It was not re-measured. The
`fetchbench replay` in that batch failed — `data/layouts/spectral.json` does not exist, the
layout is called `mincut.json` — and because the shell script had no `set -e`, the following
`echo REPLAY_DONE` ran anyway. `data/fetchbench.json` was left at its 08:24 timestamp, still
holding artefact-era numbers, and `check-numbers` passed because the stale file agreed with the
stale expectation.

Re-run properly:

| layout | fetches/token | median ms | speedup |
|---|---|---|---|
| identity | 337.5 | 6724.2 | 1.000x |
| random:1 / random:2 | 342.1 / 339.5 | 6864.8 / 7019.3 | 0.980x / 0.958x |
| frequency | 322.9 | 6685.6 | 1.006x |
| mincut | 265.9 | 6332.5 | 1.062x |
| **chain** | **240.4** | **6169.4** | **1.090x** |
| best+greedy | 238.6 | 6121.7 | **1.098x** |

So the tally is eight unchanged, not nine: cold speedup moved from 1.078x to **1.090x**, and
the replay's own fetch reduction is **28.8 %** against the 36.0 % the model predicts on holdout.
`best+greedy` now beats `chain` on the cold measurement as well as on the training split, which
it did not before.

Two things this cost. A number I had personally described as surviving the correction had not
been measured at all, and the tell — a median wall-clock time matching a previous run to one
decimal place — was visible in the output I quoted. Guarding a value against the file that
produced it cannot detect a file that was never rewritten; the missing check is on the
timestamp, and the missing habit is `set -e`.
