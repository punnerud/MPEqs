# Results

> **Corrected 2026-08-08.** These results were originally computed from a router trace that was
> an artefact: `ffn_moe_topk` is a view of llama.cpp's argsort node, so a flat backend read
> returned each token's entire 64-wide expert ranking instead of its top-k, and expert access
> came out uniform *by construction*. Every trace-derived number has now been re-measured on
> corrected traces. **Eight came back unchanged, eleven moved, and two inverted** — LRU hits
> 74.7 % where the artefact said 4.5 %, and static pinning is 0.81x LRU rather than 2.24x
> faster. The numbers below are the corrected ones; `SUMMARY.md` has the full before/after and
> the two mistakes made during the re-run itself.


All numbers from `OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf` on a MacBook Pro M3 Pro (11 cores,
36 GB, Apple NVMe), llama.cpp b8500. Raw data in `data/`. `SUMMARY.md` has the chronology,
including the two measurement errors that had to be corrected.

## The question

Can a several-hundred-gigabyte MoE run meaningfully faster on a machine that cannot hold it,
by learning from how the model is actually used?

Short answer: **yes — mostly by not fetching at all, and secondarily by making the fetches that
remain cheaper.** A 2 GiB cache at 55 % residency serves 76.5 % of requests, and a lossless byte
permutation removes 28.8 % of what is left. What fails is fetching fewer *experts* per token:
truncation, reranking and recombination all cost more quality than they save time.

An earlier version of this document said the opposite — that only the second half worked, and
that caching was dead because load balancing flattens access. That rested on a broken router
trace and is withdrawn; see the correction at the top. The surviving reason for the failures
that remain is narrower: **the experts learn near-orthogonal functions**, so no kept expert can
stand in for a dropped one.

## The model, measured

```
arch             olmoe            file size        4.21 GB
MoE layers       16               experts / layer  64
experts / token  8                expert weights   3.90 GB (92.6 % of file)
per decode token 465.0 MiB in 384 disjoint ranges
```

llama.cpp fuses each layer's experts into three tensors with the expert axis last, so one
expert is three slabs ~100 MB apart: 16 layers × 8 experts × 3 tensors = 384 ranges per token.

Bytes per expert are not uniform. Q4_K_M keeps `ffn_down_exps` at Q6_K in layers
{0,1,4,7,10,13,14,15} (4 079 616 B) and Q4_K in the rest (3 538 944 B). Every calculation here
is per-layer for that reason.

## Device, calibrated cold

`fetchbench calibrate` evicts the page cache, then regresses `t = C_fetch + C_byte · size`
over uncached reads:

```
C_fetch = 230.74 us   C_byte = 0.2856 ns/B (3.50 GB/s)   R2 = 0.99921
break-even: one extra fetch costs as much as 789 KiB of transfer
```

A second cold calibration gave 227.80 µs and 0.2760 ns/B — within 1.3 % and 3.4 %, so the
device is stable and the contaminated reading really was 35× off rather than merely unlucky.

```
```

A cold 4 KiB read takes 220 µs. Per-fetch overhead is **39 % of OLMoE's per-token cost** —
that is the entire budget layout can compete for.

> An earlier calibration read pages that were still resident and reported 6.66 µs and
> 12.8 GB/s. A 35× error, and it silently disabled the contamination guard, because the guard
> compares achieved bandwidth against this very number. `calibrate` now drops the cache itself.

Bandwidth by request size, cold, queue depth 1: 1 MiB → 1.82, 4 MiB → 2.89, 8 MiB → 3.21,
32 MiB → 3.69 GB/s.

## What works: lossless expert layout

25 000 tokens of English prose, Norwegian Wikipedia, C and Python. Search on 20 000, scored on
the held-out 5 000, interleaved token-wise.

| method | fetches/token | modelled ms/token | vs shipped |
|---|---|---|---|
| identity (shipped) | 344.5 | 213.0 | — |
| random ×3 | 341.2–343.2 | 212.3–212.7 | +0.14 % to +0.35 % |
| frequency | 296.5 | 202.1 | +5.13 % |
| mincut (spectral + KL) | 235.7 | 188.2 | +11.64 % |
| **chain (greedy affinity)** | **220.4** | **184.8** | **+13.27 %** |
| best+greedy | 206.3 | 181.6 | **+14.78 %** |

Measured cold, 40 tokens, 5 reps, after `make drop-cache`:

| layout | fetches/token | median ms | speedup |
|---|---|---|---|
| identity | 337.5 | 6724.2 | 1.000× |
| random:1 / random:2 | 342.1 / 339.5 | 6864.8 / 7019.3 | 0.980× / 0.958× |
| frequency | 322.9 | 6685.6 | 1.006× |
| mincut | 265.9 | 6332.5 | 1.062× |
| **chain** | **240.4** | **6169.4** | **1.090×** |
| best+greedy | 238.6 | 6121.7 | **1.098×** |

**28.8 % fewer fetches and 9.0 % less cold fetch time for `chain`, at exactly zero quality
cost**, against a noise floor of 0.958–1.006×. The artefact allowed only 14.4 % and 7.8 %; a
co-activation graph with real structure in it (lift 17.2 on the strongest pairs) gives the same
solver twice as much to work with.

`best+greedy` — local search on the measured objective — now wins on the cold measurement as
well as on the training split, at 1.098×. Under the artefact it lost on holdout, which was the
standing argument for keeping the split. The split still earns its place; the example no longer
does.

### Queue depth, and how it competes with layout

| queue depth | identity GB/s | chain GB/s | layout gain |
|---|---|---|---|
| 1 | 2.46 | 2.55 | 3.3 % |
| 2 | 3.77 | 3.82 | 1.3 % |
| 4 | 3.90 | 3.93 | 0.6 % |
| 8 | 3.90 | 3.93 | 0.6 % |

Parallelism is the bigger lever (1.6×) and it saturates at four. It also **eats most of
layout's advantage**, because both buy down the same per-request latency. Use both, but do
not expect them to add.

### Bigger reads do not pay

Bridging gaps to merge fetches, queue depth 1:

| gap | MiB/token | GB/s | ms/token |
|---|---|---|---|
| 0 | 465 | 2.44 | **2402** |
| 4 | 821 | 3.15 | 3279 |
| 16 | 2423 | 3.41 | 8933 |
| 63 | 3029 | 3.92 | 9727 |

Bandwidth rises 1.6× across that range; bytes rise 6.5×. For 3.9 MB experts the trade never
closes. It would for a model with small experts — at 820 KB the break-even gap is 2 — so this
is a property of the model, not of the idea.

## What does not work, and why

### Fetching fewer experts

`llama-perplexity --kl-divergence`, 24 chunks, CPU backend throughout:

| k | fetches | PPL ratio | mean KLD | same top-1 |
|---|---|---|---|---|
| 8 | 100 % | 1.005 | 0 | 100 % |
| 7 | 88 % | 1.035 | 0.057 | 90.2 % |
| 6 | 75 % | 1.084 | 0.139 | 84.7 % |
| 4 | 50 % | 1.424 | 0.549 | 72.3 % |
| 2 | 25 % | 4.394 | 2.051 | 50.1 % |
| 1 | 12 % | 143.96 | 5.915 | 19.1 % |

The lossless layout removes **28.8 %** of fetches at KLD 0. The cheapest truncation removes
12.5 % at KLD 0.057. **Layout beats truncation on both axes at once**, and by more than twice
the margin the artefact allowed.

### Reranking, rescaling, or recombining what is left

With the raw expert output vectors captured (`moetrace --vecs`), every policy can be evaluated
exactly. Relative L2 error of the FFN output against the untruncated sum:

| keep | truncate | best scalar | oracle select | best weights | oracle + best weights |
|---|---|---|---|---|---|
| 2 | 57.60 % | 56.99 % | 53.63 % | 56.89 % | **52.94 %** |
| 4 | 33.35 % | 32.96 % | 30.77 % | 32.81 % | **30.32 %** |
| 6 | 17.51 % | 17.37 % | 16.02 % | 17.27 % | **15.84 %** |

The last column is the ceiling for *any* meta-model that overrides both which experts are used
and how they are merged: it picks the best subset and solves for the least-squares-optimal
per-expert weights, fitted against the answer it is trying to reproduce. At keep=4 it moves
the error from 33.35 % to 30.32 %. **Three percentage points.**

Naive renormalisation (`w_total / w_kept`, the obvious fix for `norm_topk_prob = false`) makes
things *worse* — 51.80 % at keep=4 — because the optimal scalar is 1.049, not the 1.37 the
heuristic wants.

### Why: the experts are near-orthogonal

| rank | cos with total | cos with the other seven | share of total |
|---|---|---|---|
| 1 | 0.640 | 0.042 | 38.4 % |
| 4 | 0.340 | 0.043 | 9.7 % |
| 8 | 0.173 | 0.034 | 2.2 % |

Alignment ratio `‖Σ w v‖ / Σ w‖v‖` = **0.477**, against 0.354 for orthogonal and 1.0 for a
unanimous vote. Mean pairwise cosine: **0.034–0.044**.

The layer is not an ensemble that votes. It is eight near-orthogonal specialists. Orthogonal
components cannot be reconstructed from one another, which is exactly why no reweighting
recovers a dropped expert — and it is measured twice, once as geometry and once as the
recombination bound.

#### The memory-budget curve

OLMoE, `identity` layout, 3.63 GiB of experts, best policy at each budget:

| budget | residency | best policy | hit rate | ms/token |
|---|---|---|---|---|
| 0 | 0 % | — | 0.0 % | 199.5 |
| 256 MiB | 6.9 % | decayed | 21.6 % | 162.0 |
| 512 MiB | 13.8 % | LRU | 35.8 % | 131.2 |
| 1 GiB | 27.5 % | decayed | 51.9 % | 107.0 |
| 2 GiB | 55.1 % | decayed | 76.5 % | 48.1 |
| 3 GiB | 82.6 % | static pinned | 93.6 % | 13.6 |

**Hit rate runs well ahead of residency** — 76.5 % held at 55.1 % — because real routing has
frequency skew for a policy to collect. Under the artefact it tracked residency almost exactly,
which was the signature of uniform access and was reported at the time as "no knee, no
threshold, every gigabyte buys the same proportional speedup". That is no longer true: the
curve is convex, and the last gigabyte before full residency buys far more than the first.

## The whole stack together

OLMoE, 2 GiB budget against a 3.63 GiB expert footprint, 96 tokens, output unchanged:

| layout | policy | queue depth | hit rate | MiB/token | ms/token |
|---|---|---|---|---|---|
| identity | LRU | 1 | 74.7 % | 117.7 | 44.32 |
| chain | static | 4 | 67.6 % | 150.9 | 41.99 |
| **chain** | **decayed** | **4** | **76.5 %** | **109.7** | **31.16** |

**1.42×** from layout plus the right policy at a fixed budget. The artefact reported 3.1× for
this stack, but from a baseline of 201 ms/token that only looked that bad because LRU was
hitting 4.5 %; with a cache that works, the uncached baseline is already largely absorbed and
the remaining levers are worth proportionally less.

The honest framing is that the *cache* is nearly all of it. Against no cache at all
(199.5 ms/token) the composed stack is **6.4×**, of which the budget itself supplies most and
layout, policy choice and queue depth divide the rest.

### Measured on a second model: Qwen3.6-35B-A3B

40 layers × 256 experts at 811 KB each — 960 ranges per token for only 249 MiB, the
request-dominated case the projection singled out. Cold replay, 48 tokens, 5 reps:

| layout | fetches/token | median ms | GB/s | speedup |
|---|---|---|---|---|
| identity | 866.0 | 5038.4 | 1.40 | 1.000× |
| frequency | 644.5 | 4800.7 | 1.76 | 1.050× |
| mincut | 500.4 | 4443.2 | 1.89 | 1.134× |
| **chain** | **554.1** | **4299.4** | **1.79** | **1.172×** |

**14.7 %, against OLMoE's 8.3 %** — and now the fetch *count* falls by **36.0 %** here against
OLMoE's 28.8 %, where on the artefact it fell by only 6.6 % against 14.4 %.

| | OLMoE (64 experts) | Qwen3.6 (256 experts) |
|---|---|---|
| fetch reduction | 28.8 % | **36.0 %** |
| **measured cold gain** | **8.3 %** | **14.7 %** |
| requests per MiB | 0.83 | 3.86 |
| top-8 share of softmax mass | 43 % | 18.5 % |

**The old reading of this table has to be withdrawn.** It said fetch count was "the wrong
intermediate to judge a layout by", because Qwen gained twice as much time from a fifth as many
fetches removed. On corrected traces both models remove a similar share of fetches and Qwen
still gains more time — so fetch count is a perfectly reasonable intermediate, and what differs
between the two models is what a removed fetch is *worth*. Qwen's requests are 270 KB, a size
at which this device reaches only 1.4–1.9 GB/s, so each one is latency-dominated. The
requests-per-MiB predictor survives; the argument that fetch count misleads does not.

Note also that `mincut` removes more fetches than `chain` (500.4 against 554.1) and is still
slower (4443.2 ms against 4299.4). Fetch count is a good intermediate, not a sufficient one.

### Queue depth on Qwen3.6: layout survives it here

| queue depth | identity | chain | GB/s | layout gain |
|---|---|---|---|---|
| 1 | 5021.9 ms | 4517.0 ms | 1.80 → 1.98 | 1.112× |
| 4 | 2303.5 ms | 2113.1 ms | 3.92 → 4.24 | **1.090×** |
| 8 | 1885.9 ms | 1850.3 ms | 4.78 → 4.84 | 1.019× |

Parallelism scales to **2.7×** here against OLMoE's 1.6×, because 270 KB requests leave more
latency to overlap — and layout still adds 9 % at queue depth 4, where on OLMoE it had already
collapsed.

Against a naive serial reader on the shipped file (queue depth 1, identity, 5021.9 ms):
**2.38× at queue depth 4 with chain**, **2.71× at queue depth 8**. That is the largest result
in this study, and it is mostly parallelism with layout adding the last 9 %.

### The predictor: requests per mebibyte

| model | GB | ranges/MiB | fetch overhead | measured cold gain |
|---|---|---|---|---|
| Qwen3.6-35B-A3B IQ1_M | 10 | **3.86** | — | **1.140×** |
| Qwen3.6-35B-A3B Q2_K_XL | 12 | 3.06 | 70.2 % | — |
| Qwen3-30B-A3B Q4_K_M | 19 | 1.10 | 45.9 % | — |
| OLMoE-1B-7B Q4_K_M | 4 | 0.83 | 39.0 % | 1.078× |
| gpt-oss-20b MXFP4 | 12 | 0.47 | 26.8 % | — |
| DeepSeek-V3 IQ1_S | 186 | 0.28 | 17.5 % | — |
| Qwen3-Coder-480B Q2_K_XL | 180 | **0.18** | 12.2 % | — |

Monotonic, with both measured points on the curve. **Layout does not scale with model size; it
scales with request density**, and today's largest models have the lowest density because they
use few large experts. A 600 GB model is bandwidth-bound and layout is worth 1–3 %. A 10 GB
model with 256 tiny experts is latency-bound and layout is worth 14 %. The lever is expert
granularity, which is visible from the header before anything is downloaded.

### Generality

`ggufperm` on `qwen35moe` — a different architecture, 256 experts, IQ2_XXS, 8.39 GB rewritten
in place in 33 s. First MoE layer routes 100.0000 % identically; all six greedy generations
character-identical; `revert` restored the SHA-256 exactly.

Deep-layer drift is much larger than OLMoE's: 67.1 % same expert set, 49.2 % same rank order.
Forty layers rather than sixteen, and a far flatter router (rank 1 at 0.051, rank 8 at 0.012)
means the top-8 boundary is crowded with near-ties that a last-bit perturbation flips. The
generations are identical anyway, because the flipped pairs have almost equal gate weights —
but on an expert-rich model the exactness proof must lean on layer 0, not on the aggregate.

## Is load balancing the cause? A controlled experiment

Every result above ends at the same explanation — load-balanced routing makes expert access
near-uniform, and without skew nothing that depends on locality works. That was an inference
from two pretrained models. `experiments/sparsemem/` tests it by switching the auxiliary loss
off in a model we control.

A product-key memory net — keys resident as the index, values on disk as the payload — trained
three times from the same seed for the same number of steps, differing only in the auxiliary
term. Four-register corpus, three balanced classes, 16384 slots, 64 MB of values:

| regime | accuracy | slots touched | Gini | top-1 % share |
|---|---|---|---|---|
| none | 0.996 | **1.6 %** | 0.996 | 99.4 % |
| load balancing | 0.994 | **26.1 %** | 0.959 | 80.9 % |
| entropy penalty | 0.994 | **0.8 %** | 0.996 | 100 % |

In pinned-cache terms, the same measurement the MoE work used:

| budget | resident | none | load balancing | entropy penalty |
|---|---|---|---|---|
| 1 MB | 1.6 % | **100.0 %** | 75–84 % | **100.0 %** |
| 8 MB | 12.5 % | 100.0 % | 93–94 % | 100.0 % |
| 16 MB | 25.0 % | 100.0 % | 95–99 % | 100.0 % |

(The load-balanced arm drifts by several points between runs despite a fixed seed — MPS
reductions are not deterministic — so it is reported as a range. The other two arms sit on
100.0 % at every budget and do not move.)

**One megabyte serves every retrieval without the auxiliary loss; sixteen are needed with it,
for accuracy identical to the third decimal.** Same architecture, same seed, same steps. Load
balancing is what makes the layer expensive to fetch.

On the synthetic control — a task where small retrieval budgets genuinely lose information —
the frontier separates too, not just the footprint: the load-balanced model scores 0.847 at
k=1 against 0.888, and only catches up at the full k=32.

The caveat that keeps this honest: on a task where a small model has ample capacity, the
auxiliary loss is pure cost. At frontier scale it is doing real work keeping hundreds of
experts trained, and a collapsed router would waste most of the parameters — the entropy
penalty at 1e-2 collapsed this one to 0.253 accuracy against 0.125 chance. What is now measured
is the loss's *price*, and the target a locality-aware variant would have to beat: 16× less
resident memory at equal accuracy.

## Compression is the measurement

Holding the model fixed and varying the data, `plain` regime, 16384 slots:

| training samples | accuracy | slots used | samples per slot |
|---|---|---|---|
| 200 | 0.986 | 459 | 0.44 |
| 1000 | 0.990 | 1065 | 0.94 |
| 4300 | 0.995 | **262** | **16.40** |

**The footprint shrinks as the data grows.** Below about a thousand examples the model
memorises, roughly one slot each. Past that it cannot, is forced into shared structure, and
collapses. Holding the data and shrinking the net instead, accuracy stays between 0.994 and
0.997 from 16384 slots down to **16** — 4300 examples at 269 per slot, no loss.

Relabelling the same corpus by source file instead of register — 30 classes, all the Python
chunks from CPython stdlib and all the C from curl — raises the entropy and makes the limit
visible: accuracy holds within four points from 16384 slots down to 256, then breaks (−6 % at
64, −13 % at 16), and the model uses 1.0 % of its capacity throughout. The register task's
limit was below 16 slots; the same bytes relabelled put it near 256.

So the disk footprint a model needs is set by the ratio of data entropy to model capacity, not
by the architecture. OLMoE and Qwen3.6 need 25–55 % residency because they are trained with
load balancing on language, whose entropy dwarfs any model built so far. The controlled
experiment isolates the first factor; these sweeps isolate the second.

## Losslessness

Three claims, and the differences between them matter.

**Routing: exact.** `coact compare`, 4096 tokens: the first MoE layer matches at
**100.0000 %**. Its routing depends only on the embedding and the router weights, so there is
no floating-point excuse — a mismatch would mean the permutation is wrong. Across all layers,
98.13 % same expert set, 93.44 % same rank order.

**Behaviour: unchanged.** Twelve prompts in English, Norwegian, German, Japanese, C, Python,
SQL, JSON and mathematics: all twelve generations character-identical, argmax unmoved.

**Bits: not identical.** Worst logit deviation 6.0 × 10⁻⁴ relative. llama.cpp computes gate
weights as `SOFT_MAX` over all 64 router logits and that sum runs *in storage order*, so
reordering perturbs every gate weight in the last bits and sixteen layers compound it. A
control rules out ordinary noise: the same file at 4, 5 and 8 threads gives bit-identical
logits. Fixing it is a property of `ggml_soft_max`, not of layout.

**Reversibility: proven.** `ggufperm revert` restores SHA-256
`4ddc0e53159ed512b8dd67914a66e27bc618f694672ba43a9a0454eabd9c684f` byte for byte. Peak extra
disk across the whole cycle: zero.

## End-to-end regression

| | pp512 | tg128 |
|---|---|---|
| shipped, Metal | 1374.00 ±4.08 | 129.49 ±0.75 |
| chain, Metal | 1365.07 ±23.16 | 129.60 ±0.23 |
| shipped, `-ncmoe 16` | 196.94 | 48.02 ±6.81 |
| chain, `-ncmoe 16` | 196.59 | 51.23 |

Identical within noise. A 4.2 GB model on a 36 GB machine is entirely page cached, so nothing
is fetched and layout cannot matter. This confirms the reordering is free; it is not the
result.

## Walking an on-disk embedding index

The index need not be resident — a proposed design, not an existing MPEdb feature; see the
correction in [DESIGN-MPEE-TRANSFER.md](design/DESIGN-MPEE-TRANSFER.md) — so the cost is fetch
rounds, not bytes held. Node records are DiskANN-style: vector and neighbour ids adjacent.

| k | walk visits | reaches target | full scan | walk | break-even N |
|---|---|---|---|---|---|
| 8 | 342 | 87.0 % | 1.93 ms | 55.8 ms | 127 572 |
| 16 | 350 | 98.0 % | 1.97 ms | 50.2 ms | 111 593 |
| 32 | 377 | 99.5 % | 2.06 ms | 47.1 ms | 99 507 |

Below ~100 000 embeddings (~160 MiB of index) reading the whole index is cheaper than walking
it. Pure greedy descent reaches the target only 49 % of the time at k=32 — a property of the
descent, not the graph — so the numbers above use a beam of 4. BFS reordering makes neighbour
sets *less* contiguous (24.0 runs -> 28.2), the expander property again.

Verified with real uncached reads: walk/scan 17.9x measured against 23.0x predicted. Both sides
land below the affine cost model in the same direction, so the ratio is the reliable part.

```sh
make traversal-cost traversal-verify
```

