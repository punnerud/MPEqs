# Applying this to your own model

The procedure below is ordered so that the cheapest step that can rule the whole thing out
comes first. Step 1 costs about eight megabytes of download and answers "can this possibly
help me" before you trace anything or move a byte.

Every target is parameterised, so a different model needs no code change:

```sh
make model   REPO=unsloth/Qwen3.6-35B-A3B-GGUF \
             GGUF=Qwen3.6-35B-A3B-UD-IQ1_M.gguf \
             MODEL=models/Qwen3.6-35B-A3B-UD-IQ1_M.gguf
make pipeline MODEL=models/Qwen3.6-35B-A3B-UD-IQ1_M.gguf
```

`coact project` has been run against olmoe, qwen3_5_moe, qwen3moe, gpt-oss and deepseek2
headers with nothing special-cased, so the GGUF and MoE layers are architecture-agnostic in
practice and not just in intent. For a model with 256 experts, drop `--search-tokens` to a few
thousand: the local search is quadratic in the expert count.

## 0. Calibrate the device once

```sh
make costmodel
```

Evicts the page cache, then measures `C_fetch` and `C_byte` with uncached reads. Everything
downstream is scaled by these two numbers, so getting them from a warm file poisons the whole
study — this is why `calibrate` drops the cache itself rather than trusting you to.

Sanity check the output. On an Apple NVMe, expect `C_fetch` in the hundreds of microseconds
and 3–4 GB/s asymptotic. If you see single-digit microseconds and >10 GB/s, the eviction did
not work and you are measuring RAM.

## 1. Decide whether it is worth it, from the header alone

```sh
BYTES=40000000 ./scripts/fetch-headers.sh     # add your model to the MODELS list
make project
```

The GGUF header and tensor index sit at the front of the file, so a range request of a few
megabytes yields the complete expert geometry of a 400 GB model.

Read the **rng/MiB** column — requests per mebibyte of per-token traffic. It decides whether a
token is latency-bound or bandwidth-bound, and it is the only number here that predicts the
measured outcome:

| rng/MiB | verdict | example |
|---|---|---|
| > 3 | strong case; expect low double digits | Qwen3.6-35B-A3B, 3.86 → **1.140×** measured |
| 0.8–3 | worth doing; expect single digits | OLMoE-1B-7B, 0.83 → **1.078×** measured |
| < 0.5 | bandwidth-bound; use queue depth and a smaller quantisation instead | Qwen3-Coder-480B, 0.18 |

**Model size does not predict this.** The two 480 B-class models sit at the bottom of the
table, because they use few large experts; a 10 GB model with 256 tiny ones sits at the top.
The lever is expert granularity, and it is visible in the header before you download anything.

Do not judge a layout by fetch *count*. Qwen3.6 gets a smaller count reduction than OLMoE
(6.6 % against 14.4 %) and nearly double the time gain, because its requests are 270 KB — a
size at which this device manages 1.5 GB/s, so each one is dominated by latency.

## 2. Trace how the model actually routes

```sh
make corpus
make trace                    # 200k tokens, ~15 min, writes data/trace.bin
```

The corpus deliberately interleaves four registers (English prose, Norwegian, C, Python) in
fixed-size blocks. A single-register corpus overfits the layout to whichever experts that
register happens to use, and the holdout split will not catch it because the holdout is drawn
from the same text.

Requirements for a new architecture: nothing, if llama.cpp supports it. `moetrace` finds the
expert tensors through the GGUF metadata (`<arch>.expert_count`, `expert_used_count`,
`block_count`) and the `ffn_moe_topk` graph node. Hybrid models with interleaved dense layers
work — those layers are simply skipped.

## 3. Search for a layout and check it beats chance

```sh
make layouts
```

Read the table. **`random` and `frequency` must sit at zero.** They are the noise floor; if
your method does not clearly beat them, it has not shown anything. On OLMoE `chain` reaches
14.4 % fewer fetches while random stays within ±0.2 %.

Useful sanity metric: with `n` experts and top-`k`, a random layout averages
`k(1 − (k−1)/(n−1))` runs per layer and a perfect one averages 1. OLMoE went from 7.13 to
6.10, capturing 17 % of the theoretically available clustering. If yours captures far less,
the model's co-activation structure is weaker; far more and you should double-check the
holdout split.

Do not expect more from a longer search. Local search over both pairwise swaps and relocation,
started from the `chain` construction, improves it by 0.1 fetches per token out of 292.9. The
construction already captures what is there; what limits the gain is how much co-activation
structure the model has, not the optimiser.

## 4. Measure it cold, never warm

```sh
make fetchbench          # drops the cache first, then replays uncached
```

Two traps, both of which produced wrong numbers here before being caught:

- **`F_NOCACHE` does not evict.** It stops a descriptor from populating the cache; the kernel
  still serves pages that are already resident. Immediately after any process rewrote or read
  the file, the replay measures RAM. `fetchbench` flags any run whose throughput exceeds the
  calibrated ceiling — heed it.
- **Warm `llama-bench` will show nothing, correctly.** If the model fits in RAM, no expert is
  ever fetched and layout cannot matter. That run is the regression test, not the result.

## 5. Apply, verify, and keep the ability to undo

```sh
make baseline            # BEFORE applying: reference logits from the shipped file
make apply
make verify              # generations and argmax must be unchanged
make verify-routing      # first MoE layer must route identically — this is the real proof
make revert              # restores the shipped order, proven by SHA-256
```

`ggufperm init` records the pristine SHA-256 before anything is written, and `revert` refuses
to claim success unless the file hashes back to exactly that. `--dry-run` reports what would
move without writing.

Expect `verify` to report **generations identical, argmax unmoved, logits differing by ~1e-4
relative**. That is not a bug: llama.cpp's router softmax sums over experts in storage order,
so reordering perturbs the gate weights in their last bits. `verify-routing` is the claim that
matters, and it must be 100.0000 % at the first MoE layer.

On an expert-rich model the *aggregate* routing agreement will look alarming and is not. Qwen3.6
shows 67 % same expert set and 49 % same rank order across all layers, against OLMoE's 98 % and
93 % — forty layers instead of sixteen, and a flat router whose top-8 boundary is crowded with
near-ties. All six generations were still character-identical, because the flipped pairs have
almost equal gate weights. Judge by layer 0 and by the generations, not by the aggregate.

## 6. Never use LRU for the expert cache

If you hold experts in a fixed memory budget, the replacement policy matters more than
anything else in this document:

```sh
make policy
```

| policy | hit rate at 54 % residency | ms/token | vs LRU |
|---|---|---|---|
| LRU | 4.5 % | 203.58 | — |
| random replacement | 26.9 % | 145.35 | 1.40× |
| hybrid, half pinned | 28.4 % | 141.30 | 1.44× |
| decayed frequency (ageing) | 31.2 % | 137.06 | 1.49× |
| **static frequency pinning** | **55.4 %** | **90.82** | **2.24×** |

Load-balanced routing is near-uniform, so recency predicts nothing and LRU evicts entries
exactly as likely to be needed as the ones it admits. The sanity check is simple: **a cache
holding fraction *f* of the data should hit about *f* of the time.** Anything far below that is
doing harm, and LRU is far below it on both models tested.

Static pinning — preload the globally hottest experts from a trace, never evict — is optimal
for a stationary access distribution, and its hit rate slightly exceeds the residency fraction
because there is mild frequency skew to collect.

Every adaptive variant tested loses to it, including principled ones. Ageing (`--policy
decayed`), where scores halve periodically so a formerly hot expert falls out, reaches 31.2 %
— seven times LRU, still half of static. Splitting the budget between pinned and dynamic
(`--policy hybrid`) reaches 28.4 %, barely above random. Adaptation costs capacity relearning
a distribution that was already stationary.

That last word is the caveat. This corpus interleaves four registers every 4000 characters, so
it drifts fast and shallow; a single-user session drifts slowly and deeply, and ageing may win
there. Trace your own workload and rerun `make policy` before settling. What does not depend
on the workload is that LRU is the wrong default.

Sizing a machine is then straightforward, because the curve is linear:

| budget / expert footprint | hit rate | ms/token relative to uncached |
|---|---|---|
| 14 % | 12.9 % | 0.87 |
| 28 % | 27.3 % | 0.77 |
| 55 % | 55.4 % | 0.47 |
| 83 % | 83.7 % | 0.24 |

`ms/token ≈ (1 − cache_fraction) × uncached_ms`. There is no knee and no minimum viable
working set: every gigabyte buys the same proportional speedup. That only holds with a static
policy — LRU delivers almost none of it.

## 7. Serve it with queue depth

Parallel reads are worth more than layout — 1.6× from queue depth 1 to 4 on this device, and
it saturates there. The two overlap, since both buy down per-request latency, so measure the
combination rather than adding them:

```sh
make qdepth
```

## What not to bother with

Each of these was measured on OLMoE and failed. Re-measure on your model before dismissing
them, since the numbers above vary fivefold across architectures — but expect the same shape,
because the causes are training decisions common to nearly all MoE models.

| idea | why it fails | how to check on your model |
|---|---|---|
| Fewer experts (top-k) | k=7 costs KLD 0.057 for a 12.5 % saving; layout gives 14.4 % for free | `make kfrontier` |
| A better router | oracle reranking ceiling is 2.4 pp | `make headroom` |
| Reweighting what is left | optimal per-expert weights buy 3.0 pp; experts are near-orthogonal | `make reweight`, `make ensemble` |
| Caching experts *with LRU* | 4.5 % hit at 54 % residency — worse than chance | `make policy` |
| Prefetching from the previous layer | 0.66 bits of mutual information out of 5.96 | `make analyze` |
| Merging reads across unwanted experts | bandwidth rises 1.6×, bytes rise 6.5× | `fetchbench replay --max-gap N` |

The two root causes: load balancing during training makes expert access near-uniform, which
removes the skew that caching, prefetching and truncation all need; and the experts learn
near-orthogonal functions, which removes the redundancy that recombination and merging need.

Layout is the only lever that requires neither, because it changes nothing about the
computation — only where the bytes sit.
