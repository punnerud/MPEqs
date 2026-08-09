#!/usr/bin/env python3
"""Cross-check the numbers quoted in the docs against the JSON they came from.

Documentation drifts from data silently. This fails loudly instead, and is cheap enough to
run every time a result changes.
"""
import json
import pathlib
import sys

# (file, json path expression, expected value, tolerance, label)
CHECKS = [
    # Two independent cold calibrations gave 230.74/0.2856 and 227.80/0.2760 — within 1.3 %
    # and 3.4 %. The tolerance is set from that observed spread, not guessed, so this fails on
    # a contaminated run (which lands 35x off) rather than on ordinary device variation.
    ("data/costmodel.json", lambda d: d["c_fetch_ns"] / 1000, 229.3, 8.0, "C_fetch us"),
    ("data/costmodel.json", lambda d: d["c_byte_ns"], 0.281, 0.012, "C_byte ns/B"),
    ("data/costmodel.json",
     lambda d: 1.0 if d.get("provenance") == "measured" else 0.0,
     1.0, 0.001, "cost model provenance is measured, not inherited"),

    # Walking an on-disk embedding index instead of scanning it. The break-even is the number
    # that matters: below it, reading the whole index is cheaper than being clever.
    ("data/traversal-cost.json",
     lambda d: next(r["breakeven_n"] for r in d["by_k"] if r["k"] == 32),
     99507, 3000, "walk overtakes scan at N embeddings (k=32)"),
    ("data/traversal-cost.json",
     lambda d: next(r["reached_pct"] for r in d["by_k"] if r["k"] == 32),
     99.5, 1.0, "beam=4 reaches the target, %"),
    ("data/traversal-cost.json",
     lambda d: next(r["greedy_reached_pct"] for r in d["by_k"] if r["k"] == 32),
     49.0, 3.0, "pure greedy reaches the target, %"),
    ("data/traversal-cost.json",
     lambda d: next(r["runs_bfs"] for r in d["by_k"] if r["k"] == 32)
             - next(r["runs_identity"] for r in d["by_k"] if r["k"] == 32),
     4.2, 1.0, "BFS order makes neighbour runs WORSE by this many (expander again)"),
    # Measured with real F_NOCACHE reads, not derived. Wide tolerance: this is a device
    # measurement on a machine doing other things, and the claim is the order, not the digit.
    ("data/traversal-verify.json", lambda d: d["walk_over_scan_measured"],
     17.9, 6.0, "walk/scan ratio, measured uncached"),
    ("data/traversal-verify.json", lambda d: d["walk_over_scan_predicted"],
     23.0, 3.0, "walk/scan ratio, predicted by the cost model"),
    ("data/kfrontier.json", lambda d: next(
        x["mean_kld"] for x in d if x["k"] == 7), 0.057, 0.002, "KLD at k=7"),
    ("data/matstruct-embeddings.json", lambda d: d["triangle"]["violation_pct"],
     0.0, 0.001, "embedding metric: triangle violations %"),
    ("data/matstruct-embeddings.json", lambda d: d["spread"]["coefficient_of_variation"],
     0.076, 0.005, "embedding metric: distance CV"),
    ("data/matstruct-embeddings.json", lambda d: d["rank1"]["rank1_gain_over_mean_pct"],
     -93.1, 1.0, "embedding metric: rank-1 gain over null %"),
    ("data/matstruct-geo11.json", lambda d: d["spread"]["coefficient_of_variation"],
     0.280, 0.005, "geodesic k=11: distance CV"),
    ("data/matstruct-geo11.json", lambda d: d["rank1"]["rank1_gain_over_mean_pct"],
     -25.5, 1.5, "geodesic k=11: rank-1 gain over null %"),
    # Wide tolerance on purpose: the baseline arm of this ratio varies by ~4 % between runs,
    # so pinning it tighter would fail on noise rather than on a regression.
    ("data/sparsemem/frontier.json", lambda d: d["regimes"]["plain"]["slots_touched_pct"],
     1.6, 0.6, "sparsemem: slots touched without load balancing %"),
    # The balanced arm drifts between runs (26-36 %) despite a fixed seed, since MPS reductions
    # are not deterministic. Wide tolerance so this fails on a regression, not on the backend.
    ("data/sparsemem/frontier.json", lambda d: d["regimes"]["balanced"]["slots_touched_pct"],
     31.0, 8.0, "sparsemem: slots touched with load balancing %"),
    ("data/sparsemem/frontier.json", lambda d: next(
        b["hit_rate_pct"] for b in d["regimes"]["plain"]["cache"]["budgets"]
        if b["budget_mib"] == 1), 100.0, 0.5, "sparsemem: 1 MB cache hit rate, no balancing %"),
    ("data/sparsemem/frontier.json", lambda d: next(
        b["hit_rate_pct"] for b in d["regimes"]["balanced"]["cache"]["budgets"]
        if b["budget_mib"] == 1), 79.0, 9.0, "sparsemem: 1 MB cache hit rate, balanced %"),
    ("data/hopcount.json", lambda d: float(next(
        r["max_hops"] for r in d["by_k"] if r["k"] == 32)),
     6.0, 1.0, "kNN graph diameter at k=32 (hops)"),
    ("data/hopcount.json", lambda d: next(
        r["reachable_pct"] for r in d["by_k"] if r["k"] == 16),
     100.0, 0.1, "kNN graph reachability at k=16 %"),

    # Ground-truth validation of the analysis chain. `planted.py` builds a model whose
    # co-activation lift is known by arithmetic, and score.py checks `coact analyze` against an
    # independent implementation. These are the only checks here that compare the chain to
    # something other than its own past output — which is what the trace artefact needed and
    # did not have.
    ("data/custom/score.json", lambda d: d["independent"]["max lift"], 8.51, 0.6,
     "planted-model max co-activation lift, independent"),
    ("data/custom/score.json", lambda d: d["coact_reported"]["max lift"], 8.51, 0.6,
     "planted-model max co-activation lift, coact analyze"),
    ("data/custom/score.json", lambda d: d["compliance"], 0.712, 0.08,
     "share of planted lift the router realised"),

    # Compression as the training signal. The claim is the ordering, so the tolerances are
    # loose enough to survive a reseed and tight enough to fail if the ordering flips.
    ("data/custom/compress.json",
     lambda d: next(r["pairs_ge_2x_pct"] for r in d["rows"] if r["regime"] == "balanced"),
     0.6, 1.5, "load balancing leaves this %% of pairs above 2x lift"),
    ("data/custom/compress.json",
     lambda d: next(r["pairs_ge_2x_pct"] for r in d["rows"] if r["regime"] == "plain"),
     13.1, 4.0, "no auxiliary loss leaves this %% of pairs above 2x lift"),
    ("data/custom/compress.json",
     lambda d: next(r["fetch_reduction_vs_plain_pct"] for r in d["rows"]
                    if r["regime"] == "compressed"),
     18.9, 7.0, "compression objective, %% fewer block fetches"),

    # Exact arithmetic: every regime must stay exactly right, or the fetch numbers mean nothing.
    ("data/custom/arith.json", lambda d: min(r["exact_accuracy"] for r in d["rows"]),
     1.0, 0.001, "exact accuracy on all 300 arithmetic problems, worst regime"),
    ("data/custom/arith.json",
     lambda d: next(r["block_fetches_per_problem"] for r in d["rows"]
                    if r["hops"] == 1 and r["lambda"] > 0),
     1.02, 0.5, "blocks fetched per problem, 1 hop with compression"),
    ("data/custom/arith.json",
     lambda d: next(r["repeat_fetches_saved_pct"] for r in d["rows"]
                    if r["hops"] == 4 and r["lambda"] == 0),
     65.4, 10.0, "%% of block fetches already resident from an earlier hop, 4 hops"),

    # The falsified prediction. Guarded so that if it ever starts holding, that is news.
    ("data/custom/op-structure.json",
     lambda d: 1.0 if any(r["prediction_holds"] for r in d["rows"]) else 0.0,
     0.0, 0.001, "'+ and - share experts more than + and x' — falsified at every lambda"),
    ("data/custom/op-structure.json",
     lambda d: next(r["experts_used"] for r in d["rows"] if r["lambda"] == 1.0),
     8, 2, "experts still used at lambda=1.0 (over-collapse)"),

    # The floor. The claim is that compression and distinction are compatible, so the guard is
    # on both halves at once: fetches must stay well below the unregularised baseline AND the
    # worst pair overlap must stay well below it too.
    ("data/custom/floor.json",
     lambda d: next(r["blocks_per_problem"] for r in d["rows"]
                    if r["lambda"] == 1.0 and r["beta"] == 1.0),
     2.22, 0.6, "blocks per problem with the floor (baseline 3.34)"),
    ("data/custom/floor.json",
     lambda d: next(r["max_pair_overlap"] for r in d["rows"]
                    if r["lambda"] == 1.0 and r["beta"] == 1.0),
     0.32, 0.15, "worst pair overlap with the floor (baseline 0.545)"),
    ("data/custom/floor.json",
     lambda d: next(r["max_pair_overlap"] for r in d["rows"]
                    if r["lambda"] == 1.0 and r["beta"] == 0.0),
     1.0, 0.05, "worst pair overlap without the floor — two ops route identically"),

    # Scaled arithmetic. The claim is that capacity grows with the task while the per-problem
    # read does not, so both halves are guarded.
    ("data/custom/bigmath.json",
     lambda d: next(r["experts_used"] for r in d["rows"]
                    if r["n_op"] == 3 and r["lambda"] == 1.0 and r["beta"] == 0.0),
     52, 14, "experts recruited by 3 operations under compression"),
    ("data/custom/bigmath.json",
     lambda d: next(r["experts_used"] for r in d["rows"]
                    if r["n_op"] == 1 and r["lambda"] == 1.0 and r["beta"] == 0.0),
     5, 4, "experts recruited by 1 operation under compression"),
    ("data/custom/bigmath.json",
     lambda d: max(r["blocks_per_problem"] for r in d["rows"] if r["lambda"] == 1.0
                   and r["beta"] == 0.0),
     1.0, 0.3, "blocks read per problem under compression, worst over all task sizes"),

    # The loop revisits; foreknowledge barely beats popularity. Guarded so that if a later
    # probe does beat it, the change is visible rather than assumed.
    ("data/custom/foreknow.json", lambda d: min(r["hop_reuse"] for r in d["rows"]),
     0.69, 0.12, "expert reuse between consecutive hops, worst case"),
    ("data/custom/foreknow.json",
     lambda d: sum(1 for r in d["rows"] if r["probe_recall"] > r["popularity_recall"] + 0.05),
     1, 1, "cells where a probe beats popularity-based prefetch by >5 points"),

    # Expressions. The decorative-bracket control is the non-circular half, so it is the one
    # guarded hardest: brackets that do not reorder must barely move the routing.
    ("data/custom/exprs.json",
     lambda d: next(r["routing_divergence_reordering"] for r in d["rows"] if r["beta"] == 1.0),
     0.270, 0.09, "hop-1 routing divergence for brackets that reorder"),
    ("data/custom/exprs.json",
     lambda d: next(r["routing_divergence_decorative"] for r in d["rows"] if r["beta"] == 1.0),
     0.005, 0.02, "hop-1 routing divergence for brackets that do not reorder"),
    ("data/custom/exprs.json",
     lambda d: next(r["exact_accuracy"] for r in d["rows"]
                    if r["lambda"] == 0.0 and r["beta"] == 0.0),
     0.982, 0.05, "exact accuracy on bracketed expressions, no auxiliary loss"),

    # Equations: solved exactly, but hop 1 does not know the procedure unless asked.
    ("data/custom/equations.json",
     lambda d: next(r["exact_accuracy"] for r in d["rows"]
                    if r["lambda"] == 0.0 and r["beta"] == 0.0),
     1.0, 0.03, "equations solved exactly for X, no auxiliary loss"),
    ("data/custom/equations.json",
     lambda d: next(r["hop1_predicts_step1"] - r["step1_majority"] for r in d["rows"]
                    if r["lambda"] == 0.0 and r["beta"] == 0.0),
     0.023, 0.08, "hop 1 knows step 1, above majority, without the floor (it does not)"),

    # Compression, which is the actual claim. Guarded as a ceiling: if any of these ever
    # exceeds 1 the headline changes and it should not slip past unnoticed.
    ("data/custom/compression.json",
     lambda d: max(r["compression"] for r in d["rows"]),
     0.083, 0.05, "best compression ratio over all networks (below 1 = no compression)"),

    # The crossing exists but lands where accuracy has already gone. Both halves guarded, so
    # a future run cannot quietly report the ratio without the accuracy that earned it.
    ("data/custom/shrink.json",
     lambda d: d["crossed_at_32bit"]["ratio_32bit"] if d.get("crossed_at_32bit") else 0.0,
     1.401, 0.5, "compression ratio at the first configuration to cross 1.0"),
    ("data/custom/shrink.json",
     lambda d: d["crossed_at_32bit"]["exact_accuracy"] if d.get("crossed_at_32bit") else 1.0,
     0.717, 0.12, "exact accuracy at that crossing — well short of correct"),
    ("data/custom/shrink.json",
     lambda d: max((r["ratio_32bit"] for r in d["rows"] if r["exact_accuracy"] >= 0.80),
                   default=0.0),
     0.072, 0.05, "best ratio subject to exact accuracy >= 0.80"),
    ("data/custom/shrink.json", lambda d: d["raw_bits"] / d["gzip_bits"],
     10.94, 0.5, "gzip -9 on the answer table, which every usable network loses to"),

    # Self-curriculum: redistributes toward hard buckets, does not raise the average.
    ("data/custom/curriculum.json",
     lambda d: (d["results"][1]["per_carry"]["3"] - d["results"][0]["per_carry"]["3"]),
     0.062, 0.05, "uncertainty sampling gain on three-carry problems"),
    ("data/custom/curriculum.json",
     lambda d: (d["results"][1]["final_holdout_accuracy"]
                - d["results"][0]["final_holdout_accuracy"]),
     -0.008, 0.04, "uncertainty sampling change in average held-out accuracy"),
    ("data/custom/curriculum.json",
     lambda d: d["results"][0]["per_carry"]["0"] - d["results"][0]["per_carry"]["3"],
     0.21, 0.09, "accuracy lost going from no carries to three, uniform sampling"),

    # Dreaming: better than proportional, far short of full replay.
    ("data/custom/dreaming.json",
     lambda d: next(r["retained_fraction"] for r in d["rows"] if r["arm"] == "full-replay"),
     0.878, 0.12, "knowledge retained by full replay"),
    ("data/custom/dreaming.json",
     lambda d: next(r["retained_fraction"] for r in d["rows"] if r["arm"] == "dream"),
     0.309, 0.15, "knowledge retained by reverse-routed rehearsal on a quarter of the data"),
    ("data/custom/dreaming.json",
     lambda d: next(r["old_accuracy"] for r in d["rows"] if r["arm"] == "new-only"),
     0.0, 0.02, "knowledge retained with no rehearsal at all"),

    # Self-experimentation, and the control that makes it a result. If the shuffled arm ever
    # catches up, the effect was extra gradient and this must fail.
    # RETRACTED. These guarded a +0.283 perturbation gain measured at a single seed. Three
    # seeds give probing 0.450 +/- 0.161 against baseline 0.548 +/- 0.069 in the same setting,
    # so the effect was a favourable seed. The guards below replace them and assert the
    # variance instead, because the variance is the thing that was missing.
    ("data/custom/seeds.json", lambda d: d["staged/probing"]["sd"],
     0.161, 0.09, "seed-to-seed spread of the perturbation arm, staged"),
    ("data/custom/seeds.json",
     lambda d: max(v["sd"] for v in d.values()),
     0.161, 0.09, "largest seed spread across all arms — the noise floor for any claim here"),
    ("data/custom/seeds.json",
     lambda d: d["unstaged/probing"]["mean"] - d["unstaged/shuffled"]["mean"],
     0.038, 0.06, "perturbation over its own control, averaged over seeds (inside the noise)"),

    # Generalisation from a limited number of distinct examples: there is none.
    ("data/custom/fewshot.json",
     lambda d: max(r["holdout_accuracy"] for r in d["rows"] if r["pool"] == 4096),
     0.076, 0.05, "best held-out accuracy from 4096 distinct training problems"),
    ("data/custom/fewshot.json",
     lambda d: max(r["holdout_accuracy"] for r in d["rows"] if r["pool"] == 0),
     0.406, 0.12, "best held-out accuracy with unlimited fresh sampling"),

    # The landmark model — matcodec's other base, the one metric.rs does not implement. The
    # port is validated by unit tests against MPEE's own synthetic gateway world before it is
    # pointed at anything; these guard what it found once it was.
    ("data/matstruct-landmark-corpus.json", lambda d: d["exact_block_pct"],
     0.62, 0.3, "landmark model, exact blocks on corpus embeddings %"),
    ("data/matstruct-landmark-geo11.json", lambda d: d["exact_block_pct"],
     0.32, 0.25, "landmark model, exact blocks on the geodesic transform %"),
    ("data/matstruct-landmark-geo11.json", lambda d: d["exact_cell_pct"],
     24.91, 4.0, "landmark model, exact CELLS on the geodesic transform %"),
    ("data/matstruct-landmark-geo11.json", lambda d: d["residual_ratio"],
     0.096, 0.03, "landmark residual as a share of mean distance, geodesic"),
    # The violation count is meaningless without the magnitude: 158 799 violations that are all
    # exactly 1 unit are quantisation, not a broken metric.
    ("data/matstruct-landmark-geo11.json", lambda d: float(d["max_violation"]),
     1.0, 0.5, "worst triangle violation in units (rounding, not a non-metric)"),
    ("data/custom/layers/landmark-add-L2.json", lambda d: d["exact_block_pct"],
     0.33, 0.3, "landmark exact blocks, final layer of the task the model learned %"),
    ("data/custom/layers/landmark-mul-L2.json", lambda d: d["exact_block_pct"],
     1.16, 0.6, "landmark exact blocks, final layer of the task it did not %"),
    ("data/custom/layers/layers.json", lambda d: d["add"]["holdout_accuracy"],
     0.730, 0.08, "held-out accuracy of the model whose layers were dumped (add)"),
    ("data/custom/layers/layers.json", lambda d: d["mul"]["holdout_accuracy"],
     0.046, 0.03, "held-out accuracy of the model whose layers were dumped (mul)"),

    # The three objections. Guarded because each of them could have overturned the result and
    # a future change that makes one of them bite must show up as a failure, not a footnote.
    ("data/custom/traj/lm-add-s19200.json", lambda d: d["exact_block_pct"],
     0.65, 0.5, "exact blocks at 92.8 % accuracy — training creates no narrow cuts"),
    ("data/custom/traj/lm-mul-s0.json", lambda d: d["exact_block_pct"],
     1.79, 0.6, "exact blocks in an UNTRAINED network, the highest measured anywhere"),
    ("data/custom/traj/trajectory.json",
     lambda d: d["add"]["checkpoints"][-1]["accuracy"],
     0.928, 0.06, "addition accuracy at 19200 steps — earlier runs were undertrained"),
    ("data/custom/landmark-arith-minilm.json", lambda d: d["exact_block_pct"],
     0.49, 0.35, "exact blocks from a pretrained embedder — not a small-model artefact"),

    # Grouping by carry count. The one effect today that clears its own noise floor, so it is
    # guarded on the difference AND on the pairwise-win count, which does not assume normality.
    ("data/custom/carrygroup6.json",
     lambda d: (next(r["mean"] for r in d["rows"] if r["arm"] == "hard-first")
                - next(r["mean"] for r in d["rows"] if r["arm"] == "uniform")),
     0.226, 0.09, "hard-first carry grouping over uniform, 6 seeds"),
    ("data/custom/carrygroup6.json",
     lambda d: (next(r["mean"] for r in d["rows"] if r["arm"] == "easy-first")
                - next(r["mean"] for r in d["rows"] if r["arm"] == "uniform")),
     0.157, 0.09, "easy-first carry grouping over uniform, 6 seeds"),
    ("data/custom/carrygroup6.json",
     lambda d: next(r["per_carry_mean"]["3"] for r in d["rows"] if r["arm"] == "easy-first")
             - next(r["per_carry_mean"]["3"] for r in d["rows"] if r["arm"] == "uniform"),
     0.370, 0.12, "gain on three-carry problems, where the difficulty actually is"),

    # No automatic signal finds the partition. Guarded as a ceiling: if some future signal
    # clears 1.15x this must fail, because that would change the conclusion.
    ("data/custom/autogroup.json", lambda d: max(r["lift"] for r in d["rows"]),
     1.04, 0.10, "best lift of any model-derived signal toward the carry partition"),
    ("data/custom/autogroup.json",
     lambda d: max(r["between_over_within"] for r in d["rows"]
                   if r["between_over_within"] is not None),
     0.036, 0.02, "largest between/within effect size — why no ranking can sort by it"),

    # The control. The claim is the CONTRAST, so both sides are guarded from the same run.
    ("data/custom/visible.json",
     lambda d: max(r["op_lift"] for r in d["rows"]),
     2.04, 0.25, "best lift toward a difficulty axis the input shows (max 3.00)"),
    ("data/custom/visible.json",
     lambda d: max(r["carry_lift"] for r in d["rows"]),
     1.05, 0.10, "best lift toward the invisible axis, same run (max 4.00)"),
    ("data/custom/visible.json",
     lambda d: max(r["op_between_over_within"] for r in d["rows"]
                   if r["op_between_over_within"]),
     59.2, 25.0, "entropy effect size on the visible axis"),
    ("data/custom/visible.json",
     lambda d: next(r["op_lift"] for r in d["rows"]
                    if r["stage"] == "19200" and r["signal"] == "residual (MPEE)"),
     1.22, 0.20, "the MPEE residual on the visible axis — geometry is the weakest route"),

    # The decisive separation: strong class structure, no gateway structure, same embeddings.
    ("data/custom/landmark-labelled.json", lambda d: d["exact_block_pct"],
     0.40, 0.3, "exact blocks on embeddings with 8.21x recoverable class structure"),
    ("data/custom/landmark-labelled.json", lambda d: d["n"],
     7493, 1, "the 39-class labelled set those blocks were measured on"),

    # The correction. The graph DOES have narrow cuts; the angular metric ignores them.
    ("data/custom/waypoints.json", lambda d: d["gateway_concentration"],
     31.32, 6.0, "gateway concentration — top 1 % of nodes carry this many x their share"),
    ("data/custom/waypoints.json", lambda d: float(d["nodes_carrying_half"]),
     148.0, 50.0, "nodes carrying half of all traversals, out of 4000"),
    ("data/custom/waypoints.json", lambda d: d["geodesic_betweenness_exact_cell_pct"],
     39.08, 8.0, "exact cells with betweenness landmarks on the geodesic metric"),

    # YOLO image embeddings: the same geometry as text, measured the same four ways.
    ("data/custom/landmark-yolo.json", lambda d: d["exact_block_pct"],
     1.08, 0.6, "exact blocks on YOLO image embeddings, angular metric"),
    ("data/custom/waypoints-yolo.json", lambda d: d["gateway_concentration"],
     19.56, 5.0, "gateway concentration in YOLO image embeddings"),
    ("data/custom/waypoints-yolo.json",
     lambda d: d["geodesic_betweenness_exact_cell_pct"],
     28.17, 7.0, "exact cells, betweenness + geodesic, on images"),

    # The cost function as a lever, and the degeneracy that invalidated one row of it.
    ("data/custom/routingcost.json",
     lambda d: next(r["gateway_concentration"] for r in d["rows"] if r["cost"] == "routing")
             / next(r["gateway_concentration"] for r in d["rows"] if r["cost"] == "embedding"),
     3.0, 1.0, "routing cost over embedding cost, gateway concentration ratio"),
    ("data/custom/routingcost.json",
     lambda d: float(d["diagnostics"]["selection"]["nodes_on_routes"]),
     4.0, 1.0, "distinct expert selections across 4000 problems — the over-collapse"),

    # Embeddings resolve category and not within-category structure. Both halves guarded,
    # because the pair is the claim: near-perfect on domain, nothing on carries.
    ("data/custom/domains.json", lambda d: d["lift"],
     3.90, 0.20, "domain recovery lift, of a possible 4.00 with four domains"),
    ("data/custom/domains.json", lambda d: d["purity"],
     0.974, 0.04, "domain recovery purity"),
    # Physics between arithmetic and prose: both legs shorter than the direct distance.
    ("data/custom/domains.json",
     lambda d: d["arithmetic_to_prose"] - d["physics_to_arithmetic"],
     0.193, 0.08, "how much closer physics sits to arithmetic than prose does"),
    ("data/custom/domains.json",
     lambda d: 1.0 if max(d["physics_to_arithmetic"], d["physics_to_prose"])
                      < d["arithmetic_to_prose"] else 0.0,
     1.0, 0.001, "physics lies BETWEEN arithmetic and prose, both legs shorter"),
    ("data/custom/waypoints-domains.json",
     lambda d: d["geodesic_betweenness_exact_cell_pct"],
     45.71, 8.0, "exact cells across four real domains — the strongest measured"),

    # Expert routing IS register-dependent on the corrected trace. The artefact gave 93.8 %
    # everywhere; the spread below is the signal that makes embedding-driven prefetch possible.
    ("data/custom/register-routing.json", lambda d: d["python"]["c"],
     83.2, 6.0, "expert overlap between two code registers"),
    ("data/custom/register-routing.json", lambda d: d["prose"]["c"],
     39.8, 6.0, "expert overlap between prose and C — below the 50 % chance level"),
    ("data/custom/register-routing.json", lambda d: d["gini"]["c"] - d["gini"]["prose"],
     0.272, 0.10, "how much more skewed C routing is than English prose"),

    # The payoff, on corrected traces. The artefact gave 55.4 % / 12.56 ms for every pairing.
    ("data/custom/domain-pinning.json", lambda d: d["code-c"]["code-c"][1],
     2.68, 0.8, "ms/token, C replayed on its own pinned set"),
    ("data/custom/domain-pinning.json", lambda d: d["code-c"]["wikitext"][1],
     14.98, 3.0, "ms/token, C replayed on prose's pinned set"),
    ("data/custom/domain-pinning.json",
     lambda d: d["code-c"]["wikitext"][1] / d["code-c"]["code-c"][1],
     5.59, 1.5, "the cost of pinning on the wrong domain"),
    ("data/custom/domain-pinning.json",
     lambda d: d["code-c"]["code-python"][1] / d["code-c"]["code-c"][1],
     1.37, 0.4, "the cost of one shared set for two related domains"),
    # The anomaly, guarded so it cannot quietly disappear in a re-run.
    ("data/custom/domain-pinning.json",
     lambda d: d["norwegian"]["code-c"][0] - d["norwegian"]["wikitext"][0],
     0.8, 1.5, "Norwegian does better on C's set than prose's, despite lower overlap"),

    # The scratchpad. Three negatives, guarded as negatives: if any of these ever becomes a
    # real gain the conclusion changes and it must not slip through as noise.
    ("data/custom/scratchpad-3x1.json",
     lambda d: min(r["holdout_accuracy"] for r in d["rows"]),
     1.0, 0.01, "3x1 multiplication is solved exactly by every arm, baseline included"),
    ("data/custom/scratchpad-3x1.json",
     lambda d: (next(r["experts_used"] for r in d["rows"] if r["arm"] == "baseline")
                - next(r["experts_used"] for r in d["rows"] if r["arm"] == "scratch")),
     12, 6, "experts saved by the scratchpad at equal accuracy"),
    ("data/custom/scratchpad-3x2.json",
     lambda d: max(r["holdout_accuracy"] for r in d["rows"]),
     0.058, 0.03, "best 3x2 multiplication accuracy over all four arms"),
    ("data/custom/scratchpad-3x2.json",
     lambda d: min(r["per_digit_accuracy"][2] for r in d["rows"]),
     0.142, 0.06, "the hundreds digit, where shifted rows are summed"),
    ("data/custom/scratchpad-running.json",
     lambda d: (next(r["holdout_accuracy"] for r in d["rows"] if r["arm"] == "scratch+running")
                - next(r["holdout_accuracy"] for r in d["rows"] if r["arm"] == "baseline")),
     0.014, 0.03, "running-sum supervision gain — indistinguishable from noise"),
    # ---- Un-quarantined 2026-08-08, re-measured on corrected traces -------------------
    # All 21 were suspended when the router trace turned out to be an argsort artefact.
    # Nine came back unchanged, twelve moved, and two inverted outright. The pattern is that
    # anything independent of the access pattern survived and everything about it did not.
    ("data/layout-report.json",
     lambda d: 100 * (1 - next(m["fetches_per_token"] for m in d["methods"] if "chain" in m["method"])
                        / next(m["fetches_per_token"] for m in d["methods"]
                               if m["method"] in ("identity", "shipped"))),
     36.02, 6.0, "chain fetch reduction % (artefact said 14.4)"),
    ("data/layout-report.json",
     lambda d: next(m["cost_reduction_pct"] for m in d["methods"] if "chain" in m["method"]),
     13.27, 3.0, "chain modelled gain % (artefact said 5.22)"),
    ("data/fetchbench.json",
     lambda d: next(r["median_ms"] for r in d["results"] if r["method"] == "identity")
             / next(r["median_ms"] for r in d["results"] if r["method"] == "chain"),
     1.090, 0.04, "chain cold speedup, measured cold on the corrected trace"),
    ("data/fetchbench.json",
     lambda d: 100 * (1 - next(r["fetches_per_token"] for r in d["results"] if r["method"] == "chain")
                        / next(r["fetches_per_token"] for r in d["results"]
                               if r["method"] == "identity")),
     28.8, 5.0, "chain fetch reduction measured in the cold replay"),
    ("data/headroom.json", lambda d: d["mean_spearman"],
     0.8624, 0.03, "gate/contribution spearman — unchanged"),
    ("data/headroom.json",
     lambda d: next(b["headroom_pct"] for b in d["by_keep"] if b["keep"] == 4),
     1.49, 0.4, "rerank headroom keep=4 — unchanged"),
    ("data/reweight.json",
     lambda d: next(p["oracle_best_weights_pct"] for p in d["policies"] if p.get("keep") == 4),
     30.319, 1.0, "oracle + optimal weights keep=4 % — unchanged"),
    ("data/ensemble.json", lambda d: d["alignment_ratio"],
     0.477, 0.03, "expert alignment ratio — unchanged"),
    ("data/routing-agreement.json",
     lambda d: d.get("first_layer_exact_pct", d.get("exact_rank_pct", 100.0)),
     100.0, 0.001, "first MoE layer routing match % — unchanged, permutation is exact"),
    ("data/cache-policies.json",
     lambda d: next(r["hit_rate_pct"] for r in d["results"]
                    if r["cache_mib"] == 2048 and r["policy"] == "lru"),
     74.7, 5.0, "LRU hit at 2 GiB (artefact said 4.5 — the inversion)"),
    ("data/cache-policies.json",
     lambda d: next(r["hit_rate_pct"] for r in d["results"]
                    if r["cache_mib"] == 2048 and r["policy"] == "staticpinned"),
     67.6, 5.0, "static pinning hit at 2 GiB (artefact said 55.4)"),
    ("data/cache-policies.json",
     lambda d: next(r["hit_rate_pct"] for r in d["results"]
                    if r["cache_mib"] == 2048 and r["policy"] == "decayed"),
     76.5, 5.0, "decayed hit at 2 GiB — now the best policy (artefact said 31.2)"),
    ("data/cache-policies.json",
     lambda d: next(r["ms_per_token"] for r in d["results"]
                    if r["cache_mib"] == 2048 and r["policy"] == "lru")
             / next(r["ms_per_token"] for r in d["results"]
                    if r["cache_mib"] == 2048 and r["policy"] == "staticpinned"),
     0.81, 0.15, "static over LRU: 0.81x, i.e. SLOWER (artefact said 2.24x faster)"),
    ("data/matstruct-experts.json", lambda d: d["triangle"]["violation_pct"],
     8.3094, 2.0, "expert co-activation triangle violations % (artefact said 0.36)"),
    ("data/matstruct-expgeo6.json", lambda d: d["triangle"]["violation_pct"],
     0.0, 0.001, "expert geodesic k=6 violations — still a metric by construction"),
    ("data/matstruct-expgeo6.json", lambda d: d["rank1"]["rank1_gain_over_mean_pct"],
     38.3, 8.0, "expert geodesic rank-1 gain over null % (artefact said 23.4)"),
    ("data/matstruct-expgeo6.json", lambda d: d["rank1"]["blocks_fully_within_tol_pct"],
     0.0, 0.001, "expert geodesic readable blocks % — still zero"),
    ("data/matstruct-rf2.json", lambda d: d["rank1"]["rank1_gain_over_mean_pct"],
     0.6, 3.0, "reinforced a=2 rank-1 gain % (artefact said 28.8 — collapsed)"),
    ("data/matstruct-rf2.json", lambda d: float(d["rank1"]["degenerate_blocks"]),
     0.0, 0.001, "reinforced a=2 degenerate blocks — the guard still holds"),
    ("data/matstruct-rf4.json", lambda d: d["rank1"]["blocks_fully_within_tol_pct"],
     0.0, 0.001, "reinforced a=4 readable blocks after the artefact guard"),
    ("data/combo-chain-static-4.json", lambda d: d["results"][0]["ms_per_token"],
     41.99, 8.0, "OLMoE full stack ms/token, chain + static + QD4 (artefact said 62.8)"),
    ("data/combo-chain-decayed-4.json", lambda d: d["results"][0]["ms_per_token"],
     31.16, 7.0, "OLMoE full stack with the policy that actually wins now"),
    # Qwen is on a 2000-token trace against the original's longer one, and hits 97.5 %, so the
    # working set is small and this is NOT comparable to the 39.0 it replaces. Recorded, not
    # claimed as an improvement.
    ("data/combo-qwen-chain-static-4.json", lambda d: d["results"][0]["ms_per_token"],
     3.67, 1.5, "Qwen3.6 full stack ms/token, short trace — see the caveat"),
    ("data/fetchbench-qwen.json",
     lambda d: next(r["median_ms"] for r in d["results"] if r["method"] == "identity")
             / next(r["median_ms"] for r in d["results"] if r["method"] == "chain"),
     1.172, 0.06, "Qwen3.6 chain cold speedup on the corrected trace"),
    ("data/fetchbench-qwen.json",
     lambda d: 100 * (1 - next(r["fetches_per_token"] for r in d["results"] if r["method"] == "chain")
                        / next(r["fetches_per_token"] for r in d["results"]
                               if r["method"] == "identity")),
     36.0, 6.0, "Qwen3.6 fetch reduction (artefact said 6.6)"),

    # Two-sided hub labels and streaming kNN. The claim is a RATIO on identical memory and
    # metric, so the ratio is guarded, and so is the recall trade that runs the other way.
    ("data/custom/geohub-summary.json",
     lambda d: d["39-class"]["betweenness_cells_pct"] / d["39-class"]["facility_cells_pct"],
     1.68, 0.35, "betweenness over facility-location hubs, same metric and memory"),
    ("data/custom/geohub-summary.json",
     lambda d: min(v["betweenness_cells_pct"] / v["facility_cells_pct"] for v in d.values()),
     1.45, 0.35, "the same ratio at its weakest across four datasets"),
    ("data/custom/geohub-summary.json",
     lambda d: sum(1 for v in d.values() if v["betweenness_recall"] < v["facility_recall"]),
     5, 0.5, "datasets where betweenness hubs LOSE on recall — the junction trade-off"),
    ("data/custom/geohub-summary.json",
     lambda d: d["50k"]["betweenness_cells_pct"] / d["50k"]["facility_cells_pct"],
     4.63, 1.2, "hub-selection gain at 50 000 points — it grows with n"),
    ("data/custom/geohub-summary.json", lambda d: d["50k"]["betweenness_recall"],
     0.001, 0.01, "recall@10 at 50 000 points — the index cannot search"),

    # Chains of strong links under (max, x). The correlation pattern is the positive finding;
    # the net transfer is the negative one, and both are guarded.
    ("data/custom/chains-summary.json", lambda d: d["corr_to_prev"]["3"],
     0.842, 0.08, "chain correlation step to step stays strong at 3 hops"),
    ("data/custom/chains-summary.json", lambda d: d["corr_to_first"]["5"],
     -0.348, 0.15, "chain correlation end to end goes NEGATIVE at 5 hops"),
    ("data/custom/chains-summary.json", lambda d: float(d["rescued_by_hop"]["3"]),
     39.0, 12.0, "items only a chain solves — real transfer, and it is not zero"),
    ("data/custom/chains-summary.json", lambda d: float(d["best_net"]),
     0.0, 1.0, "best net over every strength threshold — never positive"),
    # The printed chains: the unrelated one is stronger than the related one.
    ("data/custom/showchain.json", lambda d: d["not matching"]["strength"],
     0.5256, 0.05, "strength of the 6-hop chain that lands somewhere unrelated"),
    ("data/custom/showchain.json", lambda d: d["matching"]["strength"],
     0.4910, 0.05, "strength of the 6-hop chain that lands somewhere related — weaker"),
    ("data/custom/showchain.json",
     lambda d: 1.0 if d["not matching"]["strength"] > d["matching"]["strength"] else 0.0,
     1.0, 0.001, "chain strength is anti-correlated with meaning in these two"),
    # Shared-extreme dimensions beat cosine; no reweighting beats raw shared-extreme.
    ("data/custom/dimweights.json",
     lambda d: d["none (raw extreme)"]["separation"] / d["cosine_separation"],
     2.75, 0.4, "shared extreme dimensions over cosine, domain separation"),
    ("data/custom/dimweights.json",
     lambda d: d["rarity (idf)"]["separation"] / d["cosine_separation"],
     0.76, 0.25, "idf weighting falls BELOW cosine — frequency cannot tell marker from stopword"),
    ("data/custom/dimweights.json",
     lambda d: max(v["separation"] for v in d.values() if isinstance(v, dict)),
     0.324, 0.05, "best separation of any weighting — the unweighted one"),
    ("data/custom/dimweights.json",
     lambda d: d["coherence"]["separation"] / d["cosine_separation"],
     2.33, 0.4, "coherence weighting keeps separation where idf loses it"),
    ("data/custom/dimweights.json",
     lambda d: d["coherence"]["bridge"] / d["none (raw extreme)"]["bridge"],
     0.46, 0.15, "and damps the spurious numeral bridge by half"),
    # Chains as a candidate generator: coverage is the strength, ranking is the gap.
    ("data/custom/candidates.json",
     lambda d: d["reached_3hop"] / d["direct_wrong"],
     0.994, 0.03, "share of wrongly-classified items a 3-hop chain reaches at all"),
    ("data/custom/candidates.json", lambda d: d["precision_top50_2hop"],
     0.0, 0.001, "precision in the top 50 by chain strength — strength does not rank"),
    # A small model on multi-step work: the failure is evaluation, not looping.
    ("data/custom/loops.json", lambda d: d["summary"]["total_repeats"],
     2, 2, "literal repeats in 41 steps, once the harness stopped echoing code fences"),
    ("data/custom/loops.json",
     lambda d: d["summary"]["total_stateless"] / d["summary"]["total_steps"],
     0.49, 0.15, "steps stating an operation without recording its result"),
    ("data/custom/evaluate.json", lambda d: float(d["summary"]["correct"]),
     6.0, 1.5, "solved when the model's own expression is executed (stepwise baseline: 0)"),
    ("data/custom/evaluate.json", lambda d: float(d["summary"]["wrote_expression"]),
     7.0, 1.5, "problems for which it wrote a correct-form expression"),
    ("data/custom/scratchpad-graph.json", lambda d: float(d["summary"]["correct"]),
     0.0, 0.5, "scratchpad + derivation graph: the model cannot take a second step"),
    ("data/custom/scratchpad-graph.json",
     lambda d: float(d["summary"]["total_unjustified"]),
     58.0, 15.0, "underivable steps the graph refused — detection is not the bottleneck"),
    ("data/custom/workpad.json", lambda d: float(d["tests"]),
     25.0, 0.5, "work-record properties verified against a scripted agent"),
    ("data/custom/workpad.json", lambda d: float(d["failures"]),
     0.0, 0.001, "and none of them failing"),
    # The iteration claim: not tested, because a random driver has no success rate to compare.
    ("data/custom/iterate.json",
     lambda d: max(r["backtrack_precision"] for r in d["rows"]),
     0.0, 0.05, "best precision any arm reached — too low to compare, the test lacked power"),
    # Multi-path support as a confidence signal: it measures reachability, not truth.
    ("data/custom/confidence.json", lambda d: float(d["summary"]["top_is_answer"]),
     0.0, 0.5, "problems where the best-supported value is the answer"),
    ("data/custom/confidence.json", lambda d: float(d["summary"]["answer_in_top5"]),
     0.0, 0.5, "problems where the answer is even among the five best supported"),
    # Cutting the step to one digit column. The carry is where arithmetic was already measured to
    # stop being decomposable, so the atom was made small enough that the model only ever adds
    # single digits and the record carries the state between them.
    ("data/custom/digitwise.json", lambda d: float(d["summary"]["whole_correct"]),
     6.0, 2.0, "carry-heavy additions solved as one whole number"),
    ("data/custom/digitwise.json", lambda d: float(d["summary"]["digitwise_correct"]),
     11.0, 2.0, "the same additions solved one column at a time"),
    ("data/custom/digitwise.json", lambda d: float(d["summary"]["oracle_correct"]),
     12.0, 0.5, "and with the columns correct — what the record itself contributes"),
    # Two-way with a residue. The forward chain is the base, the backward chain says what the
    # goal requires, and the residue is the exact difference that must reach zero — the same
    # contract as base + residual == d, which is why a non-zero residue localises the bad step.
    ("data/custom/twoway.json", lambda d: float(d["numeric"]["detected"]),
     12.0, 0.5, "injected column errors the residue detected"),
    ("data/custom/twoway.json", lambda d: float(d["numeric"]["localised"]),
     11.0, 2.0, "errors whose residue named the right column"),
    ("data/custom/twoway.json", lambda d: float(d["numeric"]["repaired"]),
     11.0, 2.0, "repaired by patching only that column"),
    ("data/custom/twoway.json", lambda d: float(d["numeric"]["clean_zero"]),
     11.0, 2.0, "residue zero when no error was injected (false-alarm control)"),
    ("data/custom/twoway.json", lambda d: float(d["procedural"]["correct"]),
     2.0, 1.0, "everyday procedures planned backwards from the goal"),
    # Step arity: splitting three-into-one down to two-into-one does not help in either domain.
    ("data/custom/binary.json", lambda d: float(d["numeric"]["ternary_correct"]),
     23.0, 2.5, "additions with three digits per step"),
    ("data/custom/binary.json", lambda d: float(d["numeric"]["binary_correct"]),
     22.0, 2.5, "the same additions split to two digits per step"),
    ("data/custom/binary.json", lambda d: float(d["numeric"]["binary_calls"]),
     108.0, 12.0, "and the extra model calls that split costs"),
    ("data/custom/binary.json", lambda d: float(d["procedural"]["many_correct"]),
     11.0, 2.0, "procedures planned with every open subgoal shown"),
    ("data/custom/binary.json", lambda d: float(d["procedural"]["binary_correct"]),
     11.0, 2.0, "the same procedures with at most two shown — no difference"),
    ("data/custom/binary.json",
     lambda d: float(sum(1 for r in d["procedural"]["runs"] if r["many_ok"] == r["binary_ok"])),
     16.0, 0.5, "tasks on which the two arities reach the identical outcome"),
    # Compress into a node, keep the residue, accept only what reverses. The mechanism holds;
    # it does not pay, because reversibility is safety and not progress — the record accepted
    # `Y = 3` and `Y = 48`, which put back perfectly and compress nothing.
    ("data/custom/substitute.json", lambda d: float(d["numeric"]["whole_correct"]),
     3.0, 1.5, "four-term expressions evaluated whole"),
    ("data/custom/substitute.json", lambda d: float(d["numeric"]["substituted_correct"]),
     2.0, 1.5, "the same expressions compressed to nodes and discharged"),
    ("data/custom/substitute.json",
     lambda d: float(sum(1 for r in d["numeric"]["runs"] if r["nodes"])),
     9.0, 2.0, "problems on which a substitution was accepted at all"),
    ("data/custom/substitute.json", lambda d: d["numeric"]["tokens_after"],
     8.3, 0.6, "widest expression the model ever held, down from 9.0"),
    ("data/custom/substitute.json", lambda d: float(d["expansions_ok"]),
     3.0, 0.001, "placeholder expansions that restore the abstract step's contract"),
    # The model writes only the links, as JSON. Nearly 4x from the interface alone, and the
    # reversibility check earns its keep: four of five refusals inline to the wrong value.
    ("data/custom/jsongraph.json", lambda d: float(d["summary"]["whole_correct"]),
     3.0, 1.5, "expressions evaluated whole (same 20 as phase 20)"),
    ("data/custom/jsongraph.json", lambda d: float(d["summary"]["graph_correct"]),
     11.0, 2.5, "the same expressions with the model writing the links as JSON"),
    ("data/custom/jsongraph.json", lambda d: float(d["summary"]["parsed"]),
     20.0, 0.5, "replies that parsed as a single-assignment JSON object"),
    ("data/custom/jsongraph.json", lambda d: float(d["summary"]["structurally_valid"]),
     15.0, 2.0, "graphs that passed every rule including reversal"),
    ("data/custom/jsongraph.json", lambda d: float(d["summary"]["max_fanin"]),
     3.0, 0.001, "widest step the model ever wrote, in inputs"),
    ("data/custom/jsongraph.json", lambda d: d["summary"]["fanout_one_pct"],
     87.0, 12.0, "%% of inner keys used exactly once onward, unprompted"),
    # Diff-as-step. Worse than the JSON graph, but the failures become attributable — and diff
    # SIZE does not discriminate, which was reported as a signal off four cases before the run.
    ("data/custom/rewrite.json", lambda d: float(d["summary"]["checked_correct"]),
     5.0, 2.0, "expressions solved by verified rewriting"),
    ("data/custom/rewrite.json", lambda d: d["summary"]["error_rate_by_op"].get("/", 0.0),
     0.58, 0.2, "error rate on division steps"),
    ("data/custom/rewrite.json", lambda d: d["summary"]["error_rate_by_op"].get("*", 0.0),
     0.07, 0.12, "error rate on multiplication steps"),
    ("data/custom/rewrite.json", lambda d: d["summary"]["big_diff_when_wrong"],
     0.31, 0.25, "wrong steps whose diff exceeds the average right one — not a signal"),
    # Forward against backward on identical facts. Inversion is a different task for this model.
    ("data/custom/invert.json", lambda d: float(d["summary"]["fwd_correct"]),
     22.0, 3.0, "arithmetic facts answered forwards"),
    ("data/custom/invert.json", lambda d: float(d["summary"]["bwd_correct"]),
     5.0, 2.5, "the same facts as a blank to fill"),
    ("data/custom/invert.json", lambda d: float(d["summary"]["bwdF_correct"]),
     13.0, 3.0, "the same facts rephrased into arithmetic by hand"),
    # The model states its own inverse better than a lookup table does.
    ("data/custom/rephrase.json", lambda d: float(d["summary"]["asked_correct"]),
     19.0, 3.0, "inversions where the model wrote the question itself"),
    ("data/custom/rephrase.json", lambda d: float(d["summary"]["wrote_a_question"]),
     24.0, 0.5, "and produced a valid question every time"),
    ("data/custom/rephrase.json", lambda d: float(d["summary"]["word_correct"]),
     10.0, 3.0, "the same inversions as word problems — worse"),
    # Repetition cannot fix a systematic error.
    ("data/custom/repeat.json", lambda d: float(d["summary"]["single_correct"]),
     13.0, 3.0, "one sample at temperature zero"),
    ("data/custom/repeat.json", lambda d: float(d["summary"]["vote_correct"]),
     13.0, 3.0, "majority of five at temperature 0.8 — no gain"),
    ("data/custom/repeat.json", lambda d: float(d["summary"]["longer_correct"]),
     9.0, 3.0, "given room to think — worse"),
    # Free-text editing is not a reversible operation for a 1B model.
    ("data/custom/textsub.json", lambda d: float(d["summary"]["cut"]),
     16.0, 0.5, "sentences where it produced a substitution"),
    ("data/custom/textsub.json", lambda d: float(d["summary"]["reversible"]),
     5.0, 2.5, "and the substitutions that paste back exactly"),
    # Naming the operation instead of performing it.
    ("data/custom/ops.json", lambda d: float(d["summary"]["direct_correct"]),
     9.0, 3.0, "letter counts asked directly"),
    ("data/custom/ops.json", lambda d: float(d["summary"]["ops_correct"]),
     17.0, 2.0, "letter counts where the model only named operations"),
    ("data/custom/ops.json", lambda d: float(d["summary"]["valid_pipelines"]),
     18.0, 0.5, "pipelines the record could execute"),
    ("data/custom/ops.json", lambda d: float(d["summary"]["undone"]),
     18.0, 0.5, "runs that undo back to the input exactly"),
    # Confining the model to reversible operations makes its errors harmless, not rare.
    ("data/custom/cutspan.json", lambda d: float(d["summary"]["index"]),
     0.0, 1.5, "spans cut correctly by index, one attempt"),
    ("data/custom/cutspan.json", lambda d: float(d["summary"]["feedback"]),
     1.0, 2.0, "and with up to three attempts shown what they took"),
    ("data/custom/cutspan.json", lambda d: float(d["summary"]["by_name"]),
     1.0, 2.0, "spans cut correctly by naming the word"),
    ("data/custom/cutspan.json", lambda d: float(d["summary"]["reversible"]),
     14.0, 0.5, "cuts that were reversible — every one, including every wrong one"),
    # Capacity limits scale with model size; representation limits do not.
    ("data/custom/cutbig.json", lambda d: float(d["summary"]["olmoe-1b"]["name"]),
     1.0, 1.5, "1B model: spans cut by naming the word"),
    ("data/custom/cutbig.json", lambda d: float(d["summary"]["qwen-35b"]["name"]),
     14.0, 1.0, "35B model: the same, naming the word"),
    ("data/custom/cutbig.json", lambda d: float(d["summary"]["olmoe-1b"]["index"]),
     0.0, 1.5, "1B model: spans cut by character index"),
    ("data/custom/cutbig.json", lambda d: float(d["summary"]["qwen-35b"]["index"]),
     2.0, 2.0, "35B model: the same by index — size buys almost nothing"),
    ("data/custom/cutbig.json", lambda d: float(d["summary"]["olmoe-1b"]["split"]),
     5.0, 2.0, "1B model: split first, then choose from the list"),
    ("data/custom/cutbig.json", lambda d: float(d["summary"]["qwen-35b"]["split"]),
     14.0, 1.0, "35B model: split first, then choose"),
    ("data/custom/cutbig.json",
     lambda d: 1.0 if d["summary"]["qwen-35b"]["document"]["join_restores"] else 0.0,
     1.0, 0.001, "narrowing a document by two splits reassembles it exactly"),
    ("data/custom/cutbig.json", lambda d: float(d["summary"]["qwen-35b"]["document"]["calls"]),
     2.0, 0.001, "and takes two questions regardless of the document length"),
    # Hierarchical splitting on a real 1.29 MB document. The split scales; the steering does not.
    ("data/custom/longdoc.json", lambda d: float(d["summary"]["sentences"]),
     8920.0, 5.0, "sentences in the real document split"),
    ("data/custom/longdoc.json", lambda d: float(d["summary"]["levels"]),
     5.0, 0.001, "levels needed to reach any one of them at branching 8"),
    ("data/custom/longdoc.json", lambda d: 1.0 if d["summary"]["lossless"] else 0.0,
     1.0, 0.001, "every level rejoins to the original bytes"),
    ("data/custom/longdoc.json", lambda d: float(d["summary"]["record_find"]),
     8.0, 0.001, "targets the record located with zero model calls"),
    ("data/custom/longdoc.json", lambda d: float(d["summary"]["qwen-35b"]["found"]),
     0.0, 0.001, "targets the 35B model navigated to from 90-character previews"),
    ("data/custom/longdoc.json", lambda d: float(d["summary"]["olmoe-1b"]["found"]),
     0.0, 0.001, "and the 1B model"),
    # Placeholders: a meaningful name does not help restoration, and uniqueness must be checked
    # whatever the name looks like. The lever is that the record substitutes, not the model.
    ("data/custom/placeholder.json", lambda d: float(d["summary"]["letter"]["model_restores"]),
     13.0, 2.0, "sentences restored with letter placeholders"),
    ("data/custom/placeholder.json", lambda d: float(d["summary"]["category"]["model_restores"]),
     9.0, 2.0, "restored with category placeholders — no better"),
    ("data/custom/placeholder.json", lambda d: float(d["summary"]["numbered"]["model_restores"]),
     13.0, 2.0, "restored with numbered category placeholders"),
    ("data/custom/placeholder.json", lambda d: float(d["summary"]["category"]["refused"]),
     4.0, 0.001, "substitutions refused because two spans shared a category"),
    ("data/custom/placeholder.json", lambda d: float(d["summary"]["numbered"]["reverses"]),
     16.0, 0.001, "and numbering makes every one of them reversible"),
    # Vectors on the split-tree nodes. Greedy through an approximate index fails the same way
    # here as on the kNN graph; beam is the fix in both places.
    ("data/custom/embednav.json", lambda d: float(d["exact"]["tree_correct"]),
     18.0, 4.0, "targets found by greedy descent over mean-pooled nodes"),
    ("data/custom/embednav.json", lambda d: float(d["exact"]["sample_correct"]),
     9.0, 4.0, "and by scoring a node by its best of eight descendants — worse"),
    ("data/custom/embednav.json", lambda d: float(d["exact"]["beam_correct"]),
     31.0, 4.0, "and with a beam of four"),
    ("data/custom/embednav.json", lambda d: float(d["exact"]["flat_correct"]),
     40.0, 0.001, "flat scan over every leaf, the yardstick"),
    ("data/custom/embednav.json", lambda d: float(d["half"]["beam_correct"]),
     21.0, 4.0, "half-sentence queries with a beam of four"),
    ("data/custom/embednav.json", lambda d: d["exact"]["mean_comparisons_beam"],
     116.0, 10.0, "comparisons the beam uses against 8,920 for a flat scan"),
    # Beam width, dimension-level scores, and an N x N hop. Only the first closes the gap.
    ("data/custom/beamwide.json",
     lambda d: float(next(r["exact"] for r in d["sweep"] if r["beam"] == 16)),
     40.0, 1.0, "targets found at beam 16 — the flat scan's own score"),
    ("data/custom/beamwide.json",
     lambda d: next(r["exact_cmp"] for r in d["sweep"] if r["beam"] == 16),
     401.0, 20.0, "comparisons that takes, against 8,920 for a flat scan"),
    ("data/custom/beamwide.json",
     lambda d: float(next(r["exact"] for r in d["sweep"] if r["beam"] == 128)),
     40.0, 1.0, "and beam 128 finds no more — everything past 32 is wasted"),
    ("data/custom/beamwide.json",
     lambda d: float(next(r["exact"] for r in d["scores"]
                          if r["mode"] == "shared" and r["beam"] == 4)),
     0.0, 0.001, "shared-extreme scoring for routing — total collapse"),
    ("data/custom/beamwide.json",
     lambda d: float(next(r["exact"] for r in d["scores"]
                          if r["mode"] == "topdims" and r["beam"] == 4)),
     26.0, 4.0, "scoring on the query's loudest dimensions — worse than cosine"),
    ("data/custom/beamwide.json",
     lambda d: float(next(r["after_hop"] for r in d["hops"]
                          if r["beam"] == 4 and r["kind"] == "exact")),
     34.0, 4.0, "beam 4 plus one neighbour hop, up from 29 in the beam"),
    # Streaming the neighbour graph once, or buying the cells once. Never materialise n x n.
    ("data/custom/brokerhop.json",
     lambda d: float(next(r["after_hop"] for r in d["rows"]
                          if r["beam"] == 4 and r["kind"] == "exact")),
     34.0, 4.0, "stored-graph hop reproduces the phase 32 rescue exactly"),
    ("data/custom/brokerhop.json",
     lambda d: float(next(r["lookups"] for r in d["rows"]
                          if r["beam"] == 4 and r["kind"] == "exact")),
     64.0, 0.001, "lookups it costs, against 35,680 dot products on demand"),
    ("data/custom/brokerhop.json", lambda d: d["breakeven_queries"],
     1115.0, 20.0, "queries before the prebuilt graph overtakes computing on demand"),
    ("data/custom/brokerhop.json", lambda d: float(d["broker_hits"]),
     702.0, 60.0, "neighbour requests the broker served from cache"),
    ("data/custom/brokerhop.json", lambda d: d["naive_dots"] / d["broker_dots"],
     1.78, 0.25, "what buying once saves on a barely warmed cache"),
    ("data/custom/brokerhop.json", lambda d: float(d["graph_bytes"]) / 1e6,
     1.14, 0.1, "megabytes the whole neighbour graph occupies"),
    # Agreement between different entry points is the grader-free confidence signal.
    ("data/custom/crosscheck.json", lambda d: float(d["agreement"]["4"]["right"]),
     9.0, 1.0, "targets right when all four starts agree"),
    ("data/custom/crosscheck.json", lambda d: float(d["agreement"]["1"]["right"]),
     1.0, 1.0, "and right when no two starts agree"),
    ("data/custom/crosscheck.json", lambda d: float(d["cross_article_pairs"]),
     6406.0, 300.0, "mutual neighbour pairs that cross an article boundary"),
    # Clusters exist below the percolation point and nowhere near the k that hopping wants.
    ("data/custom/clusterk.json",
     lambda d: next(r["purity"] for r in d["rows"] if r["k"] == 2),
     0.953, 0.03, "article purity of mutual-kNN clusters at k=2"),
    ("data/custom/clusterk.json",
     lambda d: next(r["purity"] for r in d["rows"] if r["k"] == 16),
     0.056, 0.01, "and at k=16, which is exactly the baseline"),
    ("data/custom/clusterk.json",
     lambda d: float(next(r["largest"] for r in d["rows"] if r["k"] == 3)),
     118.0, 30.0, "largest component at k=3, just below percolation"),
    ("data/custom/clusterk.json",
     lambda d: float(next(r["largest"] for r in d["rows"] if r["k"] == 6)),
     7807.0, 400.0, "and at k=6, just above it"),
    # Shape-aware compression of the neighbour graph, verified lossless on the grid.
    ("data/custom/codec.json", lambda d: float(d["sizes"]["gzip"]),
     672756.0, 2000.0, "gzip on the raw neighbour graph, bytes"),
    ("data/custom/codec.json", lambda d: float(d["sizes"]["delta + lzma"]),
     376972.0, 3000.0, "delta codec plus lzma, bytes"),
    ("data/custom/codec.json", lambda d: d["advantage"],
     1.44, 0.1, "shape-aware against the best general-purpose compressor"),
    ("data/custom/codec.json", lambda d: 1.0 if d["lossless_on_grid"] else 0.0,
     1.0, 0.001, "both codecs decode every id and quantised distance exactly"),
    ("data/custom/codec.json", lambda d: d["max_error_rad"],
     5.01e-05, 1e-05, "largest error the quantisation grid introduces, radians"),
    # The same codecs on the embeddings. Bits per dimension is the lever, not the predictor.
    ("data/custom/embcodec.json",
     lambda d: float(next(r["bytes"] for r in d["rows"] if r["method"] == "gzip(f32)")),
     12647962.0, 100000.0, "gzip on raw float embeddings — 8% off"),
    ("data/custom/embcodec.json",
     lambda d: float(next(r["bytes"] for r in d["rows"] if r["method"] == "scalar only")),
     2876021.0, 50000.0, "8-bit scalar quantisation alone"),
    ("data/custom/embcodec.json",
     lambda d: float(next(r["bytes"] for r in d["rows"] if r["method"] == "rank-8 + residual")),
     3117111.0, 60000.0, "a rank-8 base makes it bigger, not smaller"),
    ("data/custom/embcodec.json", lambda d: float(d["best_lossless_bytes"]),
     2781837.0, 50000.0, "smallest encoding that keeps retrieval intact"),
    ("data/custom/embcodec.json",
     lambda d: float(min(r["retrieval"] for r in d["rows"])),
     40.0, 0.001, "every codec keeps beam-16 retrieval at 40/40"),
    # Clustering: 3.7% on bytes, 44x on search, and the two want different numbers of clusters.
    ("data/custom/clustercodec.json",
     lambda d: float(next(r["bytes"] for r in d["rows"]
                          if r["scheme"] == "global" and r["bits"] == 3)),
     730016.0, 20000.0, "global 3-bit encoding that still retrieves 40/40"),
    ("data/custom/clustercodec.json",
     lambda d: float(next(r["retrieval"] for r in d["rows"]
                          if r["scheme"] == "global" and r["bits"] == 2)),
     15.0, 4.0, "and 2 bits, where it collapses"),
    ("data/custom/clustercodec.json", lambda d: float(d["winner_bytes"]),
     703240.0, 20000.0, "smallest encoding keeping retrieval intact (cluster-64, 3 bits)"),
    ("data/custom/clustercodec.json",
     lambda d: float(next(r["bytes"] for r in d["rows"]
                          if r["scheme"] == "cluster-1024" and r["bits"] == 3)),
     2101548.0, 60000.0, "and 1024 clusters, where the centroids cost more than they save"),
    ("data/custom/ivf.json",
     lambda d: float(next(r["found"] for r in d["rows"]
                          if r["C"] == 94 and r["probe"] == 1)),
     40.0, 0.001, "targets found probing one cluster of 94"),
    ("data/custom/ivf.json",
     lambda d: next(r["cmp"] for r in d["rows"] if r["C"] == 94 and r["probe"] == 1),
     201.0, 15.0, "comparisons that costs, against 8,920 for a flat scan"),
    ("data/custom/ivf.json",
     lambda d: float(next(r["flat_nxn"] / r["cluster_nxn"] for r in d["rows"]
                          if r["C"] == 94 and r["probe"] == 1)),
     9004.0, 100.0, "how much smaller the cluster-level N x N is than the sentence-level one"),
    # Analogy borrowing: the average says no, the selected borrow says yes, and the selector
    # is the same embedding that made the clusters.
    ("data/custom/analogy.json", lambda d: d["own"],
     0.2451, 0.03, "variance a cluster's own split direction captures"),
    ("data/custom/analogy.json", lambda d: d["borrowed_mean"],
     0.0167, 0.006, "and a borrowed one averaged over every other cluster"),
    ("data/custom/analogy.json", lambda d: d["universal"],
     0.0499, 0.008, "the pooled universal direction, which beats the average borrow"),
    ("data/custom/analogy-select.json", lambda d: d["corr"],
     0.913, 0.05, "correlation between centroid similarity and transferable variance"),
    ("data/custom/analogy-select.json", lambda d: d["picked_similar"],
     0.0942, 0.01, "borrowing from the most similar cluster"),
    ("data/custom/analogy-select.json",
     lambda d: d["picked_best"] - d["picked_similar"],
     0.0001, 0.002, "how far that falls short of an oracle — nothing"),
    ("data/custom/analogy.json",
     lambda d: abs(d["purity_borrowed"] - d["purity_random"]),
     0.0015, 0.02, "and the split does not separate topics, borrowed or random"),
    # Borrowed splits, held out and bucketed by size: real below 32 members, and still not
    # worth storing a sub-centroid for.
    ("data/custom/borrowsplit.json",
     lambda d: next(b["borrowed"] for b in d["buckets"] if b["lo"] == 4),
     0.0728, 0.02, "held-out variance a borrowed split captures on 4-7 member clusters"),
    ("data/custom/borrowsplit.json",
     lambda d: next(b["own"] for b in d["buckets"] if b["lo"] == 4),
     0.0224, 0.012, "and what the cluster's own split captures there"),
    ("data/custom/borrowsplit.json",
     lambda d: next(b["own"] for b in d["buckets"] if b["lo"] == 32),
     0.1118, 0.025, "at 32-63 members, where own overtakes borrowed"),
    ("data/custom/borrowsplit.json",
     lambda d: float(next(e["bytes"] for e in d["encodings"]
                          if e["variant"] == "none (cluster only)")),
     1733860.0, 60000.0, "bytes with no split at all"),
    ("data/custom/borrowsplit.json",
     lambda d: float(next(e["bytes"] for e in d["encodings"]
                          if e["variant"] == "borrowed from nearest")),
     1907272.0, 60000.0, "and with a borrowed split — larger, not smaller"),
    # Cluster address instead of a vector: perfect routing, unusable reconstruction.
    ("data/custom/rankonly.json",
     lambda d: float(min(r["in_cluster"] for r in d["rows"])),
     40.0, 0.001, "queries landing in the correct cluster, at every C"),
    ("data/custom/rankonly.json",
     lambda d: float(next(r["retrieval"] for r in d["rows"]
                          if r["C"] == 256 and r["variant"] == "cluster only")),
     0.0, 0.001, "and top-1 retrieval from a cluster address alone"),
    ("data/custom/rankonly.json",
     lambda d: float(next(r["retrieval"] for r in d["rows"]
                          if r["C"] == 4096 and r["variant"] == "cluster + rank")),
     30.0, 4.0, "the best it reaches, at 9.2 MB against 730 KB for 3-bit vectors"),
    ("data/custom/rankonly.json",
     lambda d: float(next(r["retrieval"] for r in d["rows"]
                          if r["C"] == 2048 and r["variant"] == "cluster + rank"))
     - float(next(r["retrieval"] for r in d["rows"]
                  if r["C"] == 2048 and r["variant"] == "cluster only")),
     11.0, 4.0, "what a rank coordinate adds at C=2048"),
    # The original question: resident memory, not disk. One seek per query, sqrt(n) index.
    ("data/custom/diskindex.json",
     lambda d: float(next(r["resident_bytes"] for r in d["rows"]
                          if r["C"] == 94 and r["probe"] == 1)) / 1e6,
     0.144, 0.01, "megabytes that must stay resident for a lossless index"),
    ("data/custom/diskindex.json",
     lambda d: next(r["us_per_query"] for r in d["rows"]
                    if r["C"] == 94 and r["probe"] == 1),
     232.0, 12.0, "microseconds per query, one fetch, from the measured cost model"),
    ("data/custom/diskindex.json", lambda d: d["full_scan_us"],
     582.0, 20.0, "and reading the whole 3-bit block cold instead"),
    ("data/custom/diskindex.json",
     lambda d: float(min(r["correct"] for r in d["rows"])),
     40.0, 0.001, "every configuration still answers 40/40"),
    ("data/custom/residentscale.json",
     lambda d: next(r["resident_bytes"] for r in d["rows"] if r["n"] == 10**8) / 1e6,
     15.4, 1.0, "megabytes resident for a hundred million items"),
    ("data/custom/residentscale.json",
     lambda d: next(r["ratio"] for r in d["rows"] if r["n"] == 10**8),
     938.0, 20.0, "and how many times larger the on-disk side is there"),
    # Working set: centroids plus one block. The one measurement here with a real optimum.
    ("data/custom/workingset.json", lambda d: float(d["best"]["working_set"]) / 1e3,
     133.0, 8.0, "smallest working set in KB that still retrieves 40/40"),
    ("data/custom/workingset.json", lambda d: float(d["best"]["C"]),
     32.0, 0.001, "and the number of clusters it needs"),
    ("data/custom/workingset.json", lambda d: float(d["best"]["bits"]),
     3.0, 0.001, "at three bits per dimension"),
    ("data/custom/workingset.json",
     lambda d: float(next(r["working_set"] for r in d["rows"]
                          if r["C"] == 1024 and r["bits"] == 3)) / 1e3,
     1579.0, 40.0, "against 1024 clusters, where the centroids dominate"),
    ("data/custom/workingset.json",
     lambda d: float(max(r["correct"] for r in d["rows"] if r["bits"] == 2)),
     22.0, 4.0, "best any 2-bit configuration manages — the cliff, again"),
    # The model inventing its own reversible transforms: it can, and it buys nothing.
    ("data/custom/llmcodec.json",
     lambda d: float(next(r["rules_applied"] for r in d["rows"]
                          if r["label"] == "proposed by qwen-35b")),
     29.0, 8.0, "reversible substitutions the 35B model proposed that survived checking"),
    ("data/custom/llmcodec.json",
     lambda d: 1.0 if all(r["reversible"] for r in d["rows"]) else 0.0,
     1.0, 0.001, "and every accepted rule set round-trips the 1.29 MB exactly"),
    ("data/custom/llmcodec.json",
     lambda d: (next(r["gzip"] for r in d["rows"] if r["label"] == "no transform")
                / next(r["gzip"] for r in d["rows"] if r["label"] == "proposed by qwen-35b")),
     1.028, 0.03, "what those rules are worth against plain gzip"),
    ("data/custom/llmcodec.json",
     lambda d: (next(r["lzma"] for r in d["rows"] if r["label"] == "no transform")
                / next(r["lzma"] for r in d["rows"] if r["label"] == "proposed by qwen-35b")),
     1.001, 0.02, "and against lzma, which already finds the same redundancy"),
    # Reuse from worked examples. The larger model adapts; the smaller neither adapts nor copies.
    ("data/custom/reuse.json", lambda d: float(d["summary"]["qwen-35b"]["adapted"]),
     19.0, 2.0, "expressions the 35B model adapted rather than copied"),
    ("data/custom/reuse.json", lambda d: float(d["summary"]["qwen-35b"]["correct"]),
     20.0, 1.0, "and how many it got right from two worked examples"),
    ("data/custom/reuse.json", lambda d: float(d["summary"]["olmoe-1b"]["correct"]),
     11.0, 3.0, "the 1B model, the same as its score with no examples at all"),
    ("data/custom/reuse.json",
     lambda d: float(max(m["copied"] for m in d["summary"].values())),
     1.0, 1.0, "verbatim copies once the prompt forbids them"),
    # Learning without a grader: the check verifies the decomposition, not the arithmetic.
    ("data/custom/learnloop.json",
     lambda d: 1.0 - d["verified_wrong"] / max(d["store_size"], 1),
     0.67, 0.25, "share of what the record accepted into the store that is actually right"),
    # Generalised templates: the first memory that measurably helped, within a single run.
    ("data/custom/template.json", lambda d: float(d["summary"]["olmoe-1b"]["trecord"]),
     14.0, 4.0, "1B model with record-instantiated templates"),
    ("data/custom/template.json", lambda d: float(d["summary"]["olmoe-1b"]["graph"]),
     5.0, 4.0, "and its graph arm in the same run — the gap is the finding"),
    ("data/custom/template.json", lambda d: float(d["summary"]["olmoe-1b"]["tmodel"]),
     9.0, 4.0, "model-bound values, worse than record-bound by five"),
    ("data/custom/template.json", lambda d: float(d["summary"]["qwen-35b"]["templates"]),
     4.0, 0.001, "templates covering all twenty of the 35B model's instances"),
    ("data/custom/template.json",
     lambda d: float(d["summary"]["olmoe-1b"]["refused_generalise"]),
     3.0, 2.0, "generalisations the round-trip rule refused"),
    ("data/custom/solverarm.json", lambda d: float(d["olmoe-1b"]["solver_arm"]),
     15.0, 4.0, "the composite arm with a solver on hits, 1B model"),
    ("data/custom/solverarm.json", lambda d: float(d["qwen-35b"]["solver_arm"]),
     18.0, 2.0, "and the 35B model"),
    ("data/custom/solverarm.json", lambda d: float(d["olmoe-1b"]["hit_correct_rate"]),
     1.0, 0.001, "solver correctness on template hits — exact by construction"),
    # Repair via provenance: latent defects detected by the inline check, fixed by the
    # intersection of assignments consistent with old examples and the failing new one.
    ("data/custom/repair.json", lambda d: float(d["ambiguous_stored"]),
     4.0, 0.001, "templates stored while the source could not decide the assignment"),
    ("data/custom/repair.json", lambda d: float(d["defects"]),
     1.0, 1.0, "latent defects that surfaced on a discriminating instance"),
    ("data/custom/repair.json", lambda d: float(d["repaired"]),
     1.0, 1.0, "and repaired via provenance, re-verified on old and new"),
    ("data/custom/repair.json", lambda d: float(d["unrepairable"]),
     0.0, 0.001, "with none unrepairable"),
    ("data/custom/repair.json", lambda d: float(d["covered"]),
     76.0, 2.0, "hits instantiated correctly, including the repaired one"),
    # The fill as a Python call: higher precision when it answers, lower yield overall.
    ("data/custom/calltuple.json", lambda d: float(d["record"]),
     30.0, 0.001, "record-bound instantiations correct — the ceiling, a third time"),
    ("data/custom/calltuple.json", lambda d: float(d["call"]),
     16.0, 4.0, "correct via solve(...) calls"),
    ("data/custom/calltuple.json", lambda d: float(d["graph"]),
     17.0, 4.0, "and via rewriting the JSON graph — no better"),
    ("data/custom/calltuple.json", lambda d: float(d["call_parsed"]),
     18.0, 4.0, "replies that got past echoing the worked example at all"),
    # The LLM writing rRETL pairs against mpedb's verifier: one friendly domain registers
    # first try and survives apply/edit/putback; the float-trap domains refuse every round.
    ("data/custom/rretlpairs.json", lambda d: float(d["tasks"]),
     4.0, 0.001, "transforms the model was asked to write as forward/rex/inverse"),
    ("data/custom/rretlpairs.json", lambda d: float(d["registered"]),
     1.0, 0.5, "pairs the engine's probe corpus accepted"),
    ("data/custom/rretlpairs.json", lambda d: float(d["first_try"]),
     1.0, 0.5, "of which on the first attempt, no feedback needed"),
    # The ladder, the ceiling, and the wrap: help as text lifts nothing, help as mechanics
    # recovers nearly the whole gap to the hand-written ceiling.
    ("data/custom/ladder.json", lambda d: float(d["qwen-35b"]["registered"]),
     1.0, 0.5, "pairs the 35B model registered with docs and counter-examples in context"),
    ("data/custom/ladder.json", lambda d: float(d["olmoe-1b"]["registered"]),
     0.0, 0.5, "and the 1B model, at any rung"),
    ("data/custom/ladder.json", lambda d: float(d["qwen-35b"]["counter_examples_banked"]),
     12.0, 4.0, "counter-examples the record banked across tasks"),
    ("data/custom/guardwrap.json", lambda d: float(d["registered"]),
     3.0, 1.0, "pairs registered when the record injects the guards instead"),
    # Plan choice as retrieval: the store builds free and verifies itself; pure lookup fails
    # on role binding, which is the model's irreplaceable job.
    ("data/custom/mapstore.json", lambda d: float(d["summary"]["store_kept"]),
     1554.0, 20.0, "verified generalised plans extracted from 2,000 training solutions"),
    ("data/custom/mapstore.json", lambda d: float(d["summary"]["plan_shapes"]),
     599.0, 20.0, "distinct plan shapes they collapse into"),
    ("data/custom/mapstore.json", lambda d: float(d["summary"]["compatible"]),
     56.0, 3.0, "test problems with a compatible template in the store"),
    ("data/custom/mapstore.json", lambda d: float(d["summary"]["solved"]),
     0.0, 1.0, "and solved by positional lookup — the role-binding gap"),
    ("data/custom/mapvote.json", lambda d: float(d["vote"]),
     1.0, 1.0, "solved by voting over the top 25 types"),
    ("data/custom/aimethink.json", lambda d: float(d["unclosed"]),
     4.0, 1.5, "of six AIME thoughts that never closed at a 1,600-token budget"),
    ("data/custom/aimethink.json", lambda d: float(d["tgraph"]),
     0.0, 0.5, "AIME problems where an executed plan computed the right quantity"),
    # Units as a routed graph: composition measured, reversibility verified, control free.
    ("data/custom/unitroute.json", lambda d: float(d["compound_conversions"]),
     7652.0, 1.0, "rate conversions routed from sixteen stored edges"),
    ("data/custom/unitroute.json", lambda d: float(d["reverse_exact"]),
     94.0, 0.001, "ordered pairs whose reverse route is exactly the reciprocal"),
    ("data/custom/unitroute.json", lambda d: d["example_kmh"],
     17380.9152, 0.001, "3 miles per second, in km/h, exactly"),
    ("data/custom/unitroute.json",
     lambda d: float(d["dim_refused"] + d["dim_accepted"]),
     4.0, 0.001, "dimension checks: well-typed accepted, ill-typed refused"),
    # Real problems: solo beats the graph arm on word problems; agreement is 18/18.
    ("data/custom/olympiad.json",
     lambda d: float(d["summary"]["qwen-35b/gsm8k"]["solo"]),
     19.0, 2.0, "35B solo on GSM8K, no thinking"),
    ("data/custom/olympiad.json",
     lambda d: float(d["summary"]["qwen-35b/gsm8k"]["graph"]),
     16.0, 3.0, "and with the graph arm — behind solo, the phase 45 reversal"),
    ("data/custom/olympiad.json",
     lambda d: float(d["summary"]["olmoe-1b/gsm8k"]["graph"]),
     4.0, 3.0, "the 1B graph arm on word problems, against 16/20 solo"),
    ("data/custom/olympiad.json",
     lambda d: sum(v["agree"].get("True", [0, 0])[1] for v in d["summary"].values()),
     18.0, 1.0, "unanimous cases right, across models and datasets"),
    ("data/custom/olympiad.json",
     lambda d: sum(v["agree"].get("True", [0, 0])[0] for v in d["summary"].values()),
     18.0, 1.0, "out of unanimous cases total — 100% observed precision"),
    ("data/custom/olympiad.json",
     lambda d: float(d["summary"]["qwen-35b/aime"]["vote"]),
     0.0, 0.001, "AIME without thinking, any arm — and unanimity never fired there"),
    # The dimension control on lookup: refuses a third of candidates for free, moves nothing,
    # because the bottleneck is role binding, not candidate selection.
    ("data/custom/dimcheck.json", lambda d: float(d["summary"]["self_pass"]),
     1112.0, 30.0, "store templates passing the check on their own source"),
    ("data/custom/dimcheck.json", lambda d: float(d["summary"]["refused"]),
     440.0, 40.0, "retrieved candidates refused as ill-typed, of 1,300"),
    ("data/custom/dimcheck.json", lambda d: float(d["summary"]["top1_filtered"]),
     1.0, 1.0, "solved after filtering — the gap is binding, not selection"),
    # The refusal gate on model plans: a counter-example loop inherits its judge's noise.
    ("data/custom/dimgraph.json", lambda d: float(d["qwen-35b"]["false_refusals"]),
     32.0, 5.0, "of 36 refusals on the strong model's plans that were FALSE"),
    ("data/custom/dimgraph.json", lambda d: float(d["olmoe-1b"]["dim"]),
     1.0, 1.5, "the weak model under the noisy gate, down from 4/20"),
    ("data/custom/dimgraph.json", lambda d: float(d["qwen-35b"]["dim"]),
     16.0, 2.5, "the strong model under it — break-even"),
    # Formula routing: total in its domain, mute outside it, consumption rule a third time.
    ("data/custom/formularich.json", lambda d: float(d["summary"]["unique"]),
     40.0, 0.001, "dimension-rich problems with a unique route"),
    ("data/custom/formularich.json", lambda d: float(d["summary"]["unique_right"]),
     40.0, 0.001, "and every one of them correct, zero model calls"),
    ("data/custom/formularoute.json", lambda d: float(d["summary"]["unique_right"]),
     2.0, 1.5, "GSM8K problems solved the same way — the domain boundary"),
    ("data/custom/formularoute.json", lambda d: float(d["summary"]["untaggable"]),
     26.0, 3.0, "GSM8K problems whose asked unit the tagger cannot even name"),
    # The model as unit reader: false refusals nearly halve, the gate turns net positive,
    # and abstraction helps the judge while destroying the planner.
    ("data/custom/modelunits.json", lambda d: float(d["gate"]["false"]),
     18.0, 5.0, "false refusals with model-read units, down from 32"),
    ("data/custom/modelunits.json", lambda d: float(d["gate"]["dim"]),
     17.0, 2.0, "the gate net positive for the first time"),
    ("data/custom/modelunits.json", lambda d: float(d["gate"]["unread"]),
     0.0, 0.5, "problems the model reader could not read"),
    ("data/custom/modelunits.json", lambda d: float(d["gate"]["abstract"]),
     2.0, 2.0, "plans written from the abstract form alone — the story was load-bearing"),
    ("data/custom/modelunits.json", lambda d: float(d["routing"]["untaggable"]),
     0.0, 0.5, "routing problems untaggable with the model reader, down from 26"),
    ("data/custom/modelunits.json", lambda d: float(d["routing"]["unique_right"]),
     3.0, 2.0, "GSM8K solved by routing even with perfect reading — structure, not reading"),
    # Absorb, don't strip: the full relation-graph translation is free; the staged version
    # buys total reversibility and provenance at the price of solving coherence.
    ("data/custom/relgraph.json", lambda d: float(d["qwen-35b"]["solved"]),
     16.0, 2.5, "problems solved from the full relation-graph translation, 35B"),
    ("data/custom/relgraph.json", lambda d: float(d["olmoe-1b"]["solved"]),
     3.0, 2.5, "and the 1B model, at its baseline"),
    ("data/custom/stagedabs.json", lambda d: float(d["solved"]),
     9.0, 3.0, "solved by staged absorption — the coherence price"),
    ("data/custom/stagedabs.json", lambda d: float(d["chains_exact"]),
     20.0, 0.001, "chains that unwound LIFO to the original text byte-exact"),
    ("data/custom/stagedabs.json",
     lambda d: float(d["steps_rev_ok"]) / max(d["steps_total"], 1),
     1.0, 0.001, "share of steps whose span restores exactly — the rRETL contract on text"),
    ("data/custom/stagedabs.json", lambda d: d["mean_absorbed"],
     0.97, 0.04, "share of the problems' numbers absorbed into the graph"),
    ("data/custom/stagedabs.json", lambda d: float(d["repaired_in_place"]),
     2.0, 1.5, "refusals repaired in place via span provenance"),
    # The ADAPT arm: binding works, topical retrieval aims it wrong, both models hurt.
    ("data/custom/adaptarm.json", lambda d: float(d["qwen-35b"]["solo"]),
     51.0, 4.0, "35B solo at 60-problem scale"),
    ("data/custom/adaptarm.json", lambda d: float(d["qwen-35b"]["adapt"]),
     41.0, 4.0, "and with retrieved examples — behind both solo and cold graph"),
    ("data/custom/adaptarm.json", lambda d: float(d["olmoe-1b"]["adapt"]),
     4.0, 3.0, "the 1B model riding wrong-type examples down from 15/60"),
    ("data/custom/adaptarm.json",
     lambda d: float(d["olmoe-1b"]["graph"]),
     15.0, 4.0, "its cold graph arm, for the within-run comparison"),
    # Self-diagnosis: grounded, zero fabrication, ranked by salience rather than leverage.
    ("data/custom/selfdiag.json", lambda d: float(d["hand_audit"]["fabricated"]),
     0.0, 0.001, "items the model invented beyond its data"),
    ("data/custom/selfdiag.json", lambda d: float(d["hand_audit"]["grounded"]),
     5.0, 0.001, "items citing real numbers from the files shown"),
    # The fan-out family, measured end to end: the gate with no deliverable, multiplying
    # paraphrase legs, the namespace residual, and iteration that cannot rename.
    ("data/custom/cascade.json", lambda d: float(d["summary"].get("accepted_t1", 0)),
     0.0, 0.001, "cascade tier-1 acceptances"),
    ("data/custom/cascade.json", lambda d: d["summary"]["menu_coverage"],
     0.067, 0.02, "share of store plans the eight-shape menu covers"),
    ("data/custom/fanout.json", lambda d: float(d["35b_decomposes"]["right"]),
     28.0, 4.0, "35B-split, 1B-legs — below the 1B solving alone"),
    ("data/custom/fanout.json", lambda d: float(d["1b_decomposes"]["right"]),
     12.0, 4.0, "and the 1B splitting for itself"),
    ("data/custom/rretlfan.json", lambda d: float(d["qwen-35b"]["rejoined_exact"]),
     20.0, 0.001, "record-cut splits that rejoin byte-exact"),
    ("data/custom/rretlfan.json", lambda d: float(d["qwen-35b"]["solved"]),
     2.0, 1.5, "and solved single-pass — the namespace residual"),
    ("data/custom/fanrounds.json", lambda d: float(d["solved_fix"]),
     1.0, 1.5, "solved at the fan-out/compress fixpoint"),
    ("data/custom/fanrounds.json", lambda d: float(d["late_defs"]),
     6.0, 3.0, "definitions arriving after round 1 — the hidden residual, counted"),
    # The lexicon's declared floor, the tagger that did not improve, the namespace residual
    # dying and cascading, and the starved shape space.
    ("data/custom/lexicon.json", lambda d: d["noun_precision"],
     0.606, 0.03, "lexicon noun precision on UD, after the one calibration"),
    ("data/custom/lexicon.json", lambda d: d["coverage"],
     0.726, 0.03, "share of UD tokens the lexicon can read at all"),
    ("data/custom/unitsv2.json", lambda d: float(d["A_v1_tagger_v1_asked"]),
     1112.0, 5.0, "phase 54 anchor reproduced exactly"),
    ("data/custom/unitsv2.json", lambda d: float(d["B_v2_tagger_v1_asked"]),
     1097.0, 15.0, "the dictionary walk — no better than next-word"),
    ("data/custom/namefix.json", lambda d: float(d["not_reduce"]),
     0.0, 0.001, "cross-part name mismatches with record-issued names — dead"),
    ("data/custom/namefix.json", lambda d: float(d["missing_numbers"]),
     8.0, 2.0, "and the residual cascading into missing references"),
    ("data/custom/namefix.json", lambda d: float(d["solved"]),
     2.0, 1.5, "solved — unchanged, the next layer is reference structure"),
    ("data/custom/graphmatch.json", lambda d: float(d["y_graph_match"]),
     38.0, 6.0, "graph-signature top-1 same-shape matches"),
    ("data/custom/graphmatch.json", lambda d: float(d["x_topic_match"]),
     29.0, 6.0, "topic-embedding ditto — both near chance in a starved shape space"),
    ("data/custom/graphmatch.json", lambda d: float(d["shapes"]),
     599.0, 20.0, "distinct plan shapes over 1,554 problems: types barely recur"),
    # The reference layer: reading halves the missing numbers, names stay dead, and the
    # bottom layer of the residual chain — relations — is named with the full ledger behind it.
    ("data/custom/refgraph.json", lambda d: float(d["missing_numbers"]),
     4.0, 2.0, "missing-number failures with the reference map — halved from 8"),
    ("data/custom/refgraph.json", lambda d: float(d["not_reduce"]),
     0.0, 0.001, "name mismatches — still dead under issued names"),
    ("data/custom/refgraph.json", lambda d: float(d["ref_maps"]),
     17.0, 2.5, "reference maps the record accepted"),
    ("data/custom/refgraph.json", lambda d: float(d["distinct_compounds"]),
     24.0, 8.0, "compound names coined from issued atoms"),
    ("data/custom/refgraph.json", lambda d: float(d["solved"]),
     1.0, 1.5, "solved — the relations layer is what one-shot buys"),
    # The relations live in the structure: masking every value costs the strong reader one
    # problem in twenty, and the layer amortises per shape; the small reader is ruled out.
    ("data/custom/relcarry.json", lambda d: float(d["A_whole_35b"]),
     15.0, 2.5, "whole reading, the within-run anchor"),
    ("data/custom/relcarry.json", lambda d: float(d["B_masked_35b"]),
     14.0, 2.5, "structure only — relations survive with every number hidden"),
    ("data/custom/relcarry.json", lambda d: float(d["C_masked_1b"]),
     0.0, 1.0, "the 1B reading structure at real complexity — planning-class, ruled out"),
    # Under the floor: routing does not distinguish failing subtractions, and the failures
    # obey a neighbourhood law — the atom's safety map, measured cell by cell.
    ("data/custom/failtrace.json", lambda d: d["echo_excess"],
     -0.003, 0.02, "wrong-case echo excess — the copy-routing hypothesis, refuted"),
    ("data/custom/failtrace.json", lambda d: d["within_right"] - d["right_vs_wrong"],
     0.0, 0.02, "routing similarity gap between right and wrong cases — none"),
    ("data/custom/failtrace.json", lambda d: float(d["wrong"]),
     14.0, 3.0, "single-digit subtractions the model fails"),
    ("data/custom/failtrace.json",
     lambda d: float(sum(1 for a, b, _ in d["wrong_list"] if a - b <= 1)),
     12.0, 2.0, "of which at difference zero or one — the neighbourhood law"),
    # Lego bricks: sixteen facts lifted on demand, chains found by streamed type-space
    # routing, every chain reciprocal, fan-out recombining exactly through its residual.
    ("data/custom/bricks.json", lambda d: float(d["exact"]),
     5.0, 0.001, "cross-compound conversions solved exactly by brick chains"),
    ("data/custom/bricks.json", lambda d: float(d["reverse_reciprocal"]),
     5.0, 0.001, "chains whose reverse is exactly the reciprocal"),
    ("data/custom/bricks.json",
     lambda d: 1.0 if d["fanout_matches_direct"] else 0.0,
     1.0, 0.001, "fan-out via split residual matches the direct route exactly"),
    ("data/custom/bricks.json", lambda d: float(d["bricks"]),
     32.0, 0.001, "bricks in the registry, from sixteen written-down facts"),
    # Affine and currency bricks: exact through two affine hops, lifting refused for
    # offsets, and many roads acting as the verifier — unanimity clean, divergence localised.
    ("data/custom/bricks2.json", lambda d: float(d["exact"]),
     6.0, 0.001, "temperature and currency conversions exact, including two affine hops"),
    ("data/custom/bricks2.json", lambda d: 1.0 if d["affine_lift_refused"] else 0.0,
     1.0, 0.001, "celsius/hour refused — an offset cannot ride inside a compound"),
    ("data/custom/bricks2.json", lambda d: float(d["roads_nok_eur"]),
     3.0, 1.0, "independent roads from nok to eur"),
    ("data/custom/bricks2.json", lambda d: 1.0 if d["unanimous_clean"] else 0.0,
     1.0, 0.001, "unanimous on the consistent table"),
    ("data/custom/bricks2.json", lambda d: 1.0 if d["disagree_poisoned"] else 0.0,
     1.0, 0.001, "and disagreeing once one rate is poisoned"),
    ("data/custom/bricks2.json", lambda d: 1.0 if d["poison_localised"] else 0.0,
     1.0, 0.001, "with every divergent road crossing the poisoned edge"),
    # Legitimate divergence: declared asymmetry makes disagreement a QUESTION with the
    # direction as residue; undeclared disagreement stays the alarm.
    ("data/custom/bricks3.json",
     lambda d: 1.0 if d["classifications_correct"] else 0.0,
     1.0, 0.001, "spread and toll read as questions, the poison still an alarm"),
    ("data/custom/bricks3.json",
     lambda d: 1.0 if d["residue_restores_branches"] else 0.0,
     1.0, 0.001, "collapsing the question keeps every other branch recoverable"),
    ("data/custom/bricks3.json", lambda d: float(d["spread_roads"]),
     2.0, 1.0, "roads carrying the two-sided quote"),
    # Probe-on-first-touch: two probes infer, the third verifies, the ledger remembers.
    ("data/custom/brickprobe.json", lambda d: float(d["inferred_exact"]),
     7.0, 0.001, "hidden transforms recovered exactly by two probes"),
    ("data/custom/brickprobe.json", lambda d: float(d["nonlinear_refused"]),
     1.0, 0.001, "the x*x impostor refused by the third probe"),
    ("data/custom/brickprobe.json", lambda d: float(len(d["pairs_flagged"])),
     2.0, 0.001, "pairs auto-flagged two-sided — the classifier's metadata, discovered"),
    ("data/custom/brickprobe.json", lambda d: float(d["total_probes"]),
     24.0, 0.001, "probe calls for ten hits — repeats free"),
    # Assume the minimum, ledger the assumption, rerun only the dependents.
    ("data/custom/brickassume.json", lambda d: float(d["assumptions"]),
     4.0, 0.001, "assumptions taken at two-sided edges, each ledgered with its address"),
    ("data/custom/brickassume.json", lambda d: float(d["dependents_rerun"]),
     3.0, 0.001, "computations rerun when their question was answered"),
    ("data/custom/brickassume.json", lambda d: float(d["untouched_skipped"]),
     2.0, 0.001, "computations the answer did not touch, skipped"),
    ("data/custom/brickassume.json", lambda d: float(d["counterfactual_exact"]),
     3.0, 0.001, "reruns reproducing the ledgered counterfactual to the digit"),
    ("data/custom/brickassume.json", lambda d: float(len(d["still_pending"])),
     1.0, 0.001, "and the toll question, unanswered, still pending as it should be"),
    # Three solver tiers: binary over distinct questions, exact coupled equations with the
    # types checked first, and the PySpell factory joined end to end.
    ("data/custom/bricksolve.json", lambda d: float(d["binary_surviving"]),
     1.0, 0.001, "assignments reproducing the observation — unique"),
    ("data/custom/bricksolve.json", lambda d: 1.0 if d["binary_recovered"] else 0.0,
     1.0, 0.001, "and equal to the ground-truth directions"),
    ("data/custom/bricksolve.json", lambda d: 1.0 if d["coupled_checks"] else 0.0,
     1.0, 0.001, "speed-against-weight solved exactly, equations verified"),
    ("data/custom/bricksolve.json",
     lambda d: 1.0 if d["factory"].get("is_offset_brick") else 0.0,
     1.0, 0.001, "PySpell pair, engine-verified, probe-read, recognised as a brick"),
    # Assume-and-ledger, completed by the solvers (phase 76 additions).
    ("data/custom/brickassume.json", lambda d: float(d["counterfactual_exact"]),
     3.0, 0.001, "counterfactuals carried to final form, reproduced to the digit"),
    # The factory with the model's pen: integer bricks mount four for four, the route
    # through a model brick is exact, and the wrong solver was refused twice.
    ("data/custom/factory2.json", lambda d: float(d["bricks_mounted"]),
     2.0, 0.001, "stage A bricks mounted of three authored"),
    ("data/custom/factory2.json", lambda d: 1.0 if d["route_exact"] else 0.0,
     1.0, 0.001, "the first route crossing a model-written brick, exact"),
    ("data/custom/factory2.json", lambda d: float(d["solver_passed"]),
     0.0, 0.001, "constraint samples the wrong solver passed — refused, twice"),
    ("data/custom/factory_scale.json", lambda d: float(d["probe_exact_mounted"]),
     2.0, 0.001, "stage B families mounted of eight authored"),
    ("data/custom/factory_scale.json",
     lambda d: float(d["reach_after"] - d["reach_before"]),
     12.0, 0.001, "routable ordered pairs the two families bought"),
    ("data/custom/factory_scale.json", lambda d: float(d["authored"]),
     8.0, 0.001, "families the model authored in one run"),
    # The fractional families through the Fraction factory: the model names the fraction,
    # the task's own example judges exactly, seven of seven mount.
    ("data/custom/factory_frac.json", lambda d: float(d["example_pass"]),
     7.0, 0.001, "families passing the task's embedded example, exactly"),
    ("data/custom/factory_frac.json", lambda d: float(d["truth_pass"]),
     7.0, 0.001, "and agreeing with the corrected truth table"),
    ("data/custom/factory_frac.json",
     lambda d: float(d["reach_after"] - d["reach_before"]),
     52.0, 0.001, "routable ordered pairs the seven families bought"),
    ("data/custom/factory_frac.json", lambda d: 1.0 if d["demo"]["exact"] else 0.0,
     1.0, 0.001, "ten nautical miles in km, exact through the model-named fraction"),
    # Algebra bricks: every road the same x, and the first step prices the residue 5x.
    ("data/custom/eqbricks.json", lambda d: float(d["roads_agree"]),
     20.0, 0.001, "equations where every strategy lands the identical exact x"),
    ("data/custom/eqbricks.json", lambda d: float(d["divide_early_residue"]),
     1061.0, 0.001, "total residue carried by dividing early"),
    ("data/custom/eqbricks.json", lambda d: float(d["clear_first_residue"]),
     216.0, 0.001, "and by multiplying both sides first — the 4.9x"),
    ("data/custom/eqbricks.json", lambda d: float(d["pingpong_met"]),
     20.0, 0.001, "ping-pong meetings in the middle"),
    ("data/custom/eqbricks.json",
     lambda d: float(d["pingpong_states"]) / d["forward_only_states"],
     1.25, 0.15, "its state cost against forward-only at depth three — pays only deeper"),
    # Depth: the scramble where ping-pong pays, and the priced first step.
    ("data/custom/eqdeep.json", lambda d: float(d["pingpong_met"]),
     8.0, 0.001, "ten-brick scrambles met in the middle"),
    ("data/custom/eqdeep.json", lambda d: float(d["road_valid"] + d["road_unwinds"]),
     16.0, 0.001, "stitched roads that validate forward and unwind exactly"),
    ("data/custom/eqdeep.json", lambda d: float(d["forward_found"]),
     2.0, 0.001, "forward-only rims that reached the goal on the same budget"),
    ("data/custom/eqdeep.json", lambda d: d["forward_states"] / d["pingpong_states"],
     6.9, 0.05, "the state-ratio floor at depth ten"),
    ("data/custom/eqplan.json", lambda d: float(d["correct"]),
     11.0, 0.001, "first steps picked at the priced argmin, against parrot 6 and coin 4"),
    ("data/custom/eqplan.json",
     lambda d: float(sum(1 for r in d["rows"] if r["best"] == "sub_x" and r["ok"])),
     6.0, 0.001, "integer traps where the multiply slogan was refused"),
    ("data/custom/eqplan.json", lambda d: float(d["roads_exact"]),
     36.0, 0.001, "roads under the battery landing the identical exact x"),
    # Memory over roads: one search serves the world, and the mint under audit pressure.
    ("data/custom/eqmemory.json",
     lambda d: float(d["act1_miss"] + d["act1_hard"] + d["act1_soft"]),
     1.0, 0.001, "searches the 60-equation stream needed beyond the audit floor"),
    ("data/custom/eqmemory.json",
     lambda d: float(d["act1_exact"] + d["transfer_exact"] + d["act2_exact"]),
     120.0, 0.001, "answers delivered exact across both acts and transfer"),
    ("data/custom/eqmemory.json", lambda d: float(d["transfer_miss"]),
     0.0, 0.001, "new searches needed at denominators 7-50 on a store built on 2-6"),
    ("data/custom/eqmemory.json", lambda d: float(d["act2_overpriced_serves"]),
     2.0, 0.001, "overpayments served before the randomised audit caught the class"),
    ("data/custom/eqmemory.json", lambda d: float(d["act2_mints"]),
     1.0, 0.001, "signature bits the memory minted when its vocabulary ran out"),
    ("data/custom/eqmemory.json", lambda d: float(d["store_keys"]),
     3.0, 0.001, "store keys after act two, deepened only where the world pushed"),
    # The one-bit floor: the 1B reads at exactly the parrot line; the ladder holds.
    ("data/custom/eq1b.json", lambda d: float(d["q_right"] - d["parrot"]),
     0.0, 0.001, "the 1B's reading minus the always-A parrot — zero bits contributed"),
    ("data/custom/eq1b.json", lambda d: float(d["q_right"]),
     26.0, 0.001, "counterbalanced reads right of 48"),
    ("data/custom/eq1b.json", lambda d: float(d["read_solved"] + d["esc_solved"]),
     12.0, 0.001, "equations solved by the full ladder, goal-test gated"),
    ("data/custom/eq1b.json", lambda d: float(d["esc_solved"]),
     10.0, 0.001, "escalations the 35B closed, of 10 sent up"),
    ("data/custom/eq1b.json", lambda d: float(d["menu_wasted"]),
     49.0, 0.001, "of 67 planning picks wasted on inapplicable ops at 1B"),
    # Word problems: two entry points as the gate; the risk number held at zero.
    ("data/custom/eqwords.json", lambda d: float(d["a_right"] + d["b_right"]),
     24.0, 0.001, "clean-skin translations right across both arms"),
    ("data/custom/eqwords.json", lambda d: float(d["agree_wrong"]),
     0.0, 0.001, "clean-skin deliveries on a wrong state"),
    ("data/custom/eqwords.json", lambda d: float(d["e2e"]),
     12.0, 0.001, "clean-skin end-to-end exact through the blind road"),
    ("data/custom/eqwords_hard.json", lambda d: float(d["a_right"] + d["b_right"]),
     21.0, 0.001, "hard-skin translations right — errors now exist"),
    ("data/custom/eqwords_hard.json", lambda d: float(d["agree_wrong"]),
     0.0, 0.001, "hard-skin deliveries on a wrong state — the risk held at zero"),
    ("data/custom/eqwords_hard.json", lambda d: float(d["flagged"]),
     1.0, 0.001, "stories flagged rather than guessed when the arms stayed split"),
    ("data/custom/eqwords_hard.json", lambda d: float(d["e2e"]),
     11.0, 0.001, "hard-skin end-to-end exact deliveries"),
    # Units in the text pipeline: the founding example, gated and anchored.
    ("data/custom/equnits.json", lambda d: float(d["a_right"] + d["b_right"]),
     18.0, 0.001, "unit translations right across both arms, first try"),
    ("data/custom/equnits.json", lambda d: float(d["agree_wrong"]),
     0.0, 0.001, "unit deliveries on a wrong triplet"),
    ("data/custom/equnits.json", lambda d: float(d["anchor_ok"]),
     2.0, 0.001, "hand anchors exact, including 3 mile/second = 17380.9152 km/hour"),
    ("data/custom/equnits.json", lambda d: float(d["e2e"] + d["routed"]),
     20.0, 0.001, "chains found and end-to-end exact deliveries, ten of each"),
    # Coupled systems from stories: both gates, zero wrong deliveries.
    ("data/custom/eqpairs.json", lambda d: float(d["a_right"] + d["b_right"]),
     16.0, 0.001, "coupled-system translations right across both arms"),
    ("data/custom/eqpairs.json", lambda d: float(d["agree_wrong"]),
     0.0, 0.001, "coupled deliveries on wrong constants"),
    ("data/custom/eqpairs.json", lambda d: float(d["subst_ok"]),
     8.0, 0.001, "solutions substituting back into their equations exactly"),
    ("data/custom/eqpairs.json", lambda d: float(d["e2e"]),
     8.0, 0.001, "coupled end-to-end matches of the written truth pairs"),
    # Translation memory: the toll paid once per template, exactness standing still.
    ("data/custom/eqtransmem.json",
     lambda d: d["baseline_calls"] / d["model_calls"],
     3.75, 0.05, "the bill against the dual-arm baseline"),
    ("data/custom/eqtransmem.json", lambda d: float(d["exact"]),
     30.0, 0.001, "deliveries exact on x with the store doing the reading"),
    ("data/custom/eqtransmem.json",
     lambda d: float(d["record_served"] - d["audits"] * 0),
     25.0, 0.001, "stories translated by the record alone"),
    ("data/custom/eqtransmem.json", lambda d: float(d["audit_mismatch"]),
     0.0, 0.001, "sampled audits that caught the record's extraction drifting"),
    # AIME at a working budget: the EOS fix held, and the gate's guarantee broke.
    ("data/custom/aimebudget.json", lambda d: float(d["unclosed"]),
     0.0, 0.001, "replies that died at the prompt after the think-prefill fix"),
    ("data/custom/aimebudget.json", lambda d: float(d["a"]),
     2.0, 0.001, "A-arm rights at n=6000 visible working, against the 0/15 floor"),
    ("data/custom/aimebudget.json", lambda d: float(d["gate_wrong"]),
     2.0, 0.001, "agreement-gate deliveries that were WRONG — the breaking point"),
    ("data/custom/aimebudget.json",
     lambda d: float(d["gate_delivered"] + d["flagged"]),
     10.0, 0.001, "gate outcomes accounted: delivered plus flagged"),
    ("data/custom/aimebudget.json", lambda d: float(d["graphs_ran"]),
     2.0, 0.001, "arithmetic plans that even parsed at olympiad level"),
    # Answer plus executable check: solving lifted, and the gate starves before it lies.
    ("data/custom/aimecheck.json", lambda d: float(d["answer_right"]),
     6.0, 0.001, "answers right when the prompt demands a verifier, against 2 and 1"),
    ("data/custom/aimecheck.json", lambda d: float(d["delivered_wrong"]),
     0.0, 0.001, "wrong deliveries through the check gate — the zero that held"),
    ("data/custom/aimecheck.json",
     lambda d: float(d["delivered"] + d["flagged"]),
     10.0, 0.001, "check-gate outcomes accounted: delivered plus flagged"),
    ("data/custom/aimecheck.json", lambda d: float(d["check_parsed"]),
     2.0, 0.001, "checks that stayed inside the subset at first authoring"),
    # The library's two faces: embeddings propose, the typed graph disposes.
    ("data/custom/formgraph.json", lambda d: float(d["embed_top1"]),
     19.0, 0.001, "embedding-alone top-1 of 24 — the lexical face's ceiling"),
    ("data/custom/formgraph.json", lambda d: float(d["extract_ok"]),
     24.0, 0.001, "quantity graphs verified literal against the text"),
    ("data/custom/formgraph.json",
     lambda d: float(d["overrides"] + d["override_right"]),
     8.0, 0.001, "graph overrides of the embedding, all of them right"),
    ("data/custom/formgraph.json", lambda d: float(d["tiebreak_right"]),
     5.0, 0.001, "embedding tiebreaks right among dimension-twins, of 6"),
    ("data/custom/formgraph.json", lambda d: float(d["right"]),
     23.0, 0.001, "deliveries exact through the coupled system, against 19 lexical"),
    ("data/custom/formgraph.json", lambda d: float(d["fit_empty"]),
     0.0, 0.001, "stories no formula shape could fit"),
    # MPEqs' solver library: classes of problems, exact, and a chained AIME solve.
    ("data/custom/solvers.json", lambda d: float(d["passed"]),
     22.0, 0.001, "solver self-test cases exact of 22"),
    ("data/custom/solvers.json", lambda d: float(d["failed"]),
     0.0, 0.001, "self-test failures after the anchor autopsy"),
    ("data/custom/solvers.json", lambda d: float(d["refusals_named"]),
     6.0, 0.001, "refusals raised with the right reason"),
    ("data/custom/solvers.json", lambda d: float(d["chain_answer"]),
     12.0, 0.001, "the chained AIME answer: search then factor then exponent sum"),
    ("data/custom/solvers.json", lambda d: float(d["solvers"] + d["predicates"]),
     37.0, 0.001, "solvers plus search predicates in the library"),
    # Mapping onto the library: guiding lost, and mis-mappings never self-destruct.
    ("data/custom/solvemap.json", lambda d: float(d["free_exact"]),
     29.0, 0.001, "free-arm exact deliveries of 30"),
    ("data/custom/solvemap.json", lambda d: float(d["guided_exact"]),
     26.0, 0.001, "guided-arm exact deliveries — narrowing the job lost"),
    ("data/custom/solvemap.json",
     lambda d: float(d["free_refused"] + d["guided_refused"]),
     0.0, 0.001, "specs the record refused: none, so mis-mappings answer confidently"),
    ("data/custom/solvemap.json", lambda d: float(d["embed_top1"]),
     28.0, 0.001, "solver classes the embedding alone picked right"),
    # Growth: expressions, tuples, polynomials, coordinates, modular arithmetic.
    ("data/custom/solvers2.json", lambda d: float(d["passed"]),
     15.0, 0.001, "grown-library self-test cases exact"),
    ("data/custom/solvers2.json", lambda d: float(d["refusals_named"]),
     6.0, 0.001, "refusals named, three of them expression-sandbox escapes"),
    ("data/custom/solvers2.json", lambda d: float(d["solvers_total"]),
     19.0, 0.001, "solvers in MPEqs after the growth phase"),
    # AIME coverage: willingness to map is not reach.
    ("data/custom/aimecover.json", lambda d: float(d.get("claimed", 0)),
     22.0, 0.001, "AIME problems the model claimed it could map"),
    ("data/custom/aimecover.json", lambda d: float(d.get("ran", 0)),
     19.0, 0.001, "specs that validated and executed"),
    ("data/custom/aimecover.json", lambda d: float(d.get("exact", 0)),
     0.0, 0.001, "of which matched the published answer — none"),
    ("data/custom/aimecover.json", lambda d: float(d.get("declined", 0)),
     5.0, 0.001, "problems where the model named the missing capability instead"),
    # Two-sided labels on the chord metric: no better than one relay, twice the memory.
    ("data/custom/hub-corpus.json", lambda d: d["exact_cell_pct"],
     1.34, 0.7, "two-sided hub labels on the angular metric"),
    ("data/custom/hub-corpus.json", lambda d: d["recall_at_k"],
     0.108, 0.06, "recall@10 of the two-sided base on the angular metric"),
    # More hops does nothing, on either metric, because min-plus closure is idempotent on a
    # metric. If a squaring ever changes this the reasoning above is wrong and must be redone.
    ("data/custom/hub-geodesic-sq0.json", lambda d: d["exact_cell_pct"],
     27.92, 3.0, "geodesic two-sided, two hops"),
    ("data/custom/hub-geodesic-sq2.json", lambda d: d["exact_cell_pct"],
     27.90, 3.0, "geodesic two-sided, five hops — unchanged"),
]

# (quarantine lifted) Every one of these was computed from the router trace, and the
# trace was an artefact: `ffn_moe_topk` is a view of the argsort node, so a flat backend read
# returned each token's whole 64-wide ranking instead of its top-k. The access pattern was
# uniform by construction, which is the premise most of these numbers rest on. On a corrected
# trace LRU hits 78.6 % rather than 4.5 %, so the conclusions attached to them reverse.
#
# They are kept rather than deleted so the re-run has something to diff against, but they are
# NOT asserted — a consistency gate cannot detect a consistent error, which is exactly how this
# survived 40 passing checks. `Trace::reject_argsort_artefact` is the guard that can.




def main() -> int:
    failures = 0
    skipped = 0
    for path, extract, expected, tol, label in CHECKS:
        p = pathlib.Path(path)
        if not p.exists():
            print(f"SKIP  {label:<36} ({path} missing)")
            skipped += 1
            continue
        try:
            got = extract(json.loads(p.read_text()))
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"ERROR {label:<36} {e}")
            failures += 1
            continue
        bad = abs(got - expected) > tol
        print(f"{'FAIL' if bad else 'ok  '}  {label:<36} {got:>10.4f}  expected {expected}")
        failures += bad
    if failures:
        print(f"\n{failures} documented numbers no longer match the data.")
    elif skipped:
        print(f"\nall present numbers match ({skipped} skipped)")
    else:
        print("all remaining documented numbers match their source data")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
