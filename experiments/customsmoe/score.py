#!/usr/bin/env python3
"""Score the analysis chain against the structure that was planted into the model.

`planted.py` trains a real MoE in which group g's tokens are pushed onto experts
[4g, 4g+4). With 8 groups, 32 experts and top-4, that makes the ground truth arithmetic:

    an expert fires only for its own group, so P(expert a) = 1/8
    the four experts of a group fire together, so P(a and b) = 1/8 for a pair inside a group
    lift = P(a and b) / (P(a) P(b)) = (1/8) / (1/64) = 8.0     inside a group
    lift = 0                                                    across groups

So the instrument has a number to be right about, which is precisely what the router trace
never had. That absence is how a flat-read artefact went unnoticed through forty passing checks.

Compares that prediction against `coact analyze`, and against the graph recomputed here, so a
disagreement points at which side is wrong.
"""
import itertools
import json
import re
import subprocess
import struct
import sys
from collections import Counter
from pathlib import Path


def read_moet(path):
    raw = Path(path).read_bytes()
    magic, ver, n_layer, n_expert, top_k, _, _ = struct.unpack_from("<6IQ", raw, 0)
    assert magic == 0x5445_4F4D, "not a MOET trace"
    rec = 8 + 2 * top_k + (4 * top_k if ver >= 2 else 0) + (4 * top_k if ver >= 3 else 0)
    toks = [struct.unpack_from(f"<{top_k}H", raw, off + 8)
            for off in range(32, len(raw) - rec + 1, rec)
            if struct.unpack_from("<H", raw, off)[0] == 0]
    return toks, n_expert, top_k


def main(trace="data/custom/trace-planted.bin", truth="data/custom/ground-truth.json",
         analysis="data/custom/analysis.json", out="data/custom/score.json"):
    toks, n_expert, top_k = read_moet(trace)
    gt = json.loads(Path(truth).read_text())
    group = gt["planted_group_of_expert"]
    per_group = gt["experts_per_group"]
    n_group = gt["n_group"]

    single = Counter()
    pair = Counter()
    for t in toks:
        s = set(t)
        single.update(s)
        pair.update(itertools.combinations(sorted(s), 2))
    n = len(toks)

    def split(assign):
        ins, out = [], []
        for a, b in itertools.combinations(range(n_expert), 2):
            pa, pb = single[a] / n, single[b] / n
            if pa == 0 or pb == 0:
                continue
            lift = (pair[(a, b)] / n) / (pa * pb)
            (ins if assign[a] == assign[b] else out).append(lift)
        return ins, out

    # Two ground truths, and the difference between them is the whole point. `planted` is what
    # training was asked to produce; `observed` is what the router actually settled on, which
    # `planted.py` reports as 93.8 % agreement. Scoring only against `planted` conflates two
    # different failures — an analysis chain that cannot see structure, and a model that did
    # not learn the structure it was asked for. Only the first would be a bug in this repo.
    inside, across = split(group)
    obs = gt["observed_group_of_expert"]
    inside_o, across_o = split(obs)

    predicted = float(n_group)          # 1/G divided by (1/G)^2
    mean_in = sum(inside) / len(inside)
    mean_out = sum(across) / len(across)
    print(f"{n} tokens, {n_expert} experts, top-{top_k}, "
          f"{n_group} planted groups of {per_group}\n")
    print(f"{'':>28} {'predicted':>10} {'measured':>10}")
    print(f"{'lift inside a planted group':>28} {predicted:>10.2f} {mean_in:>10.2f}")
    print(f"{'lift across groups':>28} {0.0:>10.2f} {mean_out:>10.2f}")
    print(f"{'pairs inside a group, %':>28} "
          f"{100.0 * n_group * per_group * (per_group - 1) / 2 / (n_expert * (n_expert - 1) / 2):>10.2f} "
          f"{100.0 * len(inside) / (len(inside) + len(across)):>10.2f}")

    # Does the separation survive as a decision rule? Anything above halfway between the two
    # predicted values should be an inside-group pair and nothing else.
    thr = predicted / 2
    tp = sum(1 for v in inside if v >= thr)
    fp = sum(1 for v in across if v >= thr)
    print(f"\nthresholding lift at {thr:.1f} recovers {tp}/{len(inside)} planted pairs "
          f"with {fp} false positives")

    res = {"tokens": n, "predicted_lift_inside": predicted,
           "measured_lift_inside": round(mean_in, 3),
           "measured_lift_across": round(mean_out, 3),
           "recall": round(tp / len(inside), 4),
           "false_positives": fp}

    if Path(analysis).exists():
        a = json.loads(Path(analysis).read_text())
        res["coact_analyze"] = a
        print(f"\n`coact analyze` on the same trace reported the numbers in {analysis};"
              f" its max lift should land at {predicted:.1f}, not above it.")

    Path(out).write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")

    mi_o = sum(inside_o) / len(inside_o)
    mo_o = sum(across_o) / len(across_o)
    print(f"\nagainst the router's OWN assignment ({100 * gt['plant_agreement']:.1f} % of "
          f"experts match the plant):")
    print(f"{'lift inside a group':>28} {predicted:>10.2f} {mi_o:>10.2f}")
    print(f"{'lift across groups':>28} {0.0:>10.2f} {mo_o:>10.2f}")
    res["measured_lift_inside_observed"] = round(mi_o, 3)
    res["measured_lift_across_observed"] = round(mo_o, 3)
    Path(out).write_text(json.dumps(res, indent=2))

    # Two separate questions, and conflating them was my first mistake here. Measured lift
    # inside a group is 5.89 against a predicted 8.00, and the first version called that a
    # FAIL — but a shortfall against the *ideal* can mean either "the analysis cannot see the
    # structure" or "the model never learned it". Only the first is a defect in this repo.
    #
    # They separate cleanly:
    #   compliance  = how much of the planted structure the trained router actually has,
    #                 measured from the trace by an implementation independent of the chain
    #   agreement   = whether `coact analyze` reports the same thing as that implementation
    #
    # The chain is validated by AGREEMENT. Compliance is a fact about the model.
    compliance = mean_in / predicted
    print(f"\nmodel compliance: the router realised {100 * compliance:.1f} % of the planted "
          f"lift ({mean_in:.2f} of {predicted:.2f}).")
    print("That is a property of the trained model, not of the analysis.")

    # Run the real chain and parse what it prints. Its analysis JSON does not persist the
    # lift table, and the first version of this check quietly returned PASS when it could not
    # find the fields — a check that cannot fail is not a check, which is the entire lesson of
    # the trace artefact. So invoke the binary and compare the numbers it actually reports.
    verdict = 0
    mine = {"max lift": max(inside + across),
            "pairs >= 2x lift, %": 100.0 * sum(1 for v in inside + across if v >= 2.0)
                                   / len(inside + across)}
    proc = subprocess.run(
        ["./target/release/coact", "analyze", "--trace", trace, "--out", analysis],
        capture_output=True, text=True)
    row = next((l for l in proc.stdout.splitlines()
                if re.match(r"\s*\d+\s+[\d.]+\s+[\d.]+", l)), None)
    if row is None:
        print("\nFAIL: `coact analyze` produced no lift row to compare against.")
        print(proc.stdout[-400:] or proc.stderr[-400:])
        return 1
    f = re.findall(r"[\d.]+", row)
    theirs = {"max lift": float(f[4]), "pairs >= 2x lift, %": float(f[5])}
    print(f"\n{'':>24} {'independent':>12} {'coact analyze':>14}")
    for k in mine:
        rel = abs(theirs[k] - mine[k]) / max(mine[k], 1e-9)
        ok = rel < 0.01
        verdict |= 0 if ok else 1
        print(f"{k:>24} {mine[k]:>12.2f} {theirs[k]:>14.2f}"
              f"{'  ok' if ok else f'  DISAGREES by {100 * rel:.1f} %'}")

    res["compliance"] = round(compliance, 4)
    res["independent"] = {k: round(v, 4) for k, v in mine.items()}
    res["coact_reported"] = theirs
    Path(out).write_text(json.dumps(res, indent=2))
    print("\nPASS: the analysis chain agrees with an independent implementation."
          if verdict == 0 else
          "\nFAIL: the chain disagrees with an independent implementation of the same "
          "statistic. That is a bug in the chain.")
    return verdict


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
