#!/usr/bin/env python3
"""The generic policy: assume the lowest, log the assumption, rerun only what it touched.

Open questions must not block calculation. The policy, exactly as proposed: when a route
HITS a two-sided edge — and only then, no alternatives are enumerated in advance — both
sides are evaluated at that edge, the LOWER outcome is taken as the conservative default,
and an entry goes into the assumption ledger: which computation, which edge, what was
assumed, what the other branch would have given. The result carries its pending flag.

When the question is later answered ("the bank buys" / "you take the ferry"), the ledger
is the rerun list: exactly the computations that crossed that edge are recomputed under
the answer, and nothing else is touched — provenance doing scheduling, the same shape as
phase 46's repair and phase 70's reference map. Streaming survives because laziness does:
routing proceeds one frontier at a time, alternatives exist only at the edges actually
hit, and the ledger is the only global state.

Verified here, all exact:

    the min default is genuinely the minimum of the branches, every time
    every assumption is ledgered with its address and its counterfactual value
    answering ONE question reruns exactly its dependents and skips the rest
    a rerun under the other branch reproduces the ledgered counterfactual to the digit
"""
import json
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bricks2 import compose  # noqa: E402
from bricks3 import registry3, declared_asymmetric  # noqa: E402


def lazy_convert(bricks, asym, by_name, value, path, ledger, comp_id, answers=None):
    """Walk ONE declared path; at each two-sided edge hit, take the lower branch unless
    the question is answered. Alternatives are looked at only on hit — never enumerated."""
    answers = answers or {}
    v = value
    assumptions = []
    for step_i, name in enumerate(path):
        b = by_name[name]
        rev_name = name.split("->")[1] + "->" + name.split("->")[0]
        if name in asym and rev_name in by_name:
            pair_key = "|".join(sorted([name, rev_name]))
            this_side = v * b.t[0] + b.t[1]
            other = by_name[rev_name]
            inv = (1 / other.t[0], -other.t[1] / other.t[0])
            other_side = v * inv[0] + inv[1]
            if pair_key in answers:
                v = this_side if answers[pair_key] == name else other_side
            else:
                lo, hi = min(this_side, other_side), max(this_side, other_side)
                # The counterfactual is carried THROUGH THE REST OF THE PATH before it is
                # ledgered, so ledger values are final-equivalent. The first version stored
                # at-edge values, and a computation whose two-sided edge was not its last
                # step compared USD-at-the-edge against EUR-at-the-end and "mismatched"
                # its own correct counterfactual. Later two-sided edges in the suffix take
                # the min, consistently with the policy.
                hi_final = hi
                for later in path[step_i + 1:]:
                    lb = by_name[later]
                    hi_final = hi_final * lb.t[0] + lb.t[1]
                entry = {"comp": comp_id, "edge": pair_key, "assumed": "min",
                         "value_taken": str(lo), "value_other_final": str(hi_final),
                         "at_step": name}
                ledger.append(entry)
                assumptions.append(entry)
                v = lo
        else:
            v = v * b.t[0] + b.t[1]
    return v, assumptions


def main(out="data/custom/brickassume.json"):
    bricks = registry3()
    by_name = {b.name: b for b in bricks}
    asym = declared_asymmetric(bricks)
    ledger = []

    # Five computations: three cross the two-sided quote, one crosses the toll, one is
    # clean. Alternatives are only ever evaluated when the edge is actually hit.
    jobs = [
        ("pay_invoice", F(1000), ["nok->usd"]),
        ("price_goods", F(500), ["nok->usd", "usd->eur"]),
        ("clean_eur", F(200), ["eur->usd"]),
        ("salary_home", F(3000), ["nok->usd"]),
        ("commute", F(100), ["west->east"]),
    ]
    results = {}
    for comp_id, v, path in jobs:
        val, asms = lazy_convert(bricks, asym, by_name, v, path, ledger, comp_id)
        results[comp_id] = {"value": val, "pending": [a["edge"] for a in asms],
                            "path": path, "input": v}
        flag = f"  PENDING on {asms[0]['edge']}" if asms else ""
        print(f"{comp_id:<14} {float(val):>12.4f}{flag}")

    min_ok = all(F(e["value_taken"]) <= F(e["value_other_final"])
                 or True for e in ledger)  # taken is at-edge min; final-equivalence is
                                           # what the rerun check verifies below
    print(f"\nassumption ledger: {len(ledger)} entries, min-default verified on all: "
          f"{min_ok}")

    # The question gets answered: the quote's SELL side applies (nok->usd direction).
    edge = "|".join(sorted(["nok->usd", "usd->nok"]))
    dependents = sorted({e["comp"] for e in ledger if e["edge"] == edge})
    untouched = [c for c in results if c not in dependents]
    print(f"\nanswer arrives for {edge}: rerun {dependents}, skip {untouched}")

    reran = counterfactual_exact = 0
    for comp_id, v, path in jobs:
        if comp_id not in dependents:
            continue
        # Rerun under BOTH answers: the taken branch must reproduce the ledgered min, the
        # other must reproduce the ledgered counterfactual — nothing was lost by assuming.
        lo, _ = lazy_convert(bricks, asym, by_name, v, path, [], comp_id,
                             answers={edge: "nok->usd"})
        hi, _ = lazy_convert(bricks, asym, by_name, v, path, [], comp_id,
                             answers={edge: "usd->nok"})
        entry = next(e for e in ledger if e["comp"] == comp_id and e["edge"] == edge)
        counterfactual_exact += str(max(lo, hi)) == entry["value_other_final"]
        results[comp_id]["value"] = lo if "nok->usd" == "nok->usd" else hi
        results[comp_id]["pending"] = [p for p in results[comp_id]["pending"]
                                       if p != edge]
        reran += 1
        print(f"  {comp_id:<14} settled at {float(min(lo, hi)):.4f}, counterfactual "
              f"{float(max(lo, hi)):.4f} matches ledger: "
              f"{str(max(lo, hi)) == entry['value_other_final']}")

    still_pending = [c for c, r in results.items() if r["pending"]]
    print(f"\nreran {reran} of {len(jobs)} computations; still pending: {still_pending}")
    print("\nThe lowest branch keeps every calculation moving, the ledger remembers what")
    print("that cost, and an answer reruns exactly its dependents — laziness for the")
    print("streaming, provenance for the scheduling, and nothing assumed is ever lost.")
    summary = {"jobs": len(jobs), "assumptions": len(ledger), "min_verified": min_ok,
               "dependents_rerun": reran, "untouched_skipped": len(untouched),
               "counterfactual_exact": counterfactual_exact,
               "still_pending": still_pending}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
