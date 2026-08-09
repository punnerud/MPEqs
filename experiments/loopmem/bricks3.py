#!/usr/bin/env python3
"""Legitimate divergence: when different roads SHOULD give different answers.

Phase 74 treated road disagreement as an alarm. The correction: sometimes the disagreement
is the world's, not the table's — a currency quote has a buy side and a sell side, a toll
gate charges one direction and not the other. Divergence then encodes an UNANSWERED
QUESTION (which direction? which crossing?), and the honest output is not one number but a
set of conditional answers with the question attached as a RESIDUE — collapsible the
moment the question is answered, and unwindable back to the question form, because a
residue that cannot be put back is not a residue.

The discriminator is mechanical and needs no judgement call: an edge whose two directions
are DECLARED independently with product != 1 is declared-asymmetric. Divergent roads whose
differing edges include a declared-asymmetric brick are CONDITIONAL — a question. Divergent
roads with no such edge in play are INCONSISTENT — phase 74's alarm, unchanged. Three
scenarios prove the three outcomes:

    spread     usd<->nok quoted 10.45 / 10.55: roads diverge, classified QUESTION
    toll       a transfer with a 25-crown fee one way and none back: QUESTION
    poison     the phase 74 bad cross, no declared asymmetry: ALARM, still

And the residue does residue work: answering "selling" collapses the set to one exact
value; the collapse records which branch was taken, so the conditional form is recoverable.
"""
import json
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bricks2 import ABrick, all_roads, compose, tkey  # noqa: E402


def declared_asymmetric(bricks):
    """The edges FLAGGED two-sided at declaration time — metadata, never inference.

    The first version inferred asymmetry from product != 1, and the poisoned pair's product
    is also != 1: a spread and a bug are arithmetically identical, which is exactly why the
    distinction must be knowledge about the edge, supplied by whoever declared it. A bank
    QUOTES two sides; nobody declared the poisoned cross two-sided."""
    return {b.name for b in bricks if getattr(b, "two_sided", False)}


def classify(roads, asym):
    """One transform -> ANSWER. Several -> QUESTION if a declared-asymmetric edge separates
    the branches, else ALARM. The question names the edges that decide."""
    groups = {}
    for tr, path in roads:
        groups.setdefault(tr, []).append(path)
    if len(groups) == 1:
        return {"kind": "answer", "value": next(iter(groups))}
    all_edges = [set(p) for paths in groups.values() for p in paths]
    common = set.intersection(*all_edges) if all_edges else set()
    deciding = {e for edges in all_edges for e in edges - common}
    asym_deciding = sorted(deciding & asym)
    return {
        "kind": "question" if asym_deciding else "alarm",
        "branches": [{"value": tr, "via": paths[0]} for tr, paths in groups.items()],
        "question": (f"which of {asym_deciding} applies?" if asym_deciding
                     else f"no declared asymmetry explains {sorted(deciding)[:3]}"),
    }


def registry3(spread=True, toll=True):
    b = []
    # The two-sided quote: declared independently, product deliberately != 1.
    if spread:
        b.append(ABrick("usd->nok", {"usd": 1}, {"nok": 1}, F(1045, 100)))   # bank sells nok
        b.append(ABrick("nok->usd", {"nok": 1}, {"usd": 1}, F(100, 1055)))   # bank buys nok
        b[-1].two_sided = b[-2].two_sided = True
    b.append(ABrick("eur->usd", {"eur": 1}, {"usd": 1}, F(109, 100)))
    b.append(ABrick("usd->eur", {"usd": 1}, {"eur": 1}, F(100, 109)))
    b.append(ABrick("eur->nok", {"eur": 1}, {"nok": 1}, F(109, 100) * F(105, 10)))
    b.append(ABrick("nok->eur", {"nok": 1}, {"eur": 1}, 1 / (F(109, 100) * F(105, 10))))
    # The toll: crossing east pays 25, crossing west is free. Affine, asymmetric, declared.
    if toll:
        b.append(ABrick("west->east", {"west": 1}, {"east": 1}, F(1), F(-25)))
        b.append(ABrick("east->west", {"east": 1}, {"west": 1}, F(1), F(0)))
        b[-1].two_sided = b[-2].two_sided = True
        b.append(ABrick("west->ferry", {"west": 1}, {"ferry": 1}, F(1), F(-40)))
        b.append(ABrick("ferry->east", {"ferry": 1}, {"east": 1}, F(1), F(0)))
    return b


def main(out="data/custom/bricks3.json"):
    results = {}

    # 1. The spread: nok -> nok round trips through the quote diverge from staying put,
    #    and nok -> usd via different roads carries the two-sidedness.
    bricks = registry3()
    asym = declared_asymmetric(bricks)
    print(f"declared-asymmetric edges: {sorted(asym)}\n")
    roads = all_roads(bricks, {"nok": 1}, {"usd": 1}, max_len=3)
    verdict = classify(roads, asym)
    print(f"nok -> usd: {len(roads)} roads, verdict {verdict['kind'].upper()}")
    if verdict["kind"] == "question":
        for br in verdict["branches"]:
            v = F(1000) * br["value"][0] + br["value"][1]
            print(f"    1000 nok -> {float(v):.2f} usd   via {br['via']}")
        print(f"    open question: {verdict['question']}")
    results["spread"] = verdict["kind"]

    # 2. The toll: west -> east directly (pay 25) or via the ferry (pay 40).
    roads_t = all_roads(bricks, {"west": 1}, {"east": 1}, max_len=3)
    vt = classify(roads_t, asym)
    print(f"\nwest -> east: {len(roads_t)} roads, verdict {vt['kind'].upper()}")
    for br in vt.get("branches", []):
        v = F(100) * br["value"][0] + br["value"][1]
        print(f"    100 kr in pocket -> {float(v):.0f} kr after   via {br['via']}")
    results["toll"] = vt["kind"]

    # 3. The phase 74 poison, reclassified under the discriminator: still an ALARM,
    #    because no declared asymmetry separates the branches.
    clean = [ABrick("eur->usd", {"eur": 1}, {"usd": 1}, F(109, 100)),
             ABrick("usd->eur", {"usd": 1}, {"eur": 1}, F(100, 109)),
             ABrick("nok->usd", {"nok": 1}, {"usd": 1}, F(100, 1050)),
             ABrick("usd->nok", {"usd": 1}, {"nok": 1}, F(1050, 100)),
             ABrick("nok->eur", {"nok": 1}, {"eur": 1},
                    (1 / (F(109, 100) * F(105, 10))) * F(102, 100)),   # the poison
             ABrick("eur->nok", {"eur": 1}, {"nok": 1}, F(109, 100) * F(105, 10))]
    asym_c = declared_asymmetric(clean)
    vp = classify(all_roads(clean, {"nok": 1}, {"eur": 1}, max_len=3), asym_c)
    print(f"\npoisoned table, nok -> eur: verdict {vp['kind'].upper()} — {vp['question']}")
    results["poison"] = vp["kind"]

    # 4. The residue collapses and unwinds: answer "selling nok" and the set becomes one
    #    exact value; the taken branch is recorded so the question form is recoverable.
    chosen = next(br for br in verdict["branches"] if "nok->usd" in br["via"])
    collapsed = F(1000) * chosen["value"][0] + chosen["value"][1]
    residue = {"question": verdict["question"], "taken": chosen["via"],
               "others": [b_["via"] for b_ in verdict["branches"] if b_ is not chosen]}
    results["collapsed_1000nok_usd"] = str(collapsed)
    results["residue_restores_branches"] = (
        len(residue["others"]) + 1 == len(verdict["branches"]))
    print(f"\nanswered 'selling nok': 1000 nok = {collapsed} usd = {float(collapsed):.2f}")
    print(f"residue kept: the question, the taken road, and the {len(residue['others'])} "
          f"other branch(es) — the conditional form is recoverable, so nothing was lost.")

    ok = (results["spread"] == "question" and results["toll"] == "question"
          and results["poison"] == "alarm")
    print(f"\nall three classifications correct: {ok}")
    print("Divergence now has three readings, mechanically told apart: one road is an")
    print("answer, asymmetric roads are a QUESTION carried as residue, and unexplained")
    print("disagreement stays an alarm. The unanswered direction rides with the value,")
    print("exactly as proposed.")
    results.update({"classifications_correct": ok,
                    "asym_edges": len(asym), "spread_roads": len(roads),
                    "toll_roads": len(roads_t)})
    Path(out).write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
