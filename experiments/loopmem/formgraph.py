#!/usr/bin/env python3
"""The formula library as a typed graph with embeddings attached: propose, then dispose.

The steer: away from retries, toward how problems MAP — how formulas are worded, how
the graph is built, and how embeddings connect to it. Here the library IS that
coupling. Every formula carries two faces:

    WORDINGS   how the relation is said (three phrasings each) — embedded, this is
               the lexical face the problem's text can be matched against
    SHAPE      typed slots and a result type (dimension signatures) — the graph face,
               against which the record UNIFIES mechanically

A problem arrives as text. The embedding face proposes (mask the numbers, embed,
nearest wording). The model extracts the problem's quantity graph — every stated
value with its unit, and the unit of what is asked — and the record verifies each
extracted value LITERALLY against the text (hallucinated givens die there), then
unifies dimensions against the library: which formulas take exactly these given
types and produce the asked type?

The division of authority is the point, and both directions are measured:

    fit set UNIQUE  -> the graph decides; if the embedding proposed differently,
                       the graph OVERRIDES it (adversarial pairs live here: momentum
                       and kinetic energy share every word but not their result type)
    fit set LARGER  -> dimensions genuinely cannot say (perimeter, difference and
                       average of two lengths are all m,m -> m); the embedding picks
                       WITHIN the mechanically valid set — embeddings coupled to the
                       graph, not instead of it
    fit set EMPTY   -> flag; nothing is guessed

Twelve formulas (with deliberate dimension-twins), each asked in a direct and an
indirect wording, distractor quantities in every story, truths written first, the
record computing every delivered number exactly."""
import json
import re
import subprocess
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cutbig import ask  # noqa: E402
from embednav import embed  # noqa: E402

M, S, KG, L, KR, PC, YR = "m", "second", "kg", "litre", "kr", "piece", "year"


def dim(**kw):
    return tuple(sorted(kw.items()))


UNITS = {
    "m": dim(m=1), "meter": dim(m=1), "meters": dim(m=1),
    "s": dim(second=1), "second": dim(second=1), "seconds": dim(second=1),
    "m/s": dim(m=1, second=-1),
    "kg": dim(kg=1), "kilogram": dim(kg=1), "kilograms": dim(kg=1),
    "l": dim(litre=1), "litre": dim(litre=1), "litres": dim(litre=1),
    "liter": dim(litre=1), "liters": dim(litre=1),
    "kg/l": dim(kg=1, litre=-1),
    "m^2": dim(m=2), "m2": dim(m=2), "square meters": dim(m=2),
    "square meter": dim(m=2), "sq m": dim(m=2),
    "kr": dim(kr=1), "kroner": dim(kr=1),
    "kr/piece": dim(kr=1, piece=-1),
    "piece": dim(piece=1), "pieces": dim(piece=1),
    "joule": dim(kg=1, m=2, second=-2), "joules": dim(kg=1, m=2, second=-2),
    "j": dim(kg=1, m=2, second=-2),
    "kg*m/s": dim(kg=1, m=1, second=-1), "kg·m/s": dim(kg=1, m=1, second=-1),
    "year": dim(year=1), "years": dim(year=1), "hour": dim(hour=1),
    "hours": dim(hour=1),
}

# name, input dims, output dim, exact compute, three wordings
FORMULAS = [
    ("speed", [dim(m=1), dim(second=1)], dim(m=1, second=-1),
     lambda a, b: a / b,
     ["the speed of something moving", "how fast it goes in meters per second",
      "distance divided by time"]),
    ("distance", [dim(m=1, second=-1), dim(second=1)], dim(m=1),
     lambda a, b: a * b,
     ["how far something travels at a steady speed", "the distance covered",
      "speed times time"]),
    ("momentum", [dim(kg=1), dim(m=1, second=-1)], dim(kg=1, m=1, second=-1),
     lambda a, b: a * b,
     ["the momentum of a moving object", "mass times velocity in kg*m/s",
      "how much motion it carries"]),
    ("kinetic", [dim(kg=1), dim(m=1, second=-1)], dim(kg=1, m=2, second=-2),
     lambda a, b: a * b * b / 2,
     ["the kinetic energy of a moving object", "the energy of motion in joules",
      "half the mass times the square of the speed"]),
    ("density", [dim(kg=1), dim(litre=1)], dim(kg=1, litre=-1),
     lambda a, b: a / b,
     ["the density of a substance", "how heavy each litre is",
      "mass divided by volume"]),
    ("mass_dv", [dim(kg=1, litre=-1), dim(litre=1)], dim(kg=1),
     lambda a, b: a * b,
     ["the mass from density and volume", "how many kilograms the container holds",
      "density times volume"]),
    ("area", [dim(m=1), dim(m=1)], dim(m=2),
     lambda a, b: a * b,
     ["the area of a rectangle", "how many square meters it covers",
      "length times width"]),
    ("perimeter", [dim(m=1), dim(m=1)], dim(m=1),
     lambda a, b: 2 * (a + b),
     ["the perimeter of a rectangle", "how many meters of fence go around it",
      "twice the sum of length and width"]),
    ("gap", [dim(m=1), dim(m=1)], dim(m=1),
     lambda a, b: a - b,
     ["how much longer than wide it is", "the difference between the two sides",
      "length minus width"]),
    ("mean_side", [dim(m=1), dim(m=1)], dim(m=1),
     lambda a, b: (a + b) / 2,
     ["the average side length", "the mean of length and width",
      "half of length plus width"]),
    ("price", [dim(kr=1, piece=-1), dim(piece=1)], dim(kr=1),
     lambda a, b: a * b,
     ["the total price of a purchase", "what the whole batch costs",
      "unit price times quantity"]),
    ("unitprice", [dim(kr=1), dim(piece=1)], dim(kr=1, piece=-1),
     lambda a, b: a / b,
     ["the price per piece", "what one single item costs",
      "the total divided by the count"]),
]

# (formula, values, direct story, indirect story) — truths before any call.
# Every story opens with a distractor quantity.
STORIES = [
    ("speed", (F(150), F(12)),
     "The driver is 42 years old. A car covers 150 m in 12 s. What is its speed "
     "in m/s?",
     "The driver is 42 years old. A car covers 150 m in 12 s. How many meters "
     "does it manage each second, as m/s?"),
    ("distance", (F(25), F(14)),
     "The trip began at 9 hours. A train runs at 25 m/s for 14 s. What distance "
     "in m does it cover?",
     "The trip began at 9 hours. A train runs at 25 m/s for 14 s. How many "
     "meters lie behind it after that?"),
    ("momentum", (F(6), F(14)),
     "The lab is 5 m wide. An object of 6 kg moves at 14 m/s. What is its "
     "momentum in kg*m/s?",
     "The lab is 5 m wide. An object of 6 kg moves at 14 m/s. How much motion "
     "does it carry, in kg*m/s?"),
    ("kinetic", (F(6), F(14)),
     "The lab is 5 m wide. An object of 6 kg moves at 14 m/s. What is its "
     "kinetic energy in joules?",
     "The lab is 5 m wide. An object of 6 kg moves at 14 m/s. How many joules "
     "of motion energy does it hold?"),
    ("density", (F(18), F(8)),
     "The jar label says 3 years. A syrup of 18 kg fills 8 litres. What is its "
     "density in kg/l?",
     "The jar label says 3 years. A syrup of 18 kg fills 8 litres. How many kg "
     "does each litre weigh, in kg/l?"),
    ("mass_dv", (F(3), F(7)),
     "The recipe is 2 years old. An oil of density 3 kg/l fills 7 litres. What "
     "mass in kg is that?",
     "The recipe is 2 years old. An oil of density 3 kg/l fills 7 litres. How "
     "many kilograms sit in the container?"),
    ("area", (F(30), F(20)),
     "The farm is 100 years old. A field is 30 m long and 20 m wide. What is "
     "its area in m^2?",
     "The farm is 100 years old. A field is 30 m long and 20 m wide. How many "
     "square meters does it cover?"),
    ("perimeter", (F(30), F(20)),
     "The farm is 100 years old. A field is 30 m long and 20 m wide. What is "
     "its perimeter in m?",
     "The farm is 100 years old. A field is 30 m long and 20 m wide. How many "
     "meters of fence go all the way around it?"),
    ("gap", (F(30), F(20)),
     "The farm is 100 years old. A field is 30 m long and 20 m wide. By how "
     "many m is it longer than wide?",
     "The farm is 100 years old. A field is 30 m long and 20 m wide. How many "
     "meters longer is one side than the other?"),
    ("mean_side", (F(30), F(20)),
     "The farm is 100 years old. A field is 30 m long and 20 m wide. What is "
     "the average side length in m?",
     "The farm is 100 years old. A field is 30 m long and 20 m wide. What is "
     "the mean of its two side lengths, in m?"),
    ("price", (F(3), F(40)),
     "Berit is 55 years old. One screw costs 3 kr/piece and she buys 40 pieces. "
     "What is the total in kr?",
     "Berit is 55 years old. One screw costs 3 kr/piece and she buys 40 pieces. "
     "What does the whole batch cost her, in kr?"),
    ("unitprice", (F(240), F(16)),
     "The shop is 12 years old. A box costs 240 kr and holds 16 pieces. What is "
     "the price in kr/piece?",
     "The shop is 12 years old. A box costs 240 kr and holds 16 pieces. What "
     "does one single piece cost, in kr/piece?"),
]

EXTRACT = """Read this problem. List every quantity it states, and the unit of what
it asks for.

Problem: {story}

Reply with ONLY this JSON:
{{"given": [{{"value": "<number>", "unit": "<unit>"}}, ...], "asked_unit": "<unit>"}}
Units exactly as written in the problem (like m, s, m/s, kg, joule, kr/piece, m^2)."""


def parse_unit(u):
    return UNITS.get(u.strip().lower())


def mask_numbers(text):
    return re.sub(r"\d+(?:\.\d+)?", "N", text)


def extract_graph(story):
    reply = ask("qwen-35b", EXTRACT.format(story=story), n=140)
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return None, "no JSON"
    try:
        d = json.loads(m.group(0))
        given = [(F(str(g["value"]).strip()), parse_unit(str(g["unit"])),
                  str(g["value"]).strip()) for g in d["given"]]
        asked = parse_unit(str(d["asked_unit"]))
    except Exception as e:  # noqa: BLE001
        return None, f"parse: {type(e).__name__}"
    if asked is None or any(u is None for _, u, _ in given):
        return None, "unknown unit"
    for _, _, raw in given:                # every given must be literal in the text
        if raw not in story:
            return None, f"value {raw} not in the text"
    return (given, asked), "ok"


def unify(given, asked):
    fits = []
    for name, ins, out, fn, _ in FORMULAS:
        if out != asked:
            continue
        pool = [u for _, u, _ in given]
        ok = True
        for need in ins:
            if need in pool:
                pool.remove(need)
            else:
                ok = False
                break
        if ok:
            fits.append(name)
    return fits


def bind_and_solve(fname, given):
    name, ins, out, fn, _ = next(f for f in FORMULAS if f[0] == fname)
    pool = list(given)
    args = []
    for need in ins:
        idx = next(i for i, (_, u, _) in enumerate(pool) if u == need)
        args.append(pool.pop(idx)[0])
    return fn(*args)


def main(n_inspect=0, out="data/custom/formgraph.json"):
    n_inspect = int(n_inspect)
    battery = [(fname, vals, text, kind)
               for fname, vals, direct, indirect in STORIES
               for text, kind in ((direct, "direct"), (indirect, "indirect"))]

    # The embedding face of the library, one vector per wording.
    wtexts, wowner = [], []
    for name, _, _, _, wordings in FORMULAS:
        for w in wordings:
            wtexts.append(w)
            wowner.append(name)
    wvecs = embed(wtexts)

    if n_inspect:
        for fname, vals, text, kind in (battery[0], battery[7]):
            print(f"story ({fname}/{kind}): {text}")
            print(f"  RAW extract: {ask('qwen-35b', EXTRACT.format(story=text), n=140)!r}\n")
        return

    svecs = embed([mask_numbers(t) for t, in [(b[2],) for b in battery]])
    tally = {"embed_top1": 0, "embed_direct": 0, "embed_indirect": 0,
             "extract_ok": 0, "fit_unique": 0, "fit_multi": 0, "fit_empty": 0,
             "overrides": 0, "override_right": 0, "tiebreaks": 0,
             "tiebreak_right": 0, "delivered": 0, "right": 0, "wrong": 0,
             "flagged": 0}
    rows = []
    for i, (fname, vals, text, kind) in enumerate(battery):
        sims = [sum(a * b for a, b in zip(svecs[i], wv)) for wv in wvecs]
        order = sorted(range(len(sims)), key=lambda j: -sims[j])
        prop = wowner[order[0]]
        e_ok = prop == fname
        tally["embed_top1"] += e_ok
        tally["embed_direct" if kind == "direct" else "embed_indirect"] += e_ok

        graph, why = extract_graph(text)
        verdict, chosen = "flagged", None
        if graph:
            tally["extract_ok"] += 1
            given, asked = graph
            fits = unify(given, asked)
            if not fits:
                tally["fit_empty"] += 1
                why = "no formula fits the types"
            elif len(fits) == 1:
                tally["fit_unique"] += 1
                chosen = fits[0]
                if chosen != prop:
                    tally["overrides"] += 1
                    tally["override_right"] += chosen == fname
            else:
                tally["fit_multi"] += 1
                tally["tiebreaks"] += 1
                best = max((j for j in range(len(wowner)) if wowner[j] in fits),
                           key=lambda j: sims[j])
                chosen = wowner[best]
                tally["tiebreak_right"] += chosen == fname
        if chosen:
            truth_val = next(f[3] for f in FORMULAS if f[0] == fname)(*vals)
            got = bind_and_solve(chosen, given)
            tally["delivered"] += 1
            ok = chosen == fname and got == truth_val
            tally["right"] += ok
            tally["wrong"] += not ok
            verdict = f"{chosen} = {got}" + ("" if ok else " (WRONG)")
        else:
            tally["flagged"] += 1
        rows.append({"story": fname + "/" + kind, "embed": prop, "chosen": chosen,
                     "verdict": verdict, "why": why if chosen is None else ""})
        print(f"{fname + '/' + kind:<22} embed {prop:<10} -> {verdict}"
              f"{('  [' + why + ']') if chosen is None else ''}")

    n = len(battery)
    print(f"\nembedding alone : top-1 {tally['embed_top1']}/{n} "
          f"(direct {tally['embed_direct']}/12, indirect {tally['embed_indirect']}/12)")
    print(f"graph extraction: {tally['extract_ok']}/{n} verified literal")
    print(f"unification     : unique {tally['fit_unique']}, multi "
          f"{tally['fit_multi']}, empty {tally['fit_empty']}")
    print(f"graph overrode embedding {tally['overrides']} times, right "
          f"{tally['override_right']}; embedding tiebroke {tally['tiebreaks']} "
          f"multi-fits, right {tally['tiebreak_right']}")
    print(f"delivered {tally['delivered']}/{n}: right {tally['right']}, WRONG "
          f"{tally['wrong']}, flagged {tally['flagged']}")
    print("\nThe library holds both faces of every formula — how it is said, and what")
    print("shape it has — and authority flows one way: embeddings propose, the typed")
    print("graph disposes, and only where the types genuinely cannot say does the")
    print("embedding choose, inside the set the graph already approved.")
    summary = {"n": n, **tally, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
