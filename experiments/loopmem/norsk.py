#!/usr/bin/env python3
"""Does the pipeline survive the problem being in Norwegian?

Every battery so far has been in English, which is a hidden assumption rather than a
measured property: the catalogue is English, the exemplars are English, and the mapping
step is the only place language enters. If a Norwegian problem maps as well as an English
one, then what the model is doing is reading structure rather than matching surface — and
if it does not, the deployment note is that the bank must be written in the user's
language.

Eighteen problems across six classes, each a faithful Norwegian rendering of a problem
shape the batteries already cover, with new numbers and truths computed here. Nothing
else changes: the same English catalogue, the same English exemplars, the same retrieval,
the same model. The comparison is against the same six classes measured in English at
grandmix, and against the model answering the Norwegian problems alone.
"""
import datetime as dt
import json
import math
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aimecover import EXPR_FUNCS_HELP, SCHEMAS2  # noqa: E402
from bands2 import SCHEMA_MORE  # noqa: E402
from bands3 import SCHEMA_3  # noqa: E402
from bands4 import SCHEMA_4  # noqa: E402
from bands5 import SCHEMA_5  # noqa: E402
from cutbig import ask  # noqa: E402
from embednav import embed  # noqa: E402
from gsmsolve import ARITH_SCHEMA, ask_spec_model, equal  # noqa: E402
from mapmemory import mask  # noqa: E402
from mixedretr import BANK  # noqa: E402
from newbands import SCHEMA_NEW  # noqa: E402
from olympiad import SOLO, last_number  # noqa: E402
from solvemap import PREDICATE_HELP, SCHEMAS, answer_of, parse_spec  # noqa: E402
from solvers2 import run2  # noqa: E402

PROMPT = """Map the problem onto ONE solver and fill its slots. Do NOT compute the
answer — an exact executor computes it from your spec. The problem may be in Norwegian;
the spec is always in the format below.

{examples}

Catalogue:
{catalogue}

For the search solver, conditions use these ops: {preds}
Expressions may call: {funcs}   (^ is a power; / is exact rational division)

Problem: {story}
Spec:"""


def build():
    """Eighteen Norwegian problems, truths computed here."""
    out = []

    out.append(("count", "Hvor mange heltall fra 1 til 300000 er delelige med 19 og "
                "gir rest 4 når de deles på 8?",
                str(sum(1 for n in range(1, 300001) if n % 19 == 0 and n % 8 == 4))))
    out.append(("count", "Hva er summen av alle heltall fra 1 til 90000 som er "
                "delelige med 12 og har tverrsum 15?",
                str(sum(n for n in range(1, 90001)
                        if n % 12 == 0 and sum(map(int, str(n))) == 15))))
    out.append(("count", "Hvor mange heltall fra 1 til 250000 er kvadrattall?",
                str(math.isqrt(250000))))

    out.append(("divisors", "Hvor mange positive divisorer har 22 fakultet?",
                str(math.prod(e + 1 for e in {
                    p: sum(22 // p ** k for k in range(1, 6))
                    for p in (2, 3, 5, 7, 11, 13, 17, 19)}.values()))))
    out.append(("divisors", "Hva er summen av alle positive divisorer av 277200?",
                str(sum(d for d in range(1, 277201) if 277200 % d == 0))))
    out.append(("divisors", "Hvor mange positive divisorer har 15120?",
                str(sum(1 for d in range(1, 15121) if 15120 % d == 0))))

    import itertools
    for nv, lo, hi, ev, text in [
        (2, 1, 6, lambda a, b: a + b == 8,
         "To rettferdige terninger kastes. Hva er den eksakte sannsynligheten for at "
         "summen blir 8? Svar med en brøk."),
        (2, 1, 10, lambda a, b: math.gcd(a, b) == 1,
         "To tall trekkes uavhengig og tilfeldig fra 1 til 10. Hva er den eksakte "
         "sannsynligheten for at de er innbyrdes primiske? Svar med en brøk."),
        (3, 1, 3, lambda a, b, c: a + b + c == 6,
         "Tre tall trekkes uavhengig og tilfeldig fra 1 til 3. Hva er den eksakte "
         "sannsynligheten for at summen blir 6? Svar med en brøk."),
    ]:
        space = list(itertools.product(range(lo, hi + 1), repeat=nv))
        fav = [t for t in space if ev(*t)]
        out.append(("probability", text, str(F(len(fav), len(space)))))

    for vals, text in [
        ([34, 19, 47, 22, 58], "Hva er det eksakte gjennomsnittet av 34, 19, 47, 22 "
         "og 58? Svar med en brøk."),
        ([12, 25, 8, 31, 17, 44], "Hva er den eksakte populasjonsvariansen til 12, 25, "
         "8, 31, 17 og 44? Svar med en brøk."),
    ]:
        xs = [F(v) for v in vals]
        m = sum(xs) / len(xs)
        if "gjennomsnittet" in text:
            out.append(("statistics", text, str(m)))
        else:
            out.append(("statistics", text,
                        str(sum((x - m) ** 2 for x in xs) / len(xs))))
    out.append(("statistics", "Hva er medianen av 41, 17, 63, 29, 8, 52 og 35?",
                str(sorted([41, 17, 63, 29, 8, 52, 35])[3])))

    out.append(("datetime", "Hvor mange dager er det fra 17. mai 1814 til 17. mai "
                "2026?", str((dt.date(2026, 5, 17) - dt.date(1814, 5, 17)).days)))
    out.append(("datetime", "Hvilken ukedag var 9. april 1940? Svar på engelsk, for "
                "eksempel Monday.",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
                 "Sunday"][dt.date(1940, 4, 9).weekday()]))
    out.append(("datetime", "Hvilken dato er det 1500 dager etter 1. januar 2020? Svar "
                "på formen YYYY-MM-DD.",
                (dt.date(2020, 1, 1) + dt.timedelta(days=1500)).isoformat()))

    for start, changes, text in [
        (4200, [18, -23, 11], "En beholdning på 4200 endrer seg med +18 prosent, så "
         "-23 prosent, så +11 prosent. Hva er den eksakt nå? Svar med en brøk."),
        (950, [-14, -14], "En vare til 950 kroner settes ned 14 prosent to ganger på "
         "rad. Hva er den eksakte prisen? Svar med en brøk."),
        (100, [7] * 12, "En verdi på 100 vokser 7 prosent i året i 12 år. Hva er den "
         "eksakte verdien til slutt? Svar med en brøk."),
    ]:
        v = F(start)
        for c in changes:
            v *= 1 + F(c) / 100
        out.append(("finance", text, str(v)))
    return out


def main(k=2, out="data/custom/norsk.json"):
    k = int(k)
    battery = build()
    catalogue = "\n".join(f"- {v}" for v in
                          {"arith": ARITH_SCHEMA, **SCHEMA_5, **SCHEMA_4, **SCHEMA_3,
                           **SCHEMA_MORE, **SCHEMA_NEW, **SCHEMAS,
                           **SCHEMAS2}.values())
    bvecs = embed([mask(p) for _t, p, _s in BANK])
    pvecs = embed([mask(s) for _f, s, _t in battery])

    t = {key: 0 for key in ("solo", "mpeqs", "parsed", "ran", "wrong",
                            "retrieval_hit")}
    byfam = {}
    rows = []
    for i, (fam, story, truth) in enumerate(battery):
        sims = [sum(a * b for a, b in zip(pvecs[i], bv)) for bv in bvecs]
        order = sorted(range(len(BANK)), key=lambda j: -sims[j])[:k]
        t["retrieval_hit"] += fam in [BANK[j][0] for j in order]
        examples = "\n\n".join('Example: "' + BANK[j][1] + '"\nSpec: ' + BANK[j][2]
                               for j in order)
        raw = ask("qwen-35b", SOLO.format(problem=story), n=420)
        num = last_number(raw)
        solo_ok = (equal(num, truth) if num is not None else False) or \
            (not str(truth).replace("-", "").isdigit() and str(truth) in raw)
        t["solo"] += solo_ok

        spec = parse_spec(ask_spec_model(
            "qwen-35b", PROMPT.format(story=story, catalogue=catalogue,
                                      examples=examples, preds=PREDICATE_HELP,
                                      funcs=EXPR_FUNCS_HELP), n=420))
        got, ok = None, False
        if isinstance(spec, dict) and "solver" in spec:
            t["parsed"] += 1
            res, why = run2(spec)
            if res is not None:
                t["ran"] += 1
                got = answer_of(res, spec)
                ok = str(got) == str(truth) or equal(got, truth)
                t["mpeqs"] += ok
                t["wrong"] += not ok
            else:
                got = why[:34]
        f = byfam.setdefault(fam, [0, 0, 0])
        f[0] += 1
        f[1] += solo_ok
        f[2] += ok
        rows.append({"family": fam, "truth": str(truth), "solo_ok": bool(solo_ok),
                     "mpeqs": str(got), "mpeqs_ok": bool(ok),
                     "solver": (spec or {}).get("solver")})
        print(f"{fam:<12}{'solo ok' if solo_ok else 'solo X '} "
              f"{'mpeqs ok' if ok else 'mpeqs X '} {str((spec or {}).get('solver')):<13}"
              f"{story[:36]}")

    n = len(battery)
    print(f"\nNorwegian problems, English catalogue and exemplars; the needed class was "
          f"retrieved for {t['retrieval_hit']}/{n}")
    print(f"SOLO-35B : {t['solo']}/{n}")
    print(f"MPEqs    : {t['mpeqs']}/{n}  (parsed {t['parsed']}, ran {t['ran']}, "
          f"wrong {t['wrong']})")
    for fam, (cnt, so, mp) in sorted(byfam.items()):
        print(f"  {fam:<12}{cnt:>3}{so:>6}{mp:>7}")
    print("\nThe mapping step is the only place language enters, and this is the only")
    print("measurement that says whether it reads structure or surface.")
    summary = {"n": n, **t, "byfam": byfam, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
