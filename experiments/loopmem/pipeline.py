#!/usr/bin/env python3
"""The recommended policy, run live on problems written for this phase alone.

Phase 112 derived the deployment rule from stored outcomes: map everything, execute what
validates, and let the model answer exactly what the record declined. A rule derived from
a table is a hypothesis; this runs it as a pipeline, on twenty-four problems generated
here with truths computed before any model call, and reports what each part contributed.

  MAP        retrieval picks two exemplars, the model writes one spec, the record runs it
  FALL BACK  only where the record REFUSED — never where it answered, because a wrong
             answer and a refusal are different things and conflating them is how a
             fallback turns into a coin flip
  BASELINES  the model answering alone, and the machinery with no fallback at all

Twenty-four problems across eight classes, none of them reused from an earlier battery.
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
from cutbig import ask  # noqa: E402
from embednav import embed  # noqa: E402
from gsmsolve import ARITH_SCHEMA, ask_spec_model, equal  # noqa: E402
from mapmemory import mask  # noqa: E402
from mixedretr import BANK  # noqa: E402
from newbands import SCHEMA_NEW  # noqa: E402
from olympiad import SOLO, last_number  # noqa: E402
from solvemap import PREDICATE_HELP, SCHEMAS, answer_of, parse_spec  # noqa: E402
from solvers2 import run2  # noqa: E402
from solvers3 import parse_units  # noqa: E402

PROMPT = """Map the problem onto ONE solver and fill its slots. Do NOT compute the
answer — an exact executor computes it from your spec.

{examples}

Catalogue:
{catalogue}

For the search solver, conditions use these ops: {preds}
Expressions may call: {funcs}   (^ is a power; / is exact rational division)

Problem: {story}
Spec:"""


def build():
    """Twenty-four fresh problems, truths computed here."""
    out = []
    v = F(7, 3)
    for _ in range(13):
        v = (v * F(4, 7) + F(2, 9))
    out.append(("fold", "Start with 7/3. Thirteen times in a row, multiply by 4/7 and "
                "then add 2/9. Give the exact result as a fraction.", str(v)))
    w = F(2000)
    for k in range(1, 15):
        w *= 1 + F((-1) ** k, k + 5)
    out.append(("fold", "A balance starts at 2000. In step k, for k = 1 to 14, it is "
                "multiplied by (1 + (-1)^k / (k+5)). Give the exact final balance.",
                str(w)))
    out.append(("big", "What is 246813579 times 864297531?",
                str(246813579 * 864297531)))
    out.append(("big", "What is the remainder when 13 to the power 909 is divided by "
                "1000000007?", str(pow(13, 909, 1000000007))))
    out.append(("big", "What is the greatest common divisor of 123456789098765432 and "
                "864209753086420975?",
                str(math.gcd(123456789098765432, 864209753086420975))))
    c1 = sum(1 for n in range(1, 700001) if n % 23 == 0 and n % 13 == 8)
    out.append(("count", "How many integers from 1 to 700000 are divisible by 23 and "
                "leave remainder 8 when divided by 13?", str(c1)))
    c2 = sum(n for n in range(1, 120001) if n % 17 == 0 and sum(map(int, str(n))) == 22)
    out.append(("count", "What is the sum of all integers from 1 to 120000 that are "
                "divisible by 17 and whose digits add to 22?", str(c2)))
    c3 = sum(1 for n in range(1, 400001) if math.isqrt(n) ** 2 == n and n % 3 == 1)
    out.append(("count", "How many integers from 1 to 400000 are perfect squares that "
                "leave remainder 1 when divided by 3?", str(c3)))

    def fact_fac(k):
        f = {}
        for p in range(2, k + 1):
            if all(p % q for q in range(2, int(p ** 0.5) + 1)):
                e, q = 0, p
                while q <= k:
                    e += k // q
                    q *= p
                f[p] = e
        return f

    f27 = fact_fac(27)
    out.append(("divisors", "How many positive divisors does 27 factorial have?",
                str(math.prod(e + 1 for e in f27.values()))))
    out.append(("divisors", "What is the sum of the exponents in the prime "
                "factorisation of 35 factorial?", str(sum(fact_fac(35).values()))))
    out.append(("divisors", "What is the sum of all positive divisors of 831600?",
                str(sum(d for d in range(1, 831601) if 831600 % d == 0))))

    import itertools
    for nv, lo, hi, ev, text in [
        (2, 1, 9, lambda a, b: a + b == 11,
         "Two numbers are picked independently and uniformly from 1 to 9. What is the "
         "exact probability their sum is 11? Give a fraction."),
        (3, 1, 4, lambda a, b, c: a + b + c == 8,
         "Three numbers are picked independently and uniformly from 1 to 4. What is "
         "the exact probability their sum is 8? Give a fraction."),
        (2, 1, 14, lambda a, b: math.gcd(a, b) == 2,
         "Two numbers are picked independently and uniformly from 1 to 14. What is the "
         "exact probability their greatest common divisor is exactly 2? Give a "
         "fraction."),
    ]:
        space = list(itertools.product(range(lo, hi + 1), repeat=nv))
        fav = [t for t in space if ev(*t)]
        out.append(("probability", text, str(F(len(fav), len(space)))))

    from bricks import build_registry, route
    reg = build_registry()
    for val, src, dst, text in [
        (17, "mile/hour", "cm/second", "A cyclist rides at 17 miles per hour. Exactly "
         "how many centimetres per second is that? Give a fraction."),
        (23, "litre", "gallon", "A barrel holds 23 litres. Exactly how many gallons is "
         "that? Give a fraction."),
        (4, "kg/litre", "ounce/gallon", "A liquid has density 4 kilograms per litre. "
         "Exactly what is that in ounces per gallon? Give a fraction."),
    ]:
        res, _ = route(reg, parse_units(src), parse_units(dst))
        out.append(("convert", text, str(F(val) * res[0])))

    for vals, rep, text in [
        ([23, 41, 7, 19, 33, 12], "population_variance",
         "What is the exact population variance of 23, 41, 7, 19, 33 and 12? Give a "
         "fraction."),
        ([88, 74, 96, 59, 81], "mean",
         "What is the exact mean of 88, 74, 96, 59 and 81? Give a fraction."),
    ]:
        xs = [F(v) for v in vals]
        m = sum(xs) / len(xs)
        val = m if rep == "mean" else sum((x - m) ** 2 for x in xs) / len(xs)
        out.append(("statistics", text, str(val)))

    out.append(("datetime", "How many days are there from 14 July 1789 to 9 November "
                "1989?", str((dt.date(1989, 11, 9) - dt.date(1789, 7, 14)).days)))
    out.append(("datetime", "What day of the week was 20 July 1969?",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
                 "Sunday"][dt.date(1969, 7, 20).weekday()]))
    out.append(("datetime", "What is the date 2500 days after 5 June 2011? Give it as "
                "YYYY-MM-DD.",
                (dt.date(2011, 6, 5) + dt.timedelta(days=2500)).isoformat()))

    for start, changes, text in [
        (1750, [23, -17, 9, -31, 12], "A holding of 1750 changes by +23 percent, then "
         "-17 percent, then +9 percent, then -31 percent, then +12 percent. What is it "
         "exactly now? Give a fraction."),
        (640, [-9] * 6, "A 640 kr item is discounted 9 percent six times in a row. "
         "What is the exact final price? Give a fraction."),
    ]:
        v2 = F(start)
        for c in changes:
            v2 *= 1 + F(c) / 100
        out.append(("finance", text, str(v2)))
    return out


def main(k=2, out="data/custom/pipeline.json"):
    k = int(k)
    battery = build()
    catalogue = "\n".join(f"- {v}" for v in
                          {"arith": ARITH_SCHEMA, **SCHEMA_3, **SCHEMA_MORE,
                           **SCHEMA_NEW, **SCHEMAS, **SCHEMAS2}.values())
    bvecs = embed([mask(p) for _t, p, _s in BANK])
    pvecs = embed([mask(s) for _f, s, _t in battery])

    t = {key: 0 for key in ("solo", "machine", "pipeline", "refused", "rescued",
                            "wrong_kept")}
    rows = []
    for i, (fam, story, truth) in enumerate(battery):
        sims = [sum(a * b for a, b in zip(pvecs[i], bv)) for bv in bvecs]
        order = sorted(range(len(BANK)), key=lambda j: -sims[j])[:k]
        examples = "\n\n".join('Example: "' + BANK[j][1] + '"\nSpec: ' + BANK[j][2]
                               for j in order)
        spec = parse_spec(ask_spec_model(
            "qwen-35b", PROMPT.format(story=story, catalogue=catalogue,
                                      examples=examples, preds=PREDICATE_HELP,
                                      funcs=EXPR_FUNCS_HELP), n=420))
        got, machine_ok, refused = None, False, True
        if isinstance(spec, dict) and "solver" in spec:
            res, why = run2(spec)
            if res is not None:
                refused = False
                got = answer_of(res, spec)
                machine_ok = str(got) == str(truth) or equal(got, truth)
        t["machine"] += machine_ok
        t["refused"] += refused

        raw = ask("qwen-35b", SOLO.format(problem=story), n=420)
        num = last_number(raw)
        solo_ok = (equal(num, truth) if num is not None else False) or \
            (not str(truth).replace("-", "").isdigit() and str(truth) in raw)
        t["solo"] += solo_ok

        pipe_ok = solo_ok if refused else machine_ok
        t["pipeline"] += pipe_ok
        t["rescued"] += refused and solo_ok
        t["wrong_kept"] += (not refused) and (not machine_ok)
        rows.append({"family": fam, "truth": str(truth), "machine": str(got),
                     "machine_ok": machine_ok, "refused": refused,
                     "solo_ok": bool(solo_ok), "pipeline_ok": bool(pipe_ok)})
        print(f"{fam:<12}{'refused ' if refused else ('mach ok ' if machine_ok else 'mach X  ')}"
              f"{'solo ok' if solo_ok else 'solo X '}  -> "
              f"{'RIGHT' if pipe_ok else 'wrong':<6}{story[:36]}")

    n = len(battery)
    print(f"\n{n} fresh problems, eight classes, none reused\n")
    print(f"{'the model alone':<34}{t['solo']:>5}/{n}")
    print(f"{'the machinery alone':<34}{t['machine']:>5}/{n}")
    print(f"{'the policy (model on refusal)':<34}{t['pipeline']:>5}/{n}")
    print(f"\nrecord refused {t['refused']}, of which the model rescued {t['rescued']}; "
          f"machine answers kept but wrong: {t['wrong_kept']}")
    print("\nThe policy is worth exactly the refusals the model can rescue, and costs")
    print("exactly the wrong answers the record was confident enough to produce. Both")
    print("numbers are above, on problems written after the rule was.")
    summary = {"n": n, **t, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
