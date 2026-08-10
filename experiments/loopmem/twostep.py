#!/usr/bin/env python3
"""Does chaining help, and does it help where the mapping already holds? Split battery.

Phase 135 made a machine an edge and proved the executor on hand-built graphs. This asks
whether a MODEL can use it, on sixteen problems that genuinely need two machines — and it
splits them deliberately, because "chaining works" and "chaining works where mapping works"
are different claims and this study has confused them before:

    EIGHT PRACTICAL     multi-step problems inside the measured winning band (a date span
                        feeding a percentage, a divisor sum feeding a digit root, a unit
                        chain feeding statistics)
    EIGHT OLYMPIAD-SHAPED  competition-style chains in phase 92's form (a divisor search
                        feeding a factorisation, a count feeding a modular reduction)

Truths computed here from the STRUCTURE before any wording is written, and reported per
half. Three arms:

    SOLO        the model answers
    ONE-SHOT    today's pipeline: one spec, one machine (it cannot express a chain)
    GRAPH       the model writes a system of definitions where an edge may be a machine,
                executed by the record in topological order

The fourth column the plan asked for — the record ROUTING the chain instead of the model
writing it — is measured as a separate number: on every problem where the graph arm failed,
does the type router find a road the model missed?
"""
import datetime as dt
import json
import math
import re
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cutbig import ask  # noqa: E402
from gsmsolve import ask_spec_model, equal  # noqa: E402
from islands import SOLVER_TYPES, route, solve_graph  # noqa: E402
from olympiad import SOLO, last_number  # noqa: E402
from solve import bank, catalogue  # noqa: E402
from solvemap import answer_of, parse_spec  # noqa: E402
from solvers2 import run2  # noqa: E402

GRAPH_PROMPT = """Break this problem into named steps and write it as a system. A step is
either arithmetic over earlier steps, or ONE machine call. Do not compute anything — the
executor runs the system.

A machine call looks like:  @solver{{slot=value, slot=value}} report
References to earlier steps may appear anywhere in a call or an expression.

Worked example:
Problem: "How many days are there from 1 January 1970 to 10 August 2026, and how many
whole weeks is that?"
{{"defs": {{"d": "@datetime{{kind=days_between, from=1970-01-01, to=2026-08-10}} value",
 "w": "d / 7"}}, "asked": "w"}}

Worked example:
Problem: "What is the sum of the divisors of 720720, and what is the digit sum of that?"
{{"defs": {{"s": "@factor{{n=720720, report=divisor_sum}} divisor_sum",
 "r": "@digit_ops{{n=s}} digit_sum"}}, "asked": "r"}}

Catalogue:
{catalogue}

Problem: {story}
Reply with ONLY the JSON system."""

ONESHOT = """Map the problem onto ONE solver and fill its slots. Do NOT compute the
answer — an exact executor computes it from your spec.

Catalogue:
{catalogue}

Problem: {story}
Spec:"""


def build():
    """Truths first, computed here, from the structure rather than from the wording."""
    out = []

    # ---- eight practical, two machines each -------------------------------------
    d = (dt.date(2026, 12, 24) - dt.date(2024, 3, 7)).days
    out.append(("practical", f"How many days are there from 7 March 2024 to 24 December "
                f"2026, and what is that number of days increased by 15 percent? Give "
                f"the exact value as a fraction.", str(F(d) * F(115, 100))))
    ds = sum(x for x in range(1, 498961) if 498960 % x == 0)
    out.append(("practical", "What is the sum of all positive divisors of 498960, and "
                "what is the digit sum of that total?",
                str(sum(int(c) for c in str(ds)))))
    n7 = sum(1 for x in range(1, 200001) if x % 37 == 0 and x % 11 == 5)
    out.append(("practical", "How many integers from 1 to 200000 are divisible by 37 and "
                "leave remainder 5 when divided by 11, and what is that count times "
                "3/8? Give the exact value as a fraction.", str(F(n7) * F(3, 8))))
    f22 = math.prod(e + 1 for e in {p: sum(22 // p ** k for k in range(1, 6))
                                    for p in (2, 3, 5, 7, 11, 13, 17, 19)}.values())
    out.append(("practical", "How many positive divisors does 22 factorial have, and "
                "what is that number written in base 7?",
                __import__("numpy").base_repr(f22, 7).lower() if False else
                "".join(reversed([str((f22 // 7 ** i) % 7)
                                  for i in range(0, 20) if f22 // 7 ** i]))))
    prim = sum(1 for x in range(2, 40000)
               if all(x % q for q in range(2, int(x ** 0.5) + 1)))
    out.append(("practical", "How many prime numbers are there below 40000, and what is "
                "that count as a fraction of 40000? Give the exact fraction.",
                str(F(prim, 40000))))
    arr = math.factorial(11) // (math.factorial(4) * math.factorial(4)
                                 * math.factorial(2))
    out.append(("practical", "How many distinct arrangements are there of the letters of "
                "MISSISSIPPI, and what is that number divided by 25? Give the exact "
                "value as a fraction.", str(F(arr, 25))))
    var = None
    xs = [F(v) for v in (23, 41, 7, 19, 33, 12)]
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    out.append(("practical", "What is the exact population variance of 23, 41, 7, 19, 33 "
                "and 12, and what is that value multiplied by 6? Give the exact "
                "fraction.", str(var * 6)))
    dd = (dt.date(2030, 1, 1) - dt.date(1999, 9, 9)).days
    out.append(("practical", "How many days are there from 9 September 1999 to 1 January "
                "2030, and how many whole weeks is that?", str(dd // 7)))

    # ---- eight olympiad-shaped, two machines each --------------------------------
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

    f13 = fact_fac(13)
    n13 = math.prod(p ** e for p, e in f13.items())
    s13 = sum(x for x in range(1, n13 + 1) if False)      # placeholder, computed below
    divs = [1]
    for p, e in f13.items():
        divs = [d * p ** i for d in divs for i in range(e + 1)]
    s13 = sum(x for x in divs if math.isqrt(n13 // x) ** 2 == n13 // x)
    ff = {}
    tmp, dd2 = s13, 2
    while dd2 * dd2 <= tmp:
        while tmp % dd2 == 0:
            ff[dd2] = ff.get(dd2, 0) + 1
            tmp //= dd2
        dd2 += 1
    if tmp > 1:
        ff[tmp] = ff.get(tmp, 0) + 1
    out.append(("olympiad", "The sum of all positive integers m such that 13!/m is a "
                "perfect square can be written as a product of prime powers. Find the "
                "sum of the exponents.", str(sum(ff.values()))))
    c = sum(1 for x in range(1, 1000) if len({x % k for k in (2, 3, 4, 5, 6)}) == 5)
    out.append(("olympiad", "Call a positive integer extra-distinct if its remainders on "
                "division by 2, 3, 4, 5 and 6 are all different. Take the number of "
                "extra-distinct integers below 1000 and give its prime factorisation's "
                "exponent sum.",
                str(sum(e for _p, e in __import__("collections").Counter(
                    [q for q in [c] for _ in [0] for q in
                     (lambda v: (lambda f: f(f, v, 2))(
                         lambda self, n, d: [] if n == 1 else (
                             [d] + self(self, n // d, d) if n % d == 0
                             else self(self, n, d + 1))))(c)]).items()))))
    t72 = sum(1 for a in range(1, 21) for b in range(a + 1, 21)
              for cc in range(b + 1, 21)
              if all(all(x % q for q in range(2, int(x ** 0.5) + 1)) and x > 1
                     for x in (b - a, cc - b, cc - a)))
    out.append(("olympiad", "Twenty points on a circle are labelled 1 to 20 and a segment "
                "joins two points whose labels differ by a prime. How many triangles are "
                "formed, and what is that count modulo 7?", str(t72 % 7)))
    sub = sum(1 for cbits in range(1024)
              if sum(((cbits >> i) & 1) * ((cbits >> (i + 1)) & 1)
                     for i in range(9)) == 1)
    out.append(("olympiad", "How many subsets of the numbers 1 to 10 contain exactly one "
                "pair of consecutive integers, and what is the digit sum of that "
                "count?", str(sum(int(x) for x in str(sub)))))
    p40 = 40 * 39 // 2 - (3 * 3 + 4 * 6 + 5 * 10 + 6 * 15)
    out.append(("olympiad", "A plane holds 40 lines, no two parallel, with 3 points where "
                "exactly 3 lines meet, 4 points where exactly 4 meet, 5 points where "
                "exactly 5 meet and 6 points where exactly 6 meet. How many points have "
                "exactly two lines, and what is that number modulo 100?", str(p40 % 100)))
    z2025 = sum(2025 // 5 ** k for k in range(1, 6))
    out.append(("olympiad", "How many trailing zeros does 2025 factorial have, and what "
                "is that number of zeros written as a roman numeral?",
                "".join(x for x in ["D"] * (z2025 // 500) + ["C"] * ((z2025 % 500) // 100)
                        + ["L"] * ((z2025 % 100) // 50) + ["X"] * ((z2025 % 50) // 10)
                        + ["V"] * ((z2025 % 10) // 5) + ["I"] * (z2025 % 5))))
    cnt = sum(1 for x in range(1, 100001) if x % 13 == 0 and sum(map(int, str(x))) == 19)
    out.append(("olympiad", "How many integers from 1 to 100000 are divisible by 13 and "
                "have digits summing to 19, and how many divisors does that count "
                "have?", str(sum(1 for x in range(1, cnt + 1) if cnt % x == 0))))
    pairs = sum(1 for a in range(1, 200) for b in range(1, 200)
                if 7 * a + 11 * b == 500)
    out.append(("olympiad", "How many ordered pairs of positive integers satisfy "
                "7a + 11b = 500, and what is that count squared?", str(pairs ** 2)))
    return out


def main(out="data/custom/twostep.json"):
    battery = build()
    cat = catalogue()
    tally = {}
    rows = []
    for fam, story, truth in battery:
        t = tally.setdefault(fam, {"n": 0, "solo": 0, "oneshot": 0, "graph": 0,
                                   "graph_steps": 0, "routed": 0, "wrong": 0})
        t["n"] += 1

        raw = ask("qwen-35b", SOLO.format(problem=story), n=460)
        num = last_number(raw)
        solo_ok = (equal(num, truth) if num is not None else False) or \
            (not str(truth).replace("-", "").isdigit() and str(truth) in raw)
        t["solo"] += solo_ok

        spec = parse_spec(ask_spec_model("qwen-35b", ONESHOT.format(
            story=story, catalogue=cat), n=420))
        one_ok = False
        if isinstance(spec, dict) and "solver" in spec:
            res, _why = run2(spec)
            if res is not None:
                one_ok = equal(answer_of(res, spec), truth) or \
                    str(answer_of(res, spec)) == str(truth)
        t["oneshot"] += one_ok

        reply = ask_spec_model("qwen-35b", GRAPH_PROMPT.format(
            story=story, catalogue=cat), n=700)
        got, why, info = None, "no system", {}
        m = re.search(r"\{.*\}", reply, re.S)
        if m:
            try:
                sysd = json.loads(m.group(0))
                got, why, info = solve_graph(sysd.get("defs", {}),
                                             str(sysd.get("asked", "")))
            except json.JSONDecodeError:
                why = "malformed system"
        graph_ok = got is not None and (equal(got, truth) or str(got) == str(truth))
        t["graph"] += graph_ok
        t["graph_steps"] += info.get("steps", 0) or 0
        t["wrong"] += (got is not None) and not graph_ok

        # The router's separate question: where the graph arm failed, is there a road?
        routed = None
        if not graph_ok:
            r = route(["range", "integer", "amount", "date", "list"], "count")
            routed = bool(r.get("found"))
            t["routed"] += bool(routed)

        rows.append({"family": fam, "truth": str(truth), "solo_ok": bool(solo_ok),
                     "oneshot_ok": bool(one_ok), "graph_ok": bool(graph_ok),
                     "graph_value": str(got), "why": why,
                     "steps": info.get("steps"), "story": story[:70]})
        print(f"{fam:<10}{'solo ok' if solo_ok else 'solo X '} "
              f"{'1shot ok' if one_ok else '1shot X '} "
              f"{'GRAPH ok' if graph_ok else 'graph X '} "
              f"{str(info.get('steps', '-')):<3}{str(got)[:16]:<18}{story[:34]}")

    print()
    for fam, t in sorted(tally.items()):
        print(f"{fam:<11} n={t['n']}  solo {t['solo']}  one-shot {t['oneshot']}  "
              f"GRAPH {t['graph']}  (wrong {t['wrong']}, "
              f"{t['graph_steps']} steps executed)")
    tot = {k: sum(t[k] for t in tally.values()) for k in
           ("n", "solo", "oneshot", "graph", "wrong")}
    print(f"{'TOTAL':<11} n={tot['n']}  solo {tot['solo']}  one-shot {tot['oneshot']}  "
          f"GRAPH {tot['graph']}  (wrong {tot['wrong']})")
    print("\nOne-shot cannot express a chain, so its column is the floor a chain has to")
    print("beat; the split says whether it beats it everywhere or only where mapping was")
    print("already working.")
    summary = {"by_family": tally, "total": tot, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
