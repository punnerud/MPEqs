#!/usr/bin/env python3
"""Dimension control on retrieved plans: refuse the ill-typed ones before binding executes.

Phase 51's lookup failed two ways at once — topic-not-type retrieval and role-scrambled
positional binding — and phase 52 built the control that catches a class of both: quantities
carry units, addition demands agreement, and the answer's unit must be what the question asked
for. This wires that control between retrieval and execution.

The units come from the text, mechanically: the noun following each number ("48 clips" makes a
48 of unit clips, "$5" a 5 of dollars, "3 times" a dimensionless 3). That tagging is heuristic
and its quality is MEASURED before it is used — every stored template is re-checked against its
own source problem, and the self-pass rate is reported, because a filter that refuses its own
training data would be refusing on noise.

Then the same lookup as phase 51, with the check in the loop:

  UNFILTERED   nearest compatible template, positional binding      (known: 0/60)
  DIM-FILTER   walk the retrieval list, bind each candidate WITH units, refuse any plan
               whose steps mix distinct known units under + or - or whose result unit is not
               the asked one; execute the first survivor
  DIM + VOTE   majority answer among the type-clean top 25          (unfiltered vote: 1/60)

What the check cannot catch is declared up front: a scramble among same-unit numbers types
perfectly (three clip-counts permuted still add as clips), so this is a filter on a class of
garbage, not a proof of rightness — the same asymmetry as every verifier here.
"""
import ast
import json
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from embednav import embed  # noqa: E402
from mapstore import NUM, TEST, build_store, mask, norm, run_plan  # noqa: E402

DIMLESS = {"times", "time", "percent", "%", "half", "twice", "x"}
SKIP = {"of", "more", "fewer", "less", "extra", "additional", "new", "other", "the", "his",
        "her", "their", "its", "each", "per", "total", "different", "separate", "such",
        "as", "many", "much"}


def singular(w):
    w = w.lower().strip(".,;:!?'\"")
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 4 and w[-3] in "sxh":
        return w[:-2]
    if w.endswith("s") and len(w) > 3:
        return w[:-1]
    return w


def units_of(text):
    """A unit tag per number, in order: the first plain noun after it, or dollar, or none."""
    text = norm(text)
    out = []
    for m in NUM.finditer(text):
        if m.start() > 0 and text[m.start() - 1] == "$":
            out.append("dollar")
            continue
        tail = re.findall(r"[A-Za-z%]+", text[m.end():m.end() + 60])
        unit = None
        for w in tail[:4]:
            lw = w.lower()
            if lw in DIMLESS:
                unit = None
                break
            if lw in SKIP:
                continue
            unit = singular(w)
            break
        out.append(unit)
    return out


def asked_unit(question):
    """What the question wants: the noun after 'how many', or dollars for 'how much money'."""
    q = norm(question)
    m = re.search(r"how many ([a-z]+(?: [a-z]+)?)", q, re.I)
    if m:
        for w in m.group(1).split():
            if w.lower() not in SKIP:
                return singular(w)
    if re.search(r"how much", q, re.I) and "$" in q:
        return "dollar"
    return None


class Ill(Exception):
    """The step mixes units that cannot mix."""


def type_expr(node, env):
    """The unit of an expression, or Ill. Unknown units are permissive and counted upstream."""
    if isinstance(node, ast.BinOp):
        a, b = type_expr(node.left, env), type_expr(node.right, env)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            if a is not None and b is not None and a != b:
                raise Ill(f"{a} +/- {b}")
            return a or b
        if isinstance(node.op, ast.Mult):
            if a is None:
                return b
            if b is None:
                return a
            return f"{a}*{b}"
        if isinstance(node.op, ast.Div):
            if a == b:
                return None
            if b is None:
                return a
            return f"{a}/{b}"
        return None
    if isinstance(node, ast.UnaryOp):
        return type_expr(node.operand, env)
    if isinstance(node, ast.Constant):
        return None
    if isinstance(node, ast.Name):
        return env.get(node.id)
    raise Ill("unparseable")


def plan_types_ok(steps, var_units, want):
    """Every step typed; the last step's unit must be the asked one when both are known."""
    env = dict(var_units)
    last = None
    for i, s in enumerate(steps):
        try:
            tree = ast.parse(s, mode="eval").body
            last = type_expr(tree, env)
        except Ill:
            return False
        except SyntaxError:
            return True                          # cannot parse: cannot check, let it through
        env[f"S{i}"] = last
    if want is not None and last is not None and "*" not in str(last) \
            and "/" not in str(last) and last != want:
        return False
    return True


def main(n_test=60, seed=5, out="data/custom/dimcheck.json"):
    import random
    n_test, seed = int(n_test), int(seed)
    store, kept, _, _ = build_store(2000)

    # The tagger judged against the store itself: does each template's own source pass?
    self_pass = 0
    for t in store:
        vu = {f"v{k + 1}": u for k, u in enumerate(units_of(t["question"]))}
        self_pass += plan_types_ok(t["steps"], vu, asked_unit(t["question"]))
    print(f"store: {kept} templates; {self_pass} pass the dimension check on their own "
          f"source ({100 * self_pass / kept:.0f}% — the tagger's noise floor)")

    vecs = np.array(embed([t["masked"] for t in store]), dtype=np.float32)
    rng = random.Random(seed)
    tests = []
    for line in TEST.read_text().splitlines():
        d = json.loads(line)
        tests.append((d["question"],
                      Fraction(norm(d["answer"].rsplit("#### ", 1)[-1].strip()))))
    tests = rng.sample(tests, n_test)
    qvecs = np.array(embed([mask(norm(q)) for q, _ in tests]), dtype=np.float32)

    plain = filt = vote = 0
    refused_total = 0
    rows = []
    for i, (q, truth) in enumerate(tests):
        qnums = [norm(m) for m in NUM.findall(norm(q))]
        qunits = units_of(q)
        want = asked_unit(q)
        binding = {f"v{k + 1}": qnums[k] for k in range(len(qnums))}
        var_units = {f"v{k + 1}": qunits[k] for k in range(len(qnums))}
        order = np.argsort(vecs @ qvecs[i])[::-1]
        cands = [int(j) for j in order[:200] if store[int(j)]["nvars"] == len(qnums)][:25]
        if not cands:
            rows.append({"stage": "none"})
            continue
        plain += run_plan(store[cands[0]]["steps"], binding) == truth

        clean = [j for j in cands if plan_types_ok(store[j]["steps"], var_units, want)]
        refused_total += len(cands) - len(clean)
        got_f = run_plan(store[clean[0]]["steps"], binding) if clean else None
        filt += got_f == truth

        answers = []
        for j in clean:
            g = run_plan(store[j]["steps"], binding)
            if g is not None and g.denominator == 1 and g >= 0:
                answers.append(g)
        best = Counter(answers).most_common(1)[0][0] if answers else None
        vote += best == truth
        rows.append({"q": q[:60], "candidates": len(cands), "type_clean": len(clean),
                     "filtered": str(got_f), "vote": str(best), "truth": str(truth)})

    n = len(tests)
    print(f"\n{n} test problems, lookup with the dimension control in the loop:")
    print(f"  top-1 unfiltered            : {plain}/{n}")
    print(f"  top-1 among type-clean      : {filt}/{n}")
    print(f"  vote among type-clean       : {vote}/{n}")
    print(f"  candidates refused as ill-typed: {refused_total} of "
          f"{sum(r.get('candidates', 0) for r in rows)}")
    print("\nThe check removes a class of garbage for free; what it cannot do is order the")
    print("survivors, because a same-unit scramble types perfectly. Filtering is not ranking,")
    print("and both are upstream of role binding — which remains the model's job.")
    summary = {"store": kept, "self_pass": self_pass, "tests": n,
               "top1_plain": plain, "top1_filtered": filt, "vote_filtered": vote,
               "refused": refused_total}
    Path(out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
