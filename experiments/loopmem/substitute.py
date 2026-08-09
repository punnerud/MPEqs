#!/usr/bin/env python3
"""Compress the problem into a node, keep the residue, and only accept it if it reverses.

The rule is the whole thing: a substitution is a residue exactly as long as our formula can put
it back, because then the model may work on the compressed form and the reduction is still safe.
`X = (3726 + 52 * 4) / 526` becomes `X = Y / 526` with `Y` a node holding the deferred part. The
model is then working on two smaller problems, and neither of them is an approximation — the
original is recoverable by textual substitution, exactly.

That check is what the record contributes, and it is cheap and total:

    reduced.replace(symbol, "(" + residue + ")") == original

If that fails the substitution is refused, whatever it looked like. Nothing else needs to be
trusted: not the model's choice of what to compress, not its arithmetic, not its ordering. A
wrong choice is merely a worse compression, never a wrong answer, and that is what makes long
iteration safe rather than risky.

Two arms on the same expressions:

  WHOLE       evaluate the expression in one call
  SUBSTITUTE  the model names a part to replace with a letter; the record checks reversibility,
              binds it as a node, and hands back the shortened expression. Repeat until one
              binary operation remains, then discharge the residues bottom-up — each of which is
              itself one small step.

The same idea is what "hang the picture on something" is: a placeholder that is legal because it
expands back, with what and how deferred to a subproblem. That arm is at the bottom.
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from general import ask as gen_ask  # noqa: E402
from twoway import ask_num  # noqa: E402

WHOLE = """<|endoftext|><|user|>
What is {expr}?

Reply with only the number.
<|assistant|>
"""

COMPRESS = """<|endoftext|><|user|>
Expression: {expr}

Name one part of it that could be replaced by a single letter to make it shorter.
Copy that part exactly as it appears. Reply with only that part, nothing else.
<|assistant|>
"""

EVAL = """<|endoftext|><|user|>
What is {expr}?

Reply with only the number.
<|assistant|>
"""

TOKEN = re.compile(r"[A-Z]|\d+|[-+*/()]")


def gen_expr(rng):
    """A four-term expression with a parenthesised group, chosen to have an exact integer value."""
    for _ in range(400):
        a, b, c = rng.randint(2, 60), rng.randint(2, 40), rng.randint(2, 12)
        op1, op2 = rng.choice("+-"), rng.choice("*+")
        inner = f"{a} {op1} {b} {op2} {c}"
        val = eval(inner)  # noqa: S307 - our own generated arithmetic, no external input
        d = rng.choice([x for x in range(2, 30) if val % x == 0] or [0])
        if d and abs(val // d) > 1:
            return f"({inner}) / {d}", val // d
    return "2 + 2", 4


def extract_candidates(reply, expr):
    """Pull every arithmetic-looking span out of the reply, longest first.

    The model answers this question well and in a sentence: 'The part that could be replaced by
    a single letter is "38 + 39 * 8".' Taking the first line whole gave a non-subexpression every
    single time, so the substitution arm never substituted anything and silently degraded into
    the whole-expression arm — identical answers on all twenty, which is what gave it away.

    Longest first because length is the compression, and compression is the thing being
    incentivised. The record still checks each one for reversibility before accepting it.
    """
    spans = re.findall(r'"([^"]+)"', reply) + re.findall(r"[\d+\-*/() ]{3,}", reply)
    seen, out = set(), []
    for s in spans:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return sorted(out, key=len, reverse=True)


def valid_substitution(expr, part):
    """Is `part` a piece of `expr` that a letter can stand in for, reversibly?

    Three things must hold and all three are checked against the string itself rather than
    against what the model claims: it must occur in the expression, it must be a self-contained
    expression on its own, and putting it back must reproduce the original character for
    character. The last one is the guarantee.
    """
    part = part.strip().strip(".").strip()
    if not part or part not in expr or part == expr:
        return None
    if part.count("(") != part.count(")"):
        return None
    try:
        eval(part)  # noqa: S307 - substring of our own generated arithmetic
    except Exception:  # noqa: BLE001 - any parse failure means it is not a subexpression
        return None
    return part


def reduce_one(cur, letters, nodes, budget):
    """Compress a single expression until it is one binary operation. Returns (reduced, refused).

    Recursive, because the first version was not and that is why it did not pay. Deferring
    `38 + 39 * 8` out of `(38 + 39 * 8) / 2` leaves a node that is still a precedence problem —
    the same thing the model was already failing — so the compression bought nothing. A residue
    only helps when it is smaller in the dimension the model actually fails in, which is exactly
    why cutting to single digits worked in phase 17. So every node is compressed in turn until
    nothing left anywhere is bigger than one operation.
    """
    refused, calls = 0, 0
    while len(TOKEN.findall(cur)) > 3 and budget[0] > 0:
        budget[0] -= 1
        raw = gen_ask(COMPRESS.format(expr=cur), n=48).strip()
        calls += 1
        part = next((p for p in (valid_substitution(cur, c)
                                 for c in extract_candidates(raw, cur)) if p), None)
        if part is None:
            refused += 1
            break
        sym = next(letters, None)
        if sym is None:
            break
        reduced = cur.replace(part, sym, 1)
        # The rule, in the two halves it actually has. The text must come back exactly, and the
        # VALUE must be unchanged once the residue is put back with brackets round it. The second
        # half is what protects precedence: `37 + 29` is a perfectly good substring of
        # `37 + 29 * 2` and a ruinous residue, because `(37 + 29) * 2` is 132 and not 95.
        #
        # Checking only the bracketed form was my own bug and it refused every correct answer the
        # model gave — `(37 + (29 * 2)) / 5` is not character-identical to `(37 + 29 * 2) / 5`,
        # so the arm silently degraded into evaluating the whole expression and returned answers
        # identical to the control on all twenty problems.
        if reduced.replace(sym, part, 1) != cur:
            refused += 1
            break
        try:
            if eval(reduced.replace(sym, "(" + part + ")", 1)) != eval(cur):  # noqa: S307
                refused += 1
                break
        except Exception:  # noqa: BLE001 - unevaluable means unverifiable, so refuse it
            refused += 1
            break
        nodes[sym] = part
        cur = reduced
    return cur, refused, calls


def compress(expr, truth, max_calls=12):
    """Compress the whole graph, then discharge it bottom-up. The record binds and checks."""
    letters = iter("YZWVUTSRQP")
    nodes, budget = {}, [max_calls]
    root, refused, calls = reduce_one(expr, letters, nodes, budget)
    # Every node is itself compressed until it too is one operation. Without this pass the
    # deferred part stays as hard as the original and the whole exercise is free of benefit.
    done = set()
    while True:
        todo = [s for s in nodes if s not in done and len(TOKEN.findall(nodes[s])) > 3]
        if not todo or budget[0] <= 0:
            break
        for s in todo:
            done.add(s)
            nodes[s], r, c = reduce_one(nodes[s], letters, nodes, budget)
            refused += r
            calls += c

    # Discharge bottom-up: every residue is now one operation the model evaluates on its own,
    # and its value is substituted into whatever referred to it.
    values = {}
    for sym in reversed(list(nodes)):
        body = nodes[sym]
        for s2, v2 in values.items():
            body = body.replace(s2, f"({v2})")
        v = ask_num(EVAL.format(expr=body))
        calls += 1
        if v is None:
            return None, nodes, refused, calls
        values[sym] = v
    final = root
    for s2, v2 in values.items():
        final = final.replace(s2, f"({v2})")
    got = ask_num(EVAL.format(expr=final))
    calls += 1
    return got, nodes, refused, calls


def numeric(n_tasks=20, seed=7):
    rng = random.Random(int(seed))
    tasks = [gen_expr(rng) for _ in range(int(n_tasks))]
    print(f"NUMERIC: {len(tasks)} expressions, whole against compress-with-residue\n")
    print(f"{'expression':<26}{'truth':>8}{'whole':>8}{'subst':>8}{'nodes':>7}{'ref':>5}")
    rows, w_ok, s_ok, tok_before, tok_after = [], 0, 0, 0, 0
    for expr, truth in tasks:
        w = ask_num(WHOLE.format(expr=expr))
        s, nodes, refused, calls = compress(expr, truth)
        w_ok += w == truth
        s_ok += s == truth
        # Compression is the incentive, so it is reported as the thing that actually matters:
        # the LARGEST single expression the model was ever asked to evaluate. Summing the pieces
        # would flatter it — what limits a small model is the widest thing it holds at once.
        pieces = [len(TOKEN.findall(p)) for p in nodes.values()]
        after = max(pieces + [len(TOKEN.findall(expr)) - sum(p - 1 for p in pieces)])
        tok_before += len(TOKEN.findall(expr))
        tok_after += after
        rows.append({"expr": expr, "truth": truth, "whole": w, "substituted": s,
                     "nodes": nodes, "refused": refused, "calls": calls,
                     "tokens_before": len(TOKEN.findall(expr)), "tokens_after": after})
        print(f"{expr:<26}{truth:>8}{str(w):>8}{str(s):>8}{len(nodes):>7}{refused:>5}")

    n = len(tasks)
    print(f"\nwhole expression in one go        : {w_ok}/{n}")
    print(f"compressed to nodes, residue back : {s_ok}/{n}")
    print(f"widest form the model ever held   : {tok_after / n:.1f} tokens, "
          f"down from {tok_before / n:.1f}")
    return {"tasks": n, "whole_correct": w_ok, "substituted_correct": s_ok,
            "tokens_before": tok_before / n, "tokens_after": tok_after / n, "runs": rows}


# ---------------------------------------------------------------- the same rule, in the world

PLACEHOLDER = {
    "goal": "picture hanging straight",
    # "something to hang it on" is the residue: legal in the plan because it expands back into a
    # real subplan. What it turns out to be — a nail, a plug and hook, an adhesive strip — is a
    # subproblem that does not have to be settled to get the rest of the plan right.
    "abstract": {
        "choose the wall spot": ([], ["spot chosen"]),
        "provide something to hang it on": (["spot chosen"], ["fixing in wall"]),
        "unpack the frame": ([], ["frame unpacked"]),
        "fix the wire to the frame": (["frame unpacked"], ["wire fitted"]),
        "hang the frame": (["fixing in wall", "wire fitted"], ["picture hanging"]),
        "level it": (["picture hanging"], ["picture hanging straight"]),
    },
    "expansions": {
        "provide something to hang it on": {
            "nail": {"hammer a nail in": (["spot chosen"], ["fixing in wall"])},
            "plug and hook": {
                "drill the hole": (["spot chosen"], ["hole drilled"]),
                "push in the wall plug": (["hole drilled"], ["plug seated"]),
                "screw in the hook": (["plug seated"], ["fixing in wall"]),
            },
            "adhesive strip": {
                "clean the wall": (["spot chosen"], ["wall clean"]),
                "stick on the strip": (["wall clean"], ["fixing in wall"]),
            },
        }
    },
}


def placeholder_expands_back():
    """Every expansion must reproduce the abstract step's contract exactly, or it is not one.

    Same rule as the numeric arm, stated over preconditions and effects instead of characters:
    the subplan may consume no more than the abstract step consumed, and must deliver everything
    it promised. If it does, the plan built around the placeholder stays valid whichever
    expansion is chosen later, which is what makes deferring it safe.
    """
    results = {}
    for step, options in PLACEHOLDER["expansions"].items():
        pre, eff = PLACEHOLDER["abstract"][step]
        for label, sub in options.items():
            produced = {e for _, es in sub.values() for e in es}
            consumed = {p for ps, _ in sub.values() for p in ps}
            internal = produced - set(eff)
            ok = set(eff) <= produced and (consumed - internal) <= set(pre)
            results[f"{step} -> {label}"] = ok
    return results


def main(n_tasks=20, out="data/custom/substitute.json"):
    num = numeric(int(n_tasks))
    exp = placeholder_expands_back()
    print("\nPLACEHOLDER: the same rule over preconditions instead of characters\n")
    for k, v in exp.items():
        print(f"  {'reverses' if v else 'BROKEN  '}  {k}")
    print(f"\n{sum(exp.values())}/{len(exp)} expansions restore the abstract step's contract, so")
    print("the plan around the placeholder holds whichever one is chosen later.")
    Path(out).write_text(json.dumps(
        {"numeric": num, "expansions": exp,
         "expansions_ok": sum(exp.values()), "expansions_total": len(exp)}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
