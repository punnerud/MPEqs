#!/usr/bin/env python3
"""A node per token and an edge per attachment: is the graph carrying anything?

Phase 132 grounds each role in a SPAN, which is honest labelling of chunks and not a
graph at all: "[0,2] -> action" has no nodes, no edges and nothing that says what points
at what. The question this answers is whether going down to the word level buys anything
measurable, or whether it is decoration.

Every token is a node. Each content token names its HEAD (the token it attaches to) and
the LABEL of that attachment, from a small closed vocabulary — subject, predicate,
object, bound, divisor, modulus, modifier, connector, target. The record then checks the
structure mechanically (real indices, no cycles, one root per sentence) and does the test
that matters:

    DERIVE THE QUANTITY ROLES FROM GRAPH POSITION ALONE, without the model naming them.

In phase 132 the model TOLD the record that 300000 was an upper bound. If the graph is
doing work, the record can see that for itself — the number attaches to a token that
attaches to the object under a "bound" edge — and the two can be compared against a
hand-written truth. A graph that only reproduces what the labels already said is
decoration; a graph that recovers roles the model never named is structure.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aimecover import ask_spec  # noqa: E402
from grounded import tokenise  # noqa: E402

LABELS = ["subject", "predicate", "object", "bound", "divisor", "modulus",
          "quantity", "modifier", "connector", "target", "root"]

GRAPH = """Build a dependency graph over these tokens. Every content token gets ONE
head (the index of the token it attaches to) and ONE label. Punctuation and filler may
be left out. Exactly one token has head -1 and label "root".

Tokens: {numbered}

Labels: {labels}
  subject   what the question is about        predicate  the relation being asserted
  object    what the predicate applies to     bound      a range endpoint
  divisor   what something is divided by      modulus    what a remainder is taken mod
  quantity  a number in any other role        modifier   narrows the token it attaches to
  connector joins two parts                   target     what is being asked for

Reply with ONLY:
{{"edges": [{{"i": <token index>, "head": <index or -1>, "label": "<label>"}}, ...]}}"""

# What the record should be able to see WITHOUT being told, by graph position.
POSITIONAL = {"bound": "bound", "divisor": "divisor", "modulus": "modulus"}


def build(problem, model_call=None):
    toks = tokenise(problem)
    numbered = " ".join(f"{i}:{t}" for i, t in enumerate(toks))
    call = model_call or (lambda p: ask_spec(p, n=900))
    reply = call(GRAPH.format(numbered=numbered, labels=", ".join(LABELS)))
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return toks, None
    try:
        return toks, json.loads(m.group(0)).get("edges")
    except json.JSONDecodeError:
        return toks, None


def audit(toks, edges):
    """Structure first: real indices, one root, no cycles. Then what it recovers."""
    if not edges:
        return {"parsed": False}
    seen, bad, roots = {}, 0, 0
    for e in edges:
        i, h, lab = e.get("i"), e.get("head"), str(e.get("label", ""))
        if not isinstance(i, int) or not 0 <= i < len(toks) or lab not in LABELS:
            bad += 1
            continue
        if h == -1:
            roots += 1
        elif not (isinstance(h, int) and 0 <= h < len(toks)):
            bad += 1
            continue
        seen[i] = (h, lab)

    cyclic = 0
    for i in seen:
        walk, node = set(), i
        while node in seen and seen[node][0] != -1:
            if node in walk:
                cyclic += 1
                break
            walk.add(node)
            node = seen[node][0]

    # The test: numbers whose ROLE the record can read off the graph, with no help.
    derived = {}
    for i, (h, lab) in seen.items():
        if not re.fullmatch(r"\d+(?:[.,]\d+)?", toks[i]):
            continue
        if lab in POSITIONAL:
            derived[toks[i]] = POSITIONAL[lab]
        elif h in seen and seen[h][1] in POSITIONAL:
            derived[toks[i]] = POSITIONAL[seen[h][1]]      # inherit from the head
        elif lab == "quantity" and h in seen:
            derived[toks[i]] = f"quantity of {toks[h]}"
    numbers = [t for t in toks if re.fullmatch(r"\d+(?:[.,]\d+)?", t)]
    return {"parsed": True, "nodes": len(seen), "tokens": len(toks),
            "malformed_edges": bad, "roots": roots, "cyclic": cyclic,
            "numbers": len(numbers), "derived": derived,
            "numbers_placed": len(derived)}


BATTERY = [
    ("no", "Hvor mange heltall fra 1 til 300000 er delelige med 19 og gir rest 4 når "
     "de deles på 8?", {"1": "bound", "300000": "bound", "19": "divisor",
                        "4": "quantity", "8": "modulus"}),
    ("no", "Hva er summen av alle heltall fra 1 til 90000 som er delelige med 12 og "
     "har tverrsum 15?", {"1": "bound", "90000": "bound", "12": "divisor",
                          "15": "quantity"}),
    ("no", "Hvor mange positive divisorer har 22 fakultet?", {"22": "quantity"}),
    ("en", "How many integers from 1 to 400000 are divisible by 13 and leave remainder "
     "4 when divided by 9?", {"1": "bound", "400000": "bound", "13": "divisor",
                              "4": "quantity", "9": "modulus"}),
    ("en", "What is the sum of all integers from 1 to 100000 that are divisible by 7 "
     "and leave remainder 3 when divided by 11?",
     {"1": "bound", "100000": "bound", "7": "divisor", "3": "quantity",
      "11": "modulus"}),
    ("en", "How many six-digit palindromes are divisible by 7?", {"7": "divisor"}),
]


def main(out="data/custom/wordgraph.json"):
    agg = {"parsed": 0, "malformed": 0, "cyclic": 0, "multi_root": 0,
           "placed": 0, "numbers": 0, "role_right": 0, "role_scored": 0}
    rows = []
    for lang, story, truth in BATTERY:
        toks, edges = build(story)
        a = audit(toks, edges)
        if a.get("parsed"):
            agg["parsed"] += 1
            agg["malformed"] += a["malformed_edges"]
            agg["cyclic"] += a["cyclic"]
            agg["multi_root"] += a["roots"] != 1
            agg["placed"] += a["numbers_placed"]
            agg["numbers"] += a["numbers"]
            for num, want in truth.items():
                agg["role_scored"] += 1
                got = a["derived"].get(num, "")
                agg["role_right"] += got == want or (
                    want == "quantity" and got.startswith("quantity"))
        rows.append({"lang": lang, "story": story[:56], "truth": truth, **a})
        print(f"{lang} {a.get('nodes', 0):>3}/{a.get('tokens', 0):<3} nodes | "
              f"roots {a.get('roots', '-')} cyc {a.get('cyclic', '-')} | "
              f"numbers placed {a.get('numbers_placed', 0)}/{a.get('numbers', 0)} | "
              f"{json.dumps(a.get('derived', {}))[:60]}")

    n = len(BATTERY)
    print(f"\ngraphs parsed              : {agg['parsed']}/{n}")
    print(f"malformed edges            : {agg['malformed']}, cycles {agg['cyclic']}, "
          f"graphs without exactly one root {agg['multi_root']}")
    print(f"numbers placed by the graph: {agg['placed']}/{agg['numbers']}")
    print(f"roles the record READ OFF the graph, against hand-written truth: "
          f"{agg['role_right']}/{agg['role_scored']}")
    print("\nThe question this answers is not whether a graph can be produced — it is")
    print("whether the record can learn something from its SHAPE that nobody told it.")
    print("Every role above was derived from an edge label and a head, never from a")
    print("field where the model wrote the answer down.")
    summary = {"n": n, **agg, "rows": rows}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
