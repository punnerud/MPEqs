#!/usr/bin/env python3
"""The smallest set of generic operations that compose, each with a registered inverse.

Everything measured so far points the same way. The model cannot be trusted to produce a result —
`2 - 1` came back as 2, and asked to copy a sentence with one word replaced it wrote "She Aave her
Ambrella on the Aain", 5 of 16 reversible. But it was never wrong about WHICH operation to apply:
it wrote `What is 228 - 110?` correctly for every one of 24 inversions, and its JSON graphs parsed
20 out of 20.

So the model names the operation and never performs it. Then it cannot emit anything
irreversible, because it does not emit results at all — the record executes, and the record only
owns operations it can undo.

The set is deliberately tiny, because a small set that composes covers more than a large set that
does not. Every entry declares its inverse, and the record verifies reversibility by actually
running the inverse and comparing, not by trusting the table:

    split(sep)      <-> join(sep)         a string to its parts and back
    reverse()       <-> reverse()         self-inverse
    replace(a, b)   <-> replace(b, a)     legal only when b does not already occur
    keep(x)         <-> merge(residue)    the dropped items ARE the residue
    count()          -> a number          a fold; reversible only because the input is kept

`count` is the one that is not a transform, and it is where the residue rule earns its place: the
input is retained so the fold can be undone, which is the same bookkeeping that made
`base + residual == d` lossless.

Measured on a task these models are known to fail and cannot decompose their way out of —
how many times a letter occurs in a word — because that failure is about tokenisation rather
than reasoning, so no amount of thinking fixes it and only an operation can.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from general import ask as gen_ask  # noqa: E402
from twoway import ask_num  # noqa: E402

WORDS = [
    ("antall", "l", 2), ("mississippi", "s", 4), ("strawberry", "r", 3),
    ("bookkeeper", "e", 3), ("parallel", "l", 3), ("committee", "t", 2),
    ("occurrence", "c", 3), ("assessment", "s", 4), ("beekeeper", "e", 4),
    ("aardvark", "a", 3), ("balloon", "l", 2), ("possession", "s", 4),
    ("necessary", "s", 2), ("embarrass", "r", 2), ("millennium", "n", 2),
    ("kallesnavn", "n", 2), ("skulle", "l", 2), ("innkalling", "n", 3),
]

DIRECT = """<|endoftext|><|user|>
How many times does the letter "{letter}" appear in the word "{word}"?

Reply with only the number.
<|assistant|>
"""

PLAN = """<|endoftext|><|user|>
You may not answer the question. You may only choose operations from this list, and they will
be run for you.

  split("")        break the text into single characters
  keep("x")        keep only the items equal to x
  count()          how many items there are
  reverse()        reverse the order
  join("")         put the items back into text
  plural()         fox -> foxes
  singular()       foxes -> fox
  lookup()         swap the word for its dictionary entry, fox -> cat

Write the operations to run, in order, as a JSON list of strings.

Example:
Question: how many times does "p" appear in "apple"
["split(\\"\\")", "keep(\\"p\\")", "count()"]

Question: how many times does "{letter}" appear in "{word}"
<|assistant|>
["""

# Permissive on purpose. The model writes `split(\")\"` when it means `split("")` — the escaping
# collapses but the call is unambiguous — so the argument is whatever sits between the brackets
# with quoting stripped off.
CALL = re.compile(r"([a-z]+)\(([^)]*)\)")


def clean_arg(raw):
    # Single quotes too. `keep('e')` arrived with them intact and matched nothing, scoring a
    # correct pipeline as a wrong answer — the model had chosen the right three operations.
    return raw.strip().strip("\\").strip("\"'").strip("\\").strip("\"'")


class Irreversible(Exception):
    """The record refuses an operation it cannot undo."""


def op_split(state, arg):
    if not isinstance(state, str):
        raise Irreversible("split needs text")
    return list(state) if arg == "" else state.split(arg)


def op_join(state, arg):
    if not isinstance(state, list):
        raise Irreversible("join needs a list")
    return arg.join(str(x) for x in state)


def op_reverse(state, _):
    return state[::-1]


def op_keep(state, arg):
    if not isinstance(state, list):
        raise Irreversible("keep needs a list")
    return [x for x in state if x == arg]


def op_plural(state, _):
    if not isinstance(state, str):
        raise Irreversible("plural needs text")
    return state + ("es" if state.endswith(("s", "x", "z", "ch", "sh")) else "s")


def op_singular(state, _):
    if not isinstance(state, str):
        raise Irreversible("singular needs text")
    if state.endswith("es") and state[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return state[:-2]
    return state[:-1] if state.endswith("s") else state


# A dictionary lookup is a reversible operation whenever the mapping is one-to-one, and the
# record can check that rather than assume it. This is what lets `fox -> cat` sit in the same
# set as `split` and `plural`: swapping a word for another is undoable, so it is admissible,
# and composing it with plural() gives `fox -> cats` in two steps that both invert.
LEXICON = {"fox": "cat", "dog": "wolf", "bridge": "tunnel", "kettle": "pot",
           "umbrella": "raincoat", "bakery": "dairy", "clock": "watch"}
INVERSE_LEXICON = {v: k for k, v in LEXICON.items()}


def op_lookup(state, _):
    if not isinstance(state, str):
        raise Irreversible("lookup needs text")
    if state not in LEXICON:
        raise Irreversible(f"{state!r} is not in the dictionary")
    return LEXICON[state]


def op_unlookup(state, _):
    if not isinstance(state, str) or state not in INVERSE_LEXICON:
        raise Irreversible(f"{state!r} has no entry to come back from")
    return INVERSE_LEXICON[state]


def op_count(state, _):
    if not isinstance(state, list):
        raise Irreversible("count needs a list")
    return len(state)


# name -> (function, inverse-name). The inverse is what makes the operation admissible; an entry
# with no inverse is only allowed when the record keeps the input as the residue.
OPS = {
    "split": (op_split, "join"),
    "join": (op_join, "split"),
    "reverse": (op_reverse, "reverse"),
    "lookup": (op_lookup, "unlookup"),
    "unlookup": (op_unlookup, "lookup"),
    "plural": (op_plural, "singular"),
    "singular": (op_singular, "plural"),
    "keep": (op_keep, None),      # the dropped items are the residue
    "count": (op_count, None),    # a fold; the list is the residue
}


def run(pipeline, text):
    """Execute the model's chosen operations, keeping every residue so the run can be undone."""
    state, history = text, []
    for name, arg in pipeline:
        if name not in OPS:
            raise Irreversible(f"no operation called {name}")
        fn, inv = OPS[name]
        before = state
        state = fn(state, arg or "")
        # Reversibility is checked by running the inverse, not by trusting the table. Where
        # there is no inverse the input is retained instead, which is the residue.
        if inv:
            back = OPS[inv][0](state, arg or "")
            if back != before:
                raise Irreversible(f"{name} did not undo: {back!r} is not {before!r}")
            history.append({"op": name, "arg": arg, "residue": None})
        else:
            history.append({"op": name, "arg": arg, "residue": before})
    return state, history


def undo(state, history):
    """Walk the history backwards. Every step either inverts or restores from its residue."""
    for h in reversed(history):
        if h["residue"] is not None:
            state = h["residue"]
        else:
            state = OPS[OPS[h["op"]][1]][0](state, h["arg"] or "")
    return state


def parse_pipeline(reply):
    # The prompt ends at the opening bracket, so the model continues a list instead of inventing
    # a shape. Left free, it returned {"split(\"\")": ["a","n","t","a","l"], "count()": 5} — it
    # simulated the run rather than requesting it, and got the answer wrong doing so, which is
    # the entire reason it is not allowed to touch the data.
    if not reply.lstrip().startswith("["):
        reply = "[" + reply
    m = re.search(r"\[[^\]]*\]", reply, re.S)
    if not m:
        return None
    # Read the calls straight out of the text rather than insisting on valid JSON. The model
    # produces the right operations in the right order and mangles the escaping while doing it —
    # `"split(\")\",` — and refusing that would be scoring the quoting, not the plan.
    calls = CALL.findall(m.group(0))
    return [(name, clean_arg(arg) or None) for name, arg in calls] or None


def main(out="data/custom/ops.json"):
    print(f"{len(WORDS)} letter counts. The model may only name operations.\n")
    print(f"{'word':<14}{'letter':>7}{'want':>6}{'direct':>8}{'ops':>6}{'undo':>6}  pipeline")
    tally, rows = Counter(), []
    for word, letter, want in WORDS:
        direct = ask_num(DIRECT.format(word=word, letter=letter))
        pipe = parse_pipeline(gen_ask(PLAN.format(word=word, letter=letter), n=128))
        value, why, undone = None, "no pipeline", False
        if pipe:
            try:
                value, hist = run(pipe, word)
                why = "ok"
                undone = undo(value, hist) == word
            except Irreversible as e:
                why = str(e)
        tally["direct"] += direct is not None and direct == want
        tally["ops"] += value == want
        tally["valid"] += why == "ok"
        tally["undone"] += undone
        rows.append({"word": word, "letter": letter, "want": want, "direct": direct,
                     "ops": value, "pipeline": pipe, "stage": why, "undone": undone})
        print(f"{word:<14}{letter:>7}{want:>6}{str(direct):>8}{str(value):>6}"
              f"{'ok' if undone else '.':>6}  "
              f"{' -> '.join(n + ('(' + a + ')' if a else '()') for n, a in pipe) if pipe else why}")

    n = len(WORDS)
    print(f"\nasked directly                   : {tally['direct']}/{n}")
    print(f"model names the operations       : {tally['ops']}/{n}")
    print(f"pipelines the record could run   : {tally['valid']}/{n}")
    print(f"runs that undo back to the input : {tally['undone']}/{n}")
    print("\nThe model never touches the data. It cannot produce an irreversible result because")
    print("it does not produce results — which is the point, given that free-text editing was")
    print("reversible only 5 times in 16.")
    summary = {"words": n, "direct_correct": tally["direct"], "ops_correct": tally["ops"],
               "valid_pipelines": tally["valid"], "undone": tally["undone"]}
    Path(out).write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
