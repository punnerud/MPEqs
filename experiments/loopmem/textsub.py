#!/usr/bin/env python3
"""The same residue rule, with operations the model is actually good at: cut, name, paste back.

Arithmetic kept hitting a floor that no decomposition can lift. `2 - 1` came back as 2, and there
is nothing smaller to break that into. But the residue rule never required arithmetic — it
requires an operation that can be undone, and cutting a span of text out and naming it is exactly
that. "The fox is big" becomes "The A is big" with A = "fox", and the check is that pasting A
back gives the original characters.

That is the same contract as `base + residual == d` and the same one the expression work used,
over an operation set the model has seen a great deal more of than long division.

Two directions, because the arithmetic round trip failed on the second one and that is what made
it unusable — 22/24 forward against 5/24 backward on identical facts:

  CUT      the model replaces a named span with a symbol; the record checks that pasting the
           span back reproduces the sentence exactly
  PASTE    the model is given the shortened sentence and the binding and asked to restore it;
           the record checks the restoration against the original

If both hold, then unlike inversion this is an operation the model can run in either direction,
and a work record built on it can verify itself.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from general import ask as gen_ask  # noqa: E402

SENTENCES = [
    ("The fox is big", "fox"),
    ("The old bridge crosses the river", "old bridge"),
    ("She left her umbrella on the train", "umbrella"),
    ("The kettle boiled while he read the paper", "kettle"),
    ("A heavy parcel arrived for the neighbour", "heavy parcel"),
    ("The dog barked at the postman every morning", "postman"),
    ("They painted the garden fence last summer", "garden fence"),
    ("The engine stalled halfway up the hill", "engine"),
    ("My sister borrowed the blue bicycle", "blue bicycle"),
    ("The bakery opens before dawn", "bakery"),
    ("He forgot the tickets on the kitchen table", "kitchen table"),
    ("The storm knocked out the power for two days", "storm"),
    ("A small cat slept under the parked car", "small cat"),
    ("The teacher marked the essays over the weekend", "teacher"),
    ("Rain filled the barrel behind the shed", "barrel"),
    ("The clock in the hallway runs slow", "clock"),
]

CUT = """<|endoftext|><|user|>
Replace the given part of the sentence with the letter A. Change nothing else.

Example:
Sentence: The cat sat on the mat
Part: cat
Result: The A sat on the mat

Sentence: {sentence}
Part: {part}
<|assistant|>
Result:"""

PASTE = """<|endoftext|><|user|>
Put the value back in place of the letter A. Change nothing else.

Example:
Sentence: The A sat on the mat
A: cat
Result: The cat sat on the mat

Sentence: {shortened}
A: {part}
<|assistant|>
Result:"""


def first_line(text):
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S)
    for line in text.strip().splitlines():
        line = line.strip().strip('"')
        if line:
            return re.sub(r"^Result:\s*", "", line).strip()
    return None


def reverses(shortened, part, original):
    """The rule: pasting the part back where the symbol is must give the original exactly."""
    if shortened is None or "A" not in shortened:
        return False
    return re.sub(r"\bA\b", part, shortened, count=1) == original


def main(out="data/custom/textsub.json"):
    print(f"{len(SENTENCES)} sentences. Cut a span, name it, paste it back.\n")
    print(f"{'sentence':<44}{'cut':>5}{'rev':>5}{'paste':>7}  what it wrote")
    tally, rows = Counter(), []
    for sentence, part in SENTENCES:
        short = first_line(gen_ask(CUT.format(sentence=sentence, part=part), n=64))
        rev = reverses(short, part, sentence)
        tally["cut"] += short is not None
        tally["reversible"] += rev
        # The restoration is asked of the model even when its own cut was not reversible, so the
        # two directions are measured independently rather than one gating the other.
        src = short if short else re.sub(re.escape(part), "A", sentence, count=1)
        back = first_line(gen_ask(PASTE.format(shortened=src, part=part), n=64))
        exact = back is not None and back.rstrip(".") == sentence
        tally["pasted"] += exact
        rows.append({"sentence": sentence, "part": part, "cut": short,
                     "reversible": rev, "restored": back, "exact": exact})
        print(f"{sentence:<44}{'ok' if short else '.':>5}{'ok' if rev else '.':>5}"
              f"{'ok' if exact else '.':>7}  {(short or '-')[:38]}")

    n = len(SENTENCES)
    print(f"\nproduced a substitution      : {tally['cut']}/{n}")
    print(f"and it pastes back exactly   : {tally['reversible']}/{n}")
    print(f"restored the original itself : {tally['pasted']}/{n}")
    print("\nThe arithmetic round trip was 22/24 one way and 5/24 the other, which is why it")
    print("could not check itself. This is the same question over a different operation.")
    summary = {"sentences": n, "cut": tally["cut"], "reversible": tally["reversible"],
               "pasted": tally["pasted"]}
    Path(out).write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
