#!/usr/bin/env python3
"""A placeholder the model recognises, and the uniqueness that keeps it reversible.

"The fox is big" was abbreviated to "The A is big". `A` is easy for the record and foreign to the
model — it appears in training as an article, not as a variable standing for a noun. A category
word it has seen a great deal of, "The animal is big", should sit better.

But that only works while the mapping stays one-to-one. Two animals in one sentence cannot both
become "animal", because then putting them back is a guess and the substitution has stopped being
a residue. So the record enforces the constraint the choice creates: a placeholder is admissible
only if it is not already used, and distinct spans get distinct placeholders.

Three schemes over the same sentences, with the record performing every substitution — phase 26
established that the model corrupts text it retypes, 5 of 16 reversible, so it is not asked to:

  LETTER      A, B, C
  CATEGORY    animal, vehicle, tool
  NUMBERED    animal1, animal2, when one sentence holds two of a kind

Measured on restoration, which is the direction that matters: given the shortened sentence and
the binding, can the model put it back? That was 8 of 16 with letters, so it is the number the
category form has to beat.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from general import ask as gen_ask  # noqa: E402

# (sentence, [(span, category), ...]). The last four hold two spans of the SAME category, which
# is the case that forces the uniqueness rule to do something.
CASES = [
    ("The fox is big", [("fox", "animal")]),
    ("The old bridge crosses the river", [("bridge", "structure")]),
    ("She left her umbrella on the train", [("umbrella", "object")]),
    ("The kettle boiled while he read", [("kettle", "appliance")]),
    ("A heavy parcel arrived for the neighbour", [("parcel", "object")]),
    ("The dog barked at the postman", [("dog", "animal")]),
    ("They painted the garden fence last summer", [("fence", "structure")]),
    ("The engine stalled halfway up the hill", [("engine", "machine")]),
    ("My sister borrowed the blue bicycle", [("bicycle", "vehicle")]),
    ("The bakery opens before dawn", [("bakery", "building")]),
    ("He forgot the tickets on the kitchen table", [("tickets", "object")]),
    ("The storm knocked out the power for two days", [("storm", "event")]),
    ("The fox chased the dog across the field", [("fox", "animal"), ("dog", "animal")]),
    ("The car overtook the lorry on the bridge", [("car", "vehicle"), ("lorry", "vehicle")]),
    ("The hammer lay beside the saw", [("hammer", "tool"), ("saw", "tool")]),
    ("The cat watched the bird from the window", [("cat", "animal"), ("bird", "animal")]),
]

RESTORE = """<|endoftext|><|user|>
Put the original words back. Change nothing else.

Example:
Sentence: The animal sat on the mat
animal = cat
Result: The cat sat on the mat

Sentence: {shortened}
{bindings}
<|assistant|>
Result:"""

LETTERS = "ABCDEFGH"


class Collision(Exception):
    """Two spans would share a placeholder, so putting them back would be a guess."""


def substitute(sentence, spans, scheme):
    """The record performs it. Returns (shortened, bindings) or raises on a collision.

    The check is the whole point of the numbered scheme. Under CATEGORY, "The fox chased the dog"
    would become "The animal chased the animal" and no rule recovers which was which — that is
    not an abbreviation, it is a deletion, and the record refuses it rather than producing
    something that looks restorable and is not.
    """
    shortened, bindings, used = sentence, {}, set()
    for k, (span, category) in enumerate(spans):
        if scheme == "letter":
            name = LETTERS[k]
        elif scheme == "category":
            name = category
        else:
            same = sum(1 for _, c in spans if c == category)
            name = f"{category}{k + 1}" if same > 1 else category
        if name in used:
            raise Collision(f"{name!r} already stands for {bindings[name]!r}")
        # It must also not already occur in the sentence, or restoring would hit the wrong word.
        if re.search(rf"\b{re.escape(name)}\b", shortened):
            raise Collision(f"{name!r} already appears in the sentence")
        used.add(name)
        bindings[name] = span
        shortened = re.sub(rf"\b{re.escape(span)}\b", name, shortened, count=1)
    return shortened, bindings


def restores(shortened, bindings, original):
    """The record's own reversal, character for character."""
    out = shortened
    for name, span in bindings.items():
        out = re.sub(rf"\b{re.escape(name)}\b", span, out, count=1)
    return out == original


def first_line(text):
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S)
    for line in text.strip().splitlines():
        line = line.strip().strip('"')
        if line:
            return re.sub(r"^Result:\s*", "", line).strip()
    return None


def main(out="data/custom/placeholder.json"):
    print(f"{len(CASES)} sentences, the last four holding two spans of the same kind\n")
    print(f"{'scheme':<10}{'substituted':>13}{'record reverses':>17}"
          f"{'model restores':>16}{'refused':>9}")
    summary, rows = {}, []
    for scheme in ("letter", "category", "numbered"):
        tally = Counter()
        for sentence, spans in CASES:
            try:
                shortened, bindings = substitute(sentence, spans, scheme)
            except Collision as e:
                tally["refused"] += 1
                rows.append({"scheme": scheme, "sentence": sentence, "refused": str(e)})
                continue
            tally["substituted"] += 1
            tally["reverses"] += restores(shortened, bindings, sentence)
            listing = "\n".join(f"{k} = {v}" for k, v in bindings.items())
            back = first_line(gen_ask(RESTORE.format(shortened=shortened, bindings=listing),
                                      n=64))
            ok = back is not None and back.rstrip(".") == sentence
            tally["model"] += ok
            rows.append({"scheme": scheme, "sentence": sentence, "shortened": shortened,
                         "bindings": bindings, "restored": back, "model_ok": ok})
        n = len(CASES)
        summary[scheme] = {"substituted": tally["substituted"], "reverses": tally["reverses"],
                           "model_restores": tally["model"], "refused": tally["refused"],
                           "cases": n}
        print(f"{scheme:<10}{tally['substituted']:>10}/{n:<2}{tally['reverses']:>14}/{n:<2}"
              f"{tally['model']:>13}/{n:<2}{tally['refused']:>9}")

    print("\nA refusal is the right outcome, not a failure: two spans of one category cannot")
    print("share a name and still be put back. Numbering them is what makes the category form")
    print("usable, and the letter form never had the problem because letters are already")
    print("distinct — it just gives the model nothing to hold on to.")
    Path(out).write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
