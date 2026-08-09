#!/usr/bin/env python3
"""cut by position or cut by content — and say so when the content is ambiguous.

Free-text editing failed because the model had to retype the sentence and mangled it: 5 of 16
reversible. Naming an operation fixed counting, and it fixes editing the same way — the model
gives an argument and never touches a character, so the cut is reversible by construction.

Two forms of the same operation, and which one is right is the question:

    cut(7, 10)      by position. The model must count characters, and phase 26 established it
                    cannot count reliably.
    cut("river")    by content. The record does the finding, so the model only has to know WHAT
                    it wants — which is what it tried to say unprompted, replying `cut(river,
                    end)` to a prompt that had asked for two numbers.

Content cutting has one failure the position form does not: the word may occur more than once.
That is not an error, it is a question, and the record answers it by reporting every match rather
than silently taking the first. Reporting ambiguity instead of guessing is the same discipline as
refusing an irreversible step — the record does not invent what it was not told.

Whatever the model asks for, the record returns both halves — what was taken and what is left —
so an aim that missed can be corrected, and the two together restore the original exactly.

  INDEX      cut(i, j), one attempt
  INDEX+FB   cut(i, j), up to three attempts, each shown what the last one actually took
  CONTENT    cut("word"), the record resolves it and reports every match
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from general import ask as gen_ask  # noqa: E402

# The last four repeat their target, so the ambiguity path is measured rather than assumed.
CASES = [
    ("The old bridge crosses the river", "old"),
    ("She left her umbrella on the train", "umbrella"),
    ("The kettle boiled while he read", "kettle"),
    ("A heavy parcel arrived today", "heavy"),
    ("The dog barked at the postman", "postman"),
    ("They painted the garden fence", "garden"),
    ("The engine stalled on the hill", "engine"),
    ("My sister borrowed the bicycle", "sister"),
    ("The bakery opens before dawn", "bakery"),
    ("He forgot the tickets at home", "tickets"),
    ("The cat saw the cat on the mat", "cat"),
    ("A train met a train on the bridge", "train"),
    ("The bell rang and the bell rang again", "bell"),
    ("One box beside another box", "box"),
]

EXAMPLE = """Example:
Sentence: The cat sat on the mat
Cut out the word "sat"
cut(8, 11)

"""

ASK = """<|endoftext|><|user|>
Write the operation to run as a JSON list of strings. cut(start, end) takes the characters from
start up to but not including end.

Example:
Sentence: The cat sat on the mat
Cut out the word "sat"
["cut(8, 11)"]

Sentence: {sentence}
Cut out the word "{target}"
<|assistant|>
["""

RETRY = """<|endoftext|><|user|>
Write the operation to run as a JSON list of strings. cut(start, end) takes the characters from
start up to but not including end.

Sentence: {sentence}

Your last attempt was ["cut({i}, {j})"].
It took: "{took}"
What was left: {rest}

That is not the word "{target}". Choose again.
<|assistant|>
["""

BY_NAME = """<|endoftext|><|user|>
Write the operations to run as a JSON list of strings.

Example:
Sentence: The cat sat on the mat
Remove the word "sat"
["cut(\"sat\")"]

Sentence: {sentence}
Remove the word "{target}"
<|assistant|>
["""

# Character positions are exactly what the model cannot use, and it has no way to know a word
# repeats unless it is shown. So each candidate is written out with the match marked in place,
# and the choice is made on what it can see rather than on arithmetic it cannot do.
WHICH = """<|endoftext|><|user|>
The word "{target}" appears more than once. Here is each one, marked:

{options}

Which one should be removed? Reply with only the number.
<|assistant|>
"""

PAIR = re.compile(r"(\d+)\s*,\s*(\d+)")
WORD = re.compile(r'"([^"]+)"|\'([^\']+)\'|([A-Za-z]+)')


CUT_POS = re.compile(r"cut\(\s*(\d+)\s*,\s*(\d+)\s*\)")
CUT_NAME = re.compile(r"""cut\(\s*\\?["']([^"'\\]+)\\?["']\s*\)""")


def parse_cut(reply):
    """The LAST complete call in the reply. In JSON form the model reproduces the worked example
    before writing its own answer, so the first call in the list is always `cut(8, 11)`."""
    ms = CUT_POS.findall(reply or "")
    return (int(ms[-1][0]), int(ms[-1][1])) if ms else None


def parse_word(reply):
    """Quoted text first, and never the verb.

    Two things went wrong. Ending the prompt at `cut(` forced an immediate continuation and the
    model named a salient noun instead of the one it was told, answering `cut(bridge)` when asked
    for "old", so the prefix is gone. And a bare-word pattern matched the verb itself, because
    the prefix came back as part of the reply. Unprefixed it sometimes writes `remove("bridge")`,
    so the verb is excluded by name rather than by position.
    """
    reply = reply or ""
    named = CUT_NAME.findall(reply)
    if named:
        return named[-1]
    m = re.search(r'"([^"]+)"|\'([^\']+)\'', reply)
    if m:
        return next(g for g in m.groups() if g)
    m = re.search(r"\b(?!cut\b|remove\b|word\b)([A-Za-z]+)\b", reply)
    return m.group(1) if m else None


def apply_cut(sentence, i, j):
    """Do the cut and report both halves. Reversible by construction, and checked anyway."""
    i, j = max(0, min(i, len(sentence))), max(0, min(j, len(sentence)))
    if j < i:
        i, j = j, i
    took = sentence[i:j]
    return took, sentence[:i] + "<>" + sentence[j:], sentence[:i] + took + sentence[j:] == sentence


def occurrences(sentence, word):
    """Every whole-word match. More than one is a question for the model, not a guess."""
    return [(m.start(), m.end()) for m in re.finditer(rf"\b{re.escape(word)}\b", sentence)]


def cut_by_name(sentence, target):
    """The model names the word; the record finds it and says how many candidates there were."""
    word = parse_word(gen_ask(BY_NAME.format(sentence=sentence, target=target), n=64))
    if word is None:
        return {"ok": False, "stage": "named nothing"}
    hits = occurrences(sentence, word)
    if not hits:
        # Refused rather than approximated: the record will not cut something it was not told.
        return {"ok": False, "stage": f"{word!r} does not occur", "word": word}
    ambiguous = len(hits) > 1
    if ambiguous:
        options = "\n".join(
            f"{k + 1}. {sentence[:i]}[{sentence[i:j]}]{sentence[j:]}"
            for k, (i, j) in enumerate(hits))
        pick = gen_ask(WHICH.format(target=word, options=options), n=16)
        m = re.search(r"\d+", pick or "")
        idx = int(m.group(0)) - 1 if m else 0
        idx = idx if 0 <= idx < len(hits) else 0
    else:
        idx = 0
    i, j = hits[idx]
    took, rest, reversible = apply_cut(sentence, i, j)
    return {"ok": took == target, "stage": "ok", "word": word, "took": took, "rest": rest,
            "reversible": reversible, "matches": len(hits), "ambiguous": ambiguous,
            "chose": idx + 1}


def run_index(sentence, target, rounds=1):
    attempts = []
    for attempt in range(rounds):
        if attempt == 0:
            prompt = ASK.format(sentence=sentence, target=target)
        else:
            last = attempts[-1]
            prompt = RETRY.format(sentence=sentence, target=target, i=last["i"], j=last["j"],
                                  took=last["took"], rest=last["rest"])
        span = parse_cut(gen_ask(prompt, n=64))
        if span is None:
            attempts.append({"i": None, "j": None, "took": None, "rest": None, "ok": False})
            continue
        took, rest, reversible = apply_cut(sentence, *span)
        attempts.append({"i": span[0], "j": span[1], "took": took, "rest": rest,
                         "ok": took == target, "reversible": reversible})
        if attempts[-1]["ok"]:
            break
    return attempts


def main(out="data/custom/cutspan.json"):
    print(f"{len(CASES)} sentences, the last four with the target appearing twice.\n")
    print(f"{'target':<10}{'index':>7}{'+feedback':>11}{'by name':>9}{'matches':>9}"
          f"  what the blind index attempt took")
    tally, rows = Counter(), []
    for sentence, target in CASES:
        blind = run_index(sentence, target)
        fb = run_index(sentence, target, rounds=3)
        name = cut_by_name(sentence, target)
        b_ok, f_ok = blind[-1]["ok"], any(a["ok"] for a in fb)
        tally["index"] += b_ok
        tally["feedback"] += f_ok
        tally["byname"] += name["ok"]
        tally["corrected"] += (not b_ok) and f_ok and len(fb) > 1
        tally["ambiguous"] += name.get("ambiguous", False)
        tally["ambiguous_ok"] += name.get("ambiguous", False) and name["ok"]
        tally["reversible"] += all(a.get("reversible", True) for a in blind + fb) \
            and name.get("reversible", True)
        rows.append({"sentence": sentence, "target": target, "blind": blind,
                     "feedback": fb, "by_name": name})
        print(f"{target:<10}{'ok' if b_ok else '.':>7}{'ok' if f_ok else '.':>11}"
              f"{'ok' if name['ok'] else '.':>9}{name.get('matches', 0):>9}"
              f"  {blind[-1]['took']!r}")

    n = len(CASES)
    amb = tally["ambiguous"]
    print(f"\ncut(i, j), one attempt        : {tally['index']}/{n}")
    print(f"cut(i, j), up to three        : {tally['feedback']}/{n}"
          f"   ({tally['corrected']} corrected after seeing what they took)")
    print(f'cut("word"), record resolves  : {tally["byname"]}/{n}')
    print(f"targets that occurred twice   : {amb}/{n}, "
          f"and {tally['ambiguous_ok']} of those were cut correctly")
    print(f"every cut reversible          : {tally['reversible']}/{n}")
    print("\nA cut cannot be wrong in the sense that matters — the record made it and can undo")
    print("it. It can only be aimed wrong, and there are two ways to aim: count to it, or name")
    print("it and let the record count.")
    summary = {"cases": n, "index": tally["index"], "feedback": tally["feedback"],
               "by_name": tally["byname"], "corrected": tally["corrected"],
               "ambiguous": amb, "ambiguous_correct": tally["ambiguous_ok"],
               "reversible": tally["reversible"]}
    Path(out).write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
