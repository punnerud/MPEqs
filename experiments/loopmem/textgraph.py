#!/usr/bin/env python3
"""The record's text graph: dictionary-named nodes, and the unit tagger rebuilt on nouns.

Part 1 (this run, no model): `units_of` v2. The phase 54 tagger took the word next to a
number and paid a 28% noise floor — "sold clips to 48 of her friends" tagged 48 as friends.
V2 walks forward from the number to the nearest word the LEXICON confirms as a noun,
skipping verbs, adjectives and function words it recognises as such. Measured on exactly
the phase 54 metric — self-pass over the same 1,554-template store — with the increments
separated so each change owns its delta:

    A  v1 tagger, v1 asked        (the 72% anchor, re-run)
    B  v2 tagger, v1 asked        (the number-unit tagger's own contribution)
    C  v2 tagger, v2 asked        (plus the asked-unit reader on the same lexicon)

Part 2 (phase 68): the graph itself — noun occurrences as nodes named by the text's own
lemmas, numbers attached per v2, verb edges within sentences, sentence graphs linked by
shared lemmas — and the fan-out re-run where the record HANDS OUT the node names, which is
the experiment that decides whether the phase 64 namespace residual dies here.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dimcheck import DIMLESS, SKIP, asked_unit, plan_types_ok, singular, units_of  # noqa: E402
from lexicon import Lexicon  # noqa: E402
from mapstore import NUM, build_store, norm  # noqa: E402

WORD = re.compile(r"[A-Za-z%$]+")


def units_of_v2(text, lex):
    """A unit per number: the nearest LEXICON-CONFIRMED noun after it.

    The v1 rules that were right stay: '$' before the number is dollars, an explicit
    dimensionless marker ends the search empty, and the stop-word list still skips
    determiners the lexicon happily calls nouns ('the' is not in it, but 'more' is).
    What changes is the walk: words the lexicon reads as verb/adjective/adverb and NOT
    noun are stepped over instead of trusted.
    """
    text = norm(text)
    out = []
    for m in NUM.finditer(text):
        if m.start() > 0 and text[m.start() - 1] == "$":
            out.append("dollar")
            continue
        unit = None
        for w in WORD.findall(text[m.end():m.end() + 80])[:6]:
            lw = w.lower()
            if lw in DIMLESS:
                break
            if lw in SKIP:
                continue
            lemma = lex.noun_lemma(lw)
            tag_set = {t for _, t in lex.lemmas(lw)}
            if lemma and ("n" in tag_set):
                # Ambiguity guard: a word the lexicon reads ONLY as verb/adj is skipped;
                # one with a noun reading is taken — after a number, nouny context holds.
                unit = singular(lemma)
                break
            if tag_set and "n" not in tag_set:
                continue                    # known non-noun: step over it
            # Unknown to the lexicon: v1 behaviour, take it (names like 'Natalia' or
            # invented words are units too; the lexicon cannot veto what it has never seen).
            unit = singular(lw)
            break
        out.append(unit)
    return out


def asked_unit_v2(question, lex):
    """The asked unit: first lexicon-noun after 'how many', dollars for money."""
    q = norm(question)
    m = re.search(r"how many ([a-z]+(?: [a-z]+){0,3})", q, re.I)
    if m:
        for w in m.group(1).split():
            if w.lower() in SKIP:
                continue
            lemma = lex.noun_lemma(w)
            if lemma:
                return singular(lemma)
            tag_set = {t for _, t in lex.lemmas(w)}
            if not tag_set:
                return singular(w)
        # every candidate was a known non-noun: fall through to v1's take-first behaviour
        for w in m.group(1).split():
            if w.lower() not in SKIP:
                return singular(w)
    if re.search(r"how much", q, re.I) and "$" in q:
        return "dollar"
    return None


def self_pass(store, tag_fn, ask_fn):
    ok = 0
    for t in store:
        vu = {f"v{k + 1}": u for k, u in enumerate(tag_fn(t["question"]))}
        ok += plan_types_ok(t["steps"], vu, ask_fn(t["question"]))
    return ok


def main(out="data/custom/unitsv2.json"):
    lex = Lexicon()
    store, kept, _, _ = build_store(2000)
    arms = {
        "A_v1_tagger_v1_asked": (lambda q: units_of(q), lambda q: asked_unit(q)),
        "B_v2_tagger_v1_asked": (lambda q: units_of_v2(q, lex), lambda q: asked_unit(q)),
        "C_v2_tagger_v2_asked": (lambda q: units_of_v2(q, lex),
                                 lambda q: asked_unit_v2(q, lex)),
    }
    results = {"store": kept}
    print(f"self-pass over the same {kept:,} templates as phase 54 (anchor 1,112 = 72%):\n")
    for name, (tf, af) in arms.items():
        n = self_pass(store, tf, af)
        results[name] = n
        print(f"  {name:<24} {n:>5}  ({100 * n / kept:.0f}%)")
    print("\nB minus A is the dictionary-walk's own worth on number units; C minus B is the")
    print("asked-reader's. Every downstream judge (the refusal gate, the routing triage)")
    print("inherits whichever floor this sets, declared here first.")
    Path(out).write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
