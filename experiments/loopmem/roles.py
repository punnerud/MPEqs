#!/usr/bin/env python3
"""Retrieval by ROLE rather than by word: the verb, the subject, the relations.

The bank has been keyed on masked surface text since phase 88, and two measurements say
that is the wrong key. It collapsed on Norwegian (8 of 18 hits, where the same problems
in English score 32 of 34) because a MiniLM cannot place "Hvor mange heltall" near "How
many integers" — a purely lexical failure. And it misses on olympiad problems whose
wording is unlike anything in the bank while their STRUCTURE is identical to an entry.

A problem's structure is small and says itself in four parts:

    ACTION    what is being done — count, sum, maximise, solve, convert, probability
    OBJECT    what it is done to — integers, pairs, subsets, divisors, dates, a word
    RELATIONS how the object is constrained — divisibility, primality, distinctness,
              ordering, digits, remainders, consecutiveness, an equation
    SCOPE     over what — a range, a factorial's divisors, a fixed list

That is the verb, the subject and the relations between them, which is exactly what
survives translation and rewording. This module extracts the signature with one small
model call, scores two signatures by role agreement rather than cosine, and measures
retrieval hits against the embedding baseline on three sets: Norwegian, the mixed
English battery as a control, and thirty unseen AIME problems.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aimecover import ask_spec  # noqa: E402

ACTIONS = ["count", "sum", "maximise", "minimise", "find_value", "solve_equation",
           "convert", "probability", "arrange", "check"]
OBJECTS = ["integer", "pair", "triple", "subset", "divisor", "prime", "digit_string",
           "date", "word", "matrix", "sequence_term", "amount", "shape", "set_of_sets",
           "fraction", "polynomial_root", "angle_or_length"]
RELATIONS = ["divisibility", "remainder", "primality", "distinctness", "ordering",
             "digit_property", "consecutive", "equation", "inequality", "gcd_lcm",
             "factorial", "power", "geometric", "probability_event", "percentage",
             "unit", "recurrence", "cardinality"]

EXTRACT = """Describe the STRUCTURE of this problem, not its words. Ignore the story,
the names and the numbers; say what is being done, to what, under which relations.

Problem: {problem}

Reply with ONLY this JSON:
{{"action": one of {actions},
 "object": one of {objects},
 "relations": a list of one to four of {relations},
 "scope": "range" | "divisors" | "list" | "geometry" | "none"}}"""


def extract_signature(problem, model_call=None):
    call = model_call or (lambda p: ask_spec(p, n=250))
    reply = call(EXTRACT.format(problem=" ".join(str(problem).split())[:900],
                                actions=ACTIONS, objects=OBJECTS,
                                relations=RELATIONS))
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    rels = d.get("relations") or []
    if isinstance(rels, str):
        rels = [rels]
    return {"action": str(d.get("action", "")).strip(),
            "object": str(d.get("object", "")).strip(),
            "relations": sorted({str(r).strip() for r in rels if str(r).strip()}),
            "scope": str(d.get("scope", "")).strip()}


def role_score(a, b):
    """Agreement between two signatures: the verb and the subject carry most of it,
    the relations are a set overlap, and the scope is a tiebreak. Deliberately blunt —
    a structure has few parts and they are either the same or they are not."""
    if not a or not b:
        return 0.0
    s = 0.0
    s += 3.0 if a["action"] == b["action"] else 0.0
    s += 2.0 if a["object"] == b["object"] else 0.0
    ra, rb = set(a["relations"]), set(b["relations"])
    if ra or rb:
        s += 3.0 * len(ra & rb) / max(len(ra | rb), 1)
    s += 1.0 if a["scope"] == b["scope"] else 0.0
    return s


# The bank's signatures, written by hand: this is what each exemplar IS, structurally.
BANK_SIGNATURES = {
    "fractions": {"action": "find_value", "object": "fraction",
                  "relations": ["recurrence"], "scope": "none"},
    "big": {"action": "find_value", "object": "integer",
            "relations": ["power"], "scope": "none"},
    "count": {"action": "count", "object": "integer",
              "relations": ["divisibility", "remainder"], "scope": "range"},
    "divisors": {"action": "count", "object": "divisor",
                 "relations": ["factorial"], "scope": "divisors"},
    "probability": {"action": "probability", "object": "pair",
                    "relations": ["probability_event"], "scope": "range"},
    "convert": {"action": "convert", "object": "amount",
                "relations": ["unit"], "scope": "none"},
    "statistics": {"action": "find_value", "object": "amount",
                   "relations": ["cardinality"], "scope": "list"},
    "datetime": {"action": "count", "object": "date",
                 "relations": ["cardinality"], "scope": "none"},
    "finance": {"action": "find_value", "object": "amount",
                "relations": ["percentage"], "scope": "none"},
    "sequence": {"action": "find_value", "object": "sequence_term",
                 "relations": ["recurrence"], "scope": "list"},
    "matrix": {"action": "find_value", "object": "matrix",
               "relations": ["cardinality"], "scope": "list"},
    "partition": {"action": "count", "object": "integer",
                  "relations": ["cardinality"], "scope": "list"},
    "logexp": {"action": "count", "object": "digit_string",
               "relations": ["power"], "scope": "none"},
    "basearith": {"action": "find_value", "object": "digit_string",
                  "relations": ["digit_property"], "scope": "none"},
    "approx": {"action": "find_value", "object": "fraction",
               "relations": ["inequality"], "scope": "none"},
    "strcount": {"action": "count", "object": "word",
                 "relations": ["distinctness"], "scope": "none"},
    "primes": {"action": "count", "object": "prime",
               "relations": ["primality"], "scope": "range"},
    "equation": {"action": "solve_equation", "object": "integer",
                 "relations": ["equation"], "scope": "none"},
    "kinematics": {"action": "find_value", "object": "amount",
                   "relations": ["equation"], "scope": "none"},
    "roman": {"action": "find_value", "object": "digit_string",
              "relations": ["digit_property"], "scope": "none"},
    "checksum": {"action": "check", "object": "digit_string",
                 "relations": ["digit_property", "remainder"], "scope": "none"},
    "shape": {"action": "find_value", "object": "shape",
              "relations": ["geometric"], "scope": "geometry"},
    "inclusion": {"action": "count", "object": "set_of_sets",
                  "relations": ["cardinality"], "scope": "list"},
    "formula": {"action": "find_value", "object": "amount",
                "relations": ["equation"], "scope": "none"},
}


def retrieve(sig, bank, k=2):
    """The k bank entries whose STRUCTURE agrees most with this problem's."""
    scored = [(role_score(sig, BANK_SIGNATURES.get(tag)), j)
              for j, (tag, _p, _s) in enumerate(bank)]
    scored.sort(key=lambda x: -x[0])
    return [j for _s, j in scored[:k]]
