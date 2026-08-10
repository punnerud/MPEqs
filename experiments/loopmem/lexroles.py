#!/usr/bin/env python3
"""A closed vocabulary: word to role by lookup, with no model call at all.

The observation this tests is not ours: with a small closed vocabulary the map from word
to embedding is a TABLE, so you can go out to an embedding and look the word back up, and
the network before and after stops being needed. A model already works this way — its input
matrix is a dictionary-sized word-to-vector table and its output projection is the same
table read backwards — so the closed vocabulary changes the SIZE and the INVERTIBILITY,
not the mechanism.

For this pipeline the consequence is concrete. Phase 131 spends one model call per problem
turning a sentence into a role signature (action, object, relations, scope). If the
vocabulary is closed, that call is a lookup: each word contributes its roles, a word with
several senses contributes several, and the signature is what the words add up to.

So the table is written — from what the words MEAN, in both languages, not from which
problems fail — and the same retrieval measured against the model-extracted signatures of
phase 131, on the same sets. The question is whether a table can replace a call.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from grounded import tokenise  # noqa: E402
from roles import BANK_SIGNATURES, role_score  # noqa: E402
from solve import bank  # noqa: E402

# The closed vocabulary. Each word carries what it contributes: A: action, O: object,
# R: relation, S: scope. A word may carry SEVERAL — "sum" is an action and, in
# "digit sum", part of a relation — and that ambiguity is the point rather than a defect.
LEX = {
    # actions, English then Norwegian
    "how": [("A", "count")], "many": [("A", "count")], "hvor": [("A", "count")],
    "mange": [("A", "count")], "count": [("A", "count")], "antall": [("A", "count")],
    "number": [("A", "count"), ("O", "integer")], "tall": [("O", "integer")],
    "sum": [("A", "sum"), ("R", "cardinality")], "summen": [("A", "sum")],
    "total": [("A", "sum")], "add": [("A", "sum")], "legg": [("A", "sum")],
    "least": [("A", "minimise")], "minst": [("A", "minimise")],
    "minimum": [("A", "minimise")], "smallest": [("A", "minimise")],
    "greatest": [("A", "maximise")], "largest": [("A", "maximise")],
    "maximum": [("A", "maximise")], "største": [("A", "maximise")],
    "solve": [("A", "solve_equation")], "løs": [("A", "solve_equation")],
    "convert": [("A", "convert")], "express": [("A", "convert")],
    "probability": [("A", "probability"), ("R", "probability_event")],
    "sannsynligheten": [("A", "probability"), ("R", "probability_event")],
    "chance": [("A", "probability")], "arrangements": [("A", "arrange")],
    "ways": [("A", "count"), ("R", "cardinality")], "måter": [("A", "count")],
    "valid": [("A", "check")], "what": [("A", "find_value")],
    "hva": [("A", "find_value")], "find": [("A", "find_value")],
    "value": [("A", "find_value"), ("O", "amount")],
    # objects
    "integers": [("O", "integer")], "integer": [("O", "integer")],
    "heltall": [("O", "integer")], "pairs": [("O", "pair")], "par": [("O", "pair")],
    "triples": [("O", "triple")], "subsets": [("O", "subset")],
    "delmengder": [("O", "subset")], "divisors": [("O", "divisor")],
    "divisorer": [("O", "divisor")], "primes": [("O", "prime")],
    "prime": [("O", "prime"), ("R", "primality")],
    "primtall": [("O", "prime"), ("R", "primality")],
    "date": [("O", "date")], "days": [("O", "date")], "dager": [("O", "date")],
    "dato": [("O", "date")], "weekday": [("O", "date")], "ukedag": [("O", "date")],
    "letters": [("O", "word")], "word": [("O", "word")], "bokstavene": [("O", "word")],
    "matrix": [("O", "matrix")], "determinant": [("O", "matrix")],
    "sequence": [("O", "sequence_term")], "term": [("O", "sequence_term")],
    "følge": [("O", "sequence_term")], "price": [("O", "amount")],
    "pris": [("O", "amount")], "kroner": [("O", "amount")], "kr": [("O", "amount")],
    "fraction": [("O", "fraction")], "brøk": [("O", "fraction")],
    "mean": [("O", "amount"), ("R", "cardinality")],
    "gjennomsnittet": [("O", "amount"), ("R", "cardinality")],
    "median": [("O", "amount"), ("R", "cardinality")],
    "variance": [("O", "amount"), ("R", "cardinality")],
    "variansen": [("O", "amount"), ("R", "cardinality")],
    "area": [("O", "shape"), ("R", "geometric")],
    "volume": [("O", "shape"), ("R", "geometric")],
    "circle": [("O", "shape"), ("R", "geometric")],
    "roots": [("O", "polynomial_root")],
    # relations
    "divisible": [("R", "divisibility")], "delelige": [("R", "divisibility")],
    "delelig": [("R", "divisibility")], "multiple": [("R", "divisibility")],
    "remainder": [("R", "remainder")], "rest": [("R", "remainder")],
    "modulo": [("R", "remainder")], "mod": [("R", "remainder")],
    "distinct": [("R", "distinctness")], "ulike": [("R", "distinctness")],
    "different": [("R", "distinctness")], "forskjellige": [("R", "distinctness")],
    "increasing": [("R", "ordering")], "consecutive": [("R", "consecutive")],
    "etterfølgende": [("R", "consecutive")],
    "digits": [("R", "digit_property")], "sifrene": [("R", "digit_property")],
    "tverrsum": [("R", "digit_property")], "digit": [("R", "digit_property")],
    "equals": [("R", "equation")], "equation": [("R", "equation")],
    "ligning": [("R", "equation")], "satisfy": [("R", "equation")],
    "at": [("R", "inequality")], "most": [("R", "inequality")],
    "exceeds": [("R", "inequality")], "gcd": [("R", "gcd_lcm")],
    "lcm": [("R", "gcd_lcm")], "common": [("R", "gcd_lcm")],
    "factorial": [("R", "factorial")], "fakultet": [("R", "factorial")],
    "power": [("R", "power")], "squared": [("R", "power")],
    "percent": [("R", "percentage")], "prosent": [("R", "percentage")],
    "interest": [("R", "percentage")], "rate": [("R", "unit"), ("R", "percentage")],
    "per": [("R", "unit")], "kilometres": [("R", "unit")], "miles": [("R", "unit")],
    "hour": [("R", "unit")], "metres": [("R", "unit")], "litres": [("R", "unit")],
    "recursively": [("R", "recurrence")], "previous": [("R", "recurrence")],
    "forrige": [("R", "recurrence")], "row": [("R", "recurrence")],
    # scope
    "from": [("S", "range")], "fra": [("S", "range")], "between": [("S", "range")],
    "til": [("S", "range")], "below": [("S", "range")], "under": [("S", "range")],
    "list": [("S", "list")], "these": [("S", "list")], "disse": [("S", "list")],
}


def signature_by_lookup(text):
    """The signature is what the words add up to — one table, no model call."""
    votes = {"A": Counter(), "O": Counter(), "R": Counter(), "S": Counter()}
    hits = 0
    for tok in tokenise(text):
        for kind, val in LEX.get(tok.lower(), []):
            votes[kind][val] += 1
            hits += 1
    if not hits:
        return None, 0
    action = votes["A"].most_common(1)[0][0] if votes["A"] else "find_value"
    obj = votes["O"].most_common(1)[0][0] if votes["O"] else "amount"
    rels = sorted(r for r, _c in votes["R"].most_common(3))
    scope = votes["S"].most_common(1)[0][0] if votes["S"] else "none"
    return {"action": action, "object": obj, "relations": rels, "scope": scope}, hits


def retrieve_by_role(sig, tags, k=2):
    scored = sorted(tags, key=lambda t: -role_score(sig, BANK_SIGNATURES.get(t)))
    return scored[:k]


def main(out="data/custom/lexroles.json"):
    from norsk import build as build_no
    from rolemap import mixed_battery

    d = json.loads(Path("data/custom/rolemap.json").read_text())
    tags = [t for t, _p, _s in bank()]
    sets = {"norwegian": [(f, s) for f, s, _t in build_no()],
            "mixed": mixed_battery()}

    result = {}
    for name, items in sets.items():
        model_rows = d[name]["rows"]
        tab_hits = model_hits = agree = covered = 0
        rows = []
        for i, (fam, story) in enumerate(items):
            sig, hits = signature_by_lookup(story)
            covered += hits > 0
            shown = retrieve_by_role(sig, tags) if sig else []
            tab_hits += fam in shown
            msig = model_rows[i]["signature"] if i < len(model_rows) else None
            model_hits += fam in (model_rows[i]["shown"] if i < len(model_rows) else [])
            if sig and msig:
                agree += (sig["action"] == msig["action"]) + \
                    (sig["object"] == msig["object"])
            rows.append({"family": fam, "table": sig, "model": msig, "shown": shown,
                         "lexicon_hits": hits})
        n = len(items)
        result[name] = {"n": n, "table_retrieval": tab_hits,
                        "model_retrieval": model_hits, "covered": covered,
                        "action_object_agreement": agree, "rows": rows}
        print(f"{name:<11} table {tab_hits}/{n} | model {model_hits}/{n} | "
              f"words found for {covered}/{n} | action+object agreement {agree}/{2 * n}")

    total_t = sum(r["table_retrieval"] for r in result.values())
    total_m = sum(r["model_retrieval"] for r in result.values())
    total_n = sum(r["n"] for r in result.values())
    print(f"\nTOTAL       table {total_t}/{total_n} | model {total_m}/{total_n}")
    print(f"model calls: {total_n} for the model key, ZERO for the table")
    print(f"vocabulary: {len(LEX)} words, "
          f"{sum(1 for v in LEX.values() if len(v) > 1)} of them carrying more than one "
          f"role")
    print("\nA closed vocabulary turns a call into a lookup. Whether that is a saving or")
    print("a loss is the two columns above, and the second number in each row is what")
    print("the call was buying.")
    summary = {"total_table": total_t, "total_model": total_m, "n": total_n,
               "vocabulary": len(LEX),
               "ambiguous_words": sum(1 for v in LEX.values() if len(v) > 1),
               **result}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
