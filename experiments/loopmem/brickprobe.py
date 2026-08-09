#!/usr/bin/env python3
"""Probe-on-first-touch: two calculations with different values, and a usage ledger.

The declared-asymmetry metadata of the last phase has to come from somewhere, and the
proposal supplies the discovery step: when a calculation HITS a brick for the first time,
run it twice with different values. Two probes determine an affine transform exactly —
a = f(2) - f(1), b = f(1) - a — one probe never can, because scale and offset are two
unknowns. A third value VERIFIES the inference: if f(3) is not what the fitted transform
predicts, the thing is not affine at all and is refused as a brick outright, which is how
a nonlinear impostor is caught by arithmetic rather than trust.

Probing is once per brick, ever: the inferred transform is cached, and the ledger records
that the brick was used and how often — "lagre ned at den er brukt", literally. When both
directions of a pair have been probed, their composition is checked against identity, and
a pair that does not close is FLAGGED as behaving asymmetrically — the discovery that
feeds the question/alarm classifier, found by calculation instead of documentation.

Hidden truths the probe must recover, none of them visible to it in advance:

    scale bricks, an affine brick (temperature-shaped), a two-sided quote pair,
    a fee crossing, and one impostor whose secret is x*x.
"""
import json
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


class HiddenBrick:
    """A black box: callable only. The probe never sees `secret`."""

    def __init__(self, name, fn, secret):
        self.name, self._fn, self.secret = name, fn, secret

    def __call__(self, v):
        return self._fn(v)


def hidden_registry():
    return [
        HiddenBrick("km->m", lambda v: v * 1000, ("scale", F(1000), F(0))),
        HiddenBrick("c->f", lambda v: v * F(9, 5) + 32, ("affine", F(9, 5), F(32))),
        HiddenBrick("f->c", lambda v: (v - 32) * F(5, 9), ("affine", F(5, 9), F(-160, 9))),
        HiddenBrick("usd->nok", lambda v: v * F(1045, 100), ("scale", F(1045, 100), F(0))),
        HiddenBrick("nok->usd", lambda v: v * F(100, 1055), ("scale", F(100, 1055), F(0))),
        HiddenBrick("west->east", lambda v: v - 25, ("affine", F(1), F(-25))),
        HiddenBrick("east->west", lambda v: v, ("scale", F(1), F(0))),
        HiddenBrick("sq", lambda v: v * v, ("nonlinear", None, None)),
    ]


class ProbeCache:
    def __init__(self):
        self.known = {}                    # name -> (a, b) or "refused"
        self.ledger = {}                   # name -> {"probes": n, "hits": n}

    def touch(self, brick):
        led = self.ledger.setdefault(brick.name, {"probes": 0, "hits": 0})
        led["hits"] += 1
        if brick.name in self.known:
            return self.known[brick.name]
        # First touch: two probes with different values determine (a, b); a third verifies.
        f1, f2, f3 = brick(F(1)), brick(F(2)), brick(F(3))
        led["probes"] = 3
        a = f2 - f1
        b = f1 - a
        verdict = (a, b) if f3 == a * 3 + b else "refused"
        self.known[brick.name] = verdict
        return verdict

    def pair_check(self, name_ab, name_ba):
        ta, tb = self.known.get(name_ab), self.known.get(name_ba)
        if not isinstance(ta, tuple) or not isinstance(tb, tuple):
            return None
        comp = (ta[0] * tb[0], ta[1] * tb[0] + tb[1])
        return comp == (F(1), F(0))


def main(out="data/custom/brickprobe.json"):
    bricks = {b.name: b for b in hidden_registry()}
    cache = ProbeCache()

    # A calculation that HITS bricks — some repeatedly, so the cache is exercised.
    routes = [["km->m"], ["c->f", "f->c"], ["usd->nok"], ["nok->usd"],
              ["west->east", "east->west"], ["km->m"], ["usd->nok"], ["sq"]]
    for r in routes:
        for name in r:
            cache.touch(bricks[name])

    inferred_ok = refused = 0
    print("probe verdicts against the hidden truths:\n")
    for name, b in bricks.items():
        kind, s, o = b.secret
        got = cache.known.get(name)
        if kind == "nonlinear":
            ok = got == "refused"
            refused += ok
            print(f"  {name:<12} hidden {kind:<10} -> "
                  f"{'refused (third probe disagreed)' if ok else 'MISSED'}")
        else:
            ok = got == (s, o)
            inferred_ok += ok
            print(f"  {name:<12} hidden ({kind} {s},{o})".ljust(44)
                  + f" -> inferred {'exactly' if ok else 'WRONG: ' + str(got)}")

    pairs = {"usd->nok/nok->usd": cache.pair_check("usd->nok", "nok->usd"),
             "west->east/east->west": cache.pair_check("west->east", "east->west"),
             "c->f/f->c": cache.pair_check("c->f", "f->c")}
    print("\npair closure (True = symmetric, False = flag as two-sided):")
    for k, v in pairs.items():
        print(f"  {k:<24} closes: {v}")

    total_probes = sum(l["probes"] for l in cache.ledger.values())
    total_hits = sum(l["hits"] for l in cache.ledger.values())
    print(f"\nledger: {total_hits} brick hits, {total_probes} probe calls — repeat hits "
          f"cost zero, and every brick's usage is on record:")
    for name, led in sorted(cache.ledger.items()):
        print(f"  {name:<12} hits {led['hits']}, probes {led['probes']}")

    flagged = [k for k, v in pairs.items() if v is False]
    print(f"\nauto-flagged as asymmetric pairs: {flagged}")
    print("Two probes discover, the third verifies, the cache remembers, the ledger counts,")
    print("and the pairs that do not close are exactly the ones the classifier must treat")
    print("as questions — the metadata, found by calculating instead of being told.")
    summary = {"bricks": len(bricks), "inferred_exact": inferred_ok,
               "nonlinear_refused": refused, "pairs_flagged": flagged,
               "symmetric_pairs_pass": sum(1 for v in pairs.values() if v),
               "total_hits": total_hits, "total_probes": total_probes,
               "ledger": cache.ledger}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
