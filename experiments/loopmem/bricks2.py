#!/usr/bin/env python3
"""More bricks: affine temperature, currency with rates — and many roads as a verifier.

Two extensions, each breaking something on purpose.

TEMPERATURE IS AFFINE. F = C * 9/5 + 32 is not a multiplication, so the brick generalises
to (scale, offset) with exact composition — (a,b) after (c,d) is (a*c, a*d + b) — and an
exact inverse (1/a, -b/a). The classic trap comes with it: an affine brick is only valid
for ABSOLUTE temperatures, not differences, so lifting one into a compound type (degrees
per hour) is REFUSED rather than silently wrong — the same refusal discipline as
everything else, and the same pair mpedb's verifier famously caught its own authors on.

CURRENCY HAS MANY ROADS. NOK to SEK directly, or through USD, or through EUR — and that
multiplicity is not a nuisance, it is a VERIFIER: with consistent rates every road gives
the same factor, and when one rate is poisoned the roads DISAGREE and the disagreement
localises the bad edge. Phase 34's agreement signal, reappearing at the brick level with
exact arithmetic instead of model votes: unanimity between independent paths is the
confidence, and divergence is the alarm with an address.
"""
import json
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bricks import build_registry, lift, tkey  # noqa: E402

# (scale, offset): value_dst = value_src * scale + offset. Multiplicative = offset 0.
AFFINE = [
    ("celsius", "fahrenheit", F(9, 5), F(32)),
    ("celsius", "kelvin", F(1), F(27315, 100)),
]

RATES = [  # example rates, deliberately CONSISTENT by construction (crosses derived)
    ("usd", "nok", F(105, 10)),
    ("eur", "usd", F(109, 100)),
    ("eur", "nok", F(109, 100) * F(105, 10)),      # the consistent cross
    ("gbp", "usd", F(127, 100)),
    ("gbp", "eur", F(127, 100) / F(109, 100)),
]


def compose(a, b):
    """Apply a, then b: x -> (x*sa+oa)*sb+ob."""
    (sa, oa), (sb, ob) = a, b
    return (sa * sb, oa * sb + ob)


def inverse(t):
    s, o = t
    return (1 / s, -o / s)


class ABrick:
    def __init__(self, name, src, dst, scale, offset=F(0)):
        self.name, self.src, self.dst = name, tkey(src), tkey(dst)
        self.t = (scale, offset)

    @property
    def affine(self):
        return self.t[1] != 0


def registry2():
    bricks = []
    for b in build_registry():
        bricks.append(ABrick(b.name, dict(b.src), dict(b.dst), b.factor))
    for a, b, s, o in AFFINE:
        bricks.append(ABrick(f"{a}->{b}", {a: 1}, {b: 1}, s, o))
        inv = inverse((s, o))
        bricks.append(ABrick(f"{b}->{a}", {b: 1}, {a: 1}, inv[0], inv[1]))
    for a, b, r in RATES:
        bricks.append(ABrick(f"{a}->{b}", {a: 1}, {b: 1}, r))
        bricks.append(ABrick(f"{b}->{a}", {b: 1}, {a: 1}, 1 / r))
    return bricks


def route2(bricks, src_dims, dst_dims, max_steps=10):
    """As bricks.route, generalised to affine transforms; affine bricks refuse to lift."""
    src, dst = tkey(src_dims), tkey(dst_dims)
    frontier = {src: ((F(1), F(0)), [])}
    seen = {src}
    for _ in range(max_steps):
        nxt = {}
        for t, (tr, path) in frontier.items():
            dims = dict(t)
            simple = len(dims) == 1 and list(dims.values())[0] == 1
            for b in bricks:
                if b.affine and not simple:
                    continue                      # absolute-only: no lifting an offset
                if b.affine:
                    if tkey(dims) != b.src:
                        continue
                    t2, tr2, label = b.dst, b.t, b.name
                else:
                    class _S:                     # reuse bricks.lift on the scale part
                        pass
                    _S.src, _S.dst, _S.factor, _S.name = b.src, b.dst, b.t[0], b.name
                    lifted = lift(_S, dims)
                    if lifted is None:
                        continue
                    t2, f2, label = lifted
                    tr2 = (f2, F(0))
                if t2 not in seen:
                    nxt[t2] = (compose(tr, tr2), path + [label])
        if dst in nxt:
            return nxt[dst], len(seen)
        if not nxt:
            break
        seen.update(nxt)
        frontier = nxt
    return (frontier.get(dst, (None, None))[0], None) if dst in frontier else (None, None)


def all_roads(bricks, src, dst, max_len=3):
    """EVERY simple path up to max_len between two simple types — the many roads."""
    roads = []

    def walk(cur, tr, path, used):
        if len(path) > max_len:
            return
        if cur == tkey(dst) and path:
            roads.append((tr, list(path)))
            return
        for b in bricks:
            if b.src == cur and b.name not in used:
                back = b.name.split("->")[1] + "->" + b.name.split("->")[0]
                walk(b.dst, compose(tr, b.t), path + [b.name], used | {b.name, back})

    walk(tkey(src), (F(1), F(0)), [], set())
    return roads


def main(out="data/custom/bricks2.json"):
    bricks = registry2()
    print(f"registry: {len(bricks)} bricks (units + affine temperature + currency)\n")

    tests = [
        ("37 C -> F", {"celsius": 1}, {"fahrenheit": 1}, F(37), F(986, 10)),
        ("212 F -> C", {"fahrenheit": 1}, {"celsius": 1}, F(212), F(100)),
        ("100 C -> K", {"celsius": 1}, {"kelvin": 1}, F(100), F(37315, 100)),
        ("0 F -> K (two affine hops)", {"fahrenheit": 1}, {"kelvin": 1}, F(0),
         F(45537, 180) * F(1, 1) if False else (F(0) - 32) * F(5, 9) + F(27315, 100)),
        ("100 usd -> nok", {"usd": 1}, {"nok": 1}, F(100), F(1050)),
        ("nok/litre -> usd/gallon", {"nok": 1, "litre": -1}, {"usd": 1, "gallon": -1},
         F(10), F(10) / F(105, 10) * F(3785411784, 1000000000)),
    ]
    exact = 0
    rows = []
    for name, s, d, v, want in tests:
        (tr, _), _ = route2(bricks, s, d), None
        res = route2(bricks, s, d)
        tr = res[0][0] if res[0] else None
        got = v * tr[0] + tr[1] if tr else None
        ok = got == want
        exact += ok
        rows.append({"task": name, "exact": ok, "got": str(got), "want": str(want)})
        print(f"{name:<32} {'exact' if ok else f'GOT {got} WANT {want}'}")

    # Affine round trips, exact: C -> F -> C and C -> K -> F -> C.
    rt = 0
    for chain in ((("celsius", "fahrenheit"), ("fahrenheit", "celsius")),):
        tr = (F(1), F(0))
        for a, b in chain:
            br = next(x for x in bricks if x.name == f"{a}->{b}")
            tr = compose(tr, br.t)
        rt += tr == (F(1), F(0))
    print(f"\naffine round trip exactly identity: {rt}/1")

    # Affine refuses to lift: no route for celsius/hour -> fahrenheit/hour.
    refused = route2(bricks, {"celsius": 1, "hour": -1}, {"fahrenheit": 1, "hour": -1})
    print(f"celsius/hour -> fahrenheit/hour refused (affine cannot lift): "
          f"{refused[0] is None}")

    # Many roads: nok -> eur, every path to length 3; consistent table -> unanimity.
    roads = all_roads(bricks, {"nok": 1}, {"eur": 1})
    factors = {r[0] for r in roads}
    print(f"\nnok -> eur: {len(roads)} roads found, {len(factors)} distinct factor(s) "
          f"-> {'UNANIMOUS' if len(factors) == 1 else 'DISAGREE'}")

    # Poison one rate and let the roads disagree — the alarm with an address. Two bugs in
    # the first version of this test, both the test's own: the list copy shared the brick
    # OBJECTS (mutating one mutated the clean registry too), and it poisoned eur->nok when
    # every nok->eur road uses the OTHER direction — so no road crossed the bad edge and
    # unanimity survived, correctly. A fresh registry, and the direction the roads use.
    poisoned = registry2()
    for b in poisoned:
        if b.name == "nok->eur":
            b.t = (b.t[0] * F(102, 100), F(0))     # a 2% bad quote: the arbitrage
    roads_p = all_roads(poisoned, {"nok": 1}, {"eur": 1})
    from collections import Counter
    fc = Counter(r[0] for r in roads_p)
    majority = fc.most_common(1)[0][0]
    divergent = [p for f, p in roads_p if f != majority]
    bad_named = bool(divergent) and all("nok->eur" in p for p in divergent)
    print(f"after poisoning nok->eur by 2%: {len(fc)} distinct factors "
          f"-> {'DISAGREE' if len(fc) > 1 else 'missed'}, "
          f"divergent roads all cross the poisoned edge: {bad_named}")

    summary = {"bricks": len(bricks), "exact": exact, "tests": len(tests),
               "affine_roundtrip_identity": rt,
               "affine_lift_refused": refused[0] is None,
               "roads_nok_eur": len(roads), "unanimous_clean": len(factors) == 1,
               "disagree_poisoned": len(fc) > 1,
               "poison_localised": bad_named, "rows": rows}
    print("\nMany roads are not redundancy, they are the verifier: unanimity between")
    print("independent paths is confidence, divergence is an alarm carrying the address")
    print("of the bad brick. Phase 34's law, running on exact arithmetic.")
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
