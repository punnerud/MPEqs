# What transfers from MPEE and MPEdb to an LLM stack, and what does not

Two mechanisms were candidates, both measured rather than argued.

## The mechanisms

**MPEE's streaming N×N solver.** A 50 000-customer fleet implies a 50 000 × 50 000 matrix,
~10 GB. MPEE never materialises it: it streams the cells through a 500 MB budget, collapses
regions that attach to the rest of the network through two or three connections into a single
abstract node with an interface, and prunes branches whose partial cost already loses.

**matcodec: read without decoding.** Cluster the points into regions. A cross-region block is
`D[a][b] = d(a, gw) + HWY + d(gw, b)` — additive rank-1 through a gateway — so store a rank-1
base plus an *exact* residual. Lossless by construction. The resident landmark index answers
zero-residual blocks in O(L) with no inflate at all, and supplies O(L) triangle bounds
(`cell_bounds`) for solver pruning. Measured 6.4× on a real road matrix, ~10× on single-gateway
synthetic worlds, **~1.8× on structureless points**.

**Correction, 2026-08-08.** An earlier version of this paragraph said MPEdb has the same idea at
coarser grain and cited `DESIGN-COLUMNAR` §1 for per-block min/max skipping. That document does
not exist. Searching all of `/Users/punnerud/Downloads/mpedb` finds no columnar layout, no
per-block min/max, and no vector or embedding index — in code or in design; `DESIGN-MPEE-OPT.md`
§1.6 explicitly rejects matrix compression for MPEdb because msync operates at page granularity.
The "untouched, undecoded" mechanism lives only in matcodec, which is why matcodec is the
version mirrored here. The citation was invented, and nothing downstream of it was ever
measured against MPEdb.

## The discipline, which transferred first and mattered most

`DESIGN-MPEE-SOLVER` §2 refuses to invent a selectivity factor: inputs are KNOWN, BOUNDED or
UNKNOWN, the last two are priced identically at the worst case, and no constant is made up,
because a plan optimal on an estimate can be catastrophic on the data.

That rule would have caught this project's worst bug. An early calibration read pages that were
still resident and reported `C_fetch = 6.66 µs` against the true 230.74 µs — a 35× error that
scaled every downstream number, *and* silently disabled the cache-contamination guard, because
the guard compares achieved bandwidth against that very constant. `CostModel` now carries a
`Provenance` (`Measured` or `Assumed`), defaults to `Assumed` on deserialisation so an old file
is never mistaken for a measurement, and every command that falls back says so in its output.

## Does the compression transfer? Measured: no, and the reason is specific

`matstruct probe` measures the four things matcodec depends on, in the order that can stop the
work earliest, against synthetic ground truth (a single-gateway world must give exactly zero
residual; structureless noise must not).

The one guard that changed the conclusion is the **null model**. A rank-1 base captures a
block's mean for free, so "residual is 10 % of block RMS" sounds like a fit and means nothing.
The question is whether rank-1 beats a per-block *constant*.

| | expert co-activation | embeddings, angular | embeddings, geodesic k=11 |
|---|---|---|---|
| metric | **no** — 0.36 % of triples violate | yes — 0 of 499 610 | yes, by construction |
| coefficient of variation | 0.260 | 0.076 | **0.280** |
| dynamic range | 4.14× | 1.61× | **5.85×** |
| silhouette | 0.168 | 0.051 | **0.191** |
| rank-1 over the null | +11.0 % | −93.1 % | −25.5 % |
| blocks readable without decoding | 0.00 % | 0.00 % | 0.00 % |
| deflate alone | 3.41× | 3.34× | 3.28× |

**Expert co-activation is not a metric.** Counts are similarities; `-ln(lift)` does not obey
the triangle inequality, so `cell_bounds` cannot be used at all.

**Embeddings are a perfect metric with nothing in it.** Mean angular distance 1.5163 against
π/2 = 1.5708 — textbook distance concentration in 384 dimensions. Everything is at right
angles to everything, so there are no far-apart regions and the rank-1 base is *worse than a
constant*.

**Routing beats flying.** Raw angular distance is "fly straight there"; a road network is not.
Build a kNN graph and measure shortest paths, and cross-cluster traffic must pass through
whichever points bridge the clusters. That manufactures roughly 3.7× more structure on every
axis and cuts the rank-1 penalty from −93 % to −25 %. Below k=11 the graph disconnects; above
it the geodesic converges back to the straight line, as it must.

**It still stops short, for a structural reason.** A kNN graph over a concentrated cloud is an
expander: richly connected everywhere, no narrow cuts. Road networks have narrow cuts because
geography imposes them — a river admits three bridges, a mountain range one pass. Semantic
space has no geography, so there is always another way around and no small set of points lies
on most paths between two regions.

## The other model, tested at last

Everything above scores **one** of matcodec's two bases. matcodec compresses a matrix by trying
both and keeping the smaller (`lib.rs:495-522`); `metric.rs` only implements the per-cluster
rank-1 base `col0[p] + row0[q] - c00`. The other base is min-plus through landmarks,

```
base(i, j) = min over landmarks a of  d(i, a) + d(a, j)
```

and it is the one that reaches 17.6x on MPEE's synthetic gateway world and 9.4x on real London
road data. It is also the one that matches "find the extreme points, compute N x N only between
them, and let everything else route through that skeleton" — landmarks are picked by greedy
facility location over a *sample* of pairs, O(L·n·S), so it never touches n².

`crates/matstruct/src/landmark.rs` ports it. The port is validated before use: MPEE's own
gateway generator must come out mostly index-exact and structureless points must not, both as
unit tests. Then:

| matrix | exact blocks | exact cells | residual / mean distance |
|---|---|---|---|
| corpus embeddings, raw angular | 0.62 % | 1.43 % | 0.624 |
| corpus embeddings, geodesic k=11 | 0.32 % | **24.91 %** | **0.096** |
| model layer `add`-L2, raw | 0.33 % | 0.42 % | 0.365 |
| model layer `mul`-L2, raw | 1.16 % | 2.04 % | 0.319 |
| model layer `add`-L2, geodesic | 0.30 % | 13.35 % | 0.134 |
| model layer `mul`-L2, geodesic | 0.74 % | 20.87 % | 0.099 |

**The base fits; the block does not.** On geodesic distances the landmark base is a good
approximation — residual under 10 % of mean distance, and a quarter of individual cells
reproduced *exactly*. What never happens is a whole block being exact, which is the only thing
that buys a read without decoding. Under 1 % everywhere.

That is the expander property once more, and now in its sharpest form. A block is the set of
columns in one landmark's Voronoi cell. For the block to be answerable from the index, the
skeleton must reproduce **every** pair in it. On a graph with no narrow cuts there is always
some pair that routes around the landmark, and one such pair spoils the block. Getting 25 % of
cells right and 0.3 % of blocks right is exactly the signature of a good average fit with no
clean partition underneath it.

So both of matcodec's models have now been tried on embeddings, and both fail — but for
different reasons, which is worth keeping straight. The cluster base fails because it is *worse
than a per-block constant* (rank-1 gain −93.1 %). The landmark base fails despite being a good
fit, because the all-or-nothing block criterion cannot fire without narrow cuts.

**One observation that cuts against the obvious reading.** The layers of the model that did
*not* learn its task (`mul`, 0.046 held out) are consistently *more* landmark-explainable than
those of the model that did (`add`, 0.730): 1.16 % against 0.33 % exact blocks, residual 0.319
against 0.365. More structure by this measure is not better representation — the likeliest
explanation is that a model which learned little produces less differentiated embeddings, which
a skeleton fits more easily. That is unverified, and it is a reason not to read "the landmark
base fits well" as "the network has learned something".

### Corrected 2026-08-08: the rule was wrong, and so was the diagnosis

Everything above scores the landmark base against the **angular** metric with landmarks chosen
by facility location. Betweenness centrality on the same kNN graph says the graph is *not* an
expander: the top 1 % of nodes carry 31.3 % of all traversals against 1.0 % for a flat graph,
and half of all routes pass through 148 of 4 000 points. The narrow cuts are there.

They are irrelevant to a chord metric, which is the whole error. On a road network the true
distance *is* the routed distance; between embeddings it is the straight line, and a straight
line passes through no gateway however busy. Pair betweenness-chosen landmarks with the geodesic
metric and 32 points — 0.8 % of the set — reproduce **39.08 %** of pairwise distances exactly,
against 0.91 % for the pairing measured throughout this document. Zero triangle violations,
1.3 % of matches trivially involving a landmark endpoint.

So the rule below is superseded. The correct one:

> matcodec's applicability is decided by whether **the metric being compressed is the one the
> graph induces**. Narrow cuts in the graph buy nothing for a metric that ignores them.

### The rule this yields (superseded, kept for the record)

matcodec's applicability is decided by **narrow cuts**, not by metricity, not by clusterability,
and — now that both bases have been measured — not by how well the base fits either. Both of those can be manufactured — the geodesic transform does exactly that —
while narrow cuts cannot be manufactured without inventing them. Ask of a new matrix: *is there
a small vertex set whose removal separates the graph?* If not, matcodec degrades to deflate
however the distances are defined.

Not pursued, and deliberately: pruning the kNN graph towards a spanning tree would force
bottlenecks and make matcodec work beautifully, at the cost of distances that no longer
approximate anything.

## Does the streaming solver transfer? Not to the expert layout

| graph | N | cells | streaming needed? |
|---|---|---|---|
| expert co-activation, per layer | 64–256 | 65 k | no, it fits in L2 |
| expert co-activation, global | 10 240 | 105 M | the cross-layer mutual information is 0.66 of 5.96 bits, so the graph is near block-diagonal by layer and decomposes into the per-layer problems already solved |
| embedding graph | 10⁵–10⁷ | 10¹⁰–10¹⁴ | yes — but see above for whether the payoff exists |

The region-collapse half of the solver — "a subgraph attached through two or three connections
becomes one node with an interface" — needs the same narrow cuts as matcodec, and fails for the
same reason.

## The expander property is one fact with three consequences

Everything above is the first consequence: no narrow cuts, so no gateway, so no codec. Measuring
the same graph from the other side gives the other two, and they point in opposite directions.

| consequence | measured | verdict |
|---|---|---|
| no narrow cuts -> no gateway to compress through | 0.00 % blocks readable without decoding | against |
| richly connected -> short paths | max 6 hops at k=32, 100 % reachable | for |
| no local structure -> neighbours cannot be co-located in a 1-D order | 24.0 runs in identity order, 28.2 after BFS | against |

The third is the one that surprised me. The expert layout work minimises exactly this quantity
and wins 14.4 % fewer fetches doing it, so applying it to index nodes looked like free transfer.
It is not: an expert's co-activation partners are a genuinely small, biased set, while a node's
44 graph neighbours are spread uniformly through a cloud with no local structure to exploit.
Reordering cannot create locality that the data does not have. The technique transfers; the
outcome does not.

## What traversal costs, and when it is worth it

The index does not have to be resident — keeping it on disk is a design the user proposed, not
a feature MPEdb has (see the correction above) — so the binding cost is fetch rounds, not bytes
held. `experiments/hops/traversal_cost.py` prices a walk
against a full scan of the index, and `verify_fetch.py` checks it with real `F_NOCACHE` reads.

One fetch costs 806 KiB of sequential transfer on this device, so a walk that issues 206 fetches
only pays once the index is bigger than that many fetch-equivalents. Break-even: **~100 000
embeddings, about 160 MiB**. Below it, scan. The prediction was verified at 17.9x measured
against 23.0x predicted for the losing direction.

This also closes the expert case a second way. The co-activation graph has 64–256 nodes per
layer and 10 240 globally — two to three orders of magnitude below break-even. Even if that
graph were navigable, which it is not (it is not a metric at all), walking it would cost more
than reading every gate. Two independent measurements, one conclusion.

## Fire together, wire together

The Hebbian reading of the layout work is exact. The co-activation graph *is* fire-together;
the `chain` layout is wire-together, placing what fires together adjacent on disk. Phase 3's
contribution is to compute the wiring as a **route** rather than a pairwise link. The routes
turn out to be real and informative — 3.7× more structure than the direct metric — but the
network they form has no chokepoints, which is exactly the property a gateway codec needs.

## Reproducing

```sh
make embed          # chunk the corpus and embed it with all-minilm
make probe-embeddings
make probe-geodesic
make probe-experts  # the negative control
```

The control is not decoration. If the expert graph ever reports the same structure as the
embedding graph, the probe is measuring something other than what it claims.
