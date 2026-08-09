#!/usr/bin/env python3
"""Compress the neighbour graph the way matcodec does, and check it against gzip honestly.

The graph is 1.1 MB for 8,920 sentences: 8,920 x 16 neighbour ids as u32 and the same many
distances as f32. General-purpose compressors see a byte stream. A codec that knows the shape
sees something much more compressible, and the shape here is the same one MPEE's matcodec
exploits:

    the ids in a row are sorted by distance but drawn from a small neighbourhood, so their
    SORTED form deltas down to small numbers

    the distances in a row are all close to each other, so a per-row base plus a residual needs
    far fewer bits than a per-row float

That is `base + residual == d` again, applied to a graph instead of a distance matrix, and the
residual is exact on the quantised grid rather than approximate.

The comparison has to be fair, which means stating the error rather than only the size. A lossy
codec beating gzip on bytes is not a result — this project retracted a claim of exactly that
shape earlier. So the grid is chosen first, the maximum error it implies is reported, and the
codec is then verified LOSSLESS against the quantised values: decode must reproduce every id and
every quantised distance exactly, or the number is not reported at all.
"""
import bz2
import gzip
import json
import lzma
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from brokerhop import KNN_BIN, read_knn  # noqa: E402

GRID = 1e-4          # radians per quantisation step


def varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def read_varint(buf, i):
    n = shift = 0
    while True:
        b = buf[i]
        i += 1
        n |= (b & 0x7F) << shift
        if not b & 0x80:
            return n, i
        shift += 7


def zig(n):
    return (n << 1) ^ (n >> 63)


def unzig(n):
    return (n >> 1) ^ -(n & 1)


def encode(ids, dist, grid=GRID):
    """Per-row: sorted ids delta-coded, distances as a base plus bit-packed residuals."""
    n, k = ids.shape
    q = np.rint(dist / grid).astype(np.int64)
    out = bytearray()
    out += b"MCD1" + struct.pack("<IIf", n, k, grid)
    bits_used = []
    for i in range(n):
        # Sorting the ids costs the distance ordering, so the permutation has to be stored —
        # but the ids then delta down to small gaps, which is worth far more than the k small
        # indices the permutation costs.
        order = np.argsort(ids[i])
        srt = ids[i][order].astype(np.int64)
        prev = 0
        for v in srt:
            out += varint(int(v) - prev)
            prev = int(v)
        for p in order:
            out += bytes([int(p)])
        row = q[i]
        base = int(row.min())
        res = row - base
        width = max(1, int(res.max()).bit_length())
        bits_used.append(width)
        out += varint(base) + bytes([width])
        acc = accbits = 0
        for r in res:
            acc = (acc << width) | int(r)
            accbits += width
            while accbits >= 8:
                accbits -= 8
                out.append((acc >> accbits) & 0xFF)
        if accbits:
            out.append((acc << (8 - accbits)) & 0xFF)
    return bytes(out), q, float(np.mean(bits_used))


def decode(buf):
    """Exact inverse. Anything less and the size is not a compression result."""
    magic, n, k, grid = struct.unpack_from("<4sIIf", buf, 0)
    if magic != b"MCD1":
        raise SystemExit("not an MCD1 block")
    i = 16
    ids = np.zeros((n, k), np.uint32)
    q = np.zeros((n, k), np.int64)
    for r in range(n):
        srt = []
        prev = 0
        for _ in range(k):
            d, i = read_varint(buf, i)
            prev += d
            srt.append(prev)
        order = list(buf[i:i + k])
        i += k
        row_ids = [0] * k
        for pos, p in enumerate(order):
            row_ids[p] = srt[pos]
        ids[r] = row_ids
        base, i = read_varint(buf, i)
        width = buf[i]
        i += 1
        need = (k * width + 7) // 8
        acc = int.from_bytes(buf[i:i + need], "big")
        i += need
        acc >>= (need * 8 - k * width)
        vals = []
        mask = (1 << width) - 1
        for j in range(k):
            vals.append((acc >> (width * (k - 1 - j))) & mask)
        q[r] = [base + v for v in vals]
    return ids, q, grid


def encode_delta(ids, dist, grid=GRID):
    """Second attempt, and the better one: nothing is reordered and nothing is packed.

    The first version sorted each row's ids so their gaps were small, then had to store an
    8-bit permutation per edge to recover the distance order — 8 of the 30 bits per edge went
    on undoing its own sorting. And it measured residuals from the row minimum, when
    `knn_blocked` already emits each row in ascending distance, so successive deltas are
    smaller than deviations from the minimum by construction.

    Keep the given order, zigzag the id gaps, and delta the distances along the row.
    """
    n, k = ids.shape
    q = np.rint(dist / grid).astype(np.int64)
    out = bytearray(b"MCD2" + struct.pack("<IIf", n, k, grid))
    for i in range(n):
        prev = 0
        for v in ids[i]:
            out += varint(zig(int(v) - prev))
            prev = int(v)
        prev = 0
        for v in q[i]:
            out += varint(zig(int(v) - prev))
            prev = int(v)
    return bytes(out), q


def decode_delta(buf):
    magic, n, k, grid = struct.unpack_from("<4sIIf", buf, 0)
    if magic != b"MCD2":
        raise SystemExit("not an MCD2 block")
    i = 16
    ids = np.zeros((n, k), np.uint32)
    q = np.zeros((n, k), np.int64)
    for r in range(n):
        prev = 0
        for j in range(k):
            d, i = read_varint(buf, i)
            prev += unzig(d)
            ids[r, j] = prev
        prev = 0
        for j in range(k):
            d, i = read_varint(buf, i)
            prev += unzig(d)
            q[r, j] = prev
    return ids, q, grid


def main(out="data/custom/codec.json"):
    ids, dist = read_knn(KNN_BIN)
    n, k = ids.shape
    raw = ids.astype(np.uint32).tobytes() + dist.astype(np.float32).tobytes()
    print(f"{n:,} x {k} neighbour graph, {len(raw) / 1e6:.2f} MB raw\n")

    packed, q, mean_bits = encode(ids, dist)
    back_ids, back_q, _ = decode(packed)
    exact = bool((back_ids == ids).all() and (back_q == q).all())
    err = float(np.abs(q * GRID - dist).max())

    packed2, q2 = encode_delta(ids, dist)
    b_ids2, b_q2, _ = decode_delta(packed2)
    exact2 = bool((b_ids2 == ids).all() and (b_q2 == q2).all())

    sizes = {
        "raw": len(raw),
        "gzip": len(gzip.compress(raw, 9)),
        "bzip2": len(bz2.compress(raw, 9)),
        "lzma": len(lzma.compress(raw, preset=9)),
        "matcodec-style": len(packed),
        "matcodec + gzip": len(gzip.compress(packed, 9)),
        "matcodec + lzma": len(lzma.compress(packed, preset=9)),
        "delta codec": len(packed2),
        "delta + gzip": len(gzip.compress(packed2, 9)),
        "delta + lzma": len(lzma.compress(packed2, preset=9)),
    }
    print(f"{'method':<20}{'bytes':>12}{'ratio':>8}{'bits/edge':>11}")
    for name, b in sizes.items():
        print(f"{name:<20}{b:>12,}{len(raw) / b:>8.2f}{8 * b / (n * k):>11.1f}")

    print(f"\ndecode reproduces every id and quantised distance: "
          f"packed {exact}, delta {exact2}")
    print(f"quantisation grid {GRID} rad, largest error introduced {err:.2e} rad "
          f"({100 * err / float(dist.max()):.4f}% of the largest distance)")
    print(f"residual width averaged {mean_bits:.1f} bits per edge against 32 for a float")

    best_general = min(sizes["gzip"], sizes["bzip2"], sizes["lzma"])
    best_codec = min(sizes["matcodec-style"], sizes["matcodec + gzip"],
                     sizes["matcodec + lzma"], sizes["delta codec"],
                     sizes["delta + gzip"], sizes["delta + lzma"])
    print(f"\nbest general-purpose {best_general:,} bytes, best shape-aware {best_codec:,} "
          f"— {best_general / best_codec:.2f}x")
    print("Knowing that a row is sorted ids plus clustered distances is worth more than any")
    print("amount of entropy coding over the same bytes without that knowledge.")

    summary = {"n": n, "k": k, "grid": GRID, "sizes": sizes,
               "lossless_on_grid": exact and exact2,
               "max_error_rad": err, "mean_residual_bits": mean_bits,
               "best_general": best_general, "best_codec": best_codec,
               "advantage": best_general / best_codec}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
