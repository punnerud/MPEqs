#!/usr/bin/env python3
"""Verify the traversal arithmetic with real uncached reads.

`traversal_cost.py` prices a graph walk from the fitted cost model. That model is measured, but
the walk's cost is still arithmetic on top of it: N scattered small reads against one large
sequential read. This does those reads for real, with F_NOCACHE, and checks the prediction.

No file is created — an existing model file supplies the bytes, so this measures the device and
not the filesystem's willingness to allocate. Any file at least `scan_bytes` long would do.
"""
import ctypes
import json
import os
import random
import subprocess
import sys
import time

F_NOCACHE, F_RDAHEAD = 48, 45
ALIGN = 4096


def uncached(path):
    fd = os.open(path, os.O_RDONLY)
    libc = ctypes.CDLL("libc.dylib", use_errno=True)
    for cmd, arg in ((F_NOCACHE, 1), (F_RDAHEAD, 0)):
        if libc.fcntl(fd, cmd, arg) < 0:
            raise OSError(ctypes.get_errno(), f"fcntl {cmd}")
    return fd


def pread_aligned(fd, off, size):
    """F_NOCACHE requires page-aligned offsets and lengths, so widen to the enclosing pages."""
    lo = (off // ALIGN) * ALIGN
    hi = ((off + size + ALIGN - 1) // ALIGN) * ALIGN
    return len(os.pread(fd, hi - lo, lo))


def main(path=None, cost="data/costmodel.json", traversal="data/traversal-cost.json",
         out="data/traversal-verify.json", reps=5):
    if path is None:
        # Largest available file: more room to place each repetition somewhere untouched.
        import glob
        cands = glob.glob("models/*.gguf")
        if not cands:
            sys.exit("no model file to read bytes from; pass a path")
        path = max(cands, key=os.path.getsize)   # largest: most untouched room to read from
    size = os.path.getsize(path)
    cm = json.load(open(cost))
    c_fetch, c_byte = cm["c_fetch_ns"], cm["c_byte_ns"]
    tv = json.load(open(traversal))
    row = max(tv["by_k"], key=lambda r: r["k"])

    rec, nfetch = row["record_bytes"], int(round(row["fetches_laid_out"]))
    scan = int(row["scan_mib"] * 2**20)
    print(f"backing file {path} ({size / 2**30:.1f} GiB)")
    print(f"k={row['k']}: scan {scan / 2**20:.1f} MiB sequential vs {nfetch} reads of "
          f"{rec} B scattered\n")

    # Never offset 0: that is the GGUF header, which every other tool in this repo has just
    # read, so it is the one region guaranteed to be warm. The first attempt did exactly that
    # and reported 27 GB/s for the scan — three times the device's peak, i.e. a cache hit
    # wearing a measurement's clothes. Same failure as the 35x calibration bug, same fix.
    # A fresh region per repetition, and never offset 0. Two things had to be fixed here.
    # Offset 0 is the GGUF header, which every other tool in this repo has just read, so it is
    # guaranteed warm — the first attempt reported 27 GB/s for the scan. Re-reading one region
    # `reps` times is the same bug spread over time: F_NOCACHE stops a read from *populating*
    # the cache, but a page that is already resident is still served from it, so repetition 1
    # pays for the disk and 2..5 do not. The median then reports the cache. Both are the 35x
    # calibration bug in another costume, which is why the guard below is not optional.
    # Evict first, exactly as `fetchbench calibrate` does, then discard a warm-up read: the
    # first uncached reads after an idle period measure the SSD controller waking up.
    subprocess.run(["./scripts/drop-cache.sh"], stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, check=False)

    # Scan reps live in the first half, walk offsets in the second. Without that split the two
    # measurements read each other's pages warm and the second one wins for no reason.
    rng = random.Random(11)
    half = size // 2
    stride = half // (reps + 2)
    warm = uncached(path)
    pread_aligned(warm, half // 2 // ALIGN * ALIGN, 1 << 20)
    os.close(warm)
    res = {}
    for label, mk in (
        ("scan", lambda r: [(min(half - scan, stride * (r + 1)) // ALIGN * ALIGN, scan)]),
        ("walk", lambda r: [(rng.randrange(half, size - rec), rec) for _ in range(nfetch)]),
    ):
        times, moved = [], 0
        for r in range(reps):
            reads = mk(r)
            fd = uncached(path)                    # fresh fd per rep: no page reuse at all
            t0 = time.perf_counter_ns()
            got = sum(pread_aligned(fd, o, s) for o, s in reads)
            times.append(time.perf_counter_ns() - t0)
            os.close(fd)
            moved, nreads = got, len(reads)
        med = sorted(times)[len(times) // 2] / 1e6
        gbs = moved / (med / 1e3) / 1e9
        ceiling = 1e9 / c_byte / 1e9 * 1.5          # 50 % over the calibrated asymptote
        if gbs > ceiling:
            sys.exit(f"CONTAMINATED: {label} achieved {gbs:.1f} GB/s against a calibrated "
                     f"ceiling of {ceiling:.1f} GB/s — those pages were resident. "
                     f"Run scripts/drop-cache.sh (needs sudo) and retry.")
        pred = (nreads * c_fetch + moved * c_byte) / 1e6
        res[label] = {"measured_ms": round(med, 3), "predicted_ms": round(pred, 3),
                      "bytes": moved, "fetches": nreads}
        res[label]["gb_per_s"] = round(gbs, 2)
        print(f"{label:>5}  measured {med:8.2f} ms   predicted {pred:8.2f} ms   "
              f"ratio {med / pred:5.2f}x   ({moved / 2**20:.1f} MiB at {gbs:.2f} GB/s)")

    r_meas = res["walk"]["measured_ms"] / res["scan"]["measured_ms"]
    r_pred = res["walk"]["predicted_ms"] / res["scan"]["predicted_ms"]
    print(f"\nwalk/scan  measured {r_meas:.2f}x   predicted {r_pred:.2f}x")
    print("The walk is predicted to lose at this index size, and it does." if r_meas > 1
          else "The walk wins at this index size.")
    res["walk_over_scan_measured"] = round(r_meas, 2)
    res["walk_over_scan_predicted"] = round(r_pred, 2)
    json.dump(res, open(out, "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
