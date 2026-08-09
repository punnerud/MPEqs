#!/usr/bin/env python3
"""Does the landmark structure appear as the network learns, or is it flat from the start?

`landmark.rs` found under 1 % of blocks index-exact on every layer of both models, and the
conclusion drawn was that the embedding geometry has no narrow cuts for a skeleton to exploit.
There is an obvious objection to that: embeddings only mean something once the network has been
trained, and one of the two models — `mul`, at 0.046 held out — had barely learned anything at
all. A geometry measured on a half-trained representation may say more about the training budget
than about embeddings.

So measure the trajectory instead of one point. Same architecture, same held-out sample at every
checkpoint, only the number of gradient steps varies. Three outcomes and they mean different
things:

  rises with accuracy   the negative result was about undertrained embeddings, not embeddings
  flat                  training does not create narrow cuts, and the conclusion stands
  falls                 the structure was an artefact of an undifferentiated initial geometry,
                        which is what the `mul` > `add` inversion already hinted at

The last one is worth naming in advance because it is the least comfortable and the easiest to
overlook: at step 0 the embeddings are near-random projections of a one-hot input, and a
skeleton may fit *that* better than it fits a representation the network has actually shaped.

Writes one embedding file per checkpoint for `matstruct landmark` to score.

Run with a torch venv, e.g. /Users/punnerud/Downloads/ainmt/venv/bin/python3.
"""
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from dump_layers import (HOPS, LayerMoE, TASKS, batch, block_compression,  # noqa: E402
                         sample)

CHECKPOINTS = (0, 300, 1200, 4800, 19200)


def evaluate(m, items, spec, dev, chunk=4096):
    ok = 0
    states = []
    with torch.no_grad():
        for i in range(0, len(items), chunk):
            x, sg, dg = batch(items[i:i + chunk], spec, dev)
            sl, dl, st, _, _ = m(x)
            good = sl.argmax(-1) == sg
            for j, d in enumerate(dl):
                good &= d.argmax(-1) == dg[:, j]
            ok += int(good.sum())
            states.append(st[:, HOPS - 1].to("cpu").float())
    return ok / len(items), torch.cat(states)


def main(n=8000, out="data/custom/traj", tasks="add,mul", seed=0):
    n, seed = int(n), int(seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"final-layer embeddings at steps {CHECKPOINTS}, {n:,} held-out points, {dev}\n")
    print(f"{'task':>5} {'steps':>7} {'accuracy':>9} {'file':>28}")

    summary = {}
    for task in tasks.split(","):
        spec = TASKS[task]
        # One fixed held-out sample, drawn once, reused at every checkpoint — otherwise the
        # geometry and the sample would move together and neither could be attributed.
        items = sample(random.Random(999), n, spec, holdout=True)
        torch.manual_seed(seed)
        rng = random.Random(seed)
        m = LayerMoE(spec).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3)
        rows = []
        done = 0
        t0 = time.time()
        for target in CHECKPOINTS:
            while done < target:
                b = sample(rng, 384, spec, holdout=False)
                x, sg, dg = batch(b, spec, dev)
                sl, dl, _, probs, _ = m(x)
                loss = F.cross_entropy(sl, sg)
                for j, d in enumerate(dl):
                    loss = loss + F.cross_entropy(d, dg[:, j])
                loss = loss + 0.3 * block_compression(probs)
                opt.zero_grad(); loss.backward(); opt.step()
                done += 1
            acc, st = evaluate(m, items, spec, dev)
            name = f"{task}-s{target}.f32"
            (out_dir / name).write_bytes(st.numpy().astype("<f4").tobytes())
            rows.append({"steps": target, "accuracy": round(acc, 4), "file": name})
            print(f"{task:>5} {target:>7} {acc:>9.3f} {name:>28}")
        summary[task] = {"n": n, "dim": st.shape[1], "seed": seed,
                         "checkpoints": rows, "seconds": round(time.time() - t0, 1)}
    (out_dir / "trajectory.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir}/trajectory.json and "
          f"{sum(len(v['checkpoints']) for v in summary.values())} embedding files")
    print("Score each with: matstruct landmark -e <file> --dim 64 -L 32 --rows 600")


if __name__ == "__main__":
    main(*sys.argv[1:])
