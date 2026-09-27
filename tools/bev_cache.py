#!/usr/bin/env python3
"""Precompute BEV tensors and per-cell labels for a split, so training is not
bottlenecked re-projecting 300k points per frame on every epoch.

Each frame becomes one .npz: features float16 (5, N, N), labels int8 (N, N) with -1
for empty cells. At 400x400 that is ~1.9 MB per frame compressed, against ~5 MB for
the raw cloud, and it loads in milliseconds.

  python tools/bev_cache.py data/gooseEx_3d_train --split train --platform alice \
      --labels data/challenge_labels_3d/train --out cache/train_alice
"""
import argparse
from pathlib import Path

import numpy as np

import bev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--split", default="train")
    ap.add_argument("--platform", default="alice")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cell", type=float, default=bev.CELL_M)
    ap.add_argument("--extent", type=float, default=bev.EXTENT_M)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0, help="this worker's index, 0-based")
    ap.add_argument("--of", type=int, default=1, help="total workers; frames are split round-robin")
    args = ap.parse_args()

    clouds = sorted((args.root / "lidar" / args.split).rglob("*_pcl.bin"))
    if args.platform:
        clouds = [c for c in clouds if c.parent.name.startswith(args.platform + "_")]
    if args.limit:
        clouds = clouds[: args.limit]
    if args.of > 1:
        clouds = clouds[args.shard :: args.of]   # round-robin so every worker sees every scenario
    if not clouds:
        print(f"no clouds under {args.root}/lidar/{args.split} for platform={args.platform}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    class_cells = np.zeros(bev.N_CLASSES, dtype=np.int64)

    for c in clouds:
        dst = args.out / (c.stem + ".npz")
        if dst.exists():
            skipped += 1
            continue
        lbl_path = args.labels / c.parent.name / c.name.replace("_pcl.bin", "_goose.label")
        if not lbl_path.exists():
            print(f"  no label for {c.name}, skipping")
            continue
        pts = np.fromfile(c, dtype=np.float32).reshape(-1, 4)
        lab = (np.fromfile(lbl_path, dtype=np.uint32) & 0xFFFF).astype(np.int64)
        lab[lab == 8] = 0                      # sky -> other, as the baseline does
        feats = bev.featurise(pts, args.cell, args.extent)
        target = bev.label_image(pts, lab, args.cell, args.extent)
        occupied = target >= 0
        class_cells += np.bincount(target[occupied], minlength=bev.N_CLASSES)
        np.savez_compressed(dst, feats=feats.astype(np.float16), labels=target.astype(np.int8))
        written += 1

    n = bev.grid_size(args.cell, args.extent)
    total = class_cells.sum()
    print(f"wrote {written} frames ({skipped} already cached) to {args.out}  [{n}x{n}]")
    print("cell class balance (this is what the loss has to survive):")
    for i, name in enumerate(["other", "artificial_structures", "artificial_ground",
                              "natural_ground", "obstacle", "vehicle", "vegetation", "human"]):
        share = class_cells[i] / total * 100 if total else 0.0
        print(f"  {i} {name:22s} {class_cells[i]:10,}  {share:6.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
