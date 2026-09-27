#!/usr/bin/env python3
"""Bird's-eye-view featurisation of GOOSE-Ex clouds, for a model that can be deployed.

Why BEV and not a range image: the excavator frames in GOOSE-Ex are *accumulated*
clouds, not single sweeps - a 0.1 deg x 0.1 deg direction holds ~20 returns spanning
metres of range. A spherical projection therefore throws away 66-96% of the points
(measured, see tools/range_project.py), while a 0.20 m BEV grid over +/-40 m keeps
100% of them and preserves 94.1% of per-point labels under a per-cell prediction.

Channels per cell, all cheap to compute and all meaningful to a convolution:
  0 occupancy   1 if any return fell here
  1 max z       the top of whatever is here
  2 min z       the ground under it
  3 mean intensity
  4 log density  log1p(count), which separates a wall from a leaf

Labels are the per-cell majority class; -1 marks empty cells so the loss can ignore
them. Predictions go back to points by cell membership, so accuracy is always reported
per point on the full cloud.
"""
import argparse
from pathlib import Path

import numpy as np

CELL_M = 0.20
EXTENT_M = 40.0
N_CLASSES = 8


def grid_size(cell: float = CELL_M, extent: float = EXTENT_M) -> int:
    return int(2 * extent / cell)


def cell_ids(points: np.ndarray, cell: float = CELL_M, extent: float = EXTENT_M):
    """-> (cell index per in-grid point, mask of in-grid points, grid size)."""
    n = grid_size(cell, extent)
    x, y = points[:, 0], points[:, 1]
    inside = (np.abs(x) < extent) & (np.abs(y) < extent)
    ix = ((x[inside] + extent) / cell).astype(np.int64)
    iy = ((y[inside] + extent) / cell).astype(np.int64)
    return ix * n + iy, inside, n


def featurise(points: np.ndarray, cell: float = CELL_M, extent: float = EXTENT_M) -> np.ndarray:
    cid, inside, n = cell_ids(points, cell, extent)
    z, inten = points[inside, 2], points[inside, 3]
    size = n * n

    count = np.bincount(cid, minlength=size).astype(np.float32)
    occupied = count > 0
    zsum = np.bincount(cid, weights=z, minlength=size)
    isum = np.bincount(cid, weights=inten, minlength=size)

    zmax = np.full(size, 0.0, dtype=np.float32)
    zmin = np.full(size, 0.0, dtype=np.float32)
    np.maximum.at(zmax, cid, z)          # both start at 0 and are only read where
    np.minimum.at(zmin, cid, z)          # occupied, so the init value never leaks out

    feats = np.zeros((5, size), dtype=np.float32)
    feats[0, occupied] = 1.0
    feats[1] = zmax
    feats[2] = zmin
    feats[3, occupied] = (isum[occupied] / count[occupied]).astype(np.float32) / 255.0
    feats[4] = np.log1p(count)
    return feats.reshape(5, n, n)


def label_image(points: np.ndarray, labels: np.ndarray, cell: float = CELL_M,
                extent: float = EXTENT_M) -> np.ndarray:
    """Per-cell majority class; -1 where no point fell."""
    cid, inside, n = cell_ids(points, cell, extent)
    lab = labels[inside]
    size = n * n
    counts = np.zeros((N_CLASSES, size), dtype=np.int32)
    for c in range(N_CLASSES):
        counts[c] = np.bincount(cid[lab == c], minlength=size)
    total = counts.sum(axis=0)
    out = np.full(size, -1, dtype=np.int64)
    occupied = total > 0
    out[occupied] = counts[:, occupied].argmax(axis=0)
    return out.reshape(n, n)


def scatter_to_points(pred_image: np.ndarray, points: np.ndarray, cell: float = CELL_M,
                      extent: float = EXTENT_M, fill: int = 0) -> np.ndarray:
    """Per-cell predictions -> per-point predictions, for scoring on the full cloud."""
    cid, inside, n = cell_ids(points, cell, extent)
    out = np.full(len(points), fill, dtype=np.int64)
    out[inside] = pred_image.reshape(-1)[cid]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--labels", type=Path, default=Path("data/challenge_labels_3d/val"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--platform", default="alice")
    ap.add_argument("--cell", type=float, default=CELL_M)
    ap.add_argument("--extent", type=float, default=EXTENT_M)
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()

    clouds = sorted((args.root / "lidar" / args.split).rglob("*_pcl.bin"))
    if args.platform:
        clouds = [c for c in clouds if c.parent.name.startswith(args.platform + "_")]
    clouds = clouds[: args.limit] if args.limit else clouds

    agree = total = 0
    occ = []
    for c in clouds:
        pts = np.fromfile(c, dtype=np.float32).reshape(-1, 4)
        lbl = args.labels / c.parent.name / c.name.replace("_pcl.bin", "_goose.label")
        lab = (np.fromfile(lbl, dtype=np.uint32) & 0xFFFF).astype(np.int64)
        lab[lab == 8] = 0
        img = label_image(pts, lab, args.cell, args.extent)
        back = scatter_to_points(img, pts, args.cell, args.extent)
        agree += int((back == lab).sum())
        total += len(lab)
        feats = featurise(pts, args.cell, args.extent)
        occ.append(float((feats[0] > 0).mean()))

    n = grid_size(args.cell, args.extent)
    print(f"{len(clouds)} {args.platform} clouds, {n}x{n} grid at {args.cell} m over +/-{args.extent} m")
    print(f"  cells occupied:        {np.mean(occ)*100:5.1f}%")
    print(f"  per-point label ceiling: {agree/total*100:5.2f}%   (round-trip through the grid)")
    print(f"  input tensor:          5 x {n} x {n} = {5*n*n/1e6:.2f}M values, "
          f"{5*n*n*4/1e6:.1f} MB fp32 / {5*n*n/1e6:.1f} MB int8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
