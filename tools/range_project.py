#!/usr/bin/env python3
"""Project GOOSE-Ex point clouds into range images, and back again.

A point transformer is not deployable on an embedded board - it needs custom CUDA ops
that do not exist for Maxwell-era Jetsons. A range image turns the scan into a dense
2D tensor a plain convolutional network can consume, which exports to ONNX and
quantizes to INT8 without exotic kernels.

The projection is spherical: azimuth across the columns, elevation down the rows,
nearest return wins a cell. Labels ride along so the same projection produces training
targets. Unprojection carries per-pixel predictions back to every original point via
the saved index map, so accuracy is always reported on the full cloud, never on the
image - the image is a compute convenience, not the thing being scored.

  python tools/range_project.py data/gooseEx_3d_val --split val --check
"""
import argparse
from pathlib import Path

import numpy as np

# 64 rows x 1024 columns: the excavator's sensor puts ~268k points in a scan, so this
# keeps the image dense enough to be worth convolving while staying small enough for a
# 4 GB board.
DEFAULT_H, DEFAULT_W = 64, 1024
FOV_UP_DEG, FOV_DOWN_DEG = 22.5, -22.5


def project(points: np.ndarray, h: int = DEFAULT_H, w: int = DEFAULT_W,
            fov_up: float = FOV_UP_DEG, fov_down: float = FOV_DOWN_DEG):
    """points: (N, 4) xyzi -> (image (5, h, w) float32, index map (h, w) int32).

    Channels are range, x, y, z, intensity. The index map holds the source point for
    each occupied cell and -1 elsewhere, which is what makes unprojection exact.
    """
    xyz = points[:, :3]
    depth = np.linalg.norm(xyz, axis=1)
    valid = depth > 1e-6
    xyz, depth = xyz[valid], depth[valid]
    src = np.flatnonzero(valid)
    intensity = points[valid, 3]

    yaw = -np.arctan2(xyz[:, 1], xyz[:, 0])
    pitch = np.arcsin(np.clip(xyz[:, 2] / depth, -1.0, 1.0))

    fov_up_rad, fov_down_rad = np.deg2rad(fov_up), np.deg2rad(fov_down)
    fov = fov_up_rad - fov_down_rad

    u = 0.5 * (yaw / np.pi + 1.0) * w
    v = (1.0 - (pitch - fov_down_rad) / fov) * h
    u = np.clip(np.floor(u), 0, w - 1).astype(np.int32)
    v = np.clip(np.floor(v), 0, h - 1).astype(np.int32)

    # Farthest-first so that nearer returns overwrite: the closest surface is the one
    # the machine has to see.
    order = np.argsort(-depth)
    u, v, src = u[order], v[order], src[order]
    depth_o, xyz_o, inten_o = depth[order], xyz[order], intensity[order]

    image = np.zeros((5, h, w), dtype=np.float32)
    index = np.full((h, w), -1, dtype=np.int32)
    image[0, v, u] = depth_o
    image[1, v, u] = xyz_o[:, 0]
    image[2, v, u] = xyz_o[:, 1]
    image[3, v, u] = xyz_o[:, 2]
    image[4, v, u] = inten_o
    index[v, u] = src
    return image, index


def project_labels(labels: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Per-point labels -> (h, w) int32 label image; empty cells get -1 (ignore)."""
    out = np.full(index.shape, -1, dtype=np.int32)
    occupied = index >= 0
    out[occupied] = labels[index[occupied]]
    return out


def unproject(pred_image: np.ndarray, index: np.ndarray, n_points: int,
              fill: int = 0) -> np.ndarray:
    """(h, w) predictions -> (n_points,) predictions.

    Points that no cell captured (two returns landing in one cell) keep `fill`. The
    fraction that happens is exactly the accuracy ceiling this projection imposes, so
    --check reports it rather than hiding it.
    """
    out = np.full(n_points, fill, dtype=np.int32)
    occupied = index >= 0
    out[index[occupied]] = pred_image[occupied]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--split", default="val")
    ap.add_argument("--labels", type=Path, default=None,
                    help="challenge label root, to measure the projection's ceiling")
    ap.add_argument("--height", type=int, default=DEFAULT_H)
    ap.add_argument("--width", type=int, default=DEFAULT_W)
    ap.add_argument("--platform", default=None)
    ap.add_argument("--check", action="store_true", help="report coverage, do not write")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    clouds = sorted((args.root / "lidar" / args.split).rglob("*_pcl.bin"))
    if args.platform:
        clouds = [c for c in clouds if c.parent.name.startswith(args.platform + "_")]
    if args.limit:
        clouds = clouds[: args.limit]
    if not clouds:
        print(f"no clouds under {args.root}/lidar/{args.split}")
        return 1

    covered, total, occupancy = 0, 0, []
    for c in clouds:
        pts = np.fromfile(c, dtype=np.float32).reshape(-1, 4)
        image, index = project(pts, args.height, args.width)
        covered += int((index >= 0).sum())
        total += len(pts)
        occupancy.append(float((index >= 0).mean()))

    print(f"{len(clouds)} clouds at {args.height}x{args.width}")
    print(f"  points kept by the projection: {covered/total*100:.2f}%  "
          f"({covered/1e6:.1f}M of {total/1e6:.1f}M)")
    print(f"  image cells occupied: {np.mean(occupancy)*100:.1f}% mean")
    print(f"  accuracy ceiling from collisions alone: {covered/total*100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
