#!/usr/bin/env python3
"""Deterministic sensor-degradation transforms for GOOSE 3D point clouds.

A retrofit autonomy kit lives on a machine that vibrates, works in dust, and gets
washed down. These transforms perturb a clean scan the way that environment would,
so a model trained on clean data can be scored as the conditions worsen. Every
transform is seeded per file, so a sweep is reproducible and a rerun is free.

Clouds are float32 [x, y, z, intensity]; labels are carried through unchanged except
where points are removed, in which case the same mask is applied to the labels so a
degraded cloud still has per-point ground truth.

  python tools/degrade.py --in data/gooseEx_3d_val --out data/degraded/dust_0.05 \
      --labels data/challenge_labels_3d/val --mode dust --level 0.05
"""
import argparse, json, hashlib
from pathlib import Path

import numpy as np

MODES = ("ring_dropout", "random_dropout", "mount_drift", "range_noise", "dust", "rain", "intensity_gain")


def seed_for(name: str, mode: str, level: float, base: int) -> int:
    h = hashlib.sha256(f"{base}|{mode}|{level}|{name}".encode()).digest()
    return int.from_bytes(h[:4], "little")


def ring_index(pts: np.ndarray, n_rings: int = 128) -> np.ndarray:
    """Approximate the beam (ring) each return came from.

    The .bin format carries no ring field, so rings are recovered from the elevation
    angle. Bins are equal-population (quantiles), not equal-angle: a real sensor's
    beams are spaced non-uniformly in elevation but each beam contributes roughly the
    same number of returns per revolution, so equal-angle bins would make "drop 25% of
    beams" mean "drop 9% of points" — understating the damage. Still an approximation,
    but one that keeps the level parameter honest.
    """
    r = np.linalg.norm(pts[:, :3], axis=1)
    elev = np.arcsin(np.clip(pts[:, 2] / np.maximum(r, 1e-6), -1.0, 1.0))
    edges = np.quantile(elev, np.linspace(0.0, 1.0, n_rings + 1)[1:-1])
    return np.searchsorted(edges, elev).astype(np.int32)


def apply(mode: str, level: float, pts: np.ndarray, rng: np.random.Generator):
    """Return (points, keep_mask). keep_mask is None when no point was removed."""
    xyz = pts[:, :3]
    rad = np.linalg.norm(xyz, axis=1)

    if mode == "ring_dropout":
        rings = ring_index(pts)
        uniq = np.unique(rings)
        dead = rng.random(len(uniq)) < level
        keep = ~np.isin(rings, uniq[dead])
        return pts[keep], keep

    if mode == "random_dropout":
        keep = rng.random(len(pts)) >= level
        return pts[keep], keep

    if mode == "mount_drift":
        # level is degrees; split across pitch and roll, the two axes a bolted-on
        # sensor mast actually shifts in. Rotation is about the sensor origin.
        a = np.deg2rad(level)
        cp, sp, cr, sr = np.cos(a), np.sin(a), np.cos(a / 2), np.sin(a / 2)
        pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        out = pts.copy()
        # numpy 2.0 on macOS/Accelerate raises spurious divide-by-zero, overflow and
        # invalid FP warnings from matmul even when every input and output is finite
        # (verified: identical warnings for f32 and f64 operands, output max |x| sane).
        with np.errstate(all="ignore"):
            rotated = xyz @ (roll @ pitch).T
        assert np.isfinite(rotated).all(), "rotation produced non-finite coordinates"
        out[:, :3] = rotated
        return out, None

    if mode == "range_noise":
        # level is sigma in metres along the beam.
        direction = xyz / np.maximum(rad, 1e-6)[:, None]
        out = pts.copy()
        out[:, :3] = xyz + direction * rng.normal(0.0, level, size=len(pts))[:, None]
        return out, None

    if mode == "dust":
        # Airborne particulate: a fraction of beams terminate early on a dust cloud,
        # returning a short range with weak intensity instead of the true surface.
        hit = rng.random(len(pts)) < level
        out = pts.copy()
        if hit.any():
            frac = rng.uniform(0.15, 0.6, size=hit.sum())
            out[hit, :3] = xyz[hit] * frac[:, None]
            out[hit, 3] = out[hit, 3] * rng.uniform(0.1, 0.4, size=hit.sum())
        return out, None

    if mode == "rain":
        # Attenuation grows with range: far returns are lost first, survivors dim.
        p_drop = np.clip(level * rad / 20.0, 0.0, 0.95)
        keep = rng.random(len(pts)) >= p_drop
        out = pts[keep].copy()
        out[:, 3] *= np.exp(-level * np.linalg.norm(out[:, :3], axis=1) / 40.0)
        return out, keep

    if mode == "intensity_gain":
        # Reflectivity calibration drift: multiplicative gain, clipped to the 0-255
        # range the dataset uses.
        out = pts.copy()
        out[:, 3] = np.clip(out[:, 3] * (1.0 + level), 0.0, 255.0)
        return out, None

    raise ValueError(f"unknown mode {mode}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path, required=True, help="split root containing lidar/<split>/…")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=None, help="challenge label root to carry through")
    ap.add_argument("--split", default="val")
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--level", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--platform", default=None, help="only frames from this platform, e.g. alice")
    args = ap.parse_args()

    files = sorted((args.src / "lidar" / args.split).rglob("*_pcl.bin"))
    if args.platform:
        files = [f for f in files if f.parent.name.startswith(args.platform + "_")]
    if not files:
        print(f"no frames under {args.src}/lidar/{args.split}")
        return 1

    kept_total = pts_total = 0
    for f in files:
        scenario = f.parent.name
        pts = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
        rng = np.random.default_rng(seed_for(f.name, args.mode, args.level, args.seed))
        out_pts, keep = apply(args.mode, args.level, pts, rng)

        dst = args.out / "lidar" / args.split / scenario / f.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        out_pts.astype(np.float32).tofile(dst)

        if args.labels:
            src_lbl = args.labels / scenario / f.name.replace("_pcl.bin", "_goose.label")
            if src_lbl.exists():
                lbl = np.fromfile(src_lbl, dtype=np.uint32)
                if keep is not None:
                    lbl = lbl[keep]
                dst_lbl = args.out / "labels" / args.split / scenario / src_lbl.name
                dst_lbl.parent.mkdir(parents=True, exist_ok=True)
                lbl.tofile(dst_lbl)

        pts_total += len(pts)
        kept_total += len(out_pts)

    manifest = {
        "mode": args.mode, "level": args.level, "seed": args.seed, "split": args.split,
        "platform": args.platform, "frames": len(files),
        "points_in": pts_total, "points_out": kept_total,
        "retained": round(kept_total / pts_total, 5),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{args.mode}@{args.level}: {len(files)} frames, {kept_total/1e6:.1f}M / {pts_total/1e6:.1f}M points "
          f"({manifest['retained']*100:.1f}% retained) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
