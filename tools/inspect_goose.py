#!/usr/bin/env python3
"""Inventory a GOOSE-Ex 3D split: per-platform frame counts, point statistics,
and semantic class support. Writes JSON so later runs can be diffed against it.

Format (verified against gooseEx_3d_val, 2024-10-25 release):
  lidar/<split>/<platform>_scenarioNN/*_pcl.bin    float32 [x, y, z, intensity]
  labels/<split>/<platform>_scenarioNN/*_goose.label  uint32, semantic = low 16 bits,
                                                      instance id = high 16 bits
"""
import argparse, collections, csv, json, os, sys
from pathlib import Path

import numpy as np


def load_mapping(root: Path) -> dict:
    path = root / "goose_label_mapping.csv"
    with path.open() as f:
        return {int(r["label_key"]): r["class_name"] for r in csv.DictReader(f)}


def frames(root: Path, split: str):
    for pcl in sorted((root / "lidar" / split).rglob("*_pcl.bin")):
        label = Path(str(pcl).replace("/lidar/", "/labels/").replace("_pcl.bin", "_goose.label"))
        yield pcl, (label if label.exists() else None)


def platform_of(pcl: Path) -> str:
    return pcl.parent.name.split("_scenario")[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="extracted split root, e.g. data/gooseEx_3d_val")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sample", type=int, default=0, help="stat only the first N frames (0 = all)")
    args = ap.parse_args()

    names = load_mapping(args.root)
    per_platform = collections.defaultdict(lambda: {"frames": 0, "points": 0, "scenarios": set()})
    class_points = collections.Counter()
    class_frames = collections.Counter()
    unlabeled = 0
    point_counts, intensity_max = [], 0.0

    all_frames = list(frames(args.root, args.split))
    if args.sample:
        all_frames = all_frames[: args.sample]
    if not all_frames:
        print(f"no frames under {args.root}/lidar/{args.split}", file=sys.stderr)
        return 1

    for pcl, label in all_frames:
        plat = platform_of(pcl)
        n_pts = os.path.getsize(pcl) // (4 * 4)
        per_platform[plat]["frames"] += 1
        per_platform[plat]["points"] += n_pts
        per_platform[plat]["scenarios"].add(pcl.parent.name)
        point_counts.append(n_pts)

        pts = np.fromfile(pcl, dtype=np.float32).reshape(n_pts, 4)
        intensity_max = max(intensity_max, float(pts[:, 3].max()))

        if label is None:
            unlabeled += 1
            continue
        sem = np.fromfile(label, dtype=np.uint32) & 0xFFFF
        if sem.size != n_pts:
            print(f"MISMATCH {pcl.name}: {n_pts} points, {sem.size} labels", file=sys.stderr)
        ids, counts = np.unique(sem, return_counts=True)
        for cid, cnt in zip(ids.tolist(), counts.tolist()):
            class_points[cid] += cnt
            class_frames[cid] += 1

    total_points = sum(point_counts)
    report = {
        "root": str(args.root),
        "split": args.split,
        "frames": len(all_frames),
        "unlabeled_frames": unlabeled,
        "points_total": total_points,
        "points_per_frame": {
            "min": int(min(point_counts)),
            "max": int(max(point_counts)),
            "mean": round(total_points / len(point_counts), 1),
        },
        "intensity_max": intensity_max,
        "platforms": {
            p: {
                "frames": v["frames"],
                "points": v["points"],
                "scenarios": sorted(v["scenarios"]),
                "points_per_frame_mean": round(v["points"] / v["frames"], 1),
            }
            for p, v in sorted(per_platform.items())
        },
        "classes_present": len(class_points),
        "classes": [
            {
                "id": cid,
                "name": names.get(cid, f"<unmapped {cid}>"),
                "points": class_points[cid],
                "share": round(class_points[cid] / total_points, 6),
                "frames": class_frames[cid],
            }
            for cid in sorted(class_points, key=lambda c: -class_points[c])
        ],
    }

    out = args.out or Path("results") / f"phase0_data_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"{report['frames']} frames, {total_points/1e6:.1f}M points, {report['classes_present']} classes present")
    for p, v in report["platforms"].items():
        print(f"  {p:6s} {v['frames']:4d} frames  {v['points']/1e6:7.1f}M points  {len(v['scenarios'])} scenarios")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
