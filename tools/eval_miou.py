#!/usr/bin/env python3
"""Confusion-matrix mIoU for GOOSE 3D challenge predictions, sliced by platform.

Ground truth and predictions are both SemanticKITTI-style .label files: uint32 per
point, semantic id in the low 16 bits. The challenge taxonomy is 9 keys (0-8); class
8 (sky) never occurs in lidar returns, so by default it is dropped from the mean the
same way the Pointcept baseline config does (num_classes=8).

Slices come from the scenario directory name: everything before "_scenario" is the
platform (alice = Liebherr R924 excavator, spot = quadruped, otherwise the GOOSE
offroad vehicle). A single prediction pass therefore yields overall, per-platform and
per-scenario numbers with no extra inference.

  python tools/eval_miou.py --gt data/challenge_labels_3d/val --pred runs/base/pred \
      --out results/runs/base/miou.json
"""
import argparse, json, sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

CLASS_NAMES = [
    "other", "artificial_structures", "artificial_ground", "natural_ground",
    "obstacle", "vehicle", "vegetation", "human", "sky",
]


def platform_of(scenario: str) -> str:
    if "_scenario" in scenario:
        return scenario.split("_scenario")[0]
    return "vehicle"


def read_labels(path: Path) -> np.ndarray:
    return (np.fromfile(path, dtype=np.uint32) & 0xFFFF).astype(np.int64)


def accumulate(conf: np.ndarray, gt: np.ndarray, pred: np.ndarray, k: int) -> None:
    valid = (gt >= 0) & (gt < k) & (pred >= 0) & (pred < k)
    np.add.at(conf, (gt[valid], pred[valid]), 1)


def iou_from_conf(conf: np.ndarray):
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    acc = np.where(conf.sum(1) > 0, tp / np.maximum(conf.sum(1), 1), np.nan)
    return iou, acc


def summarize(conf: np.ndarray, eval_classes) -> dict:
    iou, acc = iou_from_conf(conf)
    sel = [c for c in eval_classes if not np.isnan(iou[c])]
    total = conf.sum()
    return {
        "points": int(total),
        "mIoU": float(np.mean([iou[c] for c in sel])) if sel else float("nan"),
        "mAcc": float(np.mean([acc[c] for c in sel])) if sel else float("nan"),
        "allAcc": float(np.diag(conf).sum() / total) if total else float("nan"),
        "classes_present": len(sel),
        "per_class": {
            CLASS_NAMES[c]: {
                "iou": None if np.isnan(iou[c]) else round(float(iou[c]), 5),
                "acc": None if np.isnan(acc[c]) else round(float(acc[c]), 5),
                "gt_points": int(conf[c].sum()),
            }
            for c in eval_classes
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True, help="ground-truth label root (…/val)")
    ap.add_argument("--pred", type=Path, required=True, help="prediction label root, same layout")
    ap.add_argument("--num-classes", type=int, default=9)
    ap.add_argument("--ignore", type=int, nargs="*", default=[8],
                    help="class ids excluded from the mean (default: sky)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    k = args.num_classes
    eval_classes = [c for c in range(k) if c not in set(args.ignore)]

    gt_files = sorted(args.gt.rglob("*.label"))
    if not gt_files:
        print(f"no ground-truth labels under {args.gt}", file=sys.stderr)
        return 1

    overall = np.zeros((k, k), dtype=np.int64)
    by_platform, by_scenario = {}, {}
    missing, mismatched = [], []

    for g in gt_files:
        scenario = g.parent.name
        p = args.pred / scenario / g.name
        if not p.exists():
            missing.append(str(g.relative_to(args.gt)))
            continue
        gt, pred = read_labels(g), read_labels(p)
        if gt.size != pred.size:
            mismatched.append(f"{scenario}/{g.name}: gt {gt.size} vs pred {pred.size}")
            continue
        plat = platform_of(scenario)
        for conf in (overall,
                     by_platform.setdefault(plat, np.zeros((k, k), dtype=np.int64)),
                     by_scenario.setdefault(scenario, np.zeros((k, k), dtype=np.int64))):
            accumulate(conf, gt, pred, k)

    evaluated = len(gt_files) - len(missing) - len(mismatched)
    if evaluated == 0:
        print("nothing evaluated: every frame was missing or size-mismatched", file=sys.stderr)
        for m in mismatched[:5]:
            print("  " + m, file=sys.stderr)
        return 1

    report = OrderedDict(
        gt_root=str(args.gt), pred_root=str(args.pred),
        frames_total=len(gt_files), frames_evaluated=evaluated,
        frames_missing=len(missing), frames_mismatched=len(mismatched),
        ignored_classes=args.ignore,
        overall=summarize(overall, eval_classes),
        by_platform={p: summarize(c, eval_classes) for p, c in sorted(by_platform.items())},
        by_scenario={s: summarize(c, eval_classes) for s, c in sorted(by_scenario.items())},
    )
    if missing:
        report["missing_examples"] = missing[:10]
    if mismatched:
        report["mismatched_examples"] = mismatched[:10]

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")

    o = report["overall"]
    print(f"frames {evaluated}/{len(gt_files)}   mIoU {o['mIoU']:.4f}  mAcc {o['mAcc']:.4f}  allAcc {o['allAcc']:.4f}")
    for p, s in report["by_platform"].items():
        print(f"  {p:8s} mIoU {s['mIoU']:.4f}   ({s['points']/1e6:.1f}M points, {s['classes_present']} classes)")
    if missing or mismatched:
        print(f"  WARNING: {len(missing)} missing, {len(mismatched)} size-mismatched", file=sys.stderr)
    if args.out:
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
