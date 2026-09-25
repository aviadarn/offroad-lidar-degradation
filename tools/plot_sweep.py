#!/usr/bin/env python3
"""Render the degradation sweep: mIoU against severity, one line per condition.

Reads results/sweep/<mode>_<level>.json written by eval_miou.py plus the clean
excavator number from the baseline run, and writes a single figure plus a tidy CSV.

  python tools/plot_sweep.py --sweep results/sweep --baseline results/runs/baseline_fast.json \
      --out results/figures/degradation.png
"""
import argparse, csv, json, re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Units each condition's level is expressed in, for the x axis and the caption.
UNITS = {
    "mount_drift": "degrees of mast rotation",
    "ring_dropout": "fraction of beams dead",
    "dust": "fraction of returns hitting airborne dust",
    "rain": "attenuation strength",
    "range_noise": "ranging sigma (m)",
    "intensity_gain": "reflectivity gain error",
}
NAME_RE = re.compile(r"^(?P<mode>[a-z_]+)_(?P<level>[0-9.]+)\.json$")


def load(sweep_dir: Path, baseline: Path | None, platform: str):
    series = defaultdict(list)
    clean = None
    if baseline and baseline.exists():
        b = json.loads(baseline.read_text())
        plat = b.get("by_platform", {}).get(platform)
        if plat:
            clean = plat["mIoU"]
    for f in sorted(sweep_dir.glob("*.json")):
        m = NAME_RE.match(f.name)
        if not m:
            continue
        d = json.loads(f.read_text())
        # The sweep degrades one platform, so the overall number is that platform.
        stats = d.get("by_platform", {}).get(platform) or d["overall"]
        series[m["mode"]].append((float(m["level"]), stats["mIoU"], stats["points"]))
    for mode in series:
        series[mode].sort()
    return series, clean


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, default=Path("results/sweep"))
    ap.add_argument("--baseline", type=Path, default=Path("results/runs/baseline_fast.json"))
    ap.add_argument("--platform", default="alice")
    ap.add_argument("--out", type=Path, default=Path("results/figures/degradation.png"))
    args = ap.parse_args()

    series, clean = load(args.sweep, args.baseline, args.platform)
    if not series:
        print(f"no sweep results under {args.sweep}")
        return 1

    rows = [("condition", "level", "mIoU", "delta_vs_clean", "points")]
    for mode, pts in sorted(series.items()):
        for lvl, miou, npts in pts:
            rows.append((mode, lvl, round(miou, 5),
                         round(miou - clean, 5) if clean is not None else "", npts))
    csv_path = args.out.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)

    n = len(series)
    cols = min(3, n)
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 3.3 * rows_n), squeeze=False)
    for ax, (mode, pts) in zip(axes.flat, sorted(series.items())):
        xs = [0.0] + [p[0] for p in pts]
        ys = ([clean] if clean is not None else [pts[0][1]]) + [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", color="#a4561a")
        if clean is not None:
            ax.axhline(clean, color="#405264", lw=1, ls="--", label=f"clean {clean:.3f}")
            ax.legend(fontsize=8, frameon=False)
        ax.set_title(mode.replace("_", " "), fontsize=11)
        ax.set_xlabel(UNITS.get(mode, "level"), fontsize=9)
        ax.set_ylabel("mIoU", fontsize=9)
        ax.set_ylim(0, max(0.9, max(ys) * 1.1))
        ax.grid(alpha=0.25)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(f"PTv3 GOOSE baseline on excavator frames under sensor degradation "
                 f"({args.platform}, {len(series)} conditions)", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out} and {csv_path}")
    for mode, pts in sorted(series.items()):
        worst = min(pts, key=lambda p: p[1])
        print(f"  {mode:16s} clean {clean if clean is not None else float('nan'):.4f} "
              f"-> {worst[1]:.4f} at level {worst[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
