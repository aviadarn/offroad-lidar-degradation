#!/usr/bin/env python3
"""Run the BEV model over clouds and write per-point .label files.

The output layout matches the challenge labels exactly, so tools/eval_miou.py scores
this model with the same code path that scored PTv3. Predictions can come from the
torch checkpoint (--ckpt) or from a TensorRT engine's saved outputs (--pred-npy-dir),
which is how the Jetson's INT8 numbers get scored without torch on the board.

  python tools/predict_bev.py --ckpt results/runs/bev/best.pt \
      --clouds data/gooseEx_3d_val --split val --platform alice --out results/runs/bev/pred
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
import bev
from model_bev import BEVUNet


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--pred-npy-dir", type=Path, default=None,
                    help="directory of <frame>.npy argmax images, e.g. from the Jetson")
    ap.add_argument("--clouds", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--platform", default="alice")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if (args.ckpt is None) == (args.pred_npy_dir is None):
        raise SystemExit("pass exactly one of --ckpt or --pred-npy-dir")

    model = None
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu")
        model = BEVUNet(width=state.get("width", 32))
        model.load_state_dict(state["model"])
        model.eval().to(args.device)

    clouds = sorted((args.clouds / "lidar" / args.split).rglob("*_pcl.bin"))
    if args.platform:
        clouds = [c for c in clouds if c.parent.name.startswith(args.platform + "_")]
    if args.limit:
        clouds = clouds[: args.limit]

    written = 0
    for c in clouds:
        pts = np.fromfile(c, dtype=np.float32).reshape(-1, 4)
        if model is not None:
            x = torch.from_numpy(bev.featurise(pts)).unsqueeze(0).to(args.device)
            with torch.no_grad():
                pred_img = model(x).argmax(1)[0].cpu().numpy()
        else:
            pred_img = np.load(args.pred_npy_dir / (c.stem + ".npy"))
        per_point = bev.scatter_to_points(pred_img, pts).astype(np.uint32)
        dst = args.out / c.parent.name / c.name.replace("_pcl.bin", "_goose.label")
        dst.parent.mkdir(parents=True, exist_ok=True)
        per_point.tofile(dst)
        written += 1

    print(f"wrote {written} per-point label files to {args.out}")
    print("score with: python tools/eval_miou.py --gt data/challenge_labels_3d/val "
          f"--pred {args.out} --out results/runs/bev/miou.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
