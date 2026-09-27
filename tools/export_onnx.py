#!/usr/bin/env python3
"""Export the trained BEV network to ONNX for TensorRT 8.2 on JetPack 4.6.

Opset 11 and static shapes on purpose: TensorRT 8.2 is from 2021, and a dynamic-shape
engine buys nothing here because the BEV grid is fixed by the featuriser. Also dumps a
calibration set of real BEV tensors as raw float32, which is what the INT8 build on the
board consumes - calibrating on random noise would produce a quantisation that looks
fine in a benchmark and falls apart on a jobsite.

  python tools/export_onnx.py --ckpt results/runs/bev/best.pt --calib cache/val_alice
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from model_bev import BEVUNet, N_IN
import bev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/runs/bev/bev_unet.onnx"))
    ap.add_argument("--opset", type=int, default=11)
    ap.add_argument("--calib", type=Path, default=None, help="cache dir to draw calibration frames from")
    ap.add_argument("--calib-n", type=int, default=32)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    state = torch.load(args.ckpt, map_location="cpu")
    model = BEVUNet(width=state.get("width", 32))
    model.load_state_dict(state["model"])
    model.eval()

    n = bev.grid_size()
    dummy = torch.zeros(1, N_IN, n, n)
    torch.onnx.export(model, dummy, str(args.out), opset_version=args.opset,
                      input_names=["bev"], output_names=["logits"],
                      do_constant_folding=True)
    size_mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out}  ({size_mb:.1f} MB, opset {args.opset}, static 1x{N_IN}x{n}x{n})")

    try:
        import onnx
        m = onnx.load(str(args.out))
        onnx.checker.check_model(m)
        ops = sorted({node.op_type for node in m.graph.node})
        print(f"  checker passed. operators: {ops}")
    except ImportError:
        print("  onnx not installed locally; skipping graph check")

    if args.calib:
        cal_dir = args.out.parent / "calib"
        cal_dir.mkdir(exist_ok=True)
        files = sorted(Path(args.calib).glob("*.npz"))[: args.calib_n]
        for f in files:
            arr = np.load(f)["feats"].astype(np.float32)
            arr.tofile(cal_dir / (f.stem + ".raw"))
        print(f"  wrote {len(files)} calibration tensors to {cal_dir} "
              f"({N_IN}x{n}x{n} float32 each, real frames not noise)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
