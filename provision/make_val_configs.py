#!/usr/bin/env python3
"""Derive validation-split configs from the GOOSE PTv3 challenge config.

The shipped config tests on ["test","testEx"], which has no public labels, with a
10-way test-time augmentation (five scales x flip). Two variants are needed here:

  -val-tta   the same 10-way TTA, but on ["val"], where labels exist. This is the
             protocol the published val number (mIoU 0.8096) was produced under, so
             it is the anchor to reproduce.
  -val-fast  a single forward pass, no augmentation. ~10x cheaper, and the right
             tool for a degradation sweep where what matters is the delta between
             conditions measured the same way, not the absolute best number.
"""
import argparse, re
from pathlib import Path

SINGLE_AUG = """            aug_transform=[
                [dict(type="RandomScale", scale=[1, 1])],
            ],
"""


def patch(src: str, split: str, single_pass: bool) -> str:
    out = src.replace('split=["test","testEx"],', f'split=["{split}"],')
    if out == src:
        raise SystemExit("test split line not found - upstream config changed")
    if single_pass:
        start = out.index("            aug_transform=[")
        end = out.index("            ],\n", out.index("RandomFlip", start)) + len("            ],\n")
        out = out[:start] + SINGLE_AUG + out[end:]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=Path, default=Path("configs/goose"))
    ap.add_argument("--base", default="semseg-pt-v3m1-0-base")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    src = (args.configs / f"{args.base}.py").read_text()
    for name, single in ((f"{args.base}-{args.split}-tta", False), (f"{args.base}-{args.split}-fast", True)):
        dst = args.configs / f"{name}.py"
        dst.write_text(patch(src, args.split, single))
        n_aug = len(re.findall(r"\[dict\(type=\"RandomScale\"", dst.read_text().split("aug_transform=[")[1]))
        print(f"wrote {dst}  ({n_aug} augmentation pass{'es' if n_aug > 1 else ''})")


if __name__ == "__main__":
    main()
