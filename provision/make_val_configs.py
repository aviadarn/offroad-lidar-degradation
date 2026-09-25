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
import argparse, ast
from pathlib import Path

SINGLE_AUG = """            aug_transform=[
                [dict(type="RandomScale", scale=[1, 1])],
            ],
"""


def replace_aug_transform(src: str, replacement: str) -> str:
    """Swap the whole aug_transform=[...] block for a single-pass one.

    The block spans ten multi-line entries, so it is located by matching brackets
    from its opening '[' rather than by searching for a closing line — a naive slice
    cuts mid-expression and produces a config that only fails once the GPU is already
    rented, which is exactly what happened the first time.
    """
    key = "aug_transform=["
    start = src.index(key)
    depth, i = 0, start + len(key) - 1
    while i < len(src):
        if src[i] == "[":
            depth += 1
        elif src[i] == "]":
            depth -= 1
            if depth == 0:
                break
        i += 1
    else:
        raise SystemExit("aug_transform block never closes")
    end = i + 1
    while end < len(src) and src[end] in ",\n":       # take the trailing comma/newline too
        end += 1
    line_start = src.rindex("\n", 0, start) + 1
    return src[:line_start] + replacement + src[end:]


def patch(src: str, split: str, single_pass: bool) -> str:
    out = src.replace('split=["test","testEx"],', f'split=["{split}"],')
    if out == src:
        raise SystemExit("test split line not found - upstream config changed")
    if single_pass:
        out = replace_aug_transform(out, SINGLE_AUG)
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
        text = patch(src, args.split, single)
        # Parse before writing: a broken config otherwise surfaces only after the box
        # has loaded the model. Count the augmentation passes from the AST, not a regex.
        tree = ast.parse(text)
        n_aug = max(
            (len(node.value.elts) for node in ast.walk(tree)
             if isinstance(node, ast.keyword) and node.arg == "aug_transform"
             and isinstance(node.value, ast.List)),
            default=0,
        )
        dst.write_text(text)
        print(f"wrote {dst}  ({n_aug} augmentation pass{'es' if n_aug != 1 else ''})")


if __name__ == "__main__":
    main()
