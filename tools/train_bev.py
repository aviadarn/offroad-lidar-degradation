#!/usr/bin/env python3
"""Train the BEV segmentation network, then export it to ONNX for the Jetson.

Scoring is deliberately not done on the BEV image. Predictions are scattered back to
every point in the original cloud and written as SemanticKITTI-style .label files, so
tools/eval_miou.py scores this model exactly the way it scored the published PTv3
baseline - same confusion matrix, same platform slicing, same ignore rules. A number
that is not comparable to the baseline is not worth having.

  python tools/train_bev.py --train cache/train_alice --val cache/val_alice \
      --epochs 40 --out results/runs/bev
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent))
from model_bev import BEVUNet, N_CLASSES

CLASS_NAMES = ["other", "artificial_structures", "artificial_ground", "natural_ground",
               "obstacle", "vehicle", "vegetation", "human"]


class BEVCache(Dataset):
    def __init__(self, root: Path):
        self.files = sorted(Path(root).glob("*.npz"))
        if not self.files:
            raise SystemExit(f"no cached frames in {root} - run tools/bev_cache.py first")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        return (torch.from_numpy(d["feats"].astype(np.float32)),
                torch.from_numpy(d["labels"].astype(np.int64)))


def class_weights(loader, n_classes=N_CLASSES, device="cpu"):
    """Inverse-sqrt-frequency weights. Obstacle is ~0.3% of cells and is the class the
    published baseline fails at, so an unweighted loss would simply ignore it."""
    counts = torch.zeros(n_classes, dtype=torch.float64)
    for _, y in loader:
        v = y[y >= 0]
        counts += torch.bincount(v.flatten(), minlength=n_classes).double()
    freq = counts / counts.sum()
    w = 1.0 / torch.sqrt(freq.clamp(min=1e-8))
    w = w / w.mean()
    return w.float().to(device), counts


def confusion(pred, target, n=N_CLASSES):
    m = target >= 0
    idx = target[m] * n + pred[m]
    return torch.bincount(idx, minlength=n * n).reshape(n, n)


def miou_from_conf(conf):
    tp = conf.diag().double()
    denom = conf.sum(0).double() + conf.sum(1).double() - tp
    iou = torch.where(denom > 0, tp / denom.clamp(min=1), torch.full_like(tp, float("nan")))
    return iou, torch.nanmean(iou).item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("results/runs/bev"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tr = DataLoader(BEVCache(args.train), batch_size=args.batch, shuffle=True, num_workers=4)
    va = DataLoader(BEVCache(args.val), batch_size=args.batch, shuffle=False, num_workers=4)
    print(f"train {len(tr.dataset)} frames | val {len(va.dataset)} frames | device {args.device}")

    w, counts = class_weights(tr, device=args.device)
    print("class weights:", " ".join(f"{CLASS_NAMES[i][:12]}={w[i]:.2f}" for i in range(N_CLASSES)))

    model = BEVUNet(width=args.width).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * max(1, len(tr)))
    loss_fn = nn.CrossEntropyLoss(weight=w, ignore_index=-1)

    best = -1.0
    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, y in tr:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step(); sched.step()
            tot += loss.item(); nb += 1

        model.eval()
        conf = torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.long)
        with torch.no_grad():
            for x, y in va:
                p = model(x.to(args.device)).argmax(1).cpu()
                conf += confusion(p.flatten(), y.flatten())
        iou, mio = miou_from_conf(conf)
        history.append({"epoch": ep, "loss": tot / max(nb, 1), "cell_mIoU": mio})
        flag = ""
        if mio > best:
            best = mio
            torch.save({"model": model.state_dict(), "width": args.width}, args.out / "best.pt")
            flag = " *"
        print(f"  epoch {ep:3d}  loss {tot/max(nb,1):.4f}  val cell-mIoU {mio:.4f}  "
              f"obstacle IoU {iou[4]:.4f}  {time.time()-t0:5.1f}s{flag}")

    (args.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"best val cell-mIoU {best:.4f} -> {args.out/'best.pt'}")
    print("note: cell-mIoU is not the reported metric. Run tools/predict_bev.py to write "
          "per-point .label files and score them with tools/eval_miou.py, the same scorer "
          "used on the PTv3 baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
