#!/usr/bin/env bash
# Bare CUDA box -> GOOSE 3D challenge data + the official PTv3 baseline, ready to evaluate.
#
# Run inside the Pointcept image:
#   pointcept/pointcept:pytorch2.0.1-cuda11.7-cudnn8-devel
# That image is CUDA 11.7, so the GPU must be sm_80 or older-compatible (A100, A6000,
# V100 are fine; Ada/RTX 4090 at sm_89 needs CUDA >= 11.8 and will not run it).
#
# Downloads ~4.4 GB of point clouds + 226 MB of labels. Idempotent: every step skips
# if its output already exists, so a re-run after a dropped connection is cheap.
set -euo pipefail

ROOT=${ROOT:-/workspace}
POINTCEPT=$ROOT/Pointcept
DATA=$POINTCEPT/data/goose
EXP=$POINTCEPT/exp/goose/semseg-ptv3-challenge-goose-baseline

echo "=== 1. Pointcept (GOOSE fork) ==="
if [ ! -d "$POINTCEPT/.git" ]; then
  git clone -q -b goose https://github.com/FraunhoferIOSB/Pointcept.git "$POINTCEPT"
fi
git -C "$POINTCEPT" log -1 --format='  at %h %ad %s' --date=short

echo "=== 2. Data ==="
mkdir -p "$ROOT/dl" "$DATA"
fetch() {  # url, filename
  if [ ! -s "$ROOT/dl/$2" ]; then
    echo "  downloading $2"
    curl -sL -C - --retry 5 --retry-delay 5 --retry-all-errors -o "$ROOT/dl/$2" "$1"
  else
    echo "  have $2"
  fi
}
fetch https://goose-dataset.de/storage/goose_3d_val.zip            goose_3d_val.zip
fetch https://goose-dataset.de/storage/gooseEx_3d_val.zip          gooseEx_3d_val.zip
fetch https://zenodo.org/api/records/16942462/files/challenge_labels_3d.zip/content challenge_labels_3d.zip

if [ ! -d "$DATA/lidar/val" ]; then
  echo "  extracting point clouds"
  for z in goose_3d_val gooseEx_3d_val; do
    unzip -q -o "$ROOT/dl/$z.zip" -d "$ROOT/dl/$z"
    # Both archives carry lidar/val/<scenario>/; merge them under one root.
    src=$(find "$ROOT/dl/$z" -type d -name val -path '*lidar*' | head -1)
    mkdir -p "$DATA/lidar/val"
    cp -r "$src"/* "$DATA/lidar/val/"
  done
fi
if [ ! -d "$DATA/labels_challenge/val" ]; then
  echo "  extracting challenge labels"
  unzip -q -o "$ROOT/dl/challenge_labels_3d.zip" -d "$ROOT/dl/labels"
  mkdir -p "$DATA/labels_challenge"
  cp -r "$ROOT/dl/labels/val" "$DATA/labels_challenge/val"
  chmod -R u+rwX "$DATA/labels_challenge"   # the zip ships restrictive modes
fi

echo "=== 3. Baseline weights ==="
mkdir -p "$EXP/model"
if [ ! -s "$EXP/model/model_best.pth" ]; then
  curl -sL -o "$EXP/model/model_best.pth" \
    https://bwsyncandshare.kit.edu/s/ExDe9x5gsDCQ2kW/download/challenge_ptv3.pth
fi

echo "=== 4. Inventory ==="
python3 - <<'PY'
import os, collections
from pathlib import Path
data = Path(os.environ.get("ROOT", "/workspace")) / "Pointcept/data/goose"
pcl = sorted((data / "lidar/val").rglob("*.bin"))
lbl = sorted((data / "labels_challenge/val").rglob("*.label"))
per = collections.Counter(
    p.parent.name.split("_scenario")[0] if "_scenario" in p.parent.name else "vehicle" for p in pcl
)
print(f"  point clouds {len(pcl)}   labels {len(lbl)}")
for k, v in sorted(per.items()):
    print(f"    {k:8s} {v}")
gb = sum(p.stat().st_size for p in pcl) / 1e9
print(f"  {gb:.1f} GB of point clouds on disk")
PY

echo "=== ready ==="
echo "  cd $POINTCEPT && sh ./scripts/test.sh -g 1 -d goose -c semseg-pt-v3m1-0-base -n semseg-ptv3-challenge-goose-baseline"
