#!/usr/bin/env bash
# Rented GPU box -> trained BEV model, ready to export.
#
# The box fetches its own copy of the data rather than receiving the cache over the
# wire: GOOSE-Ex train is 8.9 GB and a datacentre link pulls it in minutes, where the
# laptop takes hours. Everything here is idempotent, so a dropped connection costs
# nothing.
set -euo pipefail

ROOT=${ROOT:-/workspace}
REPO=$ROOT/bedrock
DATA=$ROOT/data
EPOCHS=${EPOCHS:-40}
PLATFORM=${PLATFORM:-alice}

mkdir -p "$DATA"
cd "$DATA"

fetch() {
  [ -s "$2" ] || curl -sL -C - --retry 5 --retry-all-errors -o "$2" "$1"
  echo "  have $2 ($(du -h "$2" | cut -f1))"
}
echo "=== data ==="
fetch https://goose-dataset.de/storage/gooseEx_3d_train.zip gooseEx_3d_train.zip
fetch https://goose-dataset.de/storage/gooseEx_3d_val.zip   gooseEx_3d_val.zip
fetch https://zenodo.org/api/records/16942462/files/challenge_labels_3d.zip/content challenge_labels_3d.zip

python3 - <<'PY'
import zipfile, os, pathlib
for name, dest in (("gooseEx_3d_train.zip", "gooseEx_3d_train"),
                   ("gooseEx_3d_val.zip", "gooseEx_3d_val"),
                   ("challenge_labels_3d.zip", "challenge_labels_3d")):
    if pathlib.Path(dest).exists():
        print(f"  {dest} already extracted"); continue
    print(f"  extracting {name}")
    with zipfile.ZipFile(name) as z:
        z.extractall(dest)
    for root, dirs, files in os.walk(dest):
        for n in dirs + files:
            p = os.path.join(root, n)
            os.chmod(p, os.stat(p).st_mode | 0o700)
PY

echo "=== python deps ==="
pip install -q --no-input "numpy<2" torch onnx 2>&1 | tail -1

echo "=== bev cache ==="
cd "$REPO"
for split in train val; do
  src=$(find "$DATA" -maxdepth 2 -type d -name "$split" -path '*lidar*' | head -1)
  src=${src%/lidar/$split}
  lbl=$(find "$DATA/challenge_labels_3d" -maxdepth 2 -type d -name "$split" | head -1)
  python3 tools/bev_cache.py "$src" --split "$split" --platform "$PLATFORM" \
      --labels "$lbl" --out "$ROOT/cache/${split}_${PLATFORM}"
done

echo "=== train ==="
python3 tools/train_bev.py --train "$ROOT/cache/train_${PLATFORM}" \
    --val "$ROOT/cache/val_${PLATFORM}" --epochs "$EPOCHS" --batch 8 \
    --device cuda --out "$ROOT/results/bev"

echo "=== export ==="
python3 tools/export_onnx.py --ckpt "$ROOT/results/bev/best.pt" \
    --out "$ROOT/results/bev/bev_unet.onnx" --calib "$ROOT/cache/val_${PLATFORM}" --calib-n 32

echo "=== per-point predictions for scoring ==="
python3 tools/predict_bev.py --ckpt "$ROOT/results/bev/best.pt" \
    --clouds "$(find "$DATA" -maxdepth 1 -type d -name 'gooseEx_3d_val')" \
    --split val --platform "$PLATFORM" --out "$ROOT/results/bev/pred" --device cuda
python3 tools/eval_miou.py --gt "$DATA/challenge_labels_3d/val" \
    --pred "$ROOT/results/bev/pred" --out "$ROOT/results/bev/miou.json"
echo "=== done: pull $ROOT/results/bev ==="
