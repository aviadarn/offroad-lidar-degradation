#!/usr/bin/env bash
# Evaluate the published PTv3 GOOSE baseline on the val split and slice the result by
# platform. Assumes setup_box.sh has run.
set -euo pipefail

ROOT=${ROOT:-/workspace}
POINTCEPT=$ROOT/Pointcept
EXP_NAME=${EXP_NAME:-semseg-ptv3-challenge-goose-baseline}
MODE=${1:-fast}          # fast | tta
LIMIT=${LIMIT:-0}        # optional: evaluate only the first N clouds (smoke test)

cd "$POINTCEPT"
python3 "$ROOT/bedrock/provision/make_val_configs.py" --configs configs/goose

CFG="semseg-pt-v3m1-0-base-val-$MODE"
echo "=== evaluating $CFG ==="
START=$(date +%s)
sh ./scripts/test.sh -g 1 -d goose -c "$CFG" -n "$EXP_NAME" 2>&1 | tee "$ROOT/test_$MODE.log" | tail -30
echo "=== wall clock: $(( $(date +%s) - START ))s ==="

PRED_DIR="$POINTCEPT/exp/goose/$EXP_NAME/result"
echo "predictions: $(ls "$PRED_DIR" 2>/dev/null | wc -l) files in $PRED_DIR"

python3 "$ROOT/bedrock/tools/eval_miou.py" \
  --gt "$POINTCEPT/data/goose/labels_challenge/val" \
  --pred "$PRED_DIR" \
  --out "$ROOT/results/baseline_$MODE.json"
