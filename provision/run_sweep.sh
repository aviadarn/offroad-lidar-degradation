#!/usr/bin/env bash
# Degradation sweep: score the published baseline on the excavator frames as the
# sensing conditions get worse, one condition at a time.
#
# Each condition gets its own data root, config and experiment directory. That is not
# tidiness: Pointcept reuses an existing <name>_pred.npy instead of recomputing it, so
# a shared output directory would score condition 1's predictions for every condition
# after it, and every curve would come out flat.
set -euo pipefail

ROOT=${ROOT:-/workspace}
POINTCEPT=$ROOT/Pointcept
BEDROCK=$ROOT/bedrock
DATA=$POINTCEPT/data/goose
WORK=$ROOT/sweep
PLATFORM=${PLATFORM:-alice}
RESULTS=$ROOT/results/sweep
mkdir -p "$WORK" "$RESULTS"

# mode:level pairs. Levels are chosen to bracket the point where the curve bends,
# not to be evenly spaced.
CONDITIONS=${CONDITIONS:-"
mount_drift:0.25 mount_drift:0.5 mount_drift:1.0 mount_drift:2.0
ring_dropout:0.1 ring_dropout:0.25 ring_dropout:0.5
dust:0.02 dust:0.05 dust:0.10
rain:0.25 rain:0.5 rain:1.0
range_noise:0.02 range_noise:0.05 range_noise:0.10
intensity_gain:0.5
"}

for COND in $CONDITIONS; do
  MODE=${COND%%:*}; LEVEL=${COND##*:}
  TAG="${MODE}_${LEVEL}"
  OUT="$RESULTS/$TAG.json"
  if [ -s "$OUT" ]; then echo "== $TAG already done, skipping"; continue; fi

  echo "===== $TAG ====="
  CDATA="$WORK/$TAG"
  rm -rf "$CDATA"
  python3 "$BEDROCK/tools/degrade.py" --in "$DATA" --out "$CDATA" \
    --labels "$DATA/labels_challenge/val" --split val --mode "$MODE" --level "$LEVEL" \
    --platform "$PLATFORM"
  # The dataset class looks for labels under labels_challenge/, mirroring the clouds.
  mv "$CDATA/labels" "$CDATA/labels_challenge"

  CFG="configs/goose/sweep-$TAG"
  python3 - "$POINTCEPT/configs/goose/semseg-pt-v3m1-0-base-val-fast.py" "$POINTCEPT/$CFG.py" "$CDATA" <<'PY'
import sys
src, dst, data_root = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(src).read().replace('data_root = "data/goose"', f'data_root = "{data_root}"')
open(dst, "w").write(text)
PY

  EXP="sweep-$TAG"
  mkdir -p "$POINTCEPT/exp/goose/$EXP/model"
  ln -sf "$POINTCEPT/exp/goose/semseg-ptv3-challenge-goose-baseline/model/model_best.pth" \
         "$POINTCEPT/exp/goose/$EXP/model/model_best.pth"

  ( cd "$POINTCEPT" && sh ./scripts/test.sh -g 1 -d goose -c "sweep-$TAG" -n "$EXP" ) \
    > "$WORK/$TAG.log" 2>&1 || { echo "  FAILED - see $WORK/$TAG.log"; tail -5 "$WORK/$TAG.log"; continue; }

  python3 "$BEDROCK/tools/eval_miou.py" \
    --gt "$CDATA/labels_challenge/val" \
    --pred "$POINTCEPT/exp/goose/$EXP/result" \
    --out "$OUT"

  # Degraded clouds are reproducible from the seed; the predictions are not worth
  # the disk either once the confusion matrix is written.
  rm -rf "$CDATA" "$POINTCEPT/exp/goose/$EXP/result"
done

echo "===== sweep complete: $(ls "$RESULTS" | wc -l) conditions ====="
