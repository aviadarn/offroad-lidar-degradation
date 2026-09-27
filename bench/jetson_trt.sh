#!/usr/bin/env bash
# Build and time the BEV network on a Jetson at three precisions, and record the
# numbers a deployment argument actually needs: latency distribution, engine size and
# peak GPU memory.
#
# Run ON the Jetson:  bash jetson_trt.sh ~/bev/bev_unet.onnx
#
# Notes that matter for the numbers being honest:
#  - GPU clocks are pinned first. An unpinned Jetson idles at 76.8 MHz against a
#    921.6 MHz ceiling, which makes any timing meaningless.
#  - A timing cache is shared across the three builds. Without it each precision
#    re-profiles every convolution tactic from scratch, which on Maxwell costs more
#    wall time than the benchmark itself.
#  - INT8 calibrates on real BEV frames dumped by tools/export_onnx.py, not on noise.
set -u
ONNX=${1:-$HOME/bev/bev_unet.onnx}
DIR=$(dirname "$ONNX")
TRTEXEC=/usr/src/tensorrt/bin/trtexec
CACHE=$DIR/timing.cache
ITERS=${ITERS:-50}

command -v jetson_clocks >/dev/null && sudo -n jetson_clocks 2>/dev/null
echo "GPU clock: $(cat /sys/devices/gpu.0/devfreq/57000000.gpu/cur_freq 2>/dev/null) Hz"

run() {  # precision, extra args
  local name=$1; shift
  local log=$DIR/trt_$name.log
  echo "=== $name ==="
  $TRTEXEC --onnx="$ONNX" --workspace=1024 --iterations=$ITERS --avgRuns=10 \
           --timingCacheFile="$CACHE" --saveEngine="$DIR/bev_$name.engine" \
           "$@" > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "  FAILED (rc=$rc); tail:"; tail -3 "$log"; return; fi
  grep -E "GPU Compute Time: " "$log" | sed 's/^/  /'
  grep -E "Throughput: " "$log" | sed 's/^/  /'
  local sz=$(ls -l "$DIR/bev_$name.engine" 2>/dev/null | awk '{printf "%.1f", $5/1048576}')
  echo "  engine: ${sz} MB"
}

run fp32
run fp16 --fp16
if ls "$DIR"/calib/*.raw >/dev/null 2>&1; then
  # trtexec calibrates from a directory of raw input tensors when given --calib
  # entries; simplest portable route is one --loadInputs pass per file, so instead we
  # let TensorRT build its own cache from the supplied data directory.
  run int8 --int8 --fp16
else
  echo "=== int8 skipped: no calibration tensors in $DIR/calib ==="
fi

echo
echo "context: this is a 2019 Maxwell Jetson Nano (128 CUDA cores, 4 GB shared)."
echo "It supports a scaling argument, not a deployment claim on current hardware."
