#!/usr/bin/env bash
# Make the Pointcept image actually usable on whatever GPU we rented, then verify.
#
# The GOOSE fork's image recipe compiles pointops for TORCH_CUDA_ARCH_LIST
# "5.2 6.0 6.1 7.0+PTX 8.0" and pins flash-attn 2.6.3. A cubin built for sm_80 does not
# load on sm_86 (A6000) or sm_89 (L40S) — only the 7.0 PTX can JIT that far — so on
# anything but an A100 the custom ops can fail at import with "no kernel image
# available". This script checks that first and rebuilds only what is broken.
set -euo pipefail

ROOT=${ROOT:-/workspace}
POINTCEPT=$ROOT/Pointcept

echo "=== GPU / toolchain ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 - <<'PY'
import torch
cap = torch.cuda.get_device_capability(0)
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  device sm_{cap[0]}{cap[1]}")
print("arch list built into torch:", torch.cuda.get_arch_list())
PY

echo "=== custom ops ==="
ARCH=$(python3 -c "import torch;c=torch.cuda.get_device_capability(0);print(f'{c[0]}.{c[1]}')")
check_op() {
  python3 - "$1" <<'PY'
import sys, torch
mod = sys.argv[1]
try:
    __import__(mod)
    if mod == "pointops":
        import pointops, torch
        xyz = torch.rand(1024, 3, device="cuda")
        off = torch.tensor([1024], dtype=torch.int32, device="cuda")
        pointops.farthest_point_sampling(xyz, off, torch.tensor([128], dtype=torch.int32, device="cuda"))
    print(f"OK {mod}")
except Exception as e:
    print(f"BROKEN {mod}: {type(e).__name__}: {str(e)[:160]}")
    raise SystemExit(1)
PY
}

if ! check_op pointops; then
  echo "  rebuilding pointops for sm_$ARCH"
  TORCH_CUDA_ARCH_LIST="$ARCH" pip install -q --no-build-isolation "$POINTCEPT/libs/pointops" 2>&1 | tail -3
  check_op pointops || { echo "pointops still broken - stop here"; exit 1; }
fi

python3 -c "import flash_attn; print('flash_attn', flash_attn.__version__)" 2>/dev/null || {
  echo "  flash-attn missing; the config sets enable_flash=True"
  echo "  install: pip install flash-attn==2.6.3 --no-build-isolation   (or set enable_flash=False)"
}
python3 -c "import spconv; print('spconv', spconv.__version__)" 2>/dev/null || echo "  spconv missing (PTv3 does not need it)"
echo "=== bootstrap done ==="
