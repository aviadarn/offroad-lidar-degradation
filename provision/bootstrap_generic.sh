#!/usr/bin/env bash
# Bring a stock PyTorch CUDA box up to what PTv3 needs, without the Pointcept image.
#
# The published pointcept/pointcept image is ~8 GB and stalls mid-pull on every vast
# host I tried (three hosts, one layer, ~25 min each). A stock pytorch image is cached
# almost everywhere and boots in a minute; the only things it lacks are the two custom
# ops, and pointops compiles in a couple of minutes for the exact arch we rented.
set -euo pipefail

ROOT=${ROOT:-/workspace}
POINTCEPT=$ROOT/Pointcept

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
ARCH=$(python3 -c "import torch;c=torch.cuda.get_device_capability(0);print(f'{c[0]}.{c[1]}')")
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"
echo "  device arch sm_${ARCH/./}"

echo "=== repo ==="
if [ ! -d "$POINTCEPT/.git" ]; then
  git clone -q -b goose https://github.com/FraunhoferIOSB/Pointcept.git "$POINTCEPT"
fi
git -C "$POINTCEPT" log -1 --format='  at %h %s' 

echo "=== numpy floor/ceiling ==="
# spconv 2.3.6 and cumm 0.4.11 are built against the numpy 1.x C ABI. Under numpy 2.x
# the first sparse convolution dies with a bare SIGFPE inside cumm's from_numpy - no
# Python traceback, no error message, just "Floating point exception". Installing any
# of the deps below can pull numpy 2 in, so the pin is applied first and re-asserted
# after.
pip install -q --no-input "numpy<2" 2>&1 | tail -1

echo "=== python deps ==="
pip install -q --no-input \
  addict einops timm yapf termcolor plyfile scipy h5py ftfy regex \
  tensorboard tensorboardx sharedarray open3d-cpu 2>&1 | tail -2 || \
pip install -q --no-input addict einops timm yapf termcolor plyfile scipy h5py ftfy regex tensorboard tensorboardx sharedarray 2>&1 | tail -2

echo "=== pointops (compiled for sm_${ARCH/./}) ==="
python3 -c "import pointops" 2>/dev/null && echo "  already present" || \
  TORCH_CUDA_ARCH_LIST="$ARCH" pip install -q --no-build-isolation "$POINTCEPT/libs/pointops" 2>&1 | tail -3

echo "=== flash-attn ==="
if ! python3 -c "import flash_attn" 2>/dev/null; then
  # Building from source takes 30+ minutes. Try the wheel first; if it is unavailable
  # for this torch/CUDA pair, PTv3 runs without it via enable_flash=False (slower, and
  # the numbers shift slightly, which has to be reported if it happens).
  pip install -q --no-input flash-attn --no-build-isolation 2>&1 | tail -3 || echo "  flash-attn unavailable"
fi
python3 -c "import flash_attn;print('  flash_attn',flash_attn.__version__)" 2>/dev/null || echo "  NO flash-attn: config must set enable_flash=False"

pip install -q --no-input "numpy<2" 2>&1 | tail -1   # re-assert after the deps above

echo "=== verify ==="
python3 - <<'PY'
import numpy, torch, pointops
assert numpy.__version__.startswith("1."), f"numpy {numpy.__version__} breaks spconv/cumm"
print("  numpy", numpy.__version__)
xyz = torch.rand(4096, 3, device="cuda")
off = torch.tensor([4096], dtype=torch.int32, device="cuda")
new_off = torch.tensor([512], dtype=torch.int32, device="cuda")
idx = pointops.farthest_point_sampling(xyz, off, new_off)
print("  pointops OK, sampled", idx.shape[0], "points")

import spconv.pytorch as spconv
N = 4096
idx3 = torch.randint(0, 64, (N, 3), dtype=torch.int32)
indices = torch.cat([torch.zeros((N, 1), dtype=torch.int32), idx3], dim=1).cuda()
x = spconv.SparseConvTensor(torch.rand(N, 32).cuda(), indices, [64, 64, 64], 1)
out = spconv.SubMConv3d(32, 32, 3, bias=False, indice_key="v").cuda()(x)
print("  spconv OK, out", tuple(out.features.shape))
PY
echo "=== bootstrap done ==="
