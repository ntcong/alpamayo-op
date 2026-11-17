#!/usr/bin/env bash
# NOTE: This script has never been run on the Spark. Adapted from edge-setup.sh
# for an aarch64 DGX Spark with a large GPU (stock GPU mode, no hybrid).
#
# Before the first run:
#   1. On huggingface.co accept access terms for nvidia/Alpamayo2-Edge and
#      nvidia/PhysicalAI-Autonomous-Vehicles.
#   2. Create a read token (Settings -> Access Tokens).
# Re-running is safe: finished setup steps are skipped.
set -euo pipefail

# 1. Driver check
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv

# 2. Tools (uv downloads its own Python; system Python is not used)
if ! command -v git >/dev/null || ! command -v uv >/dev/null; then
    echo "git and uv are required. Please install them before proceeding." >&2
    exit 1
fi

# 3. Code, pinned to NVIDIA's commit
mkdir -p "$HOME/alpamayo" && cd "$HOME/alpamayo"
[ -d alpamayo2-edge ] || git clone https://github.com/NVlabs/alpamayo2-edge
cd alpamayo2-edge
git checkout -q 02f55a7aeabc1a2b9f7a80445c08e40bddc89f5e

# 4. Environment: PyTorch + bundled CUDA runtime
export UV_PROJECT_ENVIRONMENT=.venv
uv sync --locked --extra dataset --dev
source .venv/bin/activate

# 5. GPU check: cuda_available and bf16_supported must both print True
python - <<'PY'
import sys
import torch

print("torch:", torch.__version__, "cuda runtime:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("CUDA not available: driver too old or not loaded")
print("device:", torch.cuda.get_device_name(0))
print("bf16_supported:", torch.cuda.is_bf16_supported())
if not torch.cuda.is_bf16_supported():
    sys.exit("BF16 not supported on this GPU")
x = torch.ones(4, device="cuda", dtype=torch.bfloat16)
print("bf16 op ok:", float((x * 2).sum()))
PY

# 6. Hugging Face login
hf auth whoami >/dev/null 2>&1 || hf auth login
hf auth whoami

# 7. Stock GPU verification run
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT="outputs/physicalai-nav-gpu"
python -m alpamayo2_edge.inference \
    --model-id nvidia/Alpamayo2-Edge \
    --revision f8951d095de3589b3face1dec3f4394c8ecedb38 \
    --manifest examples/nav_samples.json \
    --output "$OUT"

cat "$OUT/report.json"
find "$OUT" -name output.json -exec cat {} \;
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
