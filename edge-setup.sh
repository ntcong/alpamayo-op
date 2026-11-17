#!/usr/bin/env bash
# Set up NVIDIA Alpamayo 2 Edge and run its public sample on a Linux PC (written for CachyOS + RTX 2060 Super).
#
# Usage: bash edge-setup.sh [hybrid|gpu|cpu]      (default: hybrid)
#   hybrid  vision encoder + projector on the CPU, language model + trajectory expert on the GPU.
#           For 8 GB cards: the plain gpu mode ran out of memory on the 2060 Super (2026-11-11).
#           Not NVIDIA's validated setup: the encoder runs in float32 on the CPU, so outputs can differ slightly.
#   gpu     NVIDIA's stock run, everything on the GPU (needs more than 8 GB).
#   cpu     NVIDIA's stock CPU run (--device cpu --dtype float32): slow, needs about 16 GB of free RAM.
#
# Before the first run:
#   1. On huggingface.co accept the access terms of nvidia/Alpamayo2-Edge and
#      nvidia/PhysicalAI-Autonomous-Vehicles (the sample clip comes from the dataset).
#   2. Create a read token (Settings -> Access Tokens); step 6 asks for it interactively.
#   3. Keep about 15 GB of disk free and close browsers / other GPU apps.
# Re-running is safe: finished setup steps are skipped.
set -euo pipefail

MODE="${1:-hybrid}"
case "$MODE" in
    hybrid | gpu | cpu) ;;
    *) echo "usage: bash edge-setup.sh [hybrid|gpu|cpu]" >&2; exit 2 ;;
esac

# 1. Driver check: must list the RTX 2060 SUPER
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv

# 2. Tools (uv downloads its own Python 3.12; system Python is not used)
if ! command -v git >/dev/null || ! command -v uv >/dev/null; then
    sudo pacman -S --needed git uv
fi

# 3. Code, pinned to the commit these steps were written against
mkdir -p "$HOME/alpamayo" && cd "$HOME/alpamayo"
[ -d alpamayo2-edge ] || git clone https://github.com/NVlabs/alpamayo2-edge
cd alpamayo2-edge
git checkout -q 02f55a7aeabc1a2b9f7a80445c08e40bddc89f5e

# 4. Environment: PyTorch 2.8.0 + bundled CUDA 12 runtime, no CUDA toolkit or flash-attn build needed
export UV_PROJECT_ENVIRONMENT=.venv
uv sync --locked --extra dataset --dev
source .venv/bin/activate

# 5. GPU check: cuda_available and bf16_supported must both print True (not needed for cpu mode)
if [ "$MODE" != cpu ]; then
    python - <<'PY'
import sys
import torch

print("torch:", torch.__version__, "cuda runtime:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("CUDA not available: driver too old for the bundled CUDA 12 runtime, or not loaded")
print("device:", torch.cuda.get_device_name(0))
print("bf16_supported:", torch.cuda.is_bf16_supported())
if not torch.cuda.is_bf16_supported():
    sys.exit("BF16 not supported on this GPU")
x = torch.ones(4, device="cuda", dtype=torch.bfloat16)
print("bf16 op ok:", float((x * 2).sum()))
PY
fi

# 6. Hugging Face login, only when not logged in yet (paste the read token when asked)
hf auth whoami >/dev/null 2>&1 || hf auth login
hf auth whoami

# 7. Hybrid launcher: NVIDIA's CLI with the vision encoder and projector moved to the CPU.
#    Kept outside NVIDIA's checkout; it changes nothing in their code.
cat > "$HOME/alpamayo/edge_hybrid.py" <<'PY'
"""Run NVIDIA's alpamayo2_edge.inference CLI with the vision encoder and projector on the CPU.

For 8 GB cards: frees about 0.9 GiB of GPU weights plus the encoder's working memory. Takes the same
arguments as `python -m alpamayo2_edge.inference` (not --accel). Not NVIDIA's validated setup: the encoder
runs in float32 on the CPU instead of BF16 on the GPU, so sampled text and trajectories can differ slightly.
"""
import time

import torch
from alpamayo2_edge import checkpoint, inference

_load_model = checkpoint.load_model


def load_model(*args, **kwargs):
    model, info = _load_model(*args, **kwargs)
    core = model.vlm.model  # Cosmos3EdgeModel: visual, projector, language_model
    visual = core.visual.to("cpu").float()
    projector = core.projector.to("cpu").float()
    # Unregister both: model.dtype reads the first parameter and must stay bfloat16 for autocast
    del core.visual, core.projector

    def get_image_features(pixel_values, image_grid_thw=None, **_):
        # Same steps as Cosmos3EdgeModel.get_image_features, on the CPU; forward() moves the result to the GPU
        start = time.perf_counter()
        grid = image_grid_thw.cpu()
        out = visual(pixel_values.to("cpu", torch.float32), grid_thw=grid, return_dict=True)
        embeds = projector(out.last_hidden_state)
        out.pooler_output = torch.split(embeds, (grid.prod(-1) // projector.spatial_merge_size**2).tolist())
        print(f"vision encoder on CPU: {time.perf_counter() - start:.1f} s for {len(grid)} images", flush=True)
        return out

    core.get_image_features = get_image_features
    torch.cuda.empty_cache()
    print(f"GPU weights after moving the vision encoder: {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    return model, info


checkpoint.load_model = load_model  # inference.main() imports it from the module at call time
inference.main()
PY

# 8. Run NVIDIA's sample: one clip, "Turn right in 30m", one trajectory
#    Watch memory in a second terminal: watch -n1 nvidia-smi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT="outputs/physicalai-nav-$MODE"
ARGS=(--model-id nvidia/Alpamayo2-Edge
      --revision f8951d095de3589b3face1dec3f4394c8ecedb38
      --manifest examples/nav_samples.json
      --output "$OUT")
case "$MODE" in
    hybrid) python "$HOME/alpamayo/edge_hybrid.py" "${ARGS[@]}" ;;
    gpu) python -m alpamayo2_edge.inference "${ARGS[@]}" ;;
    cpu) python -m alpamayo2_edge.inference "${ARGS[@]}" --device cpu --dtype float32 ;;
esac

# 9. Results: report.json = versions and settings; output.json = reasoning text and elapsed seconds;
#    the trajectory (pred_xyz, 64 waypoints) is in output.safetensors next to output.json
cat "$OUT/report.json"
find "$OUT" -name output.json -exec cat {} \;
if [ "$MODE" != cpu ]; then
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv
fi

# If hybrid still ends in "CUDA out of memory": send the 15 traceback lines above the error (they name the
# stage that ran out), then try `bash edge-setup.sh cpu`.
