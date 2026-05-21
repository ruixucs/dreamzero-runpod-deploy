#!/usr/bin/env bash
# One-shot DreamZero inference setup for RunPod (H100/H200/B200, PyTorch+CUDA 12.8 image).
#
# What this does:
#   1. Creates a Python 3.11 venv via `uv` (optional) and routes all caches into /workspace
#   2. pip install -e ./server  with PyTorch CUDA 12.8 wheels (python -m pip)
#   3. Installs hf_transfer, flash-attn (prebuilt wheel on cp312/cp311)
#   4. Verifies torch CUDA, flash-attn, and groot
#   5. Downloads the 61 GB DreamZero-DROID checkpoint (or run prefetch_models.sh for Wan2.1 too)
#
# See docs/05_deployment_b200.md for B200 second-deploy timings and gotchas.
#
# This script is idempotent: re-running it skips work that's already done.
#
# Usage (from repo root):
#   bash deployment/setup_runpod.sh
#
# Optional env vars:
#   SKIP_MODEL_DOWNLOAD=1   skip the HF model download (useful for CI / dry runs)
#   SKIP_WAN_PREFETCH=1     skip Wan2.1 prefetch (default: run prefetch if missing)
#   HF_TOKEN=...            only required if HF rate-limits anonymous downloads

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

REPO_ROOT="$(_dreamzero_repo_root)"
cd "${REPO_ROOT}"

echo "=========================================="
echo "DreamZero RunPod setup"
echo "  repo root : ${REPO_ROOT}"
echo "  GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "=========================================="

dreamzero_export_caches "${REPO_ROOT}"

# ─── Step 2: Python 3.11 venv (optional; system python3.12 may be used instead) ─
if [ ! -d "${REPO_ROOT}/.venv" ]; then
    echo ">> Creating Python 3.11 venv via uv..."
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv not found, installing..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="${HOME}/.cargo/bin:${PATH}"
    fi
    uv venv --python 3.11 "${REPO_ROOT}/.venv"
    # uv venv does not always include pip — bootstrap it
    if [ -x "${REPO_ROOT}/.venv/bin/python" ] && ! "${REPO_ROOT}/.venv/bin/python" -m pip --version >/dev/null 2>&1; then
        "${REPO_ROOT}/.venv/bin/python" -m ensurepip --upgrade 2>/dev/null || true
    fi
fi

PYTHON="$(dreamzero_select_python "${REPO_ROOT}")"
echo ">> Using Python: ${PYTHON} ($(${PYTHON} --version 2>&1))"

# ─── Step 3: install DreamZero server package ─────────────────────────────────
if ! "${PYTHON}" -c "import groot.vla.model.n1_5.sim_policy" 2>/dev/null; then
    echo ">> Installing dreamzero server package (this takes ~3–8 minutes)..."

    cd "${REPO_ROOT}/server"

    dreamzero_pip "${PYTHON}" install --ignore-installed blinker >/dev/null

    dreamzero_pip "${PYTHON}" install -e . --extra-index-url https://download.pytorch.org/whl/cu128
    cd "${REPO_ROOT}"
else
    echo ">> groot already importable, skipping pip install -e server"
fi

dreamzero_pip "${PYTHON}" install -q hf_transfer hf_xet "huggingface_hub[cli]"

dreamzero_install_flash_attn "${PYTHON}"

# ─── Step 4: dependency sanity check ──────────────────────────────────────────
echo ">> Verifying critical dependencies..."
"${PYTHON}" - <<'PY'
import importlib, sys

mods = ["torch", "flash_attn", "transformers", "diffusers", "deepspeed",
        "huggingface_hub", "groot.vla.model.n1_5.sim_policy"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "<no __version__>")
        print(f"  OK  {m}: {ver}")
    except Exception as e:
        print(f"  FAIL  {m}: {e}", file=sys.stderr)
        sys.exit(1)

import torch
print(f"  CUDA: available={torch.cuda.is_available()}, "
      f"version={torch.version.cuda}, "
      f"GPUs={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"  GPU0: {torch.cuda.get_device_name(0)} "
          f"cap={torch.cuda.get_device_capability(0)}")
PY

# ─── Step 5: model checkpoint ─────────────────────────────────────────────────
CKPT_DIR="${REPO_ROOT}/checkpoints/DreamZero-DROID"
if [ "${SKIP_MODEL_DOWNLOAD:-0}" = "1" ]; then
    echo ">> SKIP_MODEL_DOWNLOAD=1, skipping checkpoint download"
elif [ -f "${CKPT_DIR}/model.safetensors.index.json" ]; then
    SIZE=$(du -sh "${CKPT_DIR}" | cut -f1)
    echo ">> Checkpoint already at ${CKPT_DIR} (${SIZE}), skipping download"
else
    echo ">> Downloading DreamZero-DROID (~61 GB) to ${CKPT_DIR}..."
    echo "   Tip: bash deployment/prefetch_models.sh downloads Wan2.1 too (~77 GB)."
    export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
    "${PYTHON}" - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="GEAR-Dreams/DreamZero-DROID",
    repo_type="model",
    local_dir="${CKPT_DIR}",
)
PY
fi

# ─── Step 6: optional Wan2.1 prefetch (avoids ~6 min delay at server start) ───
WAN_MARKER="${HF_HOME}/hub/models--Wan-AI--Wan2.1-I2V-14B-480P"
if [ "${SKIP_WAN_PREFETCH:-0}" = "1" ]; then
    echo ">> SKIP_WAN_PREFETCH=1"
elif [ -d "${WAN_MARKER}" ] && [ -n "$(ls -A "${WAN_MARKER}/blobs" 2>/dev/null)" ]; then
    echo ">> Wan2.1 backbone already in HF cache ($(du -sh "${WAN_MARKER}" | cut -f1)), skipping"
else
    echo ">> Prefetching Wan2.1-I2V-14B-480P (~77 GB) — recommended before first server start..."
    export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
    "${PYTHON}" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Wan-AI/Wan2.1-I2V-14B-480P", repo_type="model")
print("Wan2.1 prefetch done.")
PY
fi

# Record selected python for start_server.sh
echo "${PYTHON}" > "${REPO_ROOT}/.python-interpreter"

echo ""
echo "=========================================="
echo "Setup complete."
echo "  Python interpreter saved to .python-interpreter: ${PYTHON}"
echo ""
echo "Next steps:"
echo "  1. source deployment/activate.sh"
echo "  2. bash deployment/start_server.sh"
echo "  3. tail -f logs/server-*.log"
echo "  4. Test:"
echo "       cd server && ${PYTHON} test_client_AR.py --port 5000 --num-chunks 2"
echo ""
echo "B200 / second deploy notes: docs/05_deployment_b200.md"
echo "=========================================="
