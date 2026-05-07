#!/usr/bin/env bash
# One-shot DreamZero inference setup for RunPod (H100/H200, PyTorch+CUDA 12.8 image).
#
# What this does:
#   1. Creates a Python 3.11 venv via `uv` and routes all caches into /workspace
#   2. pip install -e ./server  with PyTorch CUDA 12.8 wheels
#   3. Works around the Debian-installed `blinker` package conflict
#   4. Verifies torch CUDA, flash-attn, and key DreamZero deps load cleanly
#   5. Downloads the 61 GB DreamZero-DROID checkpoint from Hugging Face
#
# This script is idempotent: re-running it skips work that's already done.
#
# Usage (from repo root):
#   bash deployment/setup_runpod.sh
#
# Optional env vars:
#   SKIP_MODEL_DOWNLOAD=1   skip the HF model download (useful for CI / dry runs)
#   HF_TOKEN=...            only required if HF rate-limits anonymous downloads

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "=========================================="
echo "DreamZero RunPod setup"
echo "  repo root : ${REPO_ROOT}"
echo "  GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "=========================================="

# ─── Step 1: cache directories (route into /workspace) ────────────────────────
mkdir -p .cache/{uv,pip,hf,torch,triton} logs checkpoints
export UV_CACHE_DIR="${REPO_ROOT}/.cache/uv"
export PIP_CACHE_DIR="${REPO_ROOT}/.cache/pip"
export HF_HOME="${REPO_ROOT}/.cache/hf"
export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export TRITON_CACHE_DIR="${REPO_ROOT}/.cache/triton"

# ─── Step 2: Python 3.11 venv ─────────────────────────────────────────────────
if [ ! -d "${REPO_ROOT}/.venv" ]; then
    echo ">> Creating Python 3.11 venv via uv..."
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv not found, installing..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="${HOME}/.cargo/bin:${PATH}"
    fi
    uv venv --python 3.11 "${REPO_ROOT}/.venv"
fi
# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"
echo ">> Python: $(python --version)"

# ─── Step 3: install DreamZero server package ─────────────────────────────────
if ! python -c "import dreamzero" 2>/dev/null; then
    echo ">> Installing dreamzero server package (this takes ~3 minutes)..."

    # Symlink pyproject.toml from server/ so `pip install -e ./server` resolves
    # all upstream dependencies. We keep the canonical copy at repo root for
    # reference, but the install must run with server/ as the project root.
    cd "${REPO_ROOT}/server"

    # Pre-empt the Debian-installed blinker package: the upstream pyproject
    # depends on flask which depends on a newer blinker. The system blinker
    # (debian-installed) has no RECORD file so pip can't uninstall it cleanly.
    pip install --ignore-installed blinker >/dev/null

    pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
    cd "${REPO_ROOT}"
else
    echo ">> dreamzero already installed, skipping pip install"
fi

# ─── Step 4: dependency sanity check ──────────────────────────────────────────
echo ">> Verifying critical dependencies..."
python - <<'PY'
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
    print(f"  GPU0: {torch.cuda.get_device_name(0)}")
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
    echo "   This takes 10-30 minutes depending on network."
    python -m pip install -q "huggingface_hub[cli]" hf_xet
    hf download GEAR-Dreams/DreamZero-DROID \
        --repo-type model \
        --local-dir "${CKPT_DIR}"
fi

echo ""
echo "=========================================="
echo "Setup complete."
echo ""
echo "Next steps:"
echo "  1. source deployment/activate.sh"
echo "  2. bash deployment/start_server.sh"
echo "  3. Tail logs/server.log to watch warm-up"
echo "  4. Test from same machine:"
echo "       cd server && python test_client_AR.py --port 5000 --num-chunks 2"
echo "=========================================="
