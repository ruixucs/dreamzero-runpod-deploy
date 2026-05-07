#!/bin/bash
# Source this file (not execute) to activate the DreamZero venv and route all
# Python / HF / pip / triton caches into /workspace so they survive across pod
# restarts and don't blow up the small overlay disk.
#
# Usage:
#   source deployment/activate.sh

# Resolve the repo root robustly (works whether sourced from anywhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ ! -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    echo "ERROR: ${REPO_ROOT}/.venv not found. Run deployment/setup_runpod.sh first." >&2
    return 1 2>/dev/null || exit 1
fi

source "${REPO_ROOT}/.venv/bin/activate"

# Route caches into /workspace (RunPod persistent volume, ~hundreds of TB).
# The overlay rootfs is typically only ~30 GB which is too small for HF/Triton.
export UV_CACHE_DIR="${REPO_ROOT}/.cache/uv"
export PIP_CACHE_DIR="${REPO_ROOT}/.cache/pip"
export HF_HOME="${REPO_ROOT}/.cache/hf"
export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export TRITON_CACHE_DIR="${REPO_ROOT}/.cache/triton"

mkdir -p "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}"

cd "${REPO_ROOT}"
echo "DreamZero venv activated at ${REPO_ROOT}"
echo "Python: $(python --version)"
