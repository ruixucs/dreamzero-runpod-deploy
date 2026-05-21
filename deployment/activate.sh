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

# shellcheck source=_common.sh
if [ -f "${SCRIPT_DIR}/_common.sh" ]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/_common.sh"
    dreamzero_export_caches "${REPO_ROOT}"
    if [ -f "${REPO_ROOT}/.python-interpreter" ]; then
        export PYTHON="$(cat "${REPO_ROOT}/.python-interpreter")"
    else
        export PYTHON="$(dreamzero_select_python "${REPO_ROOT}")"
    fi
fi

cd "${REPO_ROOT}"
echo "DreamZero env at ${REPO_ROOT}"
if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "venv: ${VIRTUAL_ENV}"
fi
if [ -n "${PYTHON:-}" ]; then
    echo "PYTHON=${PYTHON} ($(${PYTHON} --version 2>&1))"
else
    echo "Python: $(python --version 2>&1)"
fi
echo "HF_HOME=${HF_HOME:-not set}"
