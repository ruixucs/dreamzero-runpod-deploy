#!/usr/bin/env bash
# Prefetch all Hugging Face assets needed BEFORE starting the inference server.
# Idempotent — safe to rerun on the same /workspace volume.
#
# Without this, the server downloads Wan2.1-I2V-14B-480P (~77 GB) on first boot (~6 min).
#
# Usage (from repo root):
#   bash deployment/prefetch_models.sh
#
# Optional:
#   SKIP_DROID=1   only prefetch Wan2.1 backbone
#   SKIP_WAN=1     only prefetch DreamZero-DROID checkpoint

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

REPO_ROOT="$(_dreamzero_repo_root)"
cd "${REPO_ROOT}"

dreamzero_export_caches "${REPO_ROOT}"
PYTHON="$(dreamzero_select_python "${REPO_ROOT}")"

echo "=========================================="
echo "DreamZero model prefetch"
echo "  repo   : ${REPO_ROOT}"
echo "  python : ${PYTHON} ($(${PYTHON} --version 2>&1))"
echo "  HF_HOME: ${HF_HOME}"
echo "=========================================="

dreamzero_pip "${PYTHON}" install -q "huggingface_hub[cli]" hf_transfer hf_xet

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

WAN_REPO="Wan-AI/Wan2.1-I2V-14B-480P"
DROID_DIR="${REPO_ROOT}/checkpoints/DreamZero-DROID"

if [ "${SKIP_WAN:-0}" != "1" ]; then
    echo ">> Prefetching ${WAN_REPO} (~77 GB) into HF cache..."
    "${PYTHON}" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Wan-AI/Wan2.1-I2V-14B-480P", repo_type="model")
print("Wan2.1 prefetch done.")
PY
    du -sh "${HF_HOME}/hub/models--Wan-AI--Wan2.1-I2V-14B-480P" 2>/dev/null || true
else
    echo ">> SKIP_WAN=1"
fi

if [ "${SKIP_DROID:-0}" != "1" ]; then
    if [ -f "${DROID_DIR}/model.safetensors.index.json" ]; then
        echo ">> DreamZero-DROID already at ${DROID_DIR} ($(du -sh "${DROID_DIR}" | cut -f1)), skipping"
    else
        echo ">> Downloading GEAR-Dreams/DreamZero-DROID (~61 GB) to ${DROID_DIR}..."
        "${PYTHON}" - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="GEAR-Dreams/DreamZero-DROID",
    repo_type="model",
    local_dir="${DROID_DIR}",
)
print("DreamZero-DROID prefetch done.")
PY
    fi
else
    echo ">> SKIP_DROID=1"
fi

echo ""
echo "=========================================="
echo "Prefetch complete. Start server:"
echo "  bash deployment/start_server.sh"
echo "=========================================="
