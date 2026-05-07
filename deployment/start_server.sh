#!/usr/bin/env bash
# Launch the DreamZero WebSocket inference server in the background.
#
# Defaults (override via env vars):
#   PORT=5000                                              WebSocket port
#   NPROC=1                                                # of GPUs (single H100/H200 fits the 14B model in bf16)
#   MODEL_PATH=<repo>/checkpoints/DreamZero-DROID          model checkpoint
#   LOG_FILE=<repo>/logs/server-<timestamp>.log            output log
#   ENABLE_DIT_CACHE=1                                     pass --enable-dit-cache (recommended)
#
# Usage (from repo root, with venv activated):
#   bash deployment/start_server.sh                  # background, prints PID
#   bash deployment/start_server.sh foreground       # blocks the terminal

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PORT="${PORT:-5000}"
NPROC="${NPROC:-1}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/checkpoints/DreamZero-DROID}"
ENABLE_DIT_CACHE="${ENABLE_DIT_CACHE:-1}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/logs/server-${TIMESTAMP}.log}"
mkdir -p "$(dirname "${LOG_FILE}")"

# Sanity checks
if [ ! -f "${MODEL_PATH}/model.safetensors.index.json" ]; then
    echo "ERROR: checkpoint not found at ${MODEL_PATH}" >&2
    echo "Run deployment/setup_runpod.sh first." >&2
    exit 1
fi
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "WARN: no venv active. Run 'source deployment/activate.sh' first." >&2
fi

# Build the CLI flags
CMD=(python -m torch.distributed.run --standalone --nproc_per_node="${NPROC}"
     "${REPO_ROOT}/server/socket_test_optimized_AR.py"
     --port "${PORT}"
     --model-path "${MODEL_PATH}")
if [ "${ENABLE_DIT_CACHE}" = "1" ]; then
    CMD+=(--enable-dit-cache)
fi

# GPU selection
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPU_LIST="$(seq -s, 0 $((NPROC-1)))"
    export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
fi

echo "=========================================="
echo "Starting DreamZero inference server"
echo "  port             : ${PORT}"
echo "  nproc_per_node   : ${NPROC}"
echo "  CUDA_VISIBLE_DEVS: ${CUDA_VISIBLE_DEVICES}"
echo "  model            : ${MODEL_PATH}"
echo "  log              : ${LOG_FILE}"
echo "=========================================="

if [ "${1:-background}" = "foreground" ]; then
    exec "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
else
    nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
    PID=$!
    echo "Server PID: ${PID}"
    echo "Tail logs with:  tail -f ${LOG_FILE}"
    echo "Stop with:       kill ${PID}"
    sleep 2
    if ! kill -0 "${PID}" 2>/dev/null; then
        echo "ERROR: server died immediately. Check ${LOG_FILE}" >&2
        exit 1
    fi
fi
