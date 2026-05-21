# shellcheck shell=bash
# Shared helpers for DreamZero RunPod deployment scripts.
# Source this file; do not execute directly.

_dreamzero_repo_root() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd)"
    cd "${script_dir}/.." && pwd
}

dreamzero_export_caches() {
    local repo_root="${1:?repo root required}"
    mkdir -p "${repo_root}/.cache/"{uv,pip,hf,torch,triton} "${repo_root}/logs" "${repo_root}/checkpoints"
    export UV_CACHE_DIR="${repo_root}/.cache/uv"
    export PIP_CACHE_DIR="${repo_root}/.cache/pip"
    export HF_HOME="${repo_root}/.cache/hf"
    export TORCH_HOME="${repo_root}/.cache/torch"
    export TRITON_CACHE_DIR="${repo_root}/.cache/triton"
}

# Pick a Python that can import torch after setup.
# On some RunPod images, uv venv has no pip/torch; system python3.12 works (B200 notes).
dreamzero_select_python() {
    local repo_root="${1:?repo root required}"
    local candidates=()

    if [ -x "${repo_root}/.venv/bin/python" ]; then
        candidates+=("${repo_root}/.venv/bin/python")
    fi
    if command -v python3.12 >/dev/null 2>&1; then
        candidates+=("python3.12")
    fi
    if command -v python3.11 >/dev/null 2>&1; then
        candidates+=("python3.11")
    fi
    candidates+=("python3")

    local py
    for py in "${candidates[@]}"; do
        if "${py}" -c "import torch" >/dev/null 2>&1; then
            echo "${py}"
            return 0
        fi
    done

    # Fall back to venv or system 3.12 for fresh install
    if [ -x "${repo_root}/.venv/bin/python" ]; then
        echo "${repo_root}/.venv/bin/python"
    elif command -v python3.12 >/dev/null 2>&1; then
        echo "python3.12"
    else
        echo "python3.11"
    fi
}

dreamzero_pip() {
    local py="${1:?python required}"
    shift
    # Use /tmp pip cache if network volume causes cross-device link errors
    PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/pip-cache-dreamzero}" "${py}" -m pip "$@"
}

dreamzero_install_flash_attn() {
    local py="${1:?python required}"
    if "${py}" -c "import flash_attn" 2>/dev/null; then
        echo ">> flash_attn already installed"
        return 0
    fi

    local py_tag
    py_tag="$("${py}" -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"

    # Prebuilt wheels from flash-attention releases (torch 2.8 + cu12)
    local wheel_url=""
    case "${py_tag}" in
        cp312)
            wheel_url="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
            ;;
        cp311)
            wheel_url="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
            ;;
        *)
            echo ">> No prebuilt flash-attn wheel for ${py_tag}; trying pip (may be slow)"
            dreamzero_pip "${py}" install --no-build-isolation "flash-attn==2.8.3"
            return 0
            ;;
    esac

    local wheel="/tmp/flash_attn-${py_tag}.whl"
    echo ">> Installing flash-attn 2.8.3 from prebuilt wheel (${py_tag})..."
    wget -q -O "${wheel}" "${wheel_url}"
    dreamzero_pip "${py}" install --no-build-isolation "${wheel}"
}
