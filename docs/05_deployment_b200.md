# DreamZero on NVIDIA B200 (RunPod) — Second Deployment Notes

This document records what we learned deploying **DreamZero-DROID** on a **second** RunPod pod with an **NVIDIA B200** (Blackwell, sm_100), after the first successful deployment on **H200** (Hopper, sm_90).

The goal is to make the **second and later** bring-ups fast: no surprise downloads at server start, no repeated JIT compile when `/workspace` is reused, and a clear split between **first-boot** vs **warm-cache** timelines.

---

## 1. Hardware We Used

| Resource | Value |
|---|---|
| GPU | NVIDIA B200, 183 GB VRAM, compute capability **10.0** (Blackwell) |
| Driver / CUDA (driver API) | 580.x, CUDA 13.0 reported by `nvidia-smi` |
| Toolkit | CUDA 12.8 (`nvcc`) |
| CPU / RAM | 192 cores, 2 TB |
| Persistent disk | RunPod `/workspace` network volume (~420 TB shared) |
| Python used in practice | **3.12** (system image), not the repo’s 3.11 venv |

VRAM at steady inference: **~44.7 GB** (matches H200/H100 numbers in the upstream README).

---

## 2. First Boot vs Second Boot (Same `/workspace`)

### First deployment on this B200 pod (cold cache)

| Phase | Time | What was happening |
|---|---|---|
| Install deps (`pip install -e server`) | ~6 min | PyTorch 2.8+cu128, groot, etc. (into **system** Python 3.12 — see §4) |
| Download DreamZero-DROID | ~1 min | 61 GB via `hf_transfer` / `hf_xet` |
| **Server start → `listening on 5000`** | **~15 min** | See breakdown below |
| First inference chunk | ~245 s | TorchInductor still compiling Text/Image encoders + scheduler |
| Second inference chunk | ~81 s | VAE + remaining scheduler compile |
| **Steady-state inference** | **~2.8 s / chunk** | After JIT caches warm |

**Server warm-up breakdown (first time, no prefetch):**

1. **~6 min** — Hugging Face download of `Wan-AI/Wan2.1-I2V-14B-480P` (~77 GB) into `$REPO/.cache/hf` (triggered inside `WANPolicyHead`, not by `setup_runpod.sh`).
2. **~2 min** — Load DreamZero 10 shards + Wan2.1 7 DiT shards from disk (CPU RAM ~78 GB peak during load).
3. **~6–8 min** — `torch.compile` / **TorchInductor** with **32 compile workers** for B200 (no sm_100 prebuilt kernels in cache). Log shows `Moving models to cuda` then `Torch compiling the TextEncoder, ImageEncoder, and VAE`.
4. **~1 min** — Upload weights to GPU; WebSocket binds port 5000.

So the long wait is **mostly download + first-time JIT compile**, not re-downloading DreamZero-DROID (that was already under `checkpoints/`).

### Second deployment (reuse same `/workspace` paths)

If these directories still exist on the volume:

- `checkpoints/DreamZero-DROID/` (61 GB)
- `.cache/hf/hub/models--Wan-AI--Wan2.1-I2V-14B-480P/` (~77 GB)
- `.cache/triton/` and `.cache/torch/` (Inductor / compile artifacts)

Then expect:

| Phase | Time |
|---|---|
| `bash deployment/setup_runpod.sh` | ~3–8 min (skip checkpoint; may refresh pip) |
| Server → `listening on 0.0.0.0:5000` | **~2–4 min** (load weights + lighter compile) |
| First inference after restart | Still **1–3 slow chunks** if Triton cache was cleared; then **~2.8 s** |

**Do not kill and restart the server during the first compile** — you lose progress and wait again.

---

## 3. Why Startup Felt Slow (Download vs Load vs Compile)

Use this checklist when tailing `logs/server-*.log`:

| Log signal | Meaning |
|---|---|
| `hf_hub_download` / `hf_xet` / cache growing under `.cache/hf` | **Downloading** Wan2.1 backbone (~77 GB) |
| `Loading shard: ...DreamZero-DROID/...` | **Loading** finetuned checkpoint (local) |
| `Loading shard: ...Wan2.1...` | **Loading** backbone shards (local after first download) |
| `Successfully loaded pretrained weights` | Weight load done; compile/GPU upload next |
| `Torch compiling the TextEncoder, ImageEncoder, and VAE` | **JIT compile** (CPU-heavy; many `torch/_inductor/compile_worker` children) |
| `Moving models to the cuda device` | **GPU upload** (~45 GB VRAM) |
| `server listening on 0.0.0.0:5000` | Ready for WebSocket clients |

**Inference** latency (after server is up):

| Chunk | Client time | Server `Inference Time` | Notes |
|---|---|---|---|
| Initial (1 frame) | 245 s | 244.6 s | Text + Image encoder compile dominates |
| Chunk 0 (4 frames) | 81 s | 81.0 s | VAE + scheduler compile |
| Chunk 1+ | **2.8 s** | 2.8 s | **Production-like** latency on B200 |

H100 upstream docs quote ~3 s steady state; B200 matched **~2.8 s** in bf16 without TensorRT.

---

## 4. Python Environment Gotcha (Critical for RunPod Images)

`deployment/setup_runpod.sh` creates a **uv venv** with Python 3.11, but **uv venv does not install `pip` into the venv** on this image. If you run bare `pip install`, packages go to **system** `/usr/local/lib/python3.12/dist-packages/`.

**What we did on B200 (works):**

- Run the server with **`python3.12`** (system), where `torch==2.8.0+cu128` and `groot` are importable after `pip install -e server`.
- `pyproject.toml` allows `~=3.11,<3.13` — 3.12 is valid.

**Updated scripts** use `deployment/_common.sh` to pick:

1. Repo `.venv/bin/python` if `import torch` succeeds there, else  
2. `python3.12`, else `python3.11`.

Always install with:

```bash
"$PYTHON" -m pip install ...
```

---

## 5. Dependency Fixes Specific to This Image

### 5.1 `HF_HUB_ENABLE_HF_TRANSFER=1` without `hf_transfer`

RunPod images often set `HF_HUB_ENABLE_HF_TRANSFER=1`. Server startup then crashes unless the **same** Python has `hf_transfer`:

```bash
python3.12 -m pip install hf_transfer hf_xet
```

`setup_runpod.sh` now installs these into the selected Python.

### 5.2 `flash-attn` — use the prebuilt wheel

`pip install flash-attn` failed with:

```text
Invalid cross-device link: ... (PIP_CACHE_DIR on /workspace, temp on tmpfs)
```

**Fix:** download the official wheel to `/tmp` and install:

```bash
# torch 2.8 + cu12 + cp312
wget -O /tmp/flash_attn.whl \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
python3.12 -m pip install --no-build-isolation /tmp/flash_attn.whl
```

Verified on B200: `flash_attn_func` runs in bf16 on sm_100.

### 5.3 Transformer Engine warnings

```text
Warning: Transformer Engine is not available. Falling back to FA2 backend.
```

Expected on B200/H200/H100 — TE is for GB200 in upstream docs. Safe to ignore.

---

## 6. Prefetch Before Starting the Server (Recommended)

To avoid a **6+ minute** Wan2.1 download during server boot, run once per persistent volume:

```bash
cd /workspace/dreamzero-runpod-deploy
source deployment/activate.sh   # sets HF_HOME, TORCH_HOME, TRITON_CACHE_DIR under repo
bash deployment/prefetch_models.sh
```

This downloads:

| Asset | Repo | Size (approx.) | Destination |
|---|---|---|---|
| DreamZero-DROID | `GEAR-Dreams/DreamZero-DROID` | 61 GB | `checkpoints/DreamZero-DROID/` |
| Wan2.1 I2V backbone | `Wan-AI/Wan2.1-I2V-14B-480P` | 77 GB | `$HF_HOME/hub/...` |

Idempotent — safe to rerun.

---

## 7. Recommended Commands (B200, Second Deploy)

```bash
cd /workspace/dreamzero-runpod-deploy
git pull

# One-time or after cache wipe:
bash deployment/prefetch_models.sh

# Install / verify deps (uses python -m pip + flash-attn wheel):
bash deployment/setup_runpod.sh

# Start server (uses python3.12 when venv has no torch):
bash deployment/start_server.sh
tail -f logs/server-*.log
# Wait for: INFO:websockets.server:server listening on 0.0.0.0:5000

# Smoke test (expect 2 slow chunks then ~3s):
cd server && python3.12 test_client_AR.py --host localhost --port 5000 --num-chunks 2
```

Environment variables (set automatically by `activate.sh` / scripts):

```bash
export HF_HOME="$REPO/.cache/hf"
export TORCH_HOME="$REPO/.cache/torch"
export TRITON_CACHE_DIR="$REPO/.cache/triton"
export PIP_CACHE_DIR="$REPO/.cache/pip"   # use /tmp for pip if cross-device link errors return
```

---

## 8. Differences vs H200 First Deploy

| Topic | H200 (first deploy) | B200 (second deploy) |
|---|---|---|
| Architecture | sm_90 Hopper | sm_100 Blackwell |
| VRAM | 143 GB | 183 GB |
| Steady inference | ~3 s (documented) | **~2.8 s** measured |
| First server boot | ~15–25 min (download + compile) | ~15 min (same pattern) |
| Wan2.1 at server start | Yes, if not cached | Same — **prefetch avoids this** |
| Python | 3.11 venv (when pip works) | **3.12 system** on our image |
| flash-attn | pip / MAX_JOBS build | **Prebuilt wheel** for cp312 |

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `hf_transfer package is not available` | Env var set, package missing | `python3.12 -m pip install hf_transfer` |
| Server stuck, GPU ~0 GB, many `compile_worker` processes | First-time Inductor compile | Wait; don’t restart |
| Server stuck, `.cache/hf` growing | Wan2.1 download | Run `prefetch_models.sh` next time |
| `Invalid cross-device link` installing flash-attn | pip cache on network FS | Use wheel in `/tmp` (see §5.2) |
| `import torch` fails in venv | venv never got torch | Use `python3.12` or rerun setup with `python -m pip` |
| Port 5000 closed after “listening” | Process crashed on first client | Check log for OOM; confirm 80+ GB free VRAM |

---

## 10. Evidence / Run Metadata

- **Date:** 2026-05-20  
- **Pod GPU:** NVIDIA B200  
- **Server log:** `logs/server-20260520-202440.log`  
- **Steady latency:** server `Inference Time: Total 2.826 seconds` on chunk 1  
- **Client:** `test_client_AR.py --num-chunks 2` → actions shape `(24, 8)`  

See also [`runs/b200-20260520.md`](../runs/b200-20260520.md) for a short benchmark table.
