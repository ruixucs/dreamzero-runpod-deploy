# DreamZero Inference — RunPod Deployment Guide

End-to-end recipe to bring up the DreamZero-DROID 14B inference server on a fresh RunPod pod, validated on **NVIDIA H200 (143 GB)** and **B200 (183 GB)** with PyTorch 2.8 + CUDA 12.8.

The same recipe works on any single H100 80 GB or larger; the model fits in ~45 GB of VRAM in bfloat16.

For B200-specific timings (first vs second boot, prefetch, flash-attn wheel), see [`05_deployment_b200.md`](05_deployment_b200.md).

---

## 1. RunPod Pod Setup

### Recommended Image

Pick any container with:
- **PyTorch 2.4 or newer**
- **CUDA 12.8+ runtime** (12.9 also fine — DreamZero ships its own torch wheels via `--extra-index-url`)
- Python 3.11 or 3.12 in the base image (we install our own 3.11 via `uv`)

The `runpod/pytorch:2.4.0-py3.11-cuda12.4-devel` image works. So does any "PyTorch 2.x devel" image — what matters is `nvidia-smi` reports a CUDA 12.8+ driver.

### GPU

| GPU | VRAM | Tested | Notes |
|---|---|---|---|
| B200 | 183 GB | yes | ~45 GB used; steady inference ~2.8 s/chunk — see [05](05_deployment_b200.md) |
| H200 | 143 GB | yes | ~45 GB used, plenty of headroom |
| H100 80 GB | 80 GB | upstream-tested | Targeted by upstream README |
| A100 80 GB | 80 GB | not tested | should work; flash-attn supported |
| A100 40 GB | 40 GB | will OOM | model alone is ~28 GB plus activations |

Single-GPU is enough. Multi-GPU also works (`--nproc_per_node N`) — see [§5](#5-multi-gpu-optional).

### Disk

You need ~200 GB:

| Item | Size |
|---|---|
| Repo + venv | 25 GB |
| `.cache/hf` (Wan2.1 backbone auto-download) | 60 GB |
| `checkpoints/DreamZero-DROID` | 61 GB |
| Triton/Torch caches, logs | 5 GB |
| Headroom | ~50 GB |

**Critical:** RunPod's container rootfs (`/`) is usually only ~30 GB. **Put the repo under `/workspace/`** which is the persistent network volume (typically several hundred GB or more). All scripts in this repo assume the repo is at `/workspace/dreamzero-runpod-deploy/` but adapt automatically — just stay on `/workspace`.

### Ports

The inference server listens on TCP **5000** (configurable). To reach it from a remote robot machine you need one of:
- **TCP Proxy** in the RunPod console: Edit Pod → Expose TCP Ports → add `5000` → use the assigned `proxy.runpod.net:<external>` host
- **SSH tunnel** from the client machine: `ssh -L 5000:localhost:5000 -p <ssh_port> root@<runpod_public_ip> -N`
- **Same-pod client** for sim/integration testing: just use `localhost:5000`

See [§7](#7-exposing-the-server-to-your-robot) for trade-offs.

---

## 2. One-Shot Setup

```bash
cd /workspace
git clone <YOUR-REPO-URL> dreamzero-runpod-deploy
cd dreamzero-runpod-deploy

# Optional but recommended: prefetch Wan2.1 (~77 GB) + DreamZero-DROID (61 GB)
bash deployment/prefetch_models.sh

bash deployment/setup_runpod.sh
```

That script does everything in [§3](#3-manual-setup-step-by-step) for you. It is idempotent — rerun it any time.

**Time expectations:**

| Scenario | Setup + server listen |
|---|---|
| Cold volume (no cache) | **15–25 min** (downloads + first JIT compile on B200) |
| Warm `/workspace` (prefetch done) | **5–10 min** setup; **2–4 min** server listen |

When it finishes:

```bash
source deployment/activate.sh
bash deployment/start_server.sh
tail -f logs/server-*.log
```

After ~2 minutes of warm-up logs you'll see:

```
INFO:websockets.server:server listening on 0.0.0.0:5000
```

You're live. Skip to [§6](#6-first-inference--health-check).

---

## 3. Manual Setup (Step by Step)

If you want to understand each step (or `setup_runpod.sh` failed mid-way):

### 3.1 Clone

```bash
cd /workspace
git clone <YOUR-REPO-URL> dreamzero-runpod-deploy
cd dreamzero-runpod-deploy
```

### 3.2 Python 3.11 venv

DreamZero pins Python 3.11 (see [`pyproject.toml`](../pyproject.toml) `requires-python = "~=3.11,<3.13"`). The base image's Python is usually 3.10 or 3.12, so we make our own:

```bash
# Install uv if not present
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.11 .venv
source .venv/bin/activate
python --version    # → Python 3.11.x
```

### 3.3 Route Caches into /workspace

The overlay rootfs (`/`) on RunPod is small. Force HF / pip / triton caches onto the persistent volume:

```bash
mkdir -p .cache/{uv,pip,hf,torch,triton} logs checkpoints
export UV_CACHE_DIR="$PWD/.cache/uv"
export PIP_CACHE_DIR="$PWD/.cache/pip"
export HF_HOME="$PWD/.cache/hf"
export TORCH_HOME="$PWD/.cache/torch"
export TRITON_CACHE_DIR="$PWD/.cache/triton"
```

[`deployment/activate.sh`](../deployment/activate.sh) sets all of this for you on subsequent shell sessions.

### 3.4 Work Around the Debian `blinker` Conflict

The `flask` dependency wants a newer `blinker` than what apt-installed Debian ships. The system `blinker` has no `RECORD` file, so pip refuses to uninstall it.

```bash
pip install --ignore-installed blinker
```

Skipping this step makes step 3.5 fail with `Cannot uninstall blinker 1.7.0 / The package's contents are unknown`.

### 3.5 Install DreamZero

```bash
cd server/
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
cd ..
```

Takes ~3 minutes the first time. Pulls torch 2.8.0+cu128, transformers 4.51, diffusers 0.30, deepspeed, peft, etc. — see [`server/pyproject.toml`](../server/pyproject.toml) for the full list. flash-attn is included as a transitive dependency.

### 3.6 Sanity-Check Imports

```bash
python -c "
import torch
print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'avail', torch.cuda.is_available())
import flash_attn; print('flash_attn', flash_attn.__version__)
import transformers; print('transformers', transformers.__version__)
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
print('groot OK')
"
```

Expected output (on H200):

```
torch 2.8.0+cu128 CUDA 12.8 avail True
flash_attn 2.8.3
transformers 4.51.3
groot OK
```

### 3.7 Download the Checkpoint

```bash
pip install -q "huggingface_hub[cli]" hf_xet
hf download GEAR-Dreams/DreamZero-DROID \
    --repo-type model \
    --local-dir ./checkpoints/DreamZero-DROID
```

This is ~61 GB across 10 safetensors shards. With xet enabled and a fast pod (~1 GB/s downstream) it finishes in 5-10 min. The model is publicly accessible — no `HF_TOKEN` needed.

### 3.8 (Auto, on first launch) Wan2.1-I2V-14B-480P Backbone

DreamZero is built on top of [Wan-AI/Wan2.1-I2V-14B-480P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) (~60 GB). The first time you start the inference server, it auto-downloads this backbone into `.cache/hf`. **Don't kill the server while it's downloading** — the warm-up takes ~3-4 minutes and includes this download. On subsequent starts it loads from cache in seconds.

If you'd rather pre-download:

```bash
hf download Wan-AI/Wan2.1-I2V-14B-480P \
    --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
# then export an env var so the server uses your local path:
export WAN_CKPT_DIR=$PWD/checkpoints/Wan2.1-I2V-14B-480P
```

---

## 4. Start the Inference Server

Single GPU (H200, what we tested):

```bash
source deployment/activate.sh
bash deployment/start_server.sh
```

This puts the server in the background; PID and log path are printed. To run in the foreground (Ctrl-C to stop):

```bash
bash deployment/start_server.sh foreground
```

### 4.1 What It's Doing During Warm-Up

The first ~2 minutes of `logs/server-*.log` show:

```
[INFO] Loading sharded safetensors using index: ./checkpoints/DreamZero-DROID/model.safetensors.index.json
[INFO] Loading shard: model-00001-of-00010.safetensors
...
[INFO] Successfully loaded pretrained weights
[INFO] Moving models to the cuda device and setting the dtype to bfloat16.
[INFO] Torch compiling the TextEncoder, ImageEncoder, and VAE modules
INFO:websockets.server:server listening on 0.0.0.0:5000
```

Once you see `server listening on 0.0.0.0:5000` you're ready.

The repeating `Warning: Transformer Engine is not available. Falling back to FA2 backend.` lines are **expected** on H100/H200 — TE is a GB200-only optimization.

### 4.2 GPU Memory After Warm-Up

```
$ nvidia-smi --query-gpu=memory.used,memory.free --format=csv
44570 MiB, 98598 MiB
```

About 45 GB used on H200. On H100 80 GB you'll be at ~55-60% utilization.

---

## 5. Multi-GPU (Optional)

For lower per-inference latency, two GPUs ~halves wall-clock time per chunk. The launcher already supports this:

```bash
NPROC=2 bash deployment/start_server.sh
```

This will set `CUDA_VISIBLE_DEVICES=0,1` and start `torch.distributed.run --nproc_per_node=2`. The model parallelizes across DiT layers via the device mesh DreamZero builds in [`server/socket_test_optimized_AR.py:714`](../server/socket_test_optimized_AR.py).

GB200 + TensorRT engine is faster still (~0.6 s/chunk per upstream README), but requires `transformer_engine` + a TRT-compiled model — we don't cover that here since it's H200/H100-irrelevant.

---

## 6. First Inference & Health Check

From the same machine the server is running on:

```bash
cd server/
python test_client_AR.py --port 5000 --num-chunks 2
```

Expected timing on H200:

| Phase | Time | Notes |
|---|---|---|
| WebSocket connect + metadata | < 0.1 s | |
| Initial inference (single frame) | ~120 s | Warm-up: torch compile + first VAE pass |
| Chunk 0 (4 frames) | ~42 s | Some compilation still happening |
| **Chunk 1 onwards** | **~3.5 s** | Steady-state throughput |

Output looks like:

```
[INFO]   Action shape: (24, 8), range: [-0.0444, 0.2269], time: 3.51s
[INFO] Sending reset to save video...
```

The `(24, 8)` shape is what every robot client must expect: 24 sequential actions of (7 joint targets + 1 gripper). See [`02_protocol_reference.md`](02_protocol_reference.md) for the full wire format.

A generated MP4 of the model's predicted future frames is saved to `checkpoints/DreamZero-DROID/real_world_eval_gen_<date>_<idx>/DreamZero-DROID/`. We include two examples in [`runs/sample_outputs/`](../runs/sample_outputs).

---

## 7. Exposing the Server to Your Robot

| Approach | Latency | Stability | Setup Effort |
|---|---|---|---|
| **SSH tunnel** | low | best | easy if you have SSH |
| **RunPod TCP Proxy** | ~30-100 ms | occasional disconnects | easiest in the console |
| **Same-pod client** | minimum | n/a | only useful for sim |

### 7.1 SSH Tunnel (Recommended for Real Robots)

On the **client machine** (your robot-control PC):

```bash
ssh -L 5000:localhost:5000 \
    -p <RUNPOD_TCP_PORT_22> \
    root@<RUNPOD_PUBLIC_IP> -N
```

Find `<RUNPOD_TCP_PORT_22>` and `<RUNPOD_PUBLIC_IP>` in:
- RunPod console → your pod → "Connect" tab → "Connect over SSH"
- Or inside the pod: `echo $RUNPOD_TCP_PORT_22 $RUNPOD_PUBLIC_IP`

Then in your client code use `host=localhost, port=5000`. The traffic is encrypted and the kernel handles reconnects gracefully.

### 7.2 RunPod TCP Proxy

In the RunPod console: Edit Pod → Symmetrical Port Mappings → add **5000** → restart pod. You'll get an external host like `<pod-id>-5000.proxy.runpod.net:<auto-port>`. Use that as `--host`/`--port` from your client.

Caveat: the proxy occasionally drops idle WebSocket connections. The client library [`policy_client.py`](../client/eval_utils/policy_client.py) sets `ping_interval=60s, ping_timeout=600s` which usually keeps it alive, but if you see disconnects, switch to SSH tunneling.

### 7.3 Same-Pod Client

If your robot-control code can run inside the RunPod container (e.g., for Isaac Sim eval), just `localhost:5000`. This is what `server/test_client_AR.py` does for the smoke test in §6.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot uninstall blinker 1.7.0` during pip install | Debian system blinker has no RECORD | `pip install --ignore-installed blinker` then retry |
| `No module named pip` when running `python -m pip` | The `pip` binary on PATH points to system Python, not venv | Use `python -m pip` only after `source deployment/activate.sh` |
| Server hangs at "Loading sharded safetensors..." then `OOM` | GPU < 80 GB | Use H100 80 GB or H200 |
| `flash_attn` import error | Build skipped or wrong CUDA wheel | `pip install --no-build-isolation flash-attn==2.8.3` and check `MAX_JOBS` |
| `Warning: Transformer Engine is not available` repeats forever | Expected on H100/H200 | Ignore — FA2 backend is the right choice |
| `RUNPOD_API_KEY` visible in `env` | RunPod injects it for management API | Don't share env dumps publicly; rotate key in console if leaked |
| Disk fills at ~30 GB | You're caching to overlay rootfs, not /workspace | Re-run `source deployment/activate.sh` to set `HF_HOME` etc. |
| Client reports `Failed to open a WebSocket connection: missing Connection header` | You hit the server with curl, not a WS client | Use [`policy_client.py`](../client/eval_utils/policy_client.py) or wscat |
| First inference takes 120 s, then chunk 0 takes 42 s, but everything after is 3.5 s | Expected: torch.compile lazy-compiles VAE/text encoder on first pass | This is the documented warm-up; do not kill the server |

---

## 9. Stopping & Cleaning Up

```bash
# Find the running server
ps -ef | grep socket_test_optimized_AR | grep -v grep

# Graceful stop
kill <PID>

# If unresponsive
kill -9 <PID>

# Clear generated MP4 outputs
rm -rf checkpoints/DreamZero-DROID/real_world_eval_gen_*
```

To free the entire 60 GB Wan2.1 cache (e.g., before tearing down the pod):

```bash
rm -rf .cache/hf/hub/models--Wan-AI--Wan2.1-I2V-14B-480P
```

---

## 10. What's Next

- **Connect a real robot:** see [`04_client_integration.md`](04_client_integration.md) for the Deoxys/Franka walkthrough and the gripper / control-mode conventions.
- **Understand the protocol:** [`02_protocol_reference.md`](02_protocol_reference.md) gives the byte-level WebSocket contract.
- **Why we made each design choice:** [`03_research_findings.md`](03_research_findings.md) consolidates the multi-source archaeology behind the gripper convention, action semantics, and impedance settings.
