# Client Side — DreamZero Inference Clients

This directory holds **everything you need on the robot-control machine** to talk to the DreamZero inference server.

## Two Ready-to-Use Clients

| File | When to Use |
|---|---|
| [`deoxys_client.py`](deoxys_client.py) | Franka Panda + Robotiq/Franka Hand controlled via [Deoxys](https://github.com/UT-Austin-RPL/deoxys_control). Drop-in if your stack matches DROID's. |
| [`robot_client_template.py`](robot_client_template.py) | Generic skeleton with a `RobotInterface` abstraction. Implement four hardware methods to integrate any other arm/SDK. |

Both clients implement the same WebSocket protocol; see [`../docs/02_protocol_reference.md`](../docs/02_protocol_reference.md) for the wire format.

## Directory Layout

```
client/
  ├── deoxys_client.py             ← Franka + Deoxys reference client
  ├── robot_client_template.py     ← Generic template (you fill in 4 methods)
  ├── eval_utils/                  ← Minimal copy of upstream policy client/server protocol
  │   ├── policy_client.py         ← WebsocketClientPolicy
  │   └── policy_server.py         ← PolicyServerConfig dataclass
  └── deoxys_configs/
      └── dreamzero-joint-impedance.yml   ← Tuned Deoxys controller config
```

## Install Dependencies (Robot-Control Machine)

```bash
pip install numpy opencv-python pyrealsense2 \
            websockets msgpack msgpack-numpy openpi-client
# Plus your robot SDK, for Franka:
#   - deoxys_control (UT-Austin-RPL) — built and franka-interface running on NUC
```

## Running

**Important:** the clients import `eval_utils.policy_client` from the local directory, so you must run them with `client/` as the working directory:

```bash
cd client/
# Deoxys / Franka:
python deoxys_client.py \
    --host <runpod-public-host> --port 5000 \
    --interface-cfg charmander.yml \
    --controller-cfg dreamzero-joint-impedance.yml \
    --cam-ext0 <serial> --cam-ext1 <serial> --cam-wrist <serial> \
    --prompt "pick up the red block"

# Generic template (after you implement RobotInterface):
python robot_client_template.py \
    --host <runpod-public-host> --port 5000 \
    --prompt "wave hand"
```

For the Deoxys-specific Franka integration walkthrough (controller choice, gripper convention, kp/kd tuning, frame buffering), see [`../docs/04_client_integration.md`](../docs/04_client_integration.md).

## Quick Connectivity Test (No Robot Needed)

If you want to verify the client can reach the server before plugging in hardware:

```python
import numpy as np, uuid
from eval_utils.policy_client import WebsocketClientPolicy

client = WebsocketClientPolicy(host="<runpod-host>", port=5000)
print("server config:", client.get_server_metadata())

obs = {
    "observation/exterior_image_0_left": np.zeros((180, 320, 3), dtype=np.uint8),
    "observation/exterior_image_1_left": np.zeros((180, 320, 3), dtype=np.uint8),
    "observation/wrist_image_left":      np.zeros((180, 320, 3), dtype=np.uint8),
    "observation/joint_position":   np.zeros(7, dtype=np.float32),
    "observation/cartesian_position": np.zeros(6, dtype=np.float32),
    "observation/gripper_position": np.zeros(1, dtype=np.float32),
    "prompt": "test",
    "session_id": str(uuid.uuid4()),
}
print("actions:", client.infer(obs).shape)   # → (24, 8)
```

First inference takes ~120 s (model + JIT warm-up); subsequent inferences ~3.5 s on H100/H200.
