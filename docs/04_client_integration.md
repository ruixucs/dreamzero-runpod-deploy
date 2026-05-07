# Client Integration Guide

How to wire your robot to the DreamZero inference server. Covers the Deoxys/Franka path in depth and the generic-template path for any other arm.

> Prerequisites: the inference server is up and reachable from the client machine. See [`01_deployment_runpod.md`](01_deployment_runpod.md) and §7 there for the network options.

---

## 1. Pick a Client

| Your Stack | Use | Effort |
|---|---|---|
| Franka Panda + Deoxys + Robotiq/Franka Hand | [`client/deoxys_client.py`](../client/deoxys_client.py) | ~5 minutes (set camera serials, run) |
| Franka + Polymetis / franka_ros2 | adapt [`client/robot_client_template.py`](../client/robot_client_template.py) | ~30 min (4 methods) |
| UR / xArm / KUKA / other 7-DoF | adapt [`client/robot_client_template.py`](../client/robot_client_template.py) | ~1 hour + joint-space sanity testing |
| Sim-only (Isaac Sim / MuJoCo) | Use upstream [`server/eval_utils/run_sim_eval.py`](../server/eval_utils/run_sim_eval.py) (sim-evals) | n/a |

Both clients implement the same WebSocket protocol described in [`02_protocol_reference.md`](02_protocol_reference.md).

---

## 2. Deoxys Setup (Franka Panda)

### 2.1 Hardware & Software Pre-Conditions

- Franka Panda with FCI license
- Realtime-kernel NUC running [Deoxys' `franka-interface`](https://github.com/UT-Austin-RPL/deoxys_control) (the C++ side)
- Workstation with `deoxys_control` (Python side) installed and pointing at the NUC via `charmander.yml`
- Three RGB cameras with known serials/device IDs (RealSense, ZED, or USB UVC). Each must be capable of ≥15 fps. See [`03_research_findings.md` §5](03_research_findings.md#5-camera-parameters) for format requirements.

### 2.2 Install the Tuned Controller Config

Copy the DreamZero-tuned config into Deoxys' config directory:

```bash
cd /path/to/deoxys_control/deoxys/config/
cp /path/to/dreamzero-runpod-deploy/client/deoxys_configs/dreamzero-joint-impedance.yml .
```

This config (rationale in [`03_research_findings.md` §3](03_research_findings.md#3-control-mode-joint-impedance-not-hard-position-servo)):

```yaml
controller_type: JOINT_IMPEDANCE
is_delta: false
traj_interpolator_cfg:
  traj_interpolator_type: LINEAR_JOINT_POSITION
  time_fraction: 1.0          # full-tick interpolation; default 0.3 leaves robot idle
joint_kp: [80., 80., 80., 80., 60., 100., 40.]
joint_kd: [16., 16., 16., 16.,  6.,  10.,  4.]
```

**If the robot tracks slowly:** scale kp up by 50% and kd proportionally.
**If you see oscillation:** halve kp and kd.
**Joint 6** is intentionally higher because of downstream inertia in DROID's typical poses.

### 2.3 Run the Client

On the Franka workstation (with venv that has `deoxys_control` + numpy + opencv + pyrealsense2 + openpi-client):

```bash
cd dreamzero-runpod-deploy/client/
python deoxys_client.py \
    --host  <runpod-host>          --port 5000 \
    --interface-cfg charmander.yml \
    --controller-cfg dreamzero-joint-impedance.yml \
    --controller-type JOINT_IMPEDANCE \
    --cam-ext0  <serial-1> \
    --cam-ext1  <serial-2> \
    --cam-wrist <serial-3> \
    --prompt "wave hand back and forth"
```

The client will:
1. Connect to the server, validate `PolicyServerConfig`
2. Move robot to DROID reset pose (`[0, -π/5, 0, -4π/5, 0, 3π/5, 0]`)
3. Send the initial single-frame observation
4. Loop: execute 24 actions over 24 control ticks, capture frames, request next chunk

### 2.4 Why the Initial "wave hand" Prompt?

Don't start with "pick up the red block." Start with a pure-motion prompt that doesn't involve the gripper. This isolates failures: if the arm waves, your joint pipeline works. Then graduate to grasping prompts that exercise the gripper convention.

---

## 3. The Generic Template

[`client/robot_client_template.py`](../client/robot_client_template.py) defines a `RobotInterface` ABC with four methods you implement:

```python
class RobotInterface:
    def __init__(self):
        # Init arm + gripper + 3 cameras
        ...

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        # Returns (joint_position(7,), gripper_position(1,))
        # Gripper convention: 0=open, 1=closed
        ...

    def get_images(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Returns three (180, 320, 3) RGB uint8 frames
        # OpenCV gotcha: cvtColor BGR2RGB and resize to (W=320, H=180)
        ...

    def execute_action(self, joint_target: np.ndarray, gripper_target: float) -> None:
        # Block until one control step completes (~67 ms at 15 Hz)
        # gripper_target in [0,1]; binarize at 0.5 for two-state grippers
        ...

    def is_safe(self) -> bool:
        # Joint limits, force limits, e-stop check
        ...
```

The main loop is provided; you only fill these in. See the file for full method docstrings.

### 3.1 Implementation Tips

**Joint convention:** the model assumes Franka's 7-DoF panda_joint1..7 ordering. For non-Franka arms you need to either (a) map your joints to Franka kinematics if your arm has comparable workspace, or (b) accept that zero-shot performance will degrade. DROID training never saw your arm; this is *zero-shot*, not calibrated.

**Gripper convention:** model output `>0.5 → close`, `≤0.5 → open`. For continuous-position grippers (e.g., a parallel jaw with motor encoder), you can either binarize hard (recommended) or implement a soft mapping. **Do not** linearly map `[0,1] → motor_units`; gripper outputs are bimodal and intermediate values mean "uncertain", not "half open".

**Image format:** `(180, 320, 3)` uint8 RGB. Three independent cameras, time-synchronized to the same control tick. See [`02_protocol_reference.md` §3.3](02_protocol_reference.md#33-image-format-gotchas) for the OpenCV pitfalls.

**Control rate:** the template defaults to `CONTROL_HZ = 15`. Don't go higher unless you also subsample the action chunk — the model was trained at 15 Hz.

---

## 4. Frame Buffering

The model needs **4 frames at offsets `[-23, -16, -8, 0]`** for every inference after the first. Both clients implement this with a `deque`:

```python
class FrameBuffer:
    def __init__(self, maxlen=30):                 # max(|min(offset)|, ACTION_HORIZON) + slack
        self.b0 = deque(maxlen=maxlen)             # exterior cam 0
        self.b1 = deque(maxlen=maxlen)             # exterior cam 1
        self.bw = deque(maxlen=maxlen)             # wrist cam

    def push(self, e0, e1, w):
        self.b0.append(e0); self.b1.append(e1); self.bw.append(w)

    def take(self, offsets):
        n = len(self)
        def grab(buf):
            arr = list(buf)
            return np.stack([arr[max(0, n - 1 + o)] for o in offsets], axis=0)
        return grab(self.b0), grab(self.b1), grab(self.bw)
```

The `max(0, ...)` clamp handles the early-episode case when you don't yet have 24 frames of history. The server tolerates this — it pads internally — see [`03_research_findings.md` §7](03_research_findings.md#7-open-questions--caveats).

---

## 5. Chunk-Boundary Smoothing

Each new inference returns 24 absolute joint targets. The first one (`actions[0, :7]`) is the model's notion of "where the robot should be right now", but **the robot is actually wherever it ended up after executing the previous 24 actions**, which may be slightly different. Without smoothing, switching to the new chunk produces a step-input that the impedance controller responds to with a torque spike — the arm jerks.

The reference clients do a 3-step linear blend:

```python
cur_q = np.array(robot.last_q)
target_q0 = actions[0, :7]
for k in range(blend_steps):
    alpha = (k + 1) / (blend_steps + 1)
    q_blend = (1 - alpha) * cur_q + alpha * target_q0
    robot.execute_action(q_blend, actions[0, 7])
    # capture frames during blend too
```

Then execute the remaining 24 actions normally. **Don't skip this** unless your kp is very low (in which case the impedance controller does the smoothing for you).

---

## 6. Latency & Closed-Loop Rate

A single chunk on H200:

```
inference  3.5 s  ─┐
                   ├── total = 5.1 s/chunk if synchronous
execution  1.6 s  ─┘    → ~4.7 actions/sec effective
```

The model card claims 7 Hz with **DreamZero-Flash**, which interleaves inference with execution: while the robot executes actions 0-23 of chunk N, the client already requests chunk N+1 with the *latest* observation. By the time action 23 completes, chunk N+1 is ready.

For sync clients (both reference clients here are sync), you'll get 4-5 Hz effective. For most household manipulation tasks this is fine. To go async, restructure the main loop:

```python
# Pseudo-code; not in current clients
import threading

next_chunk = None
inference_lock = threading.Lock()

def request_next_chunk(obs):
    global next_chunk
    with inference_lock:
        next_chunk = client.infer(obs)

# Main loop
while running:
    # Start next inference in background ~half-way through executing this chunk
    if step == 12 and not inference_in_flight:
        threading.Thread(target=request_next_chunk, args=(build_obs(),)).start()

    robot.execute_action(actions[step, :7], actions[step, 7])
    step += 1

    if step == 24:
        # Wait for next chunk if it's not ready yet
        with inference_lock:
            actions = next_chunk
        next_chunk = None
        step = 0
```

This is left as an extension for now.

---

## 7. Debugging Checklist

Run these in order. Each step isolates a failure mode.

| # | Test | Expected | Fix If Broken |
|---|---|---|---|
| 1 | `python -c "from eval_utils.policy_client import WebsocketClientPolicy; c = WebsocketClientPolicy('host', 5000); print(c.get_server_metadata())"` from `client/` | prints `PolicyServerConfig` dict | Network: see [§7 of deployment doc](01_deployment_runpod.md#7-exposing-the-server-to-your-robot) |
| 2 | Send all-zero observation, check action shape | `(24, 8)` | Server crashed, check `logs/server-*.log` |
| 3 | Run `--prompt "wave hand"` for 5 chunks **without robot** (replace `RobotInterface` with stubs that don't move anything) | 5 inference calls succeed, action ranges look reasonable | If actions are NaN: bad obs format |
| 4 | Plug robot in, run reset-only (`--no-reset-pose=False`, then exit) | Robot moves smoothly to DROID home pose | Joint limits, e-stop, collision |
| 5 | Run "wave hand" with robot, no objects in scene | Smooth back-and-forth motion | Chatter: lower kp; lag: raise kp |
| 6 | Run "pick up the red block" with object in scene | Approach + grasp + lift | Gripper inverted: re-read [§1 of findings](03_research_findings.md#1-gripper-convention) |
| 7 | Inspect `actions[:, 7]` over time | Approach phase ~0, grasp moment jumps to ~1, holds | If reversed, your gripper convention is flipped somewhere |

---

## 8. Common Pitfalls Recap

| Mistake | Symptom | Fix |
|---|---|---|
| BGR images sent without conversion | Model performs poorly, "as if it can't see" | `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` |
| `cv2.resize(frame, (180, 320))` | Tensor shape `(320, 180, 3)` | use `cv2.resize(frame, (320, 180))` (cv2 takes W,H) |
| Sending 4 frames on the first call | Model treats it as continuation, no warm-up | First call: single frame `(180, 320, 3)` |
| Re-using `session_id` across episodes | KV cache pollution, bad outputs | New UUID per `run_episode()` |
| Linear `2*g - 1` gripper mapping | Gripper chatter on Deoxys | Binarize at 0.5 |
| Skipping reset-to-home | Performance much worse than expected | Call `robot.reset_to_droid_pose()` before first inference |
| High kp + no chunk blend | Torque spikes every 1.6 s | Lower kp **or** add blend_steps=3 (we recommend both) |
| Ignoring `is_safe()` | Robot drives into joint limit | Implement and check every step |
