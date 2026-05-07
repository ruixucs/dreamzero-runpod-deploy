# Research Findings — Conventions Behind the Code

This document is the archaeology log: every non-obvious convention used by `deoxys_client.py` / `robot_client_template.py` is rooted in code or stats files we located somewhere in the DreamZero / sim-evals / DROID / Deoxys codebases. We capture **conclusion + evidence** here so future maintainers don't have to redo this hunt.

---

## Summary Table

| Topic | Conclusion | Primary Evidence |
|---|---|---|
| Gripper convention (state) | `0 = open, 1 = closed` | sim-evals `gripper_pos()` divides finger_joint by π/4 |
| Gripper convention (action) | Same: `0 = open, 1 = closed` | sim-evals `BinaryJointPositionZeroToOneAction` `# true: close` |
| Gripper distribution | Near-bimodal (mean=0.45, std=0.45) | `experiment_cfg/metadata.json` action.gripper_position |
| Action semantics | Server already converts relative→absolute | `groot/vla/model/n1_5/sim_policy.py:602-604` |
| Joint convention | absolute joint angles in radians | metadata.json action.joint_position stats inside Franka limits |
| Control mode | Joint impedance (PD with stiffness/damping), NOT rigid position servo | sim-evals `nvidia_droid.py` ImplicitActuatorCfg |
| Recommended kp/kd (Deoxys) | `[80,80,80,80,60,100,40] / [16,16,16,16,6,10,4]` | sim-evals stiffness 400 scaled ~20% for libfranka |
| Gripper command (Deoxys) | **Binarize at 0.5**, NOT linear `2x-1` | `deoxys_control` franka_interface.py:497-524 (Move vs Grasp APIs) |
| Reset pose | `[0, -π/5, 0, -4π/5, 0, 3π/5, 0]` | DROID `robot_env.py` `reset_joints` + sim-evals `nvidia_droid.py` init_state |
| Image format | RGB, uint8, (180, 320, 3), 3 cameras, 15 fps | `experiment_cfg/metadata.json` modalities.video |
| Frame schedule | First call: 1 frame; later: 4 frames at offsets `[-23,-16,-8,0]` | `server/test_client_AR.py:52`, `server/socket_test_optimized_AR.py:55` |

---

## 1. Gripper Convention

### 1.1 State Reading: Robotiq finger_joint Normalization

In sim-evals' DROID env, gripper state is read by:

```python
# /tmp/sim-evals/src/sim_evals/environments/droid_environment.py:196-209
def gripper_pos(env, asset_cfg=...):
    robot = env.scene[asset_cfg.name]
    joint_names = ["finger_joint"]
    joint_indices = [...]
    joint_pos = robot.data.joint_pos[0, joint_indices]
    # rescale
    joint_pos = joint_pos / (np.pi / 4)
    return joint_pos
```

For a Robotiq 2F-85: `finger_joint = 0 rad` is fully **open**, `finger_joint = π/4 rad` is fully **closed**. Dividing by π/4 thus produces:
- `0.0` → open
- `1.0` → closed

The clip is `(0, 1)`, confirming the intended range.

### 1.2 Action Convention: Same Direction

The sim-evals action term explicitly states:

```python
# /tmp/sim-evals/src/sim_evals/environments/droid_environment.py:126-141
class BinaryJointPositionZeroToOneAction(BinaryJointPositionAction):
    def process_actions(self, actions):
        ...
        if actions.dtype == torch.bool:
            # true: close, false: open
            binary_mask = actions == 0
        else:
            # true: close, false: open
            binary_mask = actions > 0.5
        self._processed_actions = torch.where(
            binary_mask, self._close_command, self._open_command
        )
```

And configured as:

```python
# same file, lines 168-173
finger_joint = BinaryJointPositionZeroToOneActionCfg(
    asset_name="robot",
    joint_names=["finger_joint"],
    open_command_expr  = {"finger_joint": 0.0},
    close_command_expr = {"finger_joint": np.pi / 4},
)
```

The action's `0` maps to `open`, `1` maps to `close`. Same convention as state. **No inversion needed in the client.**

### 1.3 Statistical Sanity Check

```json
// from server/checkpoints/DreamZero-DROID/experiment_cfg/metadata.json
"action.gripper_position": {
  "mean": 0.4534,
  "std":  0.4489,
  "q01":  0.0,
  "q99":  1.0
}
```

`std ≈ √(p·(1−p))` gives `p ≈ 0.45`, meaning the data is approximately Bernoulli-distributed at 45% closed. A mean of 0.45 with the gripper-mostly-open hypothesis (close only when grasping) is consistent with `1 = closed`.

### 1.4 Why You Must NOT Use Linear `2x − 1` Mapping for Deoxys

We initially proposed mapping DreamZero `[0, 1]` → Deoxys `[-1, +1]` linearly: `g_dx = 2*g_dz - 1`. **This is wrong.** Deoxys' gripper API has two **discrete** modes that switch at `action == 0`:

```python
# /tmp/deoxys_control/deoxys/deoxys/franka_interface/franka_interface.py:497-524
# action 0-> 1 : Grasp
# action 1-> 0 : Release
if action < 0.0:
    move_msg = FrankaGripperMoveMessage()
    move_msg.width = 0.08 * np.abs(action)   # position-only Move (no force)
    ...
elif action >= 0.0:
    grasp_msg = FrankaGripperGraspMessage()
    grasp_msg.force = 30.0                   # 30 N of grasping force
    ...
```

A continuous DreamZero output of, say, 0.4 → 0.6 → 0.5 → 0.7 would oscillate the dispatcher between *Move(width=…)* and *Grasp(force=30N)* every tick. **The gripper would chatter and force would spike chaotically.** It's literally unusable.

**Correct mapping (matches sim-evals `actions > 0.5`):**

```python
def dreamzero_gripper_action_to_deoxys(g: float) -> float:
    return 1.0 if g > 0.5 else -1.0
```

This is what [`client/deoxys_client.py`](../client/deoxys_client.py) implements.

---

## 2. Action Semantics: Relative-Trained, Absolute-Served

The training config explicitly enables relative actions:

```yaml
# server/checkpoints/DreamZero-DROID/experiment_cfg/conf.yaml:4112
relative_action: true
relative_action_keys:
  - joint_pos
  - gripper_pos
```

But the server-side decoder converts back to absolute *before* sending:

```python
# server/groot/vla/model/n1_5/sim_policy.py:602-604
# Add state to relative action to get absolute action
unnormalized_action[action_key] = unnormalized_action[action_key] + last_state
```

That's why metadata's action stats look "absolute":

```json
// metadata.json action.joint_position
"min": [-2.226, -1.668, -2.689, -1.733, -2.466, -2.285, -2.521],
"max": [ 2.427,  1.893,  2.333,  2.006,  2.418,  2.176,  2.386]
```

These are within Franka soft limits. The mean is ~0 (because they're relative *internally*), but by the time you receive `(24, 8)` over the wire, `action[i, :7]` is **the absolute target joint angle in radians** for Franka joint i+1.

**Implication for clients:** feed `action[i, :7]` directly to your joint-position controller. **Do not** add the current robot state.

---

## 3. Control Mode: Joint Impedance, Not Hard Position Servo

DROID is collected via Polymetis + libfranka, which exposes Franka's **joint impedance** mode underneath the `update_desired_joint_positions` API. From the upstream DROID `franka/robot.py`'s wrapper around Polymetis (referenced via `gs://gresearch/robotics/droid` source).

Sim-evals models this with IsaacLab `ImplicitActuatorCfg`:

```python
# /tmp/sim-evals/src/sim_evals/environments/nvidia_droid.py:53-74
actuators={
    "panda_shoulder": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-4]"],
        effort_limit=87.0,  velocity_limit=2.175,
        stiffness=400.0,    damping=80.0,
    ),
    "panda_forearm": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[5-7]"],
        effort_limit=12.0,  velocity_limit=2.61,
        stiffness=400.0,    damping=80.0,
    ),
    "gripper": ImplicitActuatorCfg(
        joint_names_expr=["finger_joint"],
        ...
    ),
},
```

Stiffness of 400 in IsaacLab's convention corresponds to a fairly **soft** PD. Mapping to Deoxys' real-hardware kp (Nm/rad), we use ~20% of these values to keep behavior gentle and reduce torque spikes at chunk boundaries:

```yaml
# client/deoxys_configs/dreamzero-joint-impedance.yml
joint_kp: [80., 80., 80., 80., 60., 100., 40.]
joint_kd: [16., 16., 16., 16.,  6.,  10.,  4.]
```

If your robot tracks slowly, scale kp up by ~50%. If it shakes, halve it. Joint 6 is bumped because of downstream inertia in DROID's typical poses.

**Why not stock Deoxys `joint-impedance-controller.yml`?** That config has `kp = [100, 100, 100, 100, 75, 150, 50]` and `time_fraction = 0.3`. The latter means the trajectory interpolator only covers 30% of each control tick, leaving the robot idle during 70% — fine for human teleop pacing, terrible for 15 Hz streamed targets.

---

## 4. Reset Pose

The DROID-canonical initial joint configuration:

```python
# upstream droid/robot_env.py
reset_joints = np.array([0, -π/5, 0, -4π/5, 0, 3π/5, 0])
```

Verified via sim-evals init_state:

```python
# /tmp/sim-evals/src/sim_evals/environments/nvidia_droid.py:38-50
joint_pos={
    "panda_joint1": 0.0,
    "panda_joint2": -1/5 * np.pi,
    "panda_joint3": 0.0,
    "panda_joint4": -4/5 * np.pi,
    "panda_joint5": 0.0,
    "panda_joint6": 3/5 * np.pi,
    "panda_joint7": 0,
    "finger_joint": 0.0,
    ...
}
```

`deoxys_client.py` calls `reset_to_droid_pose()` before the first inference. **Skipping this hurts zero-shot performance noticeably** — the DROID checkpoint has never seen non-canonical starting poses during training.

---

## 5. Camera Parameters

```json
// metadata.json modalities.video
"exterior_image_1_left": { "resolution": [320, 180], "channels": 3, "fps": 15.0 },
"exterior_image_2_left": { "resolution": [320, 180], "channels": 3, "fps": 15.0 },
"wrist_image_left":      { "resolution": [320, 180], "channels": 3, "fps": 15.0 }
```

Three cameras, all RGB uint8, all `(W, H) = (320, 180)` (numpy shape `(H, W, C) = (180, 320, 3)`), all 15 fps.

The training file confusingly mentions a 176-pixel height in some places:

```yaml
# server/checkpoints/DreamZero-DROID/experiment_cfg/conf.yaml:4467
image_resolution_height: 176
```

This is the model's **internal** input height after a server-side pad/crop. **The wire format the client must satisfy is 180**, as enforced by [`policy_server.py`](../server/eval_utils/policy_server.py) and demonstrated in [`test_client_AR.py`](../server/test_client_AR.py).

DROID's source cameras are stereo (left + right) but the model only consumes `_left`. Don't waste bandwidth on `_right` — `needs_stereo_camera=False`.

---

## 6. Frame Schedule

```python
# server/test_client_AR.py:52-53
RELATIVE_OFFSETS = [-23, -16, -8, 0]
ACTION_HORIZON   = 24
```

```python
# server/socket_test_optimized_AR.py:55
class ARDroidRoboarenaPolicy:
    FRAMES_PER_CHUNK = 4
```

**Convention:**
- First infer call of a session: **1 frame** of shape `(180, 320, 3)`.
- Every subsequent call: **4 frames** of shape `(4, 180, 320, 3)` at offsets `[-23, -16, -8, 0]` from "now".

At 15 Hz, those 4 offsets span the past ~1.53 s. Why these specific gaps? They're spaced to give a coarse temporal pyramid (24 ≈ chunk length, 16 = 2/3 chunk back, 8 = 1/3 chunk back, 0 = current). The model relies on this exact stride to interpret motion in its world-model backbone.

**Client buffering rule:** keep at least `ACTION_HORIZON + few = 30` frames per camera, append on every control tick at 15 Hz, then index `[-23, -16, -8, 0]` when constructing the next observation.

---

## 7. Open Questions / Caveats

These were not conclusively resolved but didn't matter in practice:

| Question | Status |
|---|---|
| Does the model expect zero-padding for the past frames in the very first chunk after reset? | The server replicates the first frame backward when buffer is short — see `server/socket_test_optimized_AR.py:142-147`. So clients can safely send any 4 frames they have. |
| Is the wrist camera RGB or BGR in the original training data? | Strongly RGB based on metadata format. We tested with RGB and the model's predicted videos look correct. |
| Can the model handle gripper values close to 0.5 (genuinely uncertain)? | In practice it nearly always emits values near 0 or 1. Clients should still binarize at 0.5 — see [§1.4](#14-why-you-must-not-use-linear-2x--1-mapping-for-deoxys). |
| Is the 7-DoF joint order DROID-standard (panda_joint1..7)? | Yes. Verified against `nvidia_droid.py` init_state ordering. |

---

## 8. Source Map

For anyone wanting to reverse-engineer further, here's where each fact comes from:

| File | What's there |
|---|---|
| [`server/checkpoints/DreamZero-DROID/experiment_cfg/metadata.json`](../server/checkpoints/DreamZero-DROID/experiment_cfg/metadata.json) (not in git, downloaded by setup script) | Per-key statistics (mean/std/q01/q99) and modality schemas |
| [`server/checkpoints/DreamZero-DROID/experiment_cfg/conf.yaml`](../server/checkpoints/DreamZero-DROID/experiment_cfg/conf.yaml) (same) | `relative_action: true`, `image_resolution_*`, `action_horizon: 24`, `num_frames: 33` |
| [`server/groot/vla/model/n1_5/sim_policy.py`](../server/groot/vla/model/n1_5/sim_policy.py) | Server's apply() that converts relative→absolute |
| [`server/socket_test_optimized_AR.py`](../server/socket_test_optimized_AR.py) | The actual server entry; `ARDroidRoboarenaPolicy` does frame buffering and obs/action conversion |
| [`server/eval_utils/policy_server.py`](../server/eval_utils/policy_server.py) | `PolicyServerConfig` dataclass + WS framing |
| [`server/test_client_AR.py`](../server/test_client_AR.py) | Reference frame schedule; useful for sanity |
| sim-evals `droid_environment.py` (cloned to `/tmp/sim-evals` during research) | Gripper convention + action binarization |
| sim-evals `nvidia_droid.py` | Reset pose + actuator stiffness/damping |
| deoxys_control `franka_interface.py` | Gripper Move vs Grasp dispatch |
