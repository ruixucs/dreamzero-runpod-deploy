"""DreamZero + Deoxys 客户端 (Franka Panda + Robotiq/Franka Hand)。

依赖(客户端电脑):
  pip install numpy opencv-python pyrealsense2 websockets msgpack msgpack-numpy openpi-client
  # 以及 deoxys_control 已经按官方步骤编译并在 NUC 上运行 franka-interface

并把 eval_utils/policy_client.py 和 policy_server.py 也拷贝到客户端电脑同目录。

启动:
  # 1. NUC 端先启动 franka-interface
  # 2. RunPod 服务端确认 ws://server:5000 可达
  # 3. 客户端电脑:
  python deoxys_client.py \
      --host <server-host> --port 5000 \
      --interface-cfg charmander.yml \
      --controller-cfg dreamzero-joint-impedance.yml \
      --prompt "pick up the red block"
"""
import argparse
import logging
import time
import uuid
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from eval_utils.policy_client import WebsocketClientPolicy
from eval_utils.policy_server import PolicyServerConfig

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.log_utils import get_deoxys_example_logger
from deoxys.experimental.motion_utils import joint_interpolation_traj


IMAGE_H, IMAGE_W = 180, 320
RELATIVE_OFFSETS = [-23, -16, -8, 0]
ACTION_HORIZON = 24
CONTROL_HZ = 15
DT = 1.0 / CONTROL_HZ

DROID_RESET_JOINTS = np.array(
    [0.0, -np.pi / 5, 0.0, -4 * np.pi / 5, 0.0, 3 * np.pi / 5, 0.0],
    dtype=np.float64,
)
GRIPPER_MAX_WIDTH_M = 0.08

logger = get_deoxys_example_logger()


def deoxys_gripper_state_to_dreamzero(width_m: float) -> float:
    """Deoxys 给的 gripper width 是米; 0 = closed, 0.08 = open。
    DreamZero 期待 [0,1]: 0 = open, 1 = closed。
    """
    open_ratio = float(np.clip(width_m / GRIPPER_MAX_WIDTH_M, 0.0, 1.0))
    return 1.0 - open_ratio


def dreamzero_gripper_action_to_deoxys(g: float) -> float:
    """DreamZero action [0,1] (0=open, 1=close) -> Deoxys gripper action [-1, 1]
    (action >= 0: grasp/close, action < 0: open by width = 0.08 * |action|).

    我们做硬阈值化(model gripper 输出本就近似双峰)。
    """
    if float(g) > 0.5:
        return 1.0
    return -1.0


class RealsenseCam:
    """简单的 pyrealsense2 包装。如果你用 OpenCV/V4L 摄像头, 改成对应读取即可。"""

    def __init__(self, serial: str, fps: int = 30):
        import pyrealsense2 as rs
        self._rs = rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, fps)
        self.pipeline.start(cfg)

    def read_rgb_180x320(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames(timeout_ms=200)
        c = frames.get_color_frame()
        if not c:
            return np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
        bgr = np.asanyarray(c.get_data())
        bgr = cv2.resize(bgr, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def stop(self):
        self.pipeline.stop()


class DeoxysRobot:
    def __init__(
        self,
        interface_cfg: str = "charmander.yml",
        controller_cfg: str = "dreamzero-joint-impedance.yml",
        controller_type: str = "JOINT_IMPEDANCE",
        cam_ext0_serial: str = "",
        cam_ext1_serial: str = "",
        cam_wrist_serial: str = "",
    ):
        self.controller_type = controller_type

        cfg_path = Path(config_root) / interface_cfg
        ctrl_cfg_path = Path(config_root) / controller_cfg
        if not cfg_path.exists() or not ctrl_cfg_path.exists():
            raise FileNotFoundError(
                f"Deoxys configs not found:\n  {cfg_path}\n  {ctrl_cfg_path}"
            )

        self.robot = FrankaInterface(str(cfg_path), use_visualizer=False)
        self.controller_cfg = YamlConfig(str(ctrl_cfg_path)).as_easydict()

        while self.robot.state_buffer_size == 0:
            logger.warning("Waiting for first robot state...")
            time.sleep(0.2)

        self.cam_ext0 = RealsenseCam(cam_ext0_serial)
        self.cam_ext1 = RealsenseCam(cam_ext1_serial)
        self.cam_wrist = RealsenseCam(cam_wrist_serial)

    def reset_to_droid_pose(self) -> None:
        """开始任务前移动到 DROID 默认初始位姿(reset_joints)。"""
        last_q = np.array(self.robot.last_q, dtype=np.float64)
        traj = joint_interpolation_traj(start_q=last_q, end_q=DROID_RESET_JOINTS)
        for q in traj:
            action = q.tolist() + [-1.0]   # gripper open during reset
            self.robot.control(
                controller_type=self.controller_type,
                action=action,
                controller_cfg=self.controller_cfg,
            )

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        q = np.array(self.robot.last_q, dtype=np.float32)             # (7,)
        width = float(self.robot.last_gripper_q) if hasattr(self.robot, "last_gripper_q") else 0.08
        g = np.array([deoxys_gripper_state_to_dreamzero(width)], dtype=np.float32)
        return q, g

    def get_images(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ext0 = self.cam_ext0.read_rgb_180x320()
        ext1 = self.cam_ext1.read_rgb_180x320()
        wrist = self.cam_wrist.read_rgb_180x320()
        return ext0, ext1, wrist

    def execute_action(self, joint_target: np.ndarray, gripper_action_dz: float) -> None:
        """单步执行: 把目标关节角发给 Deoxys, 同时 gripper [0,1] -> [-1,+1]。"""
        gripper_deoxys = dreamzero_gripper_action_to_deoxys(gripper_action_dz)
        action = joint_target.astype(np.float64).tolist() + [gripper_deoxys]
        self.robot.control(
            controller_type=self.controller_type,
            action=action,
            controller_cfg=self.controller_cfg,
        )

    def is_safe(self) -> bool:
        if self.robot.state_buffer_size == 0:
            return False
        q = np.array(self.robot.last_q)
        if np.any(np.isnan(q)):
            return False
        # Franka soft joint limits 简化检查
        lo = np.array([-2.7, -1.6, -2.7, -2.9, -2.7, 0.2, -2.8])
        hi = np.array([2.7, 1.6, 2.7, -0.3, 2.7, 4.2, 2.8])
        if np.any(q < lo - 0.1) or np.any(q > hi + 0.1):
            logger.error(f"Joint limit exceeded: {q}")
            return False
        return True

    def close(self):
        try:
            self.cam_ext0.stop(); self.cam_ext1.stop(); self.cam_wrist.stop()
        finally:
            self.robot.close()


class FrameBuffer:
    def __init__(self, maxlen: int = ACTION_HORIZON + 6):
        self.b0: deque = deque(maxlen=maxlen)
        self.b1: deque = deque(maxlen=maxlen)
        self.bw: deque = deque(maxlen=maxlen)

    def push(self, e0, e1, w):
        self.b0.append(e0); self.b1.append(e1); self.bw.append(w)

    def __len__(self):
        return len(self.b0)

    def take(self, offsets):
        n = len(self)
        def grab(buf):
            arr = list(buf)
            return np.stack([arr[max(0, n - 1 + o)] for o in offsets], axis=0)
        return grab(self.b0), grab(self.b1), grab(self.bw)


def run_episode(robot: DeoxysRobot, client: WebsocketClientPolicy,
                prompt: str, max_chunks: int, blend_steps: int = 3):
    session_id = str(uuid.uuid4())
    buf = FrameBuffer()
    logger.info(f"Episode start, session={session_id}, prompt={prompt!r}")

    # —— 初始单帧 ——
    e0, e1, w = robot.get_images()
    q, g = robot.get_state()
    buf.push(e0, e1, w)
    obs = {
        "observation/exterior_image_0_left": e0,
        "observation/exterior_image_1_left": e1,
        "observation/wrist_image_left": w,
        "observation/joint_position": q,
        "observation/cartesian_position": np.zeros(6, dtype=np.float32),
        "observation/gripper_position": g,
        "prompt": prompt,
        "session_id": session_id,
    }
    t0 = time.time()
    actions = client.infer(obs)
    logger.info(f"Initial inference {actions.shape} in {time.time()-t0:.2f}s")

    # —— 主循环 ——
    for chunk_idx in range(max_chunks):
        if not robot.is_safe():
            logger.error("Safety check failed, abort")
            break

        # ① 平滑过渡: 当前实际 q 到 actions[0] 做 blend_steps 步插值
        cur_q = np.array(robot.robot.last_q, dtype=np.float64)
        target_q0 = actions[0, :7].astype(np.float64)
        for k in range(blend_steps):
            alpha = (k + 1) / (blend_steps + 1)
            q_blend = (1 - alpha) * cur_q + alpha * target_q0
            t_step = time.time()
            robot.execute_action(q_blend, float(actions[0, 7]))
            e0, e1, w = robot.get_images(); buf.push(e0, e1, w)
            sleep_left = DT - (time.time() - t_step)
            if sleep_left > 0:
                time.sleep(sleep_left)

        # ② 正式执行 24 步
        for step in range(ACTION_HORIZON):
            t_step = time.time()
            robot.execute_action(actions[step, :7], float(actions[step, 7]))
            e0, e1, w = robot.get_images(); buf.push(e0, e1, w)
            sleep_left = DT - (time.time() - t_step)
            if sleep_left > 0:
                time.sleep(sleep_left)
            if not robot.is_safe():
                logger.error("Safety violated mid-execution")
                return

        # ③ 推理下一组
        e0_4, e1_4, w_4 = buf.take(RELATIVE_OFFSETS)
        q, g = robot.get_state()
        obs = {
            "observation/exterior_image_0_left": e0_4,
            "observation/exterior_image_1_left": e1_4,
            "observation/wrist_image_left": w_4,
            "observation/joint_position": q,
            "observation/cartesian_position": np.zeros(6, dtype=np.float32),
            "observation/gripper_position": g,
            "prompt": prompt,
            "session_id": session_id,
        }
        t0 = time.time()
        actions = client.infer(obs)
        logger.info(
            f"chunk {chunk_idx}: actions {actions.shape} "
            f"q_range=[{actions[:,:7].min():.2f},{actions[:,:7].max():.2f}] "
            f"g_mean={actions[:,7].mean():.2f} infer={time.time()-t0:.2f}s"
        )

    client.reset({})
    logger.info("Episode finished")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--interface-cfg", default="charmander.yml")
    p.add_argument("--controller-cfg", default="dreamzero-joint-impedance.yml")
    p.add_argument("--controller-type", default="JOINT_IMPEDANCE",
                   choices=["JOINT_IMPEDANCE", "JOINT_POSITION"])
    p.add_argument("--cam-ext0", default="")
    p.add_argument("--cam-ext1", default="")
    p.add_argument("--cam-wrist", default="")
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-chunks", type=int, default=20)
    p.add_argument("--no-reset-pose", action="store_true",
                   help="Skip moving to DROID reset pose at start")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    client = WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = client.get_server_metadata()
    server_config = PolicyServerConfig(**metadata)
    logger.info(f"Server: {server_config}")
    assert server_config.action_space == "joint_position"
    assert server_config.n_external_cameras == 2
    assert server_config.needs_wrist_camera

    robot = DeoxysRobot(
        interface_cfg=args.interface_cfg,
        controller_cfg=args.controller_cfg,
        controller_type=args.controller_type,
        cam_ext0_serial=args.cam_ext0,
        cam_ext1_serial=args.cam_ext1,
        cam_wrist_serial=args.cam_wrist,
    )
    try:
        if not args.no_reset_pose:
            robot.reset_to_droid_pose()
            time.sleep(0.5)
        run_episode(robot, client, prompt=args.prompt, max_chunks=args.max_chunks)
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        robot.close()


if __name__ == "__main__":
    main()
