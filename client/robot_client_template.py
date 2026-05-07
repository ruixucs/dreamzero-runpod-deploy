"""DreamZero 机械臂客户端模板。

把它复制到您机械臂控制电脑上,然后:
  1. 实现 RobotInterface 类的 4 个方法 (硬件驱动相关)
  2. 修改 SERVER_HOST / SERVER_PORT
  3. 运行: python robot_client_template.py --prompt "your task"

依赖(机械臂电脑上):
  pip install numpy opencv-python websockets msgpack msgpack-numpy openpi-client

通信协议: WebSocket + msgpack (与 socket_test_optimized_AR.py 服务器对接)
"""

import argparse
import logging
import time
import uuid
from collections import deque
from typing import Optional

import cv2
import numpy as np

from eval_utils.policy_client import WebsocketClientPolicy
from eval_utils.policy_server import PolicyServerConfig


IMAGE_H, IMAGE_W = 180, 320
RELATIVE_OFFSETS = [-23, -16, -8, 0]
ACTION_HORIZON = 24
CONTROL_HZ = 15  # DROID 默认 15 Hz; 改成您机械臂的控制频率


class RobotInterface:
    """硬件抽象层 —— 您必须根据自己的机械臂 SDK 实现这 4 个方法。

    示例: Franka Panda + RealSense D435 + Robotiq 2F-85
    """

    def __init__(self):
        # TODO: 在这里初始化您的机械臂、夹爪、3 个相机
        # self.arm = FrankaPanda(...)
        # self.gripper = Robotiq2F85(...)
        # self.cam_ext0 = RealsenseCamera(serial="...")
        # self.cam_ext1 = RealsenseCamera(serial="...")
        # self.cam_wrist = RealsenseCamera(serial="...")
        pass

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """读取当前机械臂关节角和夹爪位置。

        Returns:
            joint_position: (7,) float32, 7 个关节弧度(或您机械臂的单位)
            gripper_position: (1,) float32, [0,1], **0=完全张开, 1=完全闭合**
                              (DROID 约定: finger_joint / (pi/4),
                              Robotiq 2F-85 中 0 rad = open, pi/4 rad = closed)
        """
        # TODO: 替换为真实读取
        joint = np.zeros(7, dtype=np.float32)
        gripper = np.zeros(1, dtype=np.float32)
        return joint, gripper

    def get_images(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """读取 3 个相机当前帧, 返回 RGB, shape=(180, 320, 3), dtype=uint8。

        如果相机原生分辨率不同, 在这里 cv2.resize 到 (320, 180)。
        注意 OpenCV 默认 BGR, 要转 RGB!
        """
        # TODO: 替换为真实读取
        # frame_bgr = self.cam_ext0.get_color_frame()
        # ext0 = cv2.cvtColor(cv2.resize(frame_bgr, (IMAGE_W, IMAGE_H)), cv2.COLOR_BGR2RGB)
        ext0 = np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
        ext1 = np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
        wrist = np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
        return ext0, ext1, wrist

    def execute_action(self, joint_target: np.ndarray, gripper_target: float) -> None:
        """阻塞地执行一步动作 (大约 1/CONTROL_HZ 秒)。

        Args:
            joint_target: (7,) 7 个关节目标位置
            gripper_target: scalar in [0,1], **0=张开, 1=闭合** (DROID 约定)
                           sim-evals 用 > 0.5 二值化:
                             > 0.5  → close
                             <= 0.5 → open
                           真机连续控制时,直接把 [0,1] 映射到夹爪行程也可以,
                           但模型输出的 gripper 实际是双峰分布,二值化通常更稳。
        """
        # TODO: 替换为真实控制
        # self.arm.move_to_joint_positions(joint_target, asynchronous=False)
        # self.gripper.set_position(gripper_target)
        time.sleep(1.0 / CONTROL_HZ)

    def is_safe(self) -> bool:
        """安全检查 —— 力矩超限/关节限位/急停按钮等。返回 False 立即停止。"""
        return True


class FrameBuffer:
    """缓存最近的图像帧, 供模型按 [-23, -16, -8, 0] 调度取用。"""

    def __init__(self, maxlen: int = 30):
        self.buf_ext0: deque = deque(maxlen=maxlen)
        self.buf_ext1: deque = deque(maxlen=maxlen)
        self.buf_wrist: deque = deque(maxlen=maxlen)

    def push(self, ext0: np.ndarray, ext1: np.ndarray, wrist: np.ndarray) -> None:
        self.buf_ext0.append(ext0)
        self.buf_ext1.append(ext1)
        self.buf_wrist.append(wrist)

    def __len__(self) -> int:
        return len(self.buf_ext0)

    def take_recent_at_offsets(
        self, offsets: list[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """偏移 0 = 最新帧, -1 = 倒数第二帧 ..."""
        n = len(self)

        def grab(buf: deque) -> np.ndarray:
            arr = list(buf)
            picked = []
            for off in offsets:
                idx = max(0, n - 1 + off)
                picked.append(arr[idx])
            return np.stack(picked, axis=0)

        return grab(self.buf_ext0), grab(self.buf_ext1), grab(self.buf_wrist)


def run_episode(
    robot: RobotInterface,
    client: WebsocketClientPolicy,
    server_config: PolicyServerConfig,
    prompt: str,
    max_chunks: int = 100,
) -> None:
    """跑一个完整任务回合: 初始帧 -> 循环(执行 24 步 -> 推理新动作)。"""
    session_id = str(uuid.uuid4())
    logging.info(f"=== Starting episode, session_id={session_id} ===")
    logging.info(f"prompt = {prompt!r}")

    buffer = FrameBuffer(maxlen=ACTION_HORIZON + 6)

    # —— Step 1: 初始化推理(只发送 1 帧当前帧) ————————————————————————
    ext0, ext1, wrist = robot.get_images()
    joint, gripper = robot.get_state()
    buffer.push(ext0, ext1, wrist)

    obs = {
        "observation/exterior_image_0_left": ext0,           # (H, W, 3)
        "observation/exterior_image_1_left": ext1,
        "observation/wrist_image_left": wrist,
        "observation/joint_position": joint,
        "observation/cartesian_position": np.zeros(6, dtype=np.float32),
        "observation/gripper_position": gripper,
        "prompt": prompt,
        "session_id": session_id,
    }
    logging.info("Sending initial observation (1 frame)...")
    t0 = time.time()
    actions = client.infer(obs)
    logging.info(f"Initial inference: {actions.shape}, took {time.time()-t0:.2f}s")

    # —— Step 2: 主循环 ———————————————————————————————————————
    for chunk_idx in range(max_chunks):
        if not robot.is_safe():
            logging.error("Safety check failed, aborting")
            break

        # 执行 24 步动作, 同时持续缓存图像帧 (用于下次推理)
        logging.info(f"--- Chunk {chunk_idx}: executing 24 actions ---")
        for step in range(ACTION_HORIZON):
            joint_target = actions[step, :7].astype(np.float32)
            gripper_target = float(actions[step, 7])
            robot.execute_action(joint_target, gripper_target)

            # 每步采集图像入缓存(给下次模型用)
            ext0, ext1, wrist = robot.get_images()
            buffer.push(ext0, ext1, wrist)

            if not robot.is_safe():
                logging.error("Safety stop during action execution")
                return

        # 推理下一组动作 —— 用 [-23, -16, -8, 0] 取 4 帧
        ext0_4, ext1_4, wrist_4 = buffer.take_recent_at_offsets(RELATIVE_OFFSETS)
        joint, gripper = robot.get_state()

        obs = {
            "observation/exterior_image_0_left": ext0_4,     # (4, H, W, 3)
            "observation/exterior_image_1_left": ext1_4,
            "observation/wrist_image_left": wrist_4,
            "observation/joint_position": joint,
            "observation/cartesian_position": np.zeros(6, dtype=np.float32),
            "observation/gripper_position": gripper,
            "prompt": prompt,
            "session_id": session_id,
        }
        t0 = time.time()
        actions = client.infer(obs)
        logging.info(
            f"  Inference: {actions.shape}, "
            f"range=[{actions.min():.3f},{actions.max():.3f}], "
            f"took {time.time()-t0:.2f}s"
        )

    # 收尾 —— 让服务器保存可视化视频
    logging.info("Episode finished, sending reset...")
    client.reset({})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="DreamZero server hostname")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--prompt", required=True, help="Natural language task description"
    )
    parser.add_argument("--max-chunks", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    logging.info(f"Connecting to DreamZero server at {args.host}:{args.port}...")
    client = WebsocketClientPolicy(host=args.host, port=args.port)

    metadata = client.get_server_metadata()
    server_config = PolicyServerConfig(**metadata)
    logging.info(f"Server config: {server_config}")

    assert server_config.action_space == "joint_position"
    assert server_config.n_external_cameras == 2
    assert server_config.needs_wrist_camera

    robot = RobotInterface()
    try:
        run_episode(
            robot=robot,
            client=client,
            server_config=server_config,
            prompt=args.prompt,
            max_chunks=args.max_chunks,
        )
    except KeyboardInterrupt:
        logging.info("Interrupted by user")


if __name__ == "__main__":
    main()
