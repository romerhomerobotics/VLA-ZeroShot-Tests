"""
WebSocket client for Isaac Sim that streams sensor data to the remote OpenVLA-OFT server
and applies returned actions via ROS.
"""

import asyncio
import io
import json
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import rclpy
from PIL import Image
from scipy.spatial.transform import Rotation as R
import websockets

from experiments.robot.libero.libero_utils import save_rollout_video
from experiments.robot.robot_utils import set_seed_everywhere
from ros_utils.ros_utils import ROSInterface
from vla_utils.ft_config import get_config

W_FIRST = False
SERVER_URI = "ws://localhost:8000/ws_inference"


def normalize_gripper(gripper: float) -> float:
    gripper = np.clip(gripper, -1.0, 1.0)
    return float((gripper + 1.0) * 50.0)


def unnormalize_gripper(g_state: float) -> float:
    g_state = np.clip(g_state, 0.0, 0.04)
    return float((g_state / 0.04) * 2.0 - 1.0)


def encode_jpeg(image: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


async def request_actions(
    websocket: websockets.WebSocketClientProtocol,
    full_image: np.ndarray,
    wrist_image: Optional[np.ndarray],
    proprio: Dict[str, np.ndarray],
    task_description: str,
) -> List[List[float]]:
    meta = {
        "task_description": task_description,
        "proprio": {
            "eef_pos": proprio["eef_pos"].tolist(),
            "eef_quat": proprio["eef_quat"].tolist(),
            "gr_state": proprio["gr_state"].tolist(),
        },
        "has_wrist": wrist_image is not None,
    }

    await websocket.send(json.dumps(meta))
    await websocket.send(encode_jpeg(full_image))
    if wrist_image is not None:
        await websocket.send(encode_jpeg(wrist_image))

    response_raw = await websocket.recv()
    data = json.loads(response_raw)
    if "error" in data:
        raise RuntimeError(data["error"])
    return data.get("actions", [])


def build_proprio(raw_proprio: Dict[str, Any], gripper: float) -> Dict[str, np.ndarray]:
    pos = np.asarray(raw_proprio["eef_pos"], dtype=np.float32)
    quat = np.asarray(raw_proprio["eef_quat"], dtype=np.float32)
    gr = np.array([gripper], dtype=np.float32)
    return {"eef_pos": pos, "eef_quat": quat, "gr_state": gr}


async def run_episode_async(
    cfg: Any,
    ros: ROSInterface,
    task_description: str,
    max_steps: int = 800,
) -> tuple[bool, List[np.ndarray]]:
    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    replay_images: List[np.ndarray] = []
    t = 0
    success = False
    flush = True

    async with websockets.connect(SERVER_URI, max_size=None) as websocket:
        while t < max_steps:
            rclpy.spin_once(ros, timeout_sec=0.01)

            full_image, wrist_image, raw_proprio, gripper_state = ros.get_latest()
            if full_image is None or raw_proprio is None or gripper_state is None:
                await asyncio.sleep(0.01)
                continue

            gripper_model_space = unnormalize_gripper(float(gripper_state))
            proprio = build_proprio(raw_proprio, gripper_model_space)
            replay_images.append(full_image)

            if len(action_queue) == 0:
                actions = await request_actions(
                    websocket, full_image, wrist_image, proprio, task_description
                )
                if flush:
                    flush = False
                    await asyncio.sleep(0)
                    continue
                action_queue.extend(actions)

            if len(action_queue) == 0:
                await asyncio.sleep(0)
                continue

            action = np.asarray(action_queue.popleft(), dtype=np.float32)
            action_gripper = normalize_gripper(float(action[-1]))

            delta_pos = action[0:3]
            pos = proprio["eef_pos"]
            quat = proprio["eef_quat"]
            z_limit = 0.09
            if pos[2] <= z_limit:
                delta_pos[2] = 0.02

            delta_rpy = action[3:6]
            eef_rpy = R.from_quat(quat).as_euler("xyz", degrees=False)
            new_rpy = eef_rpy + delta_rpy
            new_quat = R.from_euler("xyz", new_rpy, degrees=False).as_quat()
            new_pos = pos + delta_pos
            action_pose = np.concatenate([new_pos, new_quat])

            ros.publish_action_pose(action_pose, action_gripper)

            t += 1
            await asyncio.sleep(0)

    return success, replay_images


async def main() -> None:
    cfg = get_config()
    set_seed_everywhere(cfg.seed)
    rclpy.init()
    ros = ROSInterface()

    task_description = "pick up the apple and place it in the grey basket"

    success, replay_images = await run_episode_async(
        cfg=cfg,
        ros=ros,
        task_description=task_description,
    )

    save_rollout_video(replay_images, 1, success=success, task_description=task_description)
    ros.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
