import rclpy
from collections import deque
import numpy as np
import torch

from vla_utils.ft_config import get_config
from vla_utils.ft_model_loader import load_model
from libero_utils.libero_utils import prepare_observation
from experiments.robot.libero.libero_utils import (
    save_rollout_video
)
from experiments.robot.robot_utils import (
    set_seed_everywhere,
    get_image_resize_size
)
from ros_utils.ros_utils import ROSInterface

from prismatic.vla.constants import NUM_ACTIONS_CHUNK

import traceback

W_FIRST = False

def run_episode(
    cfg,
    model,
    resize_size,
    ros: ROSInterface,
    task_description: str = "Pick up the mustard"
):
    """
    Run a single episode using OpenVLA-OFT model over ROS.
    """

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    t = 0
    replay_images = []
#    max_steps = cfg.TASK_MAX_STEPS
    max_steps = 1_000
    success = False

    try:
        while t < max_steps:
            rclpy.spin_once(ros, timeout_sec=0.01)

            # Get latest sensor data
            full_image, wrist_image, raw_proprio = ros.get_latest()
            if full_image is None or raw_proprio is None:
                print("Waiting for data")
                continue  # wait for valid data

            # Transform proprio from IsaacSim -> LIBERO
            pos = np.asarray(raw_proprio["eef_pos"], dtype=np.float32)
            quat = np.asarray(raw_proprio["eef_quat"], dtype=np.float32)
            gripper = np.asarray(raw_proprio.get("gr_state", [0.0, 0.0]), dtype=np.float32)

            proprio = {
                "eef_pos": pos,
                "eef_quat": quat,
                "gr_state": gripper
            }

            # Prepare observation
            observation, image = prepare_observation(
                image=full_image,
                wrist_image=wrist_image,
                proprio=proprio,
                resize_size=resize_size
            )
            replay_images.append(image)

            # Query model if action queue empty
            if len(action_queue) == 0:
                actions = model.predict_actions(observation, task_description)
                action_queue.extend(actions)

            # Pop action from queue
            action = action_queue.popleft()

            # Clip actions
            # norm = np.linalg.norm(action)
            # if norm > 0.2:
                # action = action * (0.2 / norm)

            # Publish to ROS
            ros.publish_action(action)

            t += 1

    except Exception as e:
        print("Episode terminated with error:")
        traceback.print_exc()

    return success, replay_images


def eval_libero():
    """
    Evaluate a fine-tuned OpenVLA-OFT policy in LIBERO via ROS.
    """

    cfg = get_config()
    set_seed_everywhere(cfg.seed)

    # Load fine-tuned model
    model = load_model(cfg)

    # Initialize ROS
    rclpy.init()
    ros = ROSInterface()

    resize_size = get_image_resize_size(cfg)
    task_description = "Pick up the mustard"

    success, replay_images = run_episode(cfg, model, resize_size, ros, task_description)

    save_rollout_video(replay_images, 1, success=success, task_description=task_description)


if __name__ == "__main__":
    # Path to your fine-tuned checkpoint folder
    eval_libero()

