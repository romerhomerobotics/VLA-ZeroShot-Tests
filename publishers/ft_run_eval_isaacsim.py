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

def normalize_gripper(gripper):
    """
    Model output: [-1, +1]
    Sim expected: 0 (close) to 100 (open)
    """
    # Clamp input
    gripper = np.clip(gripper, -1.0, 1.0)

    # Map [-1, 1] to [0, 100]
    sim_value = (gripper + 1) * 50.0   # -1->0, 1->100

    return float(sim_value)

def unnormalize_gripper(g_state):
    """
    Sim reports: 0 (closed) to 0.04 (open)
    Convert back to [-1, +1]
    """
    # Clamp incoming state
    g_state = np.clip(g_state, 0.0, 0.04)

    # Map [0, 0.04] to [-1, 1]
    model_value = (g_state / 0.04) * 2.0 - 1.0

    return float(model_value)

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
    flush = True

    try:
        while t < max_steps:
            rclpy.spin_once(ros, timeout_sec=0.01)

            # Get latest sensor data
            full_image, wrist_image, raw_proprio, gripper = ros.get_latest()
            if full_image is None or raw_proprio is None or gripper is None:
                print("Waiting for data")
                continue  # wait for valid data

            # Transform proprio from IsaacSim -> LIBERO
            pos = np.asarray(raw_proprio["eef_pos"], dtype=np.float32)
            quat = np.asarray(raw_proprio["eef_quat"], dtype=np.float32)
            gripper = float(gripper)

            gripper = unnormalize_gripper(gripper)

            proprio = {
                "eef_pos": pos,
                "eef_quat": quat,
                "gr_state": np.array([0, gripper])
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
                if flush:
                   flush = False
                   continue 
                action_queue.extend(actions)

            # Pop action from queue
            action = action_queue.popleft()
            action_gripper = action[-1]
            action_gripper = normalize_gripper(action_gripper)
            # Clip actions
            # norm = np.linalg.norm(action)
            # if norm > 0.2:
                # action = action * (0.2 / norm)

            # Publish to ROS
            ros.publish_action(action, action_gripper)

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

