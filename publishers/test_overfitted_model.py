import os
import traceback
import numpy as np
import rclpy
from collections import deque

import torch

# --- Your libraries ---
from vla_utils.ft_config import get_config
from vla_utils.ft_model_loader import load_model
from libero_utils.libero_utils import prepare_observation
from experiments.robot.libero.libero_utils import save_rollout_video
from experiments.robot.robot_utils import (
    set_seed_everywhere,
    get_image_resize_size,
)
from ros_utils.ros_utils import ROSInterface

from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# ------------------ CONFIG ------------------

NPZ_FILE = "/home/romer-vla-sim/Workspace/raw_datasets/mustard_pick_place/coupled_data_10.0hz/traj_0.npz"
TASK_DESCRIPTION = "Pick up the mustard"
W_FIRST = False


# ------------------ MAIN LOOP ------------------

def run_episode_from_npz(
    cfg,
    model,
    resize_size,
    ros: ROSInterface,
    npz_path: str,
    task_description: str = TASK_DESCRIPTION,
):
    """
    Uses a saved trajectory file as input instead of ROS sensors.
    Feeds each saved observation frame into the model and publishes predicted actions to ROS.
    """

    # Load recorded demo
    data = np.load(npz_path, allow_pickle=True)
    images = data["image"]          # (T, H, W, 3)
    wrist_images = data["image"]    # no wrist cam => reuse
    states = data["state"]          # (T, 6)   raw proprio
    T = len(images)

    print(f"[INFO] Loaded trajectory with {T} steps from {npz_path}")

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    replay_images = []

    # Debug: stop infinite loops
    max_steps = min(T, 2000)

    t = 0
    flush = True
    success = False

    try:
        while t < max_steps:

            # ----------------------------
            # Load saved image + proprio
            # ----------------------------
            full_image = images[t]
            wrist_image = wrist_images[t]

            pos = states[t][:3]    # (x,y,z)
            rpy = states[t][3:]    # (r,p,y)
            quat = np.array([0, 0, 0, 1], dtype=np.float32)   # no quat => dummy
            gripper = np.array([0.0, 0.0], dtype=np.float32)

            proprio = {
                "eef_pos": pos,
                "eef_quat": quat,
                "gr_state": gripper,
            }

            # ----------------------------
            # Prepare observation for model
            # ----------------------------
            observation, resized_image = prepare_observation(
                image=full_image,
                wrist_image=wrist_image,
                proprio=proprio,
                resize_size=resize_size,
            )

            replay_images.append(resized_image)

            # ----------------------------
            # Query model only when queue empty
            # ----------------------------
            if len(action_queue) == 0:
                actions = model.predict_actions(observation, task_description)

                # The FIRST prediction after loading is junk => ignore once
                if flush:
                    flush = False
                    t += 1
                    continue

                action_queue.extend(actions)

            # ----------------------------
            # Pop 1 action from predicted sequence
            # ----------------------------
            action = action_queue.popleft()

            # ----------------------------
            # Publish via ROS
            # ----------------------------
            ros.publish_action(action)

            t += 1

    except Exception as e:
        print("Episode terminated with error:")
        traceback.print_exc()

    return success, replay_images


# ------------------ WRAPPER ------------------

def eval_overfit_npz():
    """
    Evaluate an overfitted model using saved trajectory frames instead of live robot / IsaacSim.
    """

    cfg = get_config()
    set_seed_everywhere(cfg.seed)

    # Load your model
    model = load_model(cfg)

    # Initialize ROS publishing only
    rclpy.init()
    ros = ROSInterface()

    resize_size = get_image_resize_size(cfg)

    success, replay_images = run_episode_from_npz(
        cfg=cfg,
        model=model,
        resize_size=resize_size,
        ros=ros,
        npz_path=NPZ_FILE,
        task_description=TASK_DESCRIPTION,
    )

    save_rollout_video(
        replay_images,
        1,
        success=success,
        task_description=TASK_DESCRIPTION,
    )

    print("[DONE] Evaluation finished.")


# ------------------ ENTRY ------------------

if __name__ == "__main__":
    eval_overfit_npz()

