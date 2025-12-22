import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb
import rclpy
import cv2
import numpy as np

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

# Imports from run_libero_eval

from experiments.robot.libero.run_libero_eval import (
    validate_config,
    initialize_model,
    check_unnorm_key,
    setup_logging,
    log_message,
    process_action,
)

from libero_utils.libero_utils import (
    prepare_observation,
)
from ros_utils.ros_utils import ROSInterface
from scipy.spatial.transform import Rotation as Rot
from compare_eef_poses import R as R_umeyama, t as t_umeyama
import torch
import torch.nn as nn


class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}

task_description = 'Pick up the green cube'

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "/home/romer-vla-sim/model_weights/vla/openvla_oft"     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = "libero_object_no_noops"                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_OBJECT  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)
    TASK_MAX_STEPS = 1500

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

import numpy as np

W_FIRST = False

def quat_to_rotmat(q):
    if W_FIRST:
        w, x, y, z = q
    else:
        x, y, z, w = q
    # Normalize
    n = np.linalg.norm([x, y, z, w])
    x, y, z, w = x/n, y/n, z/n, w/n

    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ], dtype=np.float32)
    return R

def rotmat_to_quat(R):
    R = np.asarray(R, dtype=np.float64)
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            w = (R[2,1] - R[1,2]) / s
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s

    if W_FIRST:
        q = np.array([w, x, y, z], dtype=np.float32)
    else:
        q = np.array([x, y, z, w], dtype=np.float32)

    return q / np.linalg.norm(q)
# # Define the same MLP structure as during training
# class PoseMLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(7, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 7)
#         )

#     def forward(self, x):
#         out = self.net(x)
#         pos = out[:, :3]
#         quat = out[:, 3:]
#         quat = quat / torch.norm(quat, dim=1, keepdim=True)
#         return torch.cat([pos, quat], dim=1)

# mlp = PoseMLP()
# mlp.load_state_dict(torch.load("publishers/pose_mlp.pt"))
# mlp.eval()

def run_episode(
    cfg : GenerateConfig,
    model,
    resize_size,
    task_description: str = "Pick up the green cube",
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    log_file=None,
    ros: ROSInterface = None,

):
    """Run episode function similar to original implementation in OpenVLA-OFT"""
    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    max_steps = cfg.TASK_MAX_STEPS

    # Run episode
    success = False
    flush_queue = True
    try:
        while t < max_steps * 1000 + cfg.num_steps_wait:
            rclpy.spin_once(ros, timeout_sec = 0.01)

            full_image, wrist_image, raw_proprio = ros.get_latest()

            # Check data availability (raw_proprio must be checked)
            if full_image is None or wrist_image is None or raw_proprio is None:
                # don't spam logs every loop; only on first few missing frames
                print("Waiting for sensor data...")
                continue

            # --- Transform proprio from IsaacSim -> LIBERO using trained MLP ---
            pos_isaac = np.asarray(raw_proprio["eef_pos"], dtype=np.float32)
            q_isaac = np.asarray(raw_proprio["eef_quat"], dtype=np.float32)  # [x,y,z,w]

            # Prepare input for MLP
            x_input = torch.from_numpy(np.concatenate([pos_isaac, q_isaac])).float().unsqueeze(0)  # shape [1,7]

            with torch.no_grad():
                y_output = mlp(x_input).numpy().squeeze(0)

            pos_libero = y_output[:3]
            q_libero   = y_output[3:]

            proprio = {
                "eef_pos": pos_libero.astype(np.float32),
                "eef_quat": q_libero.astype(np.float32),
                "gr_state": np.asarray(raw_proprio.get("gr_state", np.array([0.0, 0.0])), dtype=np.float32),
            }

            # Prepare observation to input to the model (normalize based on LIBERO stats)
            observation, image = prepare_observation(
                image=full_image, wrist_image=wrist_image, proprio=proprio, resize_size=resize_size
            )

            replay_images.append(image)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

                if flush_queue:
                    actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                    )
                    flush_queue = False
                    
                action_queue.clear()
                action_queue.extend(actions)
            
            # Get action from queue
            action = action_queue.popleft()

            # Process action
            action = process_action(action, cfg.model_family)

            # Clip actions
            norm = np.linalg.norm(action)
            if norm > 0.2:
                action = action * (0.2 / norm)
            
            # Publish action to ROS
            ros.publish_action(action)

            t += 1
            
    except Exception as e:
        print(f"Error: {e}")

    return success, replay_images

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Init ROS
    rclpy.init()
    ros = ROSInterface()

    success, replay_images = run_episode(
            cfg,
            model,
            resize_size,
            task_description,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            ros = ros
    )

    save_rollout_video(replay_images, 1, success = success, task_description = task_description, log_file = None)


if __name__ == "__main__":
    eval_libero()
