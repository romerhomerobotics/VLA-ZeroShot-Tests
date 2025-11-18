import collections
import dataclasses
import logging
import math
import pathlib
import sys

import imageio.v3 as iio # Use v3 for modern imageio API
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
import tqdm
import tyro

# To run openvla-oft

# The necessary imports from openpi_client are REMOVED as they are not needed for dummy actions
# Also removed image_tools since we don't need to resize/pad for dummy actions (we'll save the raw obs)

# --- Constants ---
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0] # Dummy action for stability and simulation
LIBERO_ENV_RESOLUTION = 256 # resolution used to render training data

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 1  # Reduced for dummy runs (only 1 episode per task)

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/dummy_videos"  # Path to save videos
    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero_dummy_actions(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)


    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # Define max steps based on task suite (copied from original)
    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    
    # Iterate over tasks (one task is usually enough for a dummy run)
    for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc="Task Suite"):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=f"Trials for {task_description[:30]}"):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            done = False # Initialize done flag

            logging.info(f"Starting episode {task_episodes+1} with dummy actions...")
            while t < max_steps + args.num_steps_wait and not done:
                try:
                    # 💡 CORE CHANGE: Use the dummy action for ALL steps (including post-wait)
                    action = LIBERO_DUMMY_ACTION
                    
                    # Get raw image from observation for video (no need for resizing/padding for a dummy run video)
                    # IMPORTANT: rotate 180 degrees to match train preprocessing for visual consistency
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    replay_images.append(img)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action)
                    t += 1
                    
                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            # --- End of Episode ---
            task_episodes += 1
            total_episodes += 1

            if done:
                task_successes += 1
                total_successes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            
            # Using imageio.v3.imwrite which is recommended
            iio.imwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}_dummy_actions.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10, # Slower FPS for clarity
                quality=8,
                codec='libx264'
            )

            # Log current results
            logging.info(f"Episode done. Success: {done}. Total steps: {t}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log task results
        task_rate = float(task_successes) / float(task_episodes)
        total_rate = float(total_successes) / float(total_episodes)
        logging.info(f"Current task success rate: {task_rate}")
        logging.info(f"Current total success rate: {total_rate}")
        
    logging.info(f"--- FINAL RESULTS ---")
    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    # Use a single camera resolution for both width and height
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """Helper function copied from original script. Not strictly needed here but kept for completeness."""
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    # Ensure necessary modules are imported and the main function is called
    if not any(arg.startswith("--num_trials_per_task") for arg in sys.argv):
        logging.warning("Setting --num_trials_per_task=1 for faster dummy run. Override via command line if needed.")
    
    logging.basicConfig(level=logging.INFO)
    
    # Use the new function name and define a custom logger
    logger = logging.getLogger(__name__)
    
    # Tyro entry point
    tyro.cli(eval_libero_dummy_actions)
