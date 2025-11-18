import numpy as np
import logging
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import pathlib

logging.basicConfig(level=logging.INFO)

def get_camera_info_from_sim(env):
    """Retrieves camera names, positions, and orientations from the MuJoCo sim."""
    sim = env.sim
    
    # Get all camera names defined in the model
    camera_names = sim.model.camera_names
    
    print("\n--- Runtime Camera Poses (MuJoCo) ---")
    
    # Check if the number of cameras in the model matches the data arrays
    if len(camera_names) != sim.data.cam_xpos.shape[0]:
        print("Warning: Camera list length does not match data array size.")
        return

    for i, name in enumerate(camera_names):
        # Camera Position (x, y, z)
        position = sim.data.cam_xpos[i]
        
        # Camera Orientation (3x3 rotation matrix, reshape from 9 elements)
        rotation_matrix = sim.data.cam_xmat[i].reshape(3, 3)
        
        print(f"\nCamera {i} Name: **{name}**")
        print(f"  Position (XYZ): {position.round(4)}")
        print(f"  Rotation Matrix:\n{rotation_matrix.round(4)}")

def create_and_extract_libero_info():
    """
    Initializes a LIBERO environment from the 'libero_10' suite and extracts 
    the observation space, action space, and task description.
    """
    # 1. Configuration
    TASK_SUITE_NAME = "libero_spatial" 
    TASK_ID = 0  # We will use the first task in the suite
    RESOLUTION = 256
    SEED = 42

    # 2. Initialize LIBERO Task Suite
    logging.info(f"Initializing task suite: {TASK_SUITE_NAME}")
    try:
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[TASK_SUITE_NAME]()
    except KeyError:
        logging.error(f"Task suite '{TASK_SUITE_NAME}' not found.")
        return

    # 3. Get Specific Task
    task = task_suite.get_task(TASK_ID)
    task_description = task.language
    
    # 4. Initialize Environment
    logging.info(f"Setting up environment for Task {TASK_ID}: '{task_description}'")
    
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    
    env_args = {
        "bddl_file_name": task_bddl_file, 
        "camera_heights": RESOLUTION, 
        "camera_widths": RESOLUTION,
    }
    
    # Create the OffScreenRenderEnv
    env = OffScreenRenderEnv(**env_args)
    env.seed(SEED)
    
    # 5. Extract Information
    # Robot
    robot = env.robots[0]

    print("="*60)
    print(f"**Robot Model:** {robot.name}")
    print(f"**Control Frequency (Hz):** {robot.control_freq}")
    print("="*60)

    # --- Base and Joint Information ---
    print("### ⚙️ Kinematic Setup")
    print(f"**Joint Names (7-DoF Arm):** {robot.robot_joints}")
    print(f"**Base Position (XYZ):** {robot.base_pos.round(4)}")
    print(f"**Base Orientation (Quat):** {robot.base_ori.round(4)}")
    print(f"**Initial Joint Position (Rad):**\n{robot.init_qpos.round(4)}")

    # --- Controller Information ---
    config = robot.controller_config
    print("\n### 🕹️ Controller Configuration (OSC_POSE)")
    print(f"**Controller Type:** {config['type']}")
    print(f"**Policy Frequency (Hz):** {config['policy_freq']}")
    print(f"**Impedance Mode:** {config['impedance_mode']}")
    print(f"**Control Delta (Action is relative):** {config['control_delta']}")
    print(f"**Uncouple Position/Orientation:** {config['uncouple_pos_ori']}")

    # --- Action Space Limits ---
    print("\n#### Action Limits (Controller Output/Delta)")
    # Output Max/Min define the maximum change per control step
    output_max = np.array(config['output_max'])
    output_min = np.array(config['output_min'])
    output_delta = (output_max - output_min) / 2

    # Assuming the order is [dx, dy, dz, droll, dpitch, dyaw]
    print(f"**Max Output Delta (m/rad):** {output_max.round(4)}")
    print(f"  - Position delta (XYZ): ±{output_delta[:3].round(4)}")
    print(f"  - Orientation delta (Roll/Pitch/Yaw): ±{output_delta[3:].round(4)}")

    # --- Feedback Gains ---
    print("\n#### Feedback Gains")
    print(f"**Proportional Gain (Kp):** {config['kp']}")
    print(f"**Damping Ratio:** {config['damping_ratio']}")

    print("="*60)
    controller = robot.controller

    # Cameras
    get_camera_info_from_sim(env)

    

if __name__ == "__main__":
    create_and_extract_libero_info()
