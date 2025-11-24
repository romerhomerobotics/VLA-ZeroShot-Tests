import time
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import h5py
import os
from libero.libero import benchmark, get_libero_path
from scipy.spatial.transform import Rotation as R


# ------------------------------ CONFIG ------------------------------
USE_TRANSFORMED_LIBERO = True  # False -> publish Isaac poses, True -> publish Libero poses transformed to Isaac

# Umeyama transform from Libero -> Isaac (replace with your actual computed values)
R_umeyama = np.eye(3, dtype=np.float64)  # rotation
t_umeyama = np.zeros(3, dtype=np.float64)  # translation

# ------------------------------ Node ------------------------------
class PosePublisher(Node):
    def __init__(self, topic_name='/cartesian_pose_command'):
        super().__init__('eef_pose_publisher')
        self.pub = self.create_publisher(Float64MultiArray, topic_name, 10)

    def publish_pose(self, pose):
        """
        pose: np.array([x, y, z, qx, qy, qz, qw])
        """
        msg = Float64MultiArray()
        msg.data = pose.tolist()
        self.pub.publish(msg)

def load_libero_demo():
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict["libero_10"]()
    num_tasks = benchmark_instance.get_num_tasks()

    datasets_default_path = "/home/romer-vla-sim/datasets"
    demo_files = [
        os.path.join(datasets_default_path, benchmark_instance.get_task_demonstration(i))
        for i in range(num_tasks)
    ]

    example_demo_file = demo_files[0]
    print("Loading demo:", example_demo_file)

    with h5py.File(example_demo_file, "r") as f:
        images = f["data/demo_1/obs/agentview_rgb"][()]
        grp = f["data/demo_1"]
        ee_pos = grp["obs/ee_pos"][()]
        ee_ori = grp["obs/ee_ori"][()]
        actions = grp['actions'][()]

    return images, ee_pos, ee_ori, actions

# ------------------------------ Load Isaac-saved poses ------------------------------
def load_isaac_poses(npz_file):
    data = np.load(npz_file)
    ee_pos = data["recorded_ee_pos"]
    ee_quat = data["recorded_ee_quat"]
    return ee_pos, ee_quat

# ------------------------------ Quaternion utilities ------------------------------
def rpy_to_quat(rpy):
    """Convert RPY [rad] to quaternion [x, y, z, w]"""
    return R.from_euler("xyz", rpy).as_quat()

def apply_umeyama_transform(pos, quat):
    """Apply Umeyama transform R,t to position and rotate quaternion"""
    pos_trans = R_umeyama.T @ (pos - t_umeyama)

    # Rotate quaternion
    rot_mat = R.from_quat(quat).as_matrix()
    rot_mat_trans = R_umeyama.T @ rot_mat
    quat_trans = R.from_matrix(rot_mat_trans).as_quat()
    return pos_trans, quat_trans

# ------------------------------ Main ------------------------------
def main():
    # Load poses
    if USE_TRANSFORMED_LIBERO:
        images, ee_pos, ee_ori, actions = load_libero_demo()
    else:
        ee_pos, ee_ori = load_isaac_poses("eef_comparison.npz")

    # Initialize ROS
    rclpy.init()
    node = PosePublisher(topic_name='/cartesian_pose_command')

    rate_hz = 10
    dt = 1.0 / rate_hz

    # Publish poses
    for i, (pos, ori) in enumerate(zip(ee_pos, ee_ori)):
        # Convert RPY -> quaternion if necessary
        if ori.shape[0] == 3:
            quat = rpy_to_quat(ori)
        else:
            quat = ori

        # Apply Umeyama transform if using Libero -> Isaac
        if USE_TRANSFORMED_LIBERO:
            pos, quat = apply_umeyama_transform(pos, quat)

        pose_msg = np.concatenate([pos, quat])
        node.publish_pose(pose_msg)

        node.get_logger().info(f"Published step {i}: pos {pos}, quat {quat}")
        time.sleep(dt)

    node.get_logger().info("Finished publishing all poses.")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
