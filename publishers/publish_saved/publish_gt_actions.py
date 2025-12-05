import threading
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage
import h5py
import imageio
import os
import time
from libero.libero import benchmark, get_libero_path

class ActionPublisher(Node):
    def __init__(self):
        super().__init__('libero_gt_action_publisher')

        # Publisher for GT actions
        self.pub = self.create_publisher(Float64MultiArray, '/cartesian_delta_command', 10)

        # Subscriber for EEF pose
        self.sub_tf = self.create_subscription(
            TFMessage,
            '/tf',
            self._proprio_cb,
            10
        )

        self._lock = threading.Lock()
        self._latest_proprio = None

    # ------------------------------ CALLBACKS ------------------------------
    def _proprio_cb(self, tf_msg: TFMessage):
        try:
            for t in tf_msg.transforms:
                if t.child_frame_id == "eef":
                    tr = t.transform
                    pos = np.array([tr.translation.x, tr.translation.y, tr.translation.z], dtype=np.float32)
                    quat = np.array([tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w], dtype=np.float32)
                    gripper = np.array([0.0, 0.0], dtype=np.float32)

                    proprio = {"eef_pos": pos, "eef_quat": quat, "gr_state": gripper}
                    with self._lock:
                        self._latest_proprio = proprio
        except Exception as e:
            self.get_logger().error(f"Failed to process proprio msg: {e}")

    # ------------------------------ PUBLIC API ------------------------------
    def publish_action(self, action):
        msg = Float64MultiArray()
        msg.data = action.tolist()
        self.pub.publish(msg)

    def get_latest_proprio(self):
        with self._lock:
            return self._latest_proprio


# ---------------------------------------------------------
# Load LIBERO demo
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Main script
# ---------------------------------------------------------
def main():
    images, gt_ee_pos, gt_ee_ori, actions = load_libero_demo()

    # Save video of the demo
    video_writer = imageio.get_writer("output.mp4", fps=60)
    for image in images:
        video_writer.append_data(image[::-1])
    video_writer.close()

    # Initialize ROS
    rclpy.init()
    node = ActionPublisher()

    rate_hz = 10
    dt = 1.0 / rate_hz

    # Arrays to save EEF data
    recorded_ee_pos = []
    recorded_ee_quat = []
    recorded_gt_pos = []
    recorded_gt_quat = []

    node.get_logger().info("Publishing GT actions and recording EEF poses...")

    for i, (action, gt_pos, gt_quat) in enumerate(zip(actions, gt_ee_pos, gt_ee_ori)):
        rclpy.spin_once(node, timeout_sec=0.01)

        # Get current EEF
        proprio = node.get_latest_proprio()
        if proprio is not None:
            recorded_ee_pos.append(proprio["eef_pos"])
            recorded_ee_quat.append(proprio["eef_quat"])
            recorded_gt_pos.append(gt_pos)
            recorded_gt_quat.append(gt_quat)

            node.get_logger().info(f"Step {i}: GT pos {gt_pos}, EEF pos {proprio['eef_pos']}")

        # Publish GT action
        node.publish_action(action)

        time.sleep(dt)

    # Convert to numpy arrays
    recorded_ee_pos = np.array(recorded_ee_pos)
    recorded_ee_quat = np.array(recorded_ee_quat)
    recorded_gt_pos = np.array(recorded_gt_pos)
    recorded_gt_quat = np.array(recorded_gt_quat)

    # Save for later comparison
    np.savez("eef_comparison.npz",
             recorded_ee_pos=recorded_ee_pos,
             recorded_ee_quat=recorded_ee_quat,
             recorded_gt_pos=recorded_gt_pos,
             recorded_gt_rpy=recorded_gt_quat)

    node.get_logger().info("Saved EEF and GT poses to eef_comparison.npz")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

