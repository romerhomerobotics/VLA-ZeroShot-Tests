#!/usr/bin/env python3

import time
import argparse
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float64
from tf2_msgs.msg import TFMessage

from scipy.spatial.transform import Rotation as R


# ============================================================
# ROS Node
# ============================================================
class DeltaToPosePublisher(Node):
    def __init__(self, actions):
        super().__init__("delta_to_pose_publisher")

        self.actions = actions
        self._lock = threading.Lock()
        self._latest_proprio = None

        # ------------------ Publishers ------------------
        self.pose_pub = self.create_publisher(
            Float64MultiArray,
            "/cartesian_pose_command",
            10,
        )

        self.gripper_pub = self.create_publisher(
            Float64,
            "/gripper_command",
            10,
        )

        # ------------------ TF Subscriber ------------------
        self.sub_tf = self.create_subscription(
            TFMessage,
            "/tf",
            self._proprio_cb,
            10,
        )

    # --------------------------------------------------
    def _proprio_cb(self, tf_msg: TFMessage):
        try:
            for t in tf_msg.transforms:
                if t.child_frame_id == "eef":
                    tr = t.transform
                    pos = np.array(
                        [tr.translation.x, tr.translation.y, tr.translation.z],
                        dtype=np.float32,
                    )
                    quat = np.array(
                        [
                            tr.rotation.x,
                            tr.rotation.y,
                            tr.rotation.z,
                            tr.rotation.w,
                        ],
                        dtype=np.float32,
                    )

                    with self._lock:
                        self._latest_proprio = {
                            "eef_pos": pos,
                            "eef_quat": quat,
                        }
        except Exception as e:
            self.get_logger().error(f"TF parse error: {e}")

    # --------------------------------------------------
    def get_latest_proprio(self):
        with self._lock:
            return self._latest_proprio

    # --------------------------------------------------
    def publish_pose(self, pos, quat, gripper):
        msg = Float64MultiArray()
        msg.data = np.concatenate([pos, quat]).tolist()
        self.pose_pub.publish(msg)

        g = Float64()
        g.data = float(gripper) * 100
        self.gripper_pub.publish(g)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--rate", type=float, default=5.0)
    args = parser.parse_args()

    data = np.load(args.npz)
    actions = data["actions"]  # (T, 7)

    rclpy.init()
    node = DeltaToPosePublisher(actions)

    dt = 1.0 / args.rate

    node.get_logger().info("Waiting for initial TF...")
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.get_latest_proprio() is not None:
            break

    node.get_logger().info("Starting delta → pose replay")

    # ====================================================
    for t, action in enumerate(actions):
        rclpy.spin_once(node, timeout_sec=0.01)

        proprio = node.get_latest_proprio()
        if proprio is None:
            continue

        # ------------------------------------------------
        # Current pose
        pos = proprio["eef_pos"]
        quat = proprio["eef_quat"]

        # ------------------------------------------------
        # Delta from model
        dpos = action[:3] *  9.0
        drpy = action[3:6] * 9.0
        gripper = action[6]

        # ------------------------------------------------
        # Position update
        new_pos = pos + dpos

        # ------------------------------------------------
        # Orientation update: quat ⊗ delta(rpy)
        R_curr = R.from_quat(quat)
        R_delta = R.from_euler("xyz", drpy)
        R_new = R_delta * R_curr

        new_quat = R_new.as_quat()

        # ------------------------------------------------
        node.publish_pose(new_pos, new_quat, gripper)

        node.get_logger().info(
            f"[{t}] Δpos {dpos}, Δrpy {drpy}"
        )

        time.sleep(dt)

    node.get_logger().info("Replay finished")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
