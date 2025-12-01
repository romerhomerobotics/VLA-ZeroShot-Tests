import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
from std_msgs.msg import Float64MultiArray

import cv2
import numpy as np

class ROSInterface(Node):
    def __init__(self):
        super().__init__("openvla_oft_interface")

        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest_full_rgb = None
        self._latest_wrist_rgb = None
        self._latest_proprio = None
        self._proprio_count = 0

        # Subscribers
        self.sub_full = self.create_subscription(
            Image, "/tripod_2/color/image_raw", self._full_cam_cb, 10 # "/tripod_2/rgb" for libero setup 
        )
        self.sub_wrist = self.create_subscription(
            Image, "/wrist/rgb", self._wrist_cam_cb, 10 # /wrist/color/image_raw for libero setup
        )
        self.sub_tf = self.create_subscription(
            TFMessage, "/tf", self._proprio_cb, 10
        )

        # Publisher: Cartesian delta or joint vel
        self.pub_action = self.create_publisher(
            Float64MultiArray,
            "/cartesian_delta_command",
            10
        )

    # ------------------------------ CALLBACKS ------------------------------

    def _full_cam_cb(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._latest_full_rgb = rgb
        except Exception as e:
            self.get_logger().error(f"Failed to process full camera image: {e}")

    def _wrist_cam_cb(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._latest_wrist_rgb = rgb
        except Exception as e:
            self.get_logger().error(f"Failed to process wrist image: {e}")

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

    # ------------------------------ API ------------------------------

    def get_latest(self):
        with self._lock:
            return (
                self._latest_full_rgb,
                self._latest_wrist_rgb,
                self._latest_proprio,
            )

    def publish_action(self, action7):
        msg = Float64MultiArray()
        msg.data = action7.tolist()
        self.pub_action.publish(msg)

