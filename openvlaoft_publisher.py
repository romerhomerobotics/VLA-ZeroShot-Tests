#!/usr/bin/env python3
"""
openvla_oft_publisher.py

ROS2 publisher that collects main camera image, wrist camera image and proprio state,
calls the OpenVLA-OFT model to get an action chunk, and publishes the first action
as delta6 + grip in a Float64MultiArray message.

Usage:
    python3 openvla_oft_publisher.py --config ./configs/my_gen_cfg.json --task "pick up the bowl"
"""

import argparse
import time
import threading
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge

# local modules (assumes you created config.py & model_loader.py as discussed)
from config import get_config        # returns GenerateConfig
from model_loader import load_model  # returns OpenVLAOFTModel
import matplotlib.pyplot as plt

# Optional external ensembler (your original project location). If missing we continue.
try:
    from simpler_env.utils.action.action_ensemble import ActionEnsembler
except Exception:
    ActionEnsembler = None


def build_arg_parser():
    p = argparse.ArgumentParser("openvla_oft_publisher")
    p.add_argument("--config", required=False, default=None,
                   help="Path to JSON used to construct GenerateConfig (optional). If omitted, get_config() is used.")
    p.add_argument("--task", required=False, default="pick up the object",
                   help="Task description string passed to the model.")
    p.add_argument("--cam-topic", default="/tripod_2/rgb")
    p.add_argument("--wrist-topic", default="/wrist/color/image_raw")
    p.add_argument("--proprio-topic", default="/tf")
    p.add_argument("--vel-topic", default="/cartesian_delta_command")
    p.add_argument("--control-freq", type=float, default=3.0)
    p.add_argument("--image-size", type=int, default=224, help="Resize both images to this square size before inference")
    p.add_argument("--ensemble", action="store_true", default=False, help="Enable ActionEnsembler if available")
    p.add_argument("--ensemble-horizon", type=int, default=4)
    p.add_argument("--ensemble-temp", type=float, default=-0.8)
    return p


class OpenVLAOFTPublisher(Node):
    def __init__(self, cfg, model, args):
        super().__init__("openvla_oft_publisher")
        self.cfg = cfg
        self.model = model
        self.args = args

        # CvBridge & buffers
        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest_full_rgb: Optional[np.ndarray] = None
        self._latest_wrist_rgb: Optional[np.ndarray] = None
        self._latest_proprio: Optional[np.ndarray] = None
        self._proprio_count: int = 0
        self._act_count: int = 0

        # Publishers / Subscribers
        self.pub_vel = self.create_publisher(Float64MultiArray, args.vel_topic, 10)
        self.create_subscription(Image, args.cam_topic, self._full_cam_cb, 10)
        self.create_subscription(Image, args.wrist_topic, self._wrist_cam_cb, 10)
        # Expect proprio as Float64MultiArray
        from std_msgs.msg import Float64MultiArray as _F64MA
        self.create_subscription(TFMessage, args.proprio_topic, self._proprio_cb, 10)

        # Ensembler
        self._ensembler = None
        if args.ensemble:
            if ActionEnsembler is None:
                self.get_logger().warning("ActionEnsembler requested but not importable. Continuing without ensemble.")
            else:
                self._ensembler = ActionEnsembler(args.ensemble_horizon, args.ensemble_temp)
                self.get_logger().info(f"ActionEnsembler enabled: horizon={args.ensemble_horizon} temp={args.ensemble_temp}")

        # Timer for control loop
        self.control_period = 1.0 / max(args.control_freq, 1e-6)
        self._timer = self.create_timer(self.control_period, self._on_tick)
        self._img_size = args.image_size
        self._task_description = args.task

        self.get_logger().info(f"OpenVLA-OFT publisher initialized. Publishing to: {args.vel_topic}")

    # ----------------- callbacks -----------------
    def _full_cam_cb(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            cv2.imwrite("full_cam_debug.jpg", bgr)
            with self._lock:
                self._latest_full_rgb = rgb 
        except Exception as e:
            self.get_logger().error(f"Failed to process full camera image: {e}")

    def _wrist_cam_cb(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            cv2.imwrite("wrist_cam_debug.jpg", bgr)
            with self._lock:
                self._latest_wrist_rgb = rgb

        except Exception as e:
            self.get_logger().error(f"Failed to process wrist camera image: {e}")

    def _proprio_cb_jointstate(self, msg: Float64MultiArray):
        try:
            arr = np.asarray(msg.data, dtype=np.float32)
            arr8 = np.zeros((8,), dtype = np.float32)
            arr8[:-1] = arr[:-2]
            arr8[-1] = 0 # fixed gripper state for now

            with self._lock:
                arr = np.concatenate((arr, np.array([0.0])), axis = -1)
                self._latest_proprio = arr
        except Exception as e:
            self.get_logger().error(f"Failed to process proprio message: {e}")

    def _proprio_cb(self, tf_msg: TFMessage):
        try:
            # pick the transform you care about (eef)
            for t in tf_msg.transforms:
                if t.child_frame_id == "eef":
                    t = t.transform  # get the actual Transform
                    eef_pos = np.array([t.translation.x, t.translation.y, t.translation.z], dtype=np.float32)
                    quat = np.array([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w], dtype=np.float32)
                    from experiments.robot.libero.libero_utils import quat2axisangle
                    eef_axis_angle = quat2axisangle(quat)
                    gripper_state = np.array([0.0,0.0], dtype=np.float32)
                    proprio = np.concatenate((eef_pos, eef_axis_angle, gripper_state), axis=-1)

                    with self._lock:
                        self._latest_proprio = proprio

                    if self._proprio_count % 50 == 0:
                        self.get_logger().info(f"Proprio: [{proprio}]")

                    self._proprio_count += 1
        except Exception as e:
            self.get_logger().error(f"Failed to process proprio message: {e}")


    # --------------- main control tick ----------------
    def _on_tick(self):
        # Build a local snapshot of inputs
        with self._lock:
            full = None if self._latest_full_rgb is None else self._latest_full_rgb.copy()
            wrist = None if self._latest_wrist_rgb is None else self._latest_wrist_rgb.copy()
            proprio = None if self._latest_proprio is None else self._latest_proprio.copy()

        if full is None or wrist is None:
            # not ready yet
            self.get_logger().debug("Waiting for both full and wrist images...")
            return

        # Preprocess images: center / resize -> MODEL expects 224x224 (or configured)
        try:
            full_in = cv2.resize(full, (self._img_size, self._img_size), interpolation=cv2.INTER_AREA)
            wrist_in = cv2.resize(wrist, (self._img_size, self._img_size), interpolation=cv2.INTER_AREA)
        except Exception as e:
            self.get_logger().error(f"Image resize error: {e}")
            return

        # Proprio: if missing, fill with zeros of reasonable length (model expects some state length).
        if proprio is None:
            # Choose a fallback length. If model.cfg provides expected proprio size, try to use it.
            fallback_len = getattr(self.model.cfg, "proprio_dim", None)
            if fallback_len is None:
                # conservative default (your example used 8)
                fallback_len = 8
            proprio = np.zeros((fallback_len,), dtype=np.float32)

        # Build observation dict expected by your model wrapper
        observation = {
            "full_image": full_in,
            "wrist_image": wrist_in,
            "state": proprio,
            "task_description": self._task_description,
        }

        # Call model to get action chunk
        try:
            actions = self.model.predict_actions(observation, self._task_description)
        except Exception as e:
            self.get_logger().error(f"Model prediction failed: {e}")
            return

        # actions can be torch tensor or numpy. Normalize to numpy.
        if hasattr(actions, "detach"):
            try:
                actions_np = actions.detach().cpu().numpy()
            except Exception:
                actions_np = np.asarray(actions)
        else:
            actions_np = np.asarray(actions)

        # If returned shape is (batch, horizon, 7) or (horizon, 7)
        # normalize to shape: (horizon, 7)
        if actions_np.ndim == 3:
            actions_np = actions_np[0]
        if actions_np.ndim == 2 and actions_np.shape[-1] >= 7:
            # OK: (horizon, >=7)
            pass
        elif actions_np.ndim == 1 and actions_np.shape[0] >= 7:
            actions_np = actions_np[None, :]
        else:
            self.get_logger().error(f"Unexpected action shape returned by model: {actions_np.shape}")
            return

        # Optional: ensemble across horizon/time using your ActionEnsembler
        if self._ensembler is not None:
            try:
                # ActionEnsembler expects shape (horizon, 7) or similar; adapt if necessary.
                ensembled = self._ensembler.ensemble_action(actions_np)
                actions_np = np.asarray(ensembled)
            except Exception as e:
                self.get_logger().warning(f"ActionEnsembler failed, continuing with raw actions: {e}")

        # Extract first action7:
        # allowable shapes: (horizon, 7) or (1,7)
        first7 = None
        if actions_np.ndim == 2:
            first7 = actions_np[0, :7]
        elif actions_np.ndim == 1:
            first7 = actions_np[:7]
        else:
            self.get_logger().error(f"Cannot extract first action from shape {actions_np.shape}")
            return

        # Convert to delta6 (dx,dy,dz, rot_axis*angle) and grip_rel
        dx, dy, dz = map(float, first7[0:3])
        droll, dpitch, dyaw = map(float, first7[3:6])
        # Convert euler (r,p,y) to axis-angle-like representation roughly as euler2axangle does in your previous script.
        # To avoid extra dependency, we'll convert small angles approximatively: rot_ax = [droll, dpitch, dyaw] * scale
        # If you prefer exact conversion, swap in transforms3d.euler.euler2axangle as you had.
        rot_ax = np.array([droll, dpitch, dyaw], dtype=np.float64)

        # scale if user wants (you can add command-line params to scale)
        action_scale = 1.0
        delta6 = np.concatenate([np.array([dx, dy, dz], dtype=np.float64),
                                 rot_ax * action_scale], axis=-1)

        # grip: model likely outputs open probability in last dim (0..1). Map to rel pulse or +/-1
        open_abs = float(first7[6])
        # Simple mapping: open_abs < 0.5 -> close (1.0), else open (-1.0) — matches your prior script
        grip_rel = 1.0 if open_abs < 0.5 else -1.0

        action7 = np.concatenate([delta6, np.array([grip_rel], dtype=np.float64)], axis=-1)

        # Publish
        msg = Float64MultiArray()
        msg.data = action7.tolist()
        self.pub_vel.publish(msg)
        if self._act_count % 5 == 0:
            self.get_logger().info(f"Published Δ6={np.round(delta6,4).tolist()} grip={grip_rel:+.1f}")
        self._act_count += 1

    # ------------- utilities (optional) --------------
    # You can add helper methods here (abs->rel sticky gripper family, intrinsics overwrite, etc.)


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Initialize ROS
    rclpy.init()

    # Build/generate config and load model
    if args.config:
        # optional: allow user to pass a path to JSON that can be converted to GenerateConfig
        # but in your prior flow you used config.get_config() to construct a GenerateConfig
        from experiments.robot.libero.run_libero_eval import GenerateConfig
        import json, os
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config JSON not found: {args.config}")
        with open(args.config, "r") as f:
            cfg_json = json.load(f)
        # If your GenerateConfig can accept **cfg_json then call it; otherwise prefer get_config()
        cfg = GenerateConfig(**cfg_json)
    else:
        # use the local helper that returns a GenerateConfig
        cfg = get_config()

    # load model (this does the heavy lifting)
    model = load_model(cfg)

    # instantiate node
    node = OpenVLAOFTPublisher(cfg, model, args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()

