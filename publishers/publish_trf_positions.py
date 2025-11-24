import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import torch
import numpy as np

# -----------------------------
# Load MLP model
# -----------------------------
class PoseMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(7, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 7)
        )

    def forward(self, x):
        out = self.net(x)
        pos = out[:, :3]
        quat = out[:, 3:]
        quat = quat / torch.norm(quat, dim=1, keepdim=True)
        return torch.cat([pos, quat], dim=1)

# Load your trained weights here if available
mlp = PoseMLP()
mlp.load_state_dict(torch.load("pose_mlp.pt", map_location="cpu"))
mlp.eval()

# -----------------------------
# Load IsaacSim EEF poses
# -----------------------------
data = np.load("eef_comparison.npz")
ee_pos = data["recorded_ee_pos"]       # Nx3 IsaacSim pos
ee_quat = data["recorded_ee_quat"]     # Nx4 IsaacSim quaternion

X = np.hstack([ee_pos, ee_quat])
X_torch = torch.from_numpy(X).float()

# -----------------------------
# Transform poses with MLP
# -----------------------------
with torch.no_grad():
    transformed = mlp(X_torch).numpy()
    transformed_pos = transformed[:, :3]
    transformed_quat = transformed[:, 3:]

# -----------------------------
# ROS2 Node to publish poses
# -----------------------------
class PosePublisher(Node):
    def __init__(self, poses, quats, rate_hz=10):
        super().__init__("pose_publisher")
        self.pub = self.create_publisher(Float64MultiArray, "/cartesian_pose_command", 10)
        self.poses = poses
        self.quats = quats
        self.rate_hz = rate_hz
        self.index = 0
        self.timer = self.create_timer(1.0 / rate_hz, self.publish_next_pose)
        self.get_logger().info(f"Publishing {len(poses)} transformed poses at {rate_hz}Hz")

    def publish_next_pose(self):
        if self.index >= len(self.poses):
            self.get_logger().info("Finished publishing all poses.")
            return

        msg = Float64MultiArray()
        # Concatenate pos + quat
        msg.data = np.hstack([self.poses[self.index], self.quats[self.index]]).tolist()
        self.pub.publish(msg)

        self.index += 1

def main():
    rclpy.init()
    node = PosePublisher(transformed_pos, transformed_quat, rate_hz=10)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

