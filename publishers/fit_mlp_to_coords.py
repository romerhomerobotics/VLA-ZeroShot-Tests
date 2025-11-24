import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

#
# -----------------------------
#  RPY → QUATERNION CONVERSION
# -----------------------------
#
def rpy_to_quat(rpy):
    """Convert RPY (roll, pitch, yaw in radians) to quaternion [x, y, z, w]."""
    roll, pitch, yaw = rpy

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([x, y, z, w])


#
# -----------------------------
#  LOAD DATA
# -----------------------------
#
data = np.load("eef_comparison.npz")

ee_pos = data["recorded_ee_pos"]       # Nx3 IsaacSim pos
ee_quat = data["recorded_ee_quat"]     # Nx4 IsaacSim quaternion
gt_pos = data["recorded_gt_pos"]       # Nx3 LIBERO pos

# LIBERO EE ORIENTATION **IN RPY**
gt_ori_rpy = data["recorded_gt_rpy"]   # Nx3 RPY

# Convert LIBERO RPY → quaternion
gt_quat = np.array([rpy_to_quat(r) for r in gt_ori_rpy])

print(f"Isaac EE pos shape: {ee_pos.shape}")
print(f"Isaac EE ori shape: {ee_quat.shape}")
print(f"Libero EE pos shape: {gt_pos.shape}")
print(f"Libero EE pos shape: {gt_ori_rpy.shape}")


#
# -----------------------------
#  PREPARE DATASET
# -----------------------------
#
X = np.hstack([ee_pos, ee_quat])
Y = np.hstack([gt_pos, gt_quat])

# Convert to torch tensors
X_torch = torch.from_numpy(X).float()
Y_torch = torch.from_numpy(Y).float()

dataset = TensorDataset(X_torch, Y_torch)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

class PoseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        )

    def forward(self, x):
        out = self.net(x)
        # Normalize quaternion part (last 4 entries)
        pos = out[:, :3]
        quat = out[:, 3:]
        quat = quat / torch.norm(quat, dim=1, keepdim=True)
        return torch.cat([pos, quat], dim=1)

def loss_fn(pred, target):
    pos_loss = nn.functional.mse_loss(pred[:, :3], target[:, :3])
    # Quaternion loss: angle distance
    q_pred = pred[:, 3:]
    q_target = target[:, 3:]
    dot = torch.abs(torch.sum(q_pred * q_target, dim=1))
    quat_loss = torch.mean(1 - dot)
    return pos_loss + quat_loss

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PoseMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 500

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        y_pred = model(xb)
        loss = loss_fn(y_pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    total_loss /= len(dataset)
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.6f}")

torch.save(model.state_dict(), "pose_mlp.pt")

