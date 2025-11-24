import numpy as np

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
data = np.load("publishers/eef_comparison.npz")

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
#  POSITION COMPARISON
# -----------------------------
#
print("\n=== POSITION DIFFERENCE ===")
delta = ee_pos - gt_pos

print("Mean delta per axis (X, Y, Z):", np.mean(delta, axis=0))
print("Std  delta per axis (X, Y, Z):", np.std(delta, axis=0))

delta_norm = np.linalg.norm(delta, axis=1)
print("Mean position error norm:", np.mean(delta_norm))
print("Max  position error norm:", np.max(delta_norm))


#
# -----------------------------
#  ORIENTATION COMPARISON (Quat)
# -----------------------------
#
def quat_angle_error(q1, q2):
    """Returns angle (rad) between two quaternions."""
    # Normalize
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
    return 2 * np.arccos(dot)

orientation_errors = np.array([quat_angle_error(ee_quat[i], gt_quat[i]) 
                               for i in range(len(gt_quat))])

print("\n=== ORIENTATION DIFFERENCE ===")
print("Mean orientation error (deg):", np.mean(orientation_errors) * 180/np.pi)
print("Max  orientation error (deg):", np.max(orientation_errors) * 180/np.pi)


#
# -----------------------------
#  UMEYAMA ALIGNMENT (POSITION ONLY)
# -----------------------------
#
def umeyama_alignment(src, dst):
    """
    Computes R, t such that:   dst ≈ R @ src + t
    """
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Fix improper rotation
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = mu_dst - R @ mu_src
    return R, t


R, t = umeyama_alignment(gt_pos, ee_pos)

print("\n=== UMEYAMA POSE ALIGNMENT ===")
print("Estimated rotation matrix:\n", R)
print("Estimated translation vector:", t)
print()

