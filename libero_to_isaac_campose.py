import numpy as np
from scipy.spatial.transform import Rotation as R

def libero_to_isaac_pos(p):
    x, y, z = p
    return np.array([-y, x, z])

def libero_to_isaac_rot(R_lib):
    C = np.array([[0, 1, 0],
                  [-1, 0, 0],
                  [0, 0, 1]])
    return C @ R_lib @ C.T

# Input from LIBERO
cam_pos_lib = np.array([1.0, 0.0, 1.48])
R_lib = np.array([
    [-0.0, -0.2582, 0.9661],
    [ 1.0,  0.0,     0.0   ],
    [ 0.0,  0.9661,  0.2582]
])

# Convert
pos_isaac = libero_to_isaac_pos(cam_pos_lib)
R_isaac = libero_to_isaac_rot(R_lib)

# Quaternion (xyzw)
q_isaac = R.from_matrix(R_isaac).as_quat()

print("IsaacSim Position:", pos_isaac)
print("IsaacSim Quaternion (xyzw):", q_isaac)

