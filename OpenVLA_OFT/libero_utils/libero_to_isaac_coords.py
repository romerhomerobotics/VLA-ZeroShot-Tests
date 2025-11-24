import numpy as np

# IsaacSim
R_isaac = np.array([
    [-0.9998478, 0.0164766, 0.0057347],
    [-0.0140666, -0.9558074, 0.2936569],
    [0.0103197, 0.2935315, 0.9558937]
])
t_isaac = np.array([0.83196, -0.00336, 0.52816])

T_isaac = np.eye(4)
T_isaac[:3,:3] = R_isaac
T_isaac[:3,3] = t_isaac

# Libero
R_libero = np.array([
    [0, -0.2582, 0.9661],
    [1, 0, 0],
    [0, 0.9661, 0.2582]
])
t_libero = np.array([1.66, 0, 0.568])

T_libero = np.eye(4)
T_libero[:3,:3] = R_libero
T_libero[:3,3] = t_libero

# Transform Isaac -> Libero
T_isaac_to_libero = T_libero @ np.linalg.inv(T_isaac)

T_libero_to_isaac = np.linalg.inv(T_isaac_to_libero)

if __name__=="__main__":
    print("Isaac to Libero transformation")
    print(T_isaac_to_libero)

    print("Libero to Isaac transformation")
    print(T_libero_to_isaac)
