import os
import pickle
from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM
import matplotlib.pyplot as plt


cfg = GenerateConfig(
    pretrained_checkpoint = "/home/romer-vla-sim/model_weights/vla/openvla_oft",
    use_l1_regression = True,
    use_diffusion = False,
    use_film = False,
    num_images_in_input = 2,
    use_proprio = True,
    load_in_8bit = False,
    load_in_4bit = False,
    center_crop = True,
    num_open_loop_steps = NUM_ACTIONS_CHUNK,
    unnorm_key = "libero_object_no_noops",
)

# Load OpenVLA-OFT policy and inputs processor
vla = get_vla(cfg)
processor = get_processor(cfg)

# Load MLP action head to generate continuous actions (via L1 regression)
action_head = get_action_head(cfg, llm_dim=vla.llm_dim)

# Load proprio projector to map proprio to language embedding space
proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

# Load sample observation:
#   observation (dict): {
#     "full_image": primary third-person image,
#     "wrist_image": wrist-mounted camera image,
#     "state": robot proprioceptive state,
#     "task_description": task description,
#   }
with open("/home/romer-vla-sim/Workspace/fine_tuning_vlas/openvla-oft/experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as file:
    observation = pickle.load(file)

for k,v in observation.items():
    if hasattr(v, "shape"):
        print(k, v.shape)
    else: 
        print(k,v)
    if k == 'state':
        print(v)
    if k =='full_image':
        plt.imsave('img.png', v)
    if k =='wrist_image':
        plt.imsave('img1.png', v)
"""
full_image (224, 224, 3)
wrist_image (224, 224, 3)
state (8,)
task_description pick up the black bowl between the plate and the ramekin and place it on the plate
"""

# Generate robot action chunk (sequence of future actions)
actions = get_vla_action(cfg, vla, processor, observation, observation["task_description"], action_head, proprio_projector)
print("Generated action chunk:")
for act in actions:
    print(act)

"""
Generated action chunk:
[0.318 0.035 -0.050 0.000 0.006 -0.021 1.000]
[0.434 0.023 -0.092 -0.001 0.001 -0.021 0.984]
[0.511 -0.019 -0.091 -0.001 0.001 -0.026 0.965]
[0.562 -0.064 -0.103 -0.001 0.001 -0.027 0.961]
[0.592 -0.160 -0.125 0.000 0.000 -0.028 0.973]
[0.608 -0.202 -0.162 0.000 0.001 -0.030 0.992]
[0.568 -0.180 -0.212 0.001 0.001 -0.033 0.992]
[0.516 -0.123 -0.273 0.004 0.001 -0.033 0.977]
"""
