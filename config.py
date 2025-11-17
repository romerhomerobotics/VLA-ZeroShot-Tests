# config.py
"""
Creates and returns a ready-to-use GenerateConfig object for OpenVLA-OFT.
Edit the parameters here to match your experiment or deployment setup.
"""

from experiments.robot.libero.run_libero_eval import GenerateConfig
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


def get_config() -> GenerateConfig:
    """
    Construct and return a GenerateConfig for OpenVLA-OFT.
    """
    cfg = GenerateConfig(
        pretrained_checkpoint="/home/romer-vla-sim/model_weights/vla/openvla_oft",
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=8,
        unnorm_key="libero_object_no_noops",
    )
    return cfg

