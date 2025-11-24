"""
Utility for loading OpenVLA-OFT model components and running inference once.
"""

import logging
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
)
from experiments.robot.robot_utils import get_action
from prismatic.vla.constants import PROPRIO_DIM

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class OpenVLAOFTModel:
    def __init__(self, cfg):
        self.cfg = cfg

        self.vla = get_vla(cfg)
        self.processor = get_processor(cfg)

        # Needed for OFT continuous actions
        self.action_head = get_action_head(cfg, llm_dim=self.vla.llm_dim)
        self.proprio_projector = get_proprio_projector(cfg, llm_dim=self.vla.llm_dim, proprio_dim=PROPRIO_DIM)

    def predict_actions(self, observation, task_description=None):
        if task_description is None:
            task_description = observation.get("task_description", "")

        return get_action(
            cfg=self.cfg,
            model=self.vla,
            obs=observation,
            task_label=task_description,
            processor=self.processor,
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
        )


def load_model(cfg):
    """
    Convenience loader that returns a ready-to-use OpenVLAOFTModel instance.
    """
    return OpenVLAOFTModel(cfg)

