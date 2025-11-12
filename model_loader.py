"""
Utility for loading OpenVLA-OFT model components and running inference once.
"""

import logging
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
)
from prismatic.vla.constants import PROPRIO_DIM

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class OpenVLAOFTModel:
    """
    Wrapper class that holds all model components and exposes a simple API:
        model.predict_actions(observation)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        logger.info("Loading OpenVLA-OFT components...")

        self.vla = get_vla(cfg)
        self.processor = get_processor(cfg)
        self.action_head = get_action_head(cfg, llm_dim=self.vla.llm_dim)
        self.proprio_projector = get_proprio_projector(cfg, llm_dim=self.vla.llm_dim, proprio_dim=PROPRIO_DIM)

        logger.info("OpenVLA-OFT model successfully initialized.")

    def predict_actions(self, observation, task_description=None):
        """
        Run inference to generate a chunk of future actions.
        """
        if task_description is None:
            task_description = observation.get("task_description", "")

        logger.debug(f"Running inference for task: {task_description}")
        actions = get_vla_action(
            self.cfg,
            self.vla,
            self.processor,
            observation,
            task_description,
            self.action_head,
            self.proprio_projector,
        )
        return actions


def load_model(cfg):
    """
    Convenience loader that returns a ready-to-use OpenVLAOFTModel instance.
    """
    return OpenVLAOFTModel(cfg)

