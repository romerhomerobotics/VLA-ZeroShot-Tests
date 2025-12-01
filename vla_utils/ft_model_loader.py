"""
OpenVLA-OFT model loader supporting:
 - Base pretrained model
 - Fine-tuned model with LoRA + custom action head + proprio projector
"""

import logging
import os
import torch
from peft import PeftModel

from experiments.robot.openvla_utils import (
    get_vla,
    get_processor,
    get_action_head,
    get_proprio_projector,
    get_vla_ft,
)
from experiments.robot.robot_utils import get_action
from prismatic.vla.constants import PROPRIO_DIM

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class OpenVLAOFTModel:
    """
    Unified loader for both the base and fine-tuned OpenVLA-OFT models.
    """

    def __init__(self, cfg):
        self.cfg = cfg

        if not hasattr(cfg, "model_variant"):
            raise ValueError("cfg.model_variant must be 'base' or 'finetuned'.")

        if cfg.model_variant == "base":
            ckpt_dir = cfg.pretrained_checkpoint_base
            self._load_base_model(ckpt_dir)

        elif cfg.model_variant == "finetuned":
            ckpt_dir = cfg.pretrained_checkpoint_finetuned
            self._load_finetuned_model(ckpt_dir)

        else:
            raise ValueError(f"Unknown cfg.model_variant: {cfg.model_variant}")

    # ----------------------------------------------------------------------
    # LOAD BASE MODEL
    # ----------------------------------------------------------------------
    def _load_base_model(self, ckpt_dir):
        logger.info(f"[OpenVLA-OFT] Loading BASE model from: {ckpt_dir}")

        # Load HF transformer + standard weights
        self.vla = get_vla(self.cfg, override_dir=ckpt_dir)
        self.processor = get_processor(self.cfg)

        # Base action head + proprio projector
        self.action_head = get_action_head(self.cfg, llm_dim=self.vla.llm_dim)
        self.proprio_projector = get_proprio_projector(
            self.cfg, llm_dim=self.vla.llm_dim, proprio_dim=PROPRIO_DIM
        )

        self.vla.eval()
        self.action_head.eval()
        self.proprio_projector.eval()

        logger.info("[OpenVLA-OFT] Base model loaded successfully.")

    # ----------------------------------------------------------------------
    # LOAD FINE-TUNED MODEL
    # ----------------------------------------------------------------------
    def _load_finetuned_model(self, ckpt_dir):
        logger.info(f"[OpenVLA-OFT] Loading FINETUNED model from: {ckpt_dir}")

        # Step 1: Load base architecture WITHOUT weights
        base_vla = get_vla_ft(self.cfg)

        # Step 2: Apply LoRA adapter
        lora_path = os.path.join(ckpt_dir, "lora_adapter")
        if not os.path.isdir(lora_path):
            raise FileNotFoundError(f"Missing LoRA adapter: {lora_path}")

        self.vla = PeftModel.from_pretrained(
            base_vla,
            lora_path,
            torch_dtype=torch.float16,
            is_trainable=False,
        )
        self.vla.eval()

        # Step 3: Load processor/tokenizer from fine-tuned directory
        self.processor = get_processor(self.cfg,)

        # Step 4: Load trained action head
        ah_path = os.path.join(ckpt_dir, "action_head--150000_checkpoint.pt")
        if not os.path.isfile(ah_path):
            raise FileNotFoundError(f"Missing action head checkpoint: {ah_path}")

        self.action_head = get_action_head(self.cfg, llm_dim=self.vla.llm_dim)
        raw_sd = torch.load(ah_path, map_location="cpu")
        clean_sd = {k.replace("module.", ""): v for k, v in raw_sd.items()}
        self.action_head.load_state_dict(clean_sd) 
        self.action_head.eval()

        # Step 5: Load proprio projector
        pp_path = os.path.join(ckpt_dir, "proprio_projector--150000_checkpoint.pt")
        if not os.path.isfile(pp_path):
            raise FileNotFoundError(f"Missing proprio projector: {pp_path}")

        self.proprio_projector = get_proprio_projector(
            self.cfg, llm_dim=self.vla.llm_dim, proprio_dim=PROPRIO_DIM
        )
        raw_sd = torch.load(pp_path, map_location="cpu")
        clean_sd = {k.replace("module.", ""): v for k, v in raw_sd.items()}
        self.proprio_projector.load_state_dict(clean_sd)
        self.proprio_projector.eval()

        logger.info("[OpenVLA-OFT] Finetuned model loaded successfully.")

    # ----------------------------------------------------------------------
    # RUN INFERENCE
    # ----------------------------------------------------------------------
    @torch.no_grad()
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


# ----------------------------------------------------------------------
# Convenience loader
# ----------------------------------------------------------------------
def load_model(cfg):
    return OpenVLAOFTModel(cfg)

