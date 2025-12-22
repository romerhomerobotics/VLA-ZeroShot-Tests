"""
OpenVLA-OFT model loader supporting:
 - Base pretrained model
 - Fine-tuned model with LoRA + custom action head + proprio projector
"""

import logging
import os
import glob
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

print("Overwrite proprio dim")
PROPRIO_DIM = 7

from transformers import AutoModelForVision2Seq


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

import types
import torch

def fixed_process_vision_features(self, pixel_values, language_embeddings, use_film):
    """
    Custom method to handle 12-channel (2-image) input by reshaping.
    """
    if pixel_values.shape[1] == 12:
        # Reshape (Batch, 12, H, W) -> (Batch*2, 6, H, W)
        b, c, h, w = pixel_values.shape
        pixel_values_reshaped = pixel_values.view(b * 2, 6, h, w)
        
        # Pass through backbone
        patch_features = self.vision_backbone(pixel_values_reshaped) 
        
        # Reshape back: (Batch*2, Seq, Dim) -> (Batch, Seq*2, Dim)
        patch_features = patch_features.view(b, -1, patch_features.shape[-1])
        return patch_features
    else:
        # Fallback to original logic for single images
        return self.vision_backbone(pixel_values)

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

        # ---------------------------------------------------------
        # 1. Detect MERGED model (full weights)
        # ---------------------------------------------------------
        merged_shards = glob.glob(os.path.join(ckpt_dir, "loar_adapter"))
        #TODO
        is_merged = False # Will put this into the config 
        if is_merged:
            logger.info("[OpenVLA-OFT] Detected MERGED checkpoint (full model weights).")

            # Load full merged VLA model
            self.vla = AutoModelForVision2Seq.from_pretrained(
                ckpt_dir,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            self.vla.eval().cuda()

            # Append deataset statistics the norm_stats
            import json
            with open(os.path.join(ckpt_dir, 'dataset_statistics.json'),'r') as f:
                stats = json.load(f)

            self.vla.norm_stats[list(stats.keys())[0]] = stats[list(stats.keys())[0]]

            # Processor
            self.processor = get_processor(self.cfg)

            # No separate heads needed
            self.action_head = None
            self.proprio_projector = None

            logger.info("[OpenVLA-OFT] Loaded merged model successfully (no extra heads).")
            return

        # ---------------------------------------------------------
        # 2. Otherwise → LoRA finetune directory with separate heads
        # ---------------------------------------------------------
        logger.info("[OpenVLA-OFT] Detected non-merged checkpoint (LoRA + separate heads).")

        # Load base model + LoRA
        self.vla = get_vla(self.cfg)
        self.vla.eval()

        # Processor
        self.processor = get_processor(self.cfg)

        # ------------------------------
        # Load action head
        # ------------------------------
        ah_pattern = os.path.join(ckpt_dir, "action_head--*_checkpoint.pt")
        ah_paths = glob.glob(ah_pattern)
        if not ah_paths:
            raise FileNotFoundError(f"Missing action head checkpoint: {ah_pattern}")

        ah_path = ah_paths[0]
        logger.info(f"[OpenVLA-OFT] Loading Action Head: {os.path.basename(ah_path)}")

        self.action_head = get_action_head(self.cfg, llm_dim=self.vla.llm_dim)
        sd = torch.load(ah_path, map_location="cpu")
        self.action_head.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
        self.action_head.eval()

        # ------------------------------
        # Load proprio projector
        # ------------------------------
        pp_pattern = os.path.join(ckpt_dir, "proprio_projector--*_checkpoint.pt")
        pp_paths = glob.glob(pp_pattern)
        if not pp_paths:
            raise FileNotFoundError(f"Missing proprio projector checkpoint: {pp_pattern}")

        pp_path = pp_paths[0]
        logger.info(f"[OpenVLA-OFT] Loading Proprio Projector: {os.path.basename(pp_path)}")

        self.proprio_projector = get_proprio_projector(
            self.cfg, llm_dim=self.vla.llm_dim, proprio_dim=PROPRIO_DIM
        )
        sd = torch.load(pp_path, map_location="cpu")
        self.proprio_projector.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
        self.proprio_projector.eval()

        logger.info("[OpenVLA-OFT] Loaded LoRA finetuned model + heads successfully.")


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

