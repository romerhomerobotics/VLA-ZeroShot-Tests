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
import json

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
        logger.info(f"[OpenVLA-OFT] Loading model from: {ckpt_dir}")

        # 1. Determine if we are loading a merged or un-merged model
        # Usually, merged models contain a 'config.json', 
        # while LoRA folders contain 'adapter_config.json'kkk   
        is_lora = False

        if not is_lora:
            # --- PATH A: LOAD MERGED MODEL ---
            logger.info("[OpenVLA-OFT] Loading MERGED checkpoint (Full Weights).")
            
            self.vla = get_vla(self.cfg)
            # AutoModelForVision2Seq.from_pretrained(
                # ckpt_dir,
                # config = self.cfg,
                # torch_dtype=torch.bfloat16,
                # low_cpu_mem_usage=True,
                # local_files_only = True,
                # trust_remote_code=True
            # )
        else:
            # --- PATH B: LOAD BASE + LORA SEPARATELY ---
            logger.info("[OpenVLA-OFT] Loading BASE model + LoRA adapter.")
            # Load the base model first (using path from your config)
            base_model = get_vla(self.cfg) 

            lora_dir = os.path.join(ckpt_dir, "lora_adapter")

            # Wrap with PeftModel to load LoRA weights
            self.vla = PeftModel.from_pretrained(base_model, lora_dir)
            # Optional: Merge them in memory now for faster inference
            # self.vla = self.vla.merge_and_unload()

        self.vla.eval().cuda()
        self.processor = get_processor(self.cfg)

        # ---------------------------------------------------------
        # 2. Load External Adaptors (Proprio & Action Head)
        # ---------------------------------------------------------
        # NOTE: Even if the VLA is merged, you likely still need these 
        # if they were trained as separate MLP modules.

        self.action_head = self._load_extra_module(
            ckpt_dir, "action_head", 
            get_action_head(self.cfg, llm_dim=self.vla.llm_dim)
        )

        self.proprio_projector = self._load_extra_module(
            ckpt_dir, "proprio_projector", 
            get_proprio_projector(self.cfg, llm_dim=self.vla.llm_dim, proprio_dim=PROPRIO_DIM)
        )

        # 3. Handle Dataset Statistics (Crucial for OpenVLA inference)
        stats_path = os.path.join(ckpt_dir, 'dataset_statistics.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            # Ensure stats are loaded into the model for normalization
            if self.cfg.unnorm_key not in list(stats.keys()): 
                key = list(stats.keys())[0]
                print(f"Unnorm key {self.cfg.unnorm_key} not found")
                print(f"Using the default {key}")
            else:
                key = self.cfg.unnorm_key
            self.vla.norm_stats[key] = stats[key]
            logger.info(f"[OpenVLA-OFT] Loaded normalization stats for: {key}")



    def _load_extra_module(self, ckpt_dir, prefix, module_instance):
        """Helper to find and load .pt checkpoints for MLP heads"""
        pattern = os.path.join(ckpt_dir, f"{prefix}--*_checkpoint.pt")
        paths = glob.glob(pattern)
        if not paths:
            logger.warning(f"No checkpoint found for {prefix}, skipping...")
            return None

        path = paths[0]
        logger.info(f"[OpenVLA-OFT] Loading {prefix} from {os.path.basename(path)}")
        sd = torch.load(path, map_location="cpu")
        # Clean 'module.' prefix if saved with DataParallel
        cleaned_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        module_instance.load_state_dict(cleaned_sd)
        return module_instance.eval().cuda()


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

