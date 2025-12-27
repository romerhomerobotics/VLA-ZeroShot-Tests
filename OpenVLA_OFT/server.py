import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from experiments.robot.libero.run_libero_eval import (
    initialize_model,
    validate_config,
    process_action,
)
from experiments.robot.libero.libero_utils import quat2axisangle
from experiments.robot.openvla_utils import resize_image_for_policy
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    set_seed_everywhere,
)


# ------------------------------ CONFIG ------------------------------


task_description = "Pick up the green cube"


@dataclass
class GenerateConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = "/dl_scratch2/romerhomerobotics/fine_tuned_models/runs_orange60_white_tablev2/2/openvla+orange60_white_tablev2+b4+lr-0.0001+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state--8000_chkpt"
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    lora_rank: int = 32
    unnorm_key: Union[str, Path] = "libero_object_no_noops"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    task_suite_name: str = "libero_object"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256
    TASK_MAX_STEPS: int = 1500
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"
    seed: int = 7


# ------------------------------ MODEL LOADING ------------------------------


logger = logging.getLogger("openvla_server")
logging.basicConfig(level=logging.INFO)


class InferenceContext:
    def __init__(self) -> None:
        self.cfg = GenerateConfig()
        validate_config(self.cfg)
        set_seed_everywhere(self.cfg.seed)

        logger.info("Loading OpenVLA model once at startup...")
        (
            self.model,
            self.action_head,
            self.proprio_projector,
            self.noisy_action_projector,
            self.processor,
        ) = initialize_model(self.cfg)

        self.model = self.model.to("cuda")
        self.resize_size = get_image_resize_size(self.cfg)
        logger.info("Model ready on CUDA")

    def make_observation(
        self,
        full_rgb: np.ndarray,
        wrist_rgb: Optional[np.ndarray],
        state_vec: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        full_resized = resize_image_for_policy(full_rgb, self.resize_size)
        wrist_resized = (
            resize_image_for_policy(wrist_rgb, self.resize_size)
            if wrist_rgb is not None
            else None
        )
        observation = {
            "full_image": full_resized,
            "state": state_vec.astype(np.float32),
        }
        if wrist_resized is not None:
            observation["wrist_image"] = wrist_resized
        return observation

    def run_inference(
        self,
        full_rgb: np.ndarray,
        wrist_rgb: Optional[np.ndarray],
        state_vec: np.ndarray,
    ) -> list:
        observation = self.make_observation(full_rgb, wrist_rgb, state_vec)
        with torch.no_grad():
            actions = get_action(
                self.cfg,
                self.model,
                observation,
                task_description,
                processor=self.processor,
                action_head=self.action_head,
                proprio_projector=self.proprio_projector,
                noisy_action_projector=self.noisy_action_projector,
                use_film=self.cfg.use_film,
            )
        action = np.array(actions[0], dtype=np.float32)
        action = process_action(action, self.cfg.model_family)
        norm = np.linalg.norm(action)
        if norm > 0.2:
            action = action * (0.2 / norm)
        return action.tolist()


CTX = InferenceContext()
app = FastAPI()


# ------------------------------ WEBSOCKET ENDPOINT ------------------------------


@app.websocket("/ws_inference")
async def websocket_inference(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected")
    try:
        while True:
            payload = await websocket.receive_text()
            data = json.loads(payload)

            full_b64 = data.get("full_jpeg_b64")
            wrist_b64 = data.get("wrist_jpeg_b64")
            proprio = data.get("proprio", {})

            if not full_b64:
                await websocket.send_text(json.dumps({"error": "missing_full_image"}))
                continue

            def decode_image(b64_str: str) -> Optional[np.ndarray]:
                try:
                    buf = base64.b64decode(b64_str)
                    arr = np.frombuffer(buf, dtype=np.uint8)
                    bgr_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if bgr_img is None:
                        return None
                    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                except Exception:
                    return None

            full_rgb = decode_image(full_b64)
            wrist_rgb = decode_image(wrist_b64) if wrist_b64 else None

            if full_rgb is None:
                await websocket.send_text(json.dumps({"error": "decode_full_failed"}))
                continue

            pos = np.array(proprio.get("eef_pos", [0, 0, 0]), dtype=np.float32)
            quat = np.array(proprio.get("eef_quat", [0, 0, 0, 1]), dtype=np.float32)
            gr = np.array(proprio.get("gr_state", [0, 0]), dtype=np.float32)
            axis_angle = quat2axisangle(quat)
            state_vec = np.concatenate([pos, axis_angle, gr], dtype=np.float32)

            try:
                action = CTX.run_inference(full_rgb, wrist_rgb, state_vec)
                await websocket.send_text(json.dumps(action))
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Inference failed")
                await websocket.send_text(json.dumps({"error": str(exc)}))
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("WebSocket loop failed: %s", exc)
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
