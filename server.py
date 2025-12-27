"""
WebSocket inference server for OpenVLA-OFT.
Loads the model once at startup and serves actions over FastAPI.
"""

import asyncio
import io
import json
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image
import uvicorn

from vla_utils.ft_config import get_config
from vla_utils.ft_model_loader import load_model
from libero_utils.libero_utils import prepare_observation
from experiments.robot.robot_utils import set_seed_everywhere, get_image_resize_size

app = FastAPI()

cfg = get_config()
set_seed_everywhere(cfg.seed)
resize_size = get_image_resize_size(cfg)

model = load_model(cfg)


def decode_image(jpeg_bytes: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(jpeg_bytes)).convert("RGB"))


def build_proprio(meta: Dict[str, Any]) -> Dict[str, np.ndarray]:
    proprio = meta.get("proprio") or {}
    eef_pos = np.asarray(proprio.get("eef_pos", []), dtype=np.float32)
    eef_quat = np.asarray(proprio.get("eef_quat", []), dtype=np.float32)
    gr_state = np.asarray(proprio.get("gr_state", []), dtype=np.float32)
    return {"eef_pos": eef_pos, "eef_quat": eef_quat, "gr_state": gr_state}


def run_inference(observation: Dict[str, Any], task_description: str) -> List[List[float]]:
    actions = model.predict_actions(observation, task_description)
    return [np.asarray(a).tolist() for a in actions]


@app.websocket("/ws_inference")
async def ws_inference(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            meta_text = await websocket.receive_text()
            meta = json.loads(meta_text)

            image_bytes = await websocket.receive_bytes()
            wrist_bytes: Optional[bytes] = None
            if meta.get("has_wrist"):
                wrist_bytes = await websocket.receive_bytes()

            full_image = decode_image(image_bytes)
            wrist_image = decode_image(wrist_bytes) if wrist_bytes else None
            proprio = build_proprio(meta)
            task_description = meta.get("task_description") or "Pick up the mustard"

            observation, _ = prepare_observation(
                image=full_image,
                wrist_image=wrist_image,
                proprio=proprio,
                resize_size=resize_size,
            )

            loop = asyncio.get_running_loop()
            actions = await loop.run_in_executor(
                None, run_inference, observation, task_description
            )

            await websocket.send_json({"actions": actions})
        except WebSocketDisconnect:
            break
        except Exception as exc:
            await websocket.send_json({"error": str(exc)})
            continue


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
