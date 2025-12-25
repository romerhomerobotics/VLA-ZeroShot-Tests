import asyncio
import base64
import json

import cv2
import numpy as np
import rclpy
import websockets

from ros_utils.ros_utils import ROSInterface

WS_URL = "ws://localhost:8000/ws_inference"
JPEG_QUALITY = 85


async def send_receive_loop(ros: ROSInterface) -> None:
    while rclpy.ok():
        try:
            async with websockets.connect(WS_URL, max_size=None) as websocket:
                while rclpy.ok():
                    rclpy.spin_once(ros, timeout_sec=0.01)
                    full_image, wrist_image, proprio = ros.get_latest()
                    if full_image is None or proprio is None:
                        await asyncio.sleep(0.01)
                        continue

                    bgr = cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR)
                    ok, buf = cv2.imencode(
                        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                    )
                    if not ok:
                        continue
                    payload = {
                        "full_jpeg_b64": base64.b64encode(buf).decode("ascii"),
                        "proprio": {
                            "eef_pos": proprio["eef_pos"].tolist(),
                            "eef_quat": proprio["eef_quat"].tolist(),
                            "gr_state": proprio.get("gr_state", np.zeros(2, dtype=np.float32)).tolist(),
                        },
                    }

                    if wrist_image is not None:
                        wrist_bgr = cv2.cvtColor(wrist_image, cv2.COLOR_RGB2BGR)
                        ok_w, buf_w = cv2.imencode(
                            ".jpg",
                            wrist_bgr,
                            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
                        )
                        if ok_w:
                            payload["wrist_jpeg_b64"] = base64.b64encode(buf_w).decode("ascii")

                    await websocket.send(json.dumps(payload))

                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue

                    action = json.loads(message)
                    if isinstance(action, dict) and "error" in action:
                        continue

                    action_np = np.asarray(action, dtype=np.float32)
                    ros.publish_action(action_np)
        except (websockets.ConnectionClosedError, ConnectionRefusedError):
            await asyncio.sleep(1.0)
        except Exception:
            await asyncio.sleep(1.0)


async def main() -> None:
    rclpy.init()
    ros = ROSInterface()
    try:
        await send_receive_loop(ros)
    finally:
        ros.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
