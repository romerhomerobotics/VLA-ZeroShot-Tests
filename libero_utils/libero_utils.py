from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from experiments.robot.openvla_utils import (
    resize_image_for_policy
)
from experiments.robot.libero.libero_utils import (
    quat2axisangle
)



def prepare_observation(
    image: np.ndarray,
    wrist_image: Optional[np.ndarray],
    proprio: Dict[str, np.ndarray],
    resize_size: Union[int, Tuple[int, int]],
):
    """Prepare observation taken from IsaacSim in the same way as Libero.

    If `wrist_image` is None, we build a single-image observation.
    """

    image_resized = resize_image_for_policy(image, resize_size)
    observation = {
        "full_image": image_resized,
        "state": np.concatenate(
            (proprio["eef_pos"], quat2axisangle(proprio["eef_quat"]), proprio["gr_state"])
        ),
    }

    if wrist_image is not None:
        wrist_image_resized = resize_image_for_policy(wrist_image, resize_size)
        observation["wrist_image"] = wrist_image_resized

    return observation, image

