import dataclasses

import numpy as np


@dataclasses.dataclass
class Frame:
    frame_index: int
    frame_image: np.ndarray
    propositional_probability: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class FrameWindow:
    frame_window_idx: int
    frame_image_set: list = dataclasses.field(default_factory=list)
