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
    states: list = dataclasses.field(default_factory=list)
    transitions: list = dataclasses.field(default_factory=list)

    def get_propositional_confidence(self):
        propositional_confidence = [
            []
            for i in range(
                len(self.frame_image_set[0].propositional_probability.keys())
            )
        ]
        propositional_confidence
        for frame in self.frame_image_set:
            frame: Frame
            idx = 0
            for prop in frame.propositional_probability.keys():
                propositional_confidence[idx].append(
                    frame.propositional_probability[prop]
                )
                idx += 1
        self.propositional_confidence = propositional_confidence
        return propositional_confidence

    def update_states(self, states):
        self.states = states
        return states
