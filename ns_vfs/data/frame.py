import dataclasses  # noqa: D100
import random
from typing import List, Optional

import numpy as np
import torchvision.datasets as datasets

@dataclasses.dataclass
class Frame:
    """Frame class."""

    frame_index: int
    frame_image: np.ndarray
    object_detection: dict = dataclasses.field(default_factory=dict)
    propositional_probability: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class FrameWindow:
    """Frame window class."""

    frame_window_idx: int
    frame_image_set: list = dataclasses.field(default_factory=list)
    states: list = dataclasses.field(default_factory=list)
    transitions: list = dataclasses.field(default_factory=list)
    verification_result: bool = False

    def get_propositional_confidence(self):
        """Get propositional confidence."""
        propositional_confidence = [
            [] for i in range(len(self.frame_image_set[0].propositional_probability.keys()))
        ]
        propositional_confidence
        for frame in self.frame_image_set:
            frame: Frame
            idx = 0
            for prop in frame.propositional_probability.keys():
                propositional_confidence[idx].append(frame.propositional_probability[prop])
                idx += 1
        self.propositional_confidence = propositional_confidence
        return propositional_confidence

    def update_states(self, states):
        """Update states."""
        self.states = states
        return states


@dataclasses.dataclass
class BenchmarkRawImage:
    """Benchmark image frame class."""

    unique_labels: list
    labels: List[str]
    images: List[np.ndarray]

    def sample_image_from_label(self, labels: list, proposition: list) -> np.ndarray:
        """Sample image from label."""
        image_of_frame = []
        img_to_label = {}
        for prop in proposition:
            img_to_label[prop] = [i for i, value in enumerate(self.labels) if value == prop]

        label_idx = 0
        for lable in labels:
            if lable is None:
                while True:
                    random_idx = random.randrange(len(self.images))
                    if self.labels[random_idx] not in proposition:
                        break
                labels[label_idx] = self.labels[random_idx]
                image_of_frame.append(self.images[random_idx])
            else:
                random_idx = random.choice(img_to_label[lable])
                image_of_frame.append(self.images[random_idx])

            label_idx += 1
        return labels, image_of_frame


@dataclasses.dataclass
class BenchmarkLTLFrame:
    """Benchmark image frame class.

    ground_truth (bool): Ground truth answer of LTL condition for frames
    ltl_frame (str): LTL formula
    number_of_frame (int): Number of frame
    frames_of_interest (list): List of frames that satisfy LTL condition
    - [[0]] -> Frame 0 satisfy LTL condition;
      [[4,5,6,7]] -> Frame 4 to 7 satisfy LTL condition
      [[0],[4,5,6,7]] -> Frame 0 and Frame 4 to 7 satisfy LTL condition.
    labels_of_frame: list of labels of frame
    """

    ground_truth: bool
    ltl_formula: str
    proposition: list
    number_of_frame: int
    frames_of_interest: Optional[List[List[int]]]
    labels_of_frames: List[str]
    images_of_frames: List[np.ndarray] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        """Post init."""
        self.frames_of_interest = self.group_consecutive_indices(data=self.labels_of_frames)

    def group_consecutive_indices(self, data) -> list:
        """Group consecutive indices.

        Args:
        data (list): List of data.
        """
        groups = []
        group = []

        for i, v in enumerate(data):
            if v is not None:
                if not group or i - 1 == group[-1]:
                    group.append(i)
                else:
                    groups.append(group)
                    group = [i]
        if group:
            groups.append(group)

        return groups

    def save_image(self, path="/opt/Neuro-Symbolic-Video-Frame-Search/artifacts") -> None:
        """Save image to path.

        Args:
        path (str, optional): Path to save image. Defaults to "/opt/Neuro-Symbolic-Video-Frame-Search/artifacts".
        """
        from PIL import Image

        for idx, img in enumerate(self.images_of_frames):
            Image.fromarray(img).save(f"{path}/{idx}.png")

@dataclasses.dataclass
class BenchmarkRawImageDataset:
    """Benchmark image frame class with a torchvision dataset for large datasets"""

    unique_labels: list
    labels: List[str]
    dataset: datasets

    def sample_image_from_label(self, labels: list, proposition: list) -> np.ndarray:
        """Sample image from label."""
        image_of_frame = []
        img_to_label = {}
        for prop in proposition:
            img_to_label[prop] = [i for i, value in enumerate(self.labels) if value == prop]

        label_idx = 0
        for lable in labels:
            if lable is None:
                while True:
                    random_idx = random.randrange(len(self.images))
                    if self.labels[random_idx] not in proposition:
                        break
                labels[label_idx] = self.labels[random_idx]
                image_of_frame.append(self.dataset[random_idx][0])
            else:
                random_idx = random.choice(img_to_label[lable])
                image_of_frame.append(self.dataset[random_idx][0])

            label_idx += 1
        return labels, image_of_frame