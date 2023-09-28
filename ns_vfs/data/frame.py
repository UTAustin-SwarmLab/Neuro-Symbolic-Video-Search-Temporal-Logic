import dataclasses  # noqa: D100
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ns_vfs.common.frame_grouping import combine_consecutive_lists


@dataclasses.dataclass
class Frame:
    """Frame class."""

    frame_index: int
    frame_image: np.ndarray
    annotated_image: Dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    real_frame_idx: Optional[int] = None
    object_detection: dict = dataclasses.field(default_factory=dict)
    propositional_probability: dict = dataclasses.field(default_factory=dict)

    @property
    def propositional_confidence(self):
        """Get propositional confidence."""
        return list(self.propositional_probability.values())


@dataclasses.dataclass
class FramesofInterest:
    """Frame class."""

    ltl_formula: str
    foi_list: List[List[int]] = dataclasses.field(default_factory=list)
    frame_images: List[np.ndarray] = dataclasses.field(default_factory=list)
    annotated_images: List[np.ndarray] = dataclasses.field(default_factory=list)
    frame_idx_to_real_idx: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if "!" in self.ltl_formula:
            self._reverse_search = True
        else:
            self._reverse_search = False

    def save_annotated_images(self, annotated_image: Dict[str, np.ndarray]):
        for a_img in list(annotated_image.values()):
            self.annotated_images.append(a_img)

    def reorder_frame_of_interest(self):
        if self._reverse_search:
            flattened_list = [item for sublist in self.foi_list for item in sublist]
            self.foi_list = [x for x in range(len(self.frame_images)) if x not in flattened_list]
        self.foi_list = combine_consecutive_lists(self.foi_list)

    def save_frames(self, path):
        from PIL import Image

        root_path = Path(path)
        frame_path = root_path / "frame"
        annotation_path = root_path / "annotation"

        frame_path.mkdir(parents=True, exist_ok=True)
        annotation_path.mkdir(parents=True, exist_ok=True)

        for idx, img in enumerate(self.frame_images):
            Image.fromarray(img).save(f"{frame_path}/{idx}.png")
            if self.annotated_image_images[idx] is not None:
                Image.fromarray(self.annotated_image_images[idx]).save(
                    f"{annotation_path}/{idx}_annotated.png"
                )


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
        self.frames_of_interest = combine_consecutive_lists(data=self.frames_of_interest)

    def save_frames(self, path="/opt/Neuro-Symbolic-Video-Frame-Search/artifacts") -> None:
        """Save image to path.

        Args:
        path (str, optional): Path to save image.
        """
        from PIL import Image

        for idx, img in enumerate(self.images_of_frames):
            Image.fromarray(img).save(f"{path}/{idx}.png")

    def save(self, save_path: str = "/opt/Neuro-Symbolic-Video-Frame-Search/artifacts") -> None:
        """Save the current instance to a pickle file."""
        import pickle

        """Save the current instance to a pickle file."""
        with open(save_path, "wb") as f:
            pickle.dump(self, f)
