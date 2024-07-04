from __future__ import annotations

import random
import warnings

import numpy as np
from ultralytics import YOLO

from ns_vfs.data.detected_object import DetectedObject
from ns_vfs.model.vision.object_detection._base import (
    ComputerVisionObjectDetector,
)

warnings.filterwarnings("ignore")


class DummyVisionModel(ComputerVisionObjectDetector):
    """Dummy Vision Model."""

    def __init__(
        self,
        detection_probability: float,
        random_greater_than: float | None = None,
        random_prob_less_than: float | None = None,
    ) -> None:
        self.name = "dummy"
        self._detection_probability = detection_probability
        if random_greater_than:
            self._detection_probability = random.uniform(random_greater_than, 1)
        elif random_prob_less_than:
            self._detection_probability = random.uniform(
                0, random_prob_less_than
            )

    def load_model(self, weight_path) -> YOLO:
        """Load weight.

        Args:
            weight_path (str): Path to weight file.

        Returns:
            None
        """
        pass

    def _parse_class_name(self, class_names: list[str]) -> list[str]:
        """Parse class name.

        Args:
            class_names (list[str]): List of class names.

        Returns:
            list[str]: List of class names.
        """
        pass

    def detect(
        self, frame_img: np.ndarray, classes: list, ground_truth: bool = True
    ) -> any:
        """Detect object in frame.

        Args:
            frame_img (np.ndarray): Frame image.
            classes (list[str]): List of class names.

        Returns:
            any: Detections.
        """
        self._labels = []

        self._detection = None

        self._confidence = 0

        self._size = 0

        if ground_truth:
            return DetectedObject(
                name=classes[0],
                model_name=self.name,
                confidence_of_all_obj=[self._detection_probability - 0.1],
                probability_of_all_obj=[self._detection_probability],
                all_obj_detected=None,
                number_of_detection=1,
                is_detected=True,
                supervision_detections=None,
            )
        else:
            return DetectedObject(
                name=classes[0],
                model_name=self.name,
                confidence_of_all_obj=[0],
                probability_of_all_obj=[0],
                all_obj_detected=None,
                number_of_detection=0,
                is_detected=True,
                supervision_detections=None,
            )

    def get_confidence_score(
        self, frame_img: np.ndarray, true_lable: str
    ) -> any:
        pass
