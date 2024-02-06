from __future__ import annotations

import warnings

import numpy as np
from ultralytics import YOLO

from ns_vfs.model.vision.object_detection._base import (
    ComputerVisionObjectDetector,
)

warnings.filterwarnings("ignore")


class DummyVisionModel(ComputerVisionObjectDetector):
    """Yolo."""

    def __init__(self) -> None:
        self.name = "dummy"

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

    def detect(self, frame_img: np.ndarray, classes: list) -> any:
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

        return None

    def get_confidence_score(self, frame_img: np.ndarray, true_lable: str) -> any:
        pass
