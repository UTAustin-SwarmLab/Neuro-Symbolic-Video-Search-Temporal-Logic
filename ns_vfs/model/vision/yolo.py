from __future__ import annotations

import warnings

from ultralytics import YOLO
from omegaconf import DictConfig

from ns_vfs.model.vision._base import ComputerVisionDetector

warnings.filterwarnings("ignore")
import numpy as np


class Yolo(ComputerVisionDetector):
    """Yolo"""

    def __init__(
        self,
        config: DictConfig,
        weight_path: str
    ) -> None:
        self.model = self.load_model(weight_path)
        self._config = config

    def load_model(self, weight_path) -> Model:
        """Load weight.

        Args:
            weight_path (str): Path to weight file.

        Returns:
            None
        """
        return YOLO(
            weight_path
        )

    def _parse_class_name(self, class_names: list[str]) -> list[str]:
        """Parse class name.

        Args:
            class_names (list[str]): List of class names.

        Returns:
            list[str]: List of class names.
        """
        return [f"all {class_name}s" for class_name in class_names]

    def detect(self, frame_img: np.ndarray, classes: list) -> any:
        """Detect object in frame.

        Args:
            frame_img (np.ndarray): Frame image.
            classes (list[str]): List of class names.

        Returns:
            any: Detections.
        """
        print("here")
        print(frame_img)
        print(classes)
        detections = self.model.predict(
            source=frame_img,
            classes=classes,
            save=True
        )

        return detections
