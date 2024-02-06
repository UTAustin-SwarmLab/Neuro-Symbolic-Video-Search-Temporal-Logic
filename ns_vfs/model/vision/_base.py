import abc
from pathlib import Path
from typing import Union

import numpy as np
import supervision as sv


class ComputerVisionModel(abc.ABC):
    @abc.abstractmethod
    def load_model(self) -> any:
        """Load weight."""


class ComputerVisionDetector(ComputerVisionModel):
    """Computer Vision Detector."""

    def __init__(self, weight_path: Union[Path, str], gpu_number: int = 0) -> None:
        """Computer Vision Detector.

        Args:
            weight_path (str): Path to weight file.
        """
        self._weight_path = weight_path
        self.load_model(weight_path)

    def load_model(self, weight_path):
        """Load weight."""
        self._weight = weight_path

    def get_weight(self):
        """Get weight."""
        return self._weight

    def get_labels(self) -> list:
        """Return sv.Detections"""
        return self._labels

    def get_detections(self) -> sv.Detections:
        """Return sv.Detections"""
        return self._detection

    def get_confidence(self) -> np.ndarray:
        return self._confidence

    def get_size(self) -> int:
        return self._size

    def _sigmoid_mapping_estimation_function(self, x, a=1, k=1, x0=0) -> float:
        """Sigmoid function.

        Args:
            x (float): Input.
            k (int, optional): Steepness of the function. Defaults to 1.
            x0 (int, optional): Midpoint of the function. Defaults to 0.

        Returns:
            float: Sigmoid function.
        """
        return 1 / (1 + np.exp(-k * (x - x0)))

    @abc.abstractmethod
    def detect(self, frame) -> any:
        """Detect object in frame."""
        pass
