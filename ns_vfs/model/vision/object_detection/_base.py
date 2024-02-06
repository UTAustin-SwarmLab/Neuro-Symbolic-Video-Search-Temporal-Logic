import abc
from pathlib import Path
from typing import Union

import numpy as np

from ns_vfs.model.vision._base import ComputerVisionModel


class ComputerVisionObjectDetector(ComputerVisionModel):
    """Computer Vision Detector."""

    def __init__(self, weight_path: Union[Path, str], gpu_number: int = 0) -> None:
        """Computer Vision Detector.

        Args:
            weight_path (str): Path to weight file.
        """
        self._weight_path = weight_path
        self.load_model(weight_path)

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
