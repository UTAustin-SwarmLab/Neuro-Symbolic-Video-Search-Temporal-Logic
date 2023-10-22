from __future__ import annotations

from mmdet.apis import DetInferencer
from omegaconf import DictConfig
import supervision as sv
import numpy as np
import warnings

from ns_vfs.model.vision._base import ComputerVisionDetector

warnings.filterwarnings("ignore")


class MMDetection(ComputerVisionDetector):
    """MMDetection"""

    def __init__(
        self,
        config: DictConfig,
        config_path: str,
        weight_path: str
    ) -> None:
        self.model = self.load_model(weight_path, config_path)
        self._config = config

    def load_model(self, weight_path, config_path) -> DetInferencer:
        """Load weight.

        Args:
            weight_path (str): Path to weight file.

        Returns:
            None
        """
        init_args = {'model': config_path, 'weights': weight_path, 'device': 'cpu', 'palette': 'none'}

        return DetInferencer(**init_args)

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

        return None

