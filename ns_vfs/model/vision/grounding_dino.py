from __future__ import annotations

import warnings

from groundingdino.util.inference import Model
from omegaconf import DictConfig

from ns_vfs.model.vision._base import ComputerVisionDetector

warnings.filterwarnings("ignore")
import numpy as np


class GroundingDino(ComputerVisionDetector):
    """Grounding Dino."""

    def __init__(
        self,
        config: DictConfig,
        weight_path: str,
        config_path: str,
    ) -> None:
        self.model = self.load_model(weight_path, config_path)
        self._config = config

    def load_model(self, weight_path, config_path) -> Model:
        """Load weight.

        Args:
            weight_path (str): Path to weight file.
            config_path (str): Path to config file.


        Returns:
            None
        """
        return Model(
            model_config_path=config_path, model_checkpoint_path=weight_path
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
        detections = self.model.predict_with_classes(
            image=frame_img,
            classes=self._parse_class_name(class_names=classes),
            box_threshold=self._config.BOX_TRESHOLD,
            text_threshold=self._config.TEXT_TRESHOLD,
        )

        return detections