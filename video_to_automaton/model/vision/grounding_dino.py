from __future__ import annotations

import warnings

from _base import ComputerVisionDetector
from groundingdino.util import inference
from omegaconf import DictConfig

warnings.filterwarnings("ignore")


class GroundingDino(ComputerVisionDetector):
    """Grounding Dino."""

    def __init__(
        self, weight_path: str, config_path: str, config: DictConfig
    ) -> None:
        self.model = self.load_model(weight_path, config_path)
        self._config = config

    def load_model(self, weight_path, config_path) -> inference.Model:
        """Load weight.

        Args:
            weight_path (str): Path to weight file.
            config_path (str): Path to config file.


        Returns:
            None
        """
        return inference.Model(
            model_config_path=config_path, model_checkpoint_path=weight_path
        )

    def _parse_class_name(class_names: list[str]) -> list[str]:
        """Parse class name.

        Args:
            class_names (list[str]): List of class names.

        Returns:
            list[str]: List of class names.
        """
        return [f"all {class_name}s" for class_name in class_names]

    def detect(self, frame_img, classes) -> any:
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
