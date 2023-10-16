from __future__ import annotations

import warnings

import numpy as np
from groundingdino.util.inference import Model
from omegaconf import DictConfig

from ns_vfs.model.vision._base import ComputerVisionDetector

warnings.filterwarnings("ignore")


class GroundingDino(ComputerVisionDetector):
    """Grounding Dino."""

    def __init__(
        self,
        config: DictConfig,
        weight_path: str,
        config_path: str,
        device: str = "cuda:0",
    ) -> None:
        self.model = self.load_model(weight_path, config_path, device)
        self._config = config

    def load_model(self, weight_path, config_path, device) -> Model:
        """Load weight.

        Args:
            weight_path (str): Path to weight file.
            config_path (str): Path to config file.


        Returns:
            None
        """
        return Model(model_config_path=config_path, model_checkpoint_path=weight_path, device=device)

    def _parse_class_name(self, class_names: list[str]) -> list[str]:
        """Parse class name.

        Args:
            class_names (list[str]): List of class names.

        Returns:
            list[str]: List of class names.
        """
        result = []
        for class_name in class_names:
            c = class_name.replace("_", " ")
            result.append(f"all {c}s")
        return result

    def detect(self, frame_img: np.ndarray, classes: list) -> any:
        """Detect object in frame.

        Args:
            frame_img (np.ndarray): Frame image.
            classes (list[str]): List of class names.

        Returns:
            any: Detections.
        """
        detected_obj = self.model.predict_with_classes(
            image=frame_img,
            classes=self._parse_class_name(class_names=classes),
            box_threshold=self._config.BOX_TRESHOLD,
            text_threshold=self._config.TEXT_TRESHOLD,
        )

        self._labels = [
            f"{classes[class_id] if class_id is not None else None} {confidence:0.2f}"
            for _, _, confidence, class_id, _ in detected_obj
        ]

        self._detection = detected_obj

        self._confidence = detected_obj.confidence

        self._size = len(detected_obj)

        return detected_obj

    def get_confidence_score(self, frame_img: np.ndarray, true_label: str) -> any:
        max_conf = 0
        detected_obj = self.model.predict_with_classes(
            image=frame_img,
            classes=self._parse_class_name(class_names=[true_label]),
            box_threshold=self._config.BOX_TRESHOLD,
            text_threshold=self._config.TEXT_TRESHOLD,
        )
        all_detected_object_list = detected_obj.class_id
        all_detected_object_confidence = detected_obj.confidence

        for i in range(len(all_detected_object_list)):
            if all_detected_object_list[i] == 0:  # we pass only one class lable -> class id = 0
                if all_detected_object_confidence[i] > max_conf:
                    max_conf = all_detected_object_confidence[i]
        return max_conf
