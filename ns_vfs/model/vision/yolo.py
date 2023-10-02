from __future__ import annotations


import warnings

import numpy as np
import supervision as sv
from omegaconf import DictConfig
from ultralytics import YOLO

from ns_vfs.model.vision._base import ComputerVisionDetector

warnings.filterwarnings("ignore")


class Yolo(ComputerVisionDetector):

    """Yolo."""

    def __init__(self, config: DictConfig, weight_path: str) -> None:
        self.model = self.load_model(weight_path)
        self._config = config

    def load_model(self, weight_path) -> YOLO:

        """Load weight.

        Args:
            weight_path (str): Path to weight file.

        Returns:
            None
        """
        return YOLO(weight_path)

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
        classes_reversed = {v: k for k, v in self.model.names.items()}
        class_ids = [classes_reversed[c] for c in classes]
        detected_obj = self.model.predict(source=frame_img, classes=class_ids)


        self._labels = []
        for i in range(len(detected_obj[0].boxes)):
            class_id = int(detected_obj[0].boxes.cls[i])
            confidence = float(detected_obj[0].boxes.conf[i])
            self._labels.append(
                f"{detected_obj[0].names[class_id] if class_id is not None else None} {confidence:0.2f}"
            )

        self._detection = sv.Detections(xyxy=detected_obj[0].boxes.xyxy.cpu().detach().numpy())
        
        self._confidence = detected_obj[0].boxes.conf.cpu().detach().numpy()

        self._size = len(detected_obj[0].boxes)

        return detected_obj
