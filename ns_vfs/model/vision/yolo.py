from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import supervision as sv
import torch
from omegaconf import DictConfig
from ultralytics import YOLO

from ns_vfs.model.vision._base import ComputerVisionDetector

warnings.filterwarnings("ignore")


class Yolo(ComputerVisionDetector):
    """Yolo."""

    def __init__(self, config: DictConfig, weight_path: Path) -> None:
        """Init."""
        if isinstance(weight_path, str):
            weight_path = Path(weight_path)
        self.model = self.load_model(weight_path)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self._config = config
        self._classes_reversed = {v: k for k, v in self.model.names.items()}

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
        class_ids = [
            self._classes_reversed[c.replace("_", " ")] for c in classes
        ]
        detected_obj = self.model.predict(source=frame_img, classes=class_ids)

        self._labels = []
        for i in range(len(detected_obj[0].boxes)):
            class_id = int(detected_obj[0].boxes.cls[i])
            confidence = float(detected_obj[0].boxes.conf[i])
            self._labels.append(
                f"{detected_obj[0].names[class_id] if class_id is not None else None} {confidence:0.2f}"
            )

        self._detection = sv.Detections(
            xyxy=detected_obj[0].boxes.xyxy.cpu().detach().numpy()
        )

        self._confidence = detected_obj[0].boxes.conf.cpu().detach().numpy()

        self._size = len(detected_obj[0].boxes)

        return detected_obj

    def _mapping_probability(
        self,
        confidence_per_video: float,
        true_threshold=0.60,
        false_threshold=0.40,
        a=0.971,
        k=7.024,
        x0=0.117,
    ) -> float:
        """Mapping probability.

        Args:
            confidence_per_video (float): Confidence per video.
            true_threshold (float, optional): True threshold. Defaults to 0.64.
            false_threshold (float, optional): False threshold. Defaults to 0.38.

        Returns:
            float: Mapped probability.
        """
        if confidence_per_video >= true_threshold:
            return 1
        elif confidence_per_video < false_threshold:
            return 0
        else:
            return round(
                self._sigmoid_mapping_estimation_function(
                    confidence_per_video,
                    a=a,
                    k=k,
                    x0=x0,
                ),
                2,
            )

    def get_confidence_score(
        self, frame_img: np.ndarray, true_label: str
    ) -> any:
        max_conf = 0
        class_ids = [
            self._classes_reversed[c.replace("_", " ")] for c in [true_label]
        ]
        detected_obj = self.model.predict(source=frame_img, classes=class_ids)[
            0
        ]
        all_detected_object_list = detected_obj.boxes.cls
        all_detected_object_confidence = detected_obj.boxes.conf

        for i in range(len(all_detected_object_list)):
            if all_detected_object_list[i] == class_ids[0]:
                if all_detected_object_confidence[i] > max_conf:
                    max_conf = all_detected_object_confidence[i].cpu().item()
        return max_conf
