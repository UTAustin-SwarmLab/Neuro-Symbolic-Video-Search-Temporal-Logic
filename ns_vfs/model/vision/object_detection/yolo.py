from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

from ns_vfs.data.detected_object import DetectedObject
from ns_vfs.model.vision.object_detection._base import (
    ComputerVisionObjectDetector,
)

warnings.filterwarnings("ignore")


class Yolo(ComputerVisionObjectDetector):
    """Yolo."""

    def __init__(self, weight_path: Path, gpu_number: int = 0) -> None:
        """Initialization."""
        if isinstance(weight_path, str):
            weight_path = Path(weight_path)
        self.model = self.load_model(weight_path)
        device = f"cuda:{gpu_number}" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self._available_classes = {v: k for k, v in self.model.names.items()}

    def load_model(self, weight_path) -> YOLO:
        """Load weight.

        Args:
            weight_path (str): Path to weight file.

        Returns:
            None
        """
        return YOLO(weight_path)

    def validate_object(self, object_name: str) -> bool:
        """Validate object name.

        Args:
            object_name (str): Object name.

        Returns:
            bool: True if object name is valid.
        """
        return object_name.replace("_", " ") in list(
            self._available_classes.keys()
        )

    def _parse_class_name(self, class_names: list[str]) -> list[str]:
        """Doest not need to parse class name."""
        ...

    def get_bounding_boxes(self, detected_obj) -> list:
        """Get bounding boxes.

        Args:
            detected_obj (DetectedObject): Detected object.

        Returns:
            list: Bounding boxes.
        """
        bboxes = []
        for row in detected_obj[0].boxes.data.cpu().numpy():
            bbox = row[:4].tolist()
            bboxes.append(bbox)
        return bboxes

    def detect(self, frame_img: np.ndarray, classes: list) -> any:
        """Detect object in frame.

        Args:
            frame_img (np.ndarray): Frame image.
            classes (list[str]): List of class names.

        Returns:
            any: Detections.
        """
        class_name = classes[0].replace("_", " ")
        class_ids = [
            self._available_classes[c.replace("_", " ")] for c in classes
        ]
        detected_obj = self.model.predict(source=frame_img, classes=class_ids)

        supervision_detections = sv.Detections(
            xyxy=detected_obj[0].boxes.xyxy.cpu().detach().numpy()
        )

        confidence_from_model = (
            detected_obj[0].boxes.conf.cpu().detach().numpy()
        )

        num_detections = len(detected_obj[0].boxes)
        if num_detections > 0:
            is_detected = True
        else:
            is_detected = False

        probability = []
        for confidence in confidence_from_model:
            probability.append(self._mapping_probability(confidence))

        return DetectedObject(
            name=class_name,
            model_name="yolo",
            confidence_of_all_obj=list(confidence_from_model),
            probability_of_all_obj=list(probability),
            all_obj_detected=detected_obj,
            number_of_detection=num_detections,
            is_detected=is_detected,
            supervision_detections=supervision_detections,
            bounding_box_of_all_obj=self.get_bounding_boxes(detected_obj),
        )

    def _mapping_probability(
        self,
        confidence_per_video: float,
        true_threshold=0.60,  # 0.60,
        false_threshold=0.40,  # 0.40,
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
            return 1.0
        elif confidence_per_video < false_threshold:
            return 0.0
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
