from __future__ import annotations

import warnings

import numpy as np
import torch

from ns_vfs.data.detected_object import DetectedObject
from ns_vfs.model.vision.object_detection._base import (
    ComputerVisionObjectDetector,
)

from transformers import AutoImageProcessor, VideoMAEForVideoClassification
import torch.nn.functional as F
import av

warnings.filterwarnings("ignore")


class VideoMAE(ComputerVisionObjectDetector):
    """VideoMAE."""

    def __init__(self) -> None:
        """Initialization."""
        self.model = self.load_model()

    def load_model(self) -> None:
        """Load weight.

        Args:
            None

        Returns:
            None
        """
        return VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics"
        )

    def detect(self, frames: list) -> any:
        """Detect object in frame.

        Args:
            frames (list[np.ndarray]): Frame list.

        Returns:
            any: Detections.
        """
        image_processor = AutoImageProcessor.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics"
        )
        av_images = [
            av.VideoFrame.from_ndarray(image, format="rgb24")
            for image in frames
        ]
        video = np.stack([x.to_ndarray(format="rgb24") for x in av_images])

        inputs = image_processor(list(video), return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)

        predicted_label = logits.argmax(-1).item()
        detection = self.model.config.id2label[predicted_label]
        confidence_from_model = probabilities[0][predicted_label].item()
        print(f"DETECTION: {detection} - {confidence_from_model}")

        return DetectedObject(
            name=detection.replace(" ", "_"),
            model_name="videomae",
            confidence_of_all_obj=[confidence_from_model],
            probability_of_all_obj=[confidence_from_model],
            all_obj_detected=detection,
            number_of_detection=1,
            is_detected=(confidence_from_model > 0.8),
            supervision_detections=None,
            bounding_box_of_all_obj=None,
        )
