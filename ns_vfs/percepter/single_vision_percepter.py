from __future__ import annotations

import numpy as np

from ns_vfs.data.detected_object import DetectedObject
from ns_vfs.model.vision._base import ComputerVisionModel
from ns_vfs.percepter._base import VisionPercepter


class SingleVisionPercepter(VisionPercepter):
    def __init__(self, cv_models: dict[str, ComputerVisionModel]):
        super().__init__()
        self.cv_model = cv_models

    def perceive(
        self,
        image: np.ndarray,
        object_of_interest: list[str],
        ground_truth_object: list[str] | str | None = None,
    ) -> dict[str, DetectedObject]:
        """Perceive the environment and return the perception."""
        detected_objects = {}
        for object in object_of_interest:
            if ground_truth_object:
                # Dummy CV model is being used.
                if isinstance(ground_truth_object, str):
                    ground_truth_object = [ground_truth_object]
                if object in ground_truth_object:
                    detected_objects[object] = self.cv_model.detect(
                        frame_img=image, classes=[object], ground_truth=True
                    )
                else:
                    detected_objects[object] = self.cv_model.detect(
                        frame_img=image, classes=[object], ground_truth=False
                    )
            else:
                detected_object = self.cv_model.detect(
                    frame_img=image, classes=[object]
                )
                detected_objects[object] = detected_object
        return detected_objects
