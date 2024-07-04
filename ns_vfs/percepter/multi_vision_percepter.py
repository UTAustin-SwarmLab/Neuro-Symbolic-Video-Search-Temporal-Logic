from typing import Dict, List

import numpy as np

from ns_vfs.model.vision._base import ComputerVisionModel
from ns_vfs.percepter._base import VisionPercepter


class MultiVisionPercepter(VisionPercepter):
    def __init__(self, cv_models: Dict[str, ComputerVisionModel]):
        super().__init__()
        self._cv_models = cv_models

    def perceive(self, image: np.ndarray, object_of_interest: List[str]):
        """Perceive the environment and return the perception."""
        detected_objects = {}
        for object in object_of_interest:
            for model_name, cv_model in self._cv_models.items():
                if cv_model.validate_object(object):
                    detected_object = cv_model.detect(
                        frame_img=image, classes=[object]
                    )
                    detected_objects[object] = detected_object
                    break
        return detected_objects
