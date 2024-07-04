from __future__ import annotations


from ns_vfs.data.detected_object import DetectedObject
from ns_vfs.percepter._base import VisionPercepter


class ActivityPercepter(VisionPercepter):
    def __init__(self, model):
        super().__init__()
        self.activity_model = model

    def perceive(
        self,
        images: list,
        object_of_interest: str,
    ) -> dict[str, DetectedObject]:
        """Perceive the environment and return the perception."""
        detected_objects = {}
        detected_object = self.activity_model.detect(frames=images)
        return detected_object
