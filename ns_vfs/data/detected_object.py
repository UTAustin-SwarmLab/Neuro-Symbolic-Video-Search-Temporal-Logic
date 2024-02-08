from __future__ import annotations

import dataclasses  # noqa: D100
from typing import Any, List, Optional

import supervision as sv

from ns_vfs.enums.status import Status


@dataclasses.dataclass
class DetectedObject:
    """Detected Object class."""

    name: str
    confidence: float = 0.0
    probability: float = 0.0
    confidence_of_all_obj: Optional[List[float]] = None
    probability_of_all_obj: Optional[List[float]] = None
    bounding_box_of_all_obj: Optional[List[Any]] = None
    all_obj_detected: Optional[List[Any]] = None
    number_of_detection: int = 0
    is_detected: bool | Status = Status.UNKNOWN
    model_name: Optional[str] = None
    supervision_detections: Optional[sv.Detections] = None

    def __post_init__(self):
        if len(self.confidence_of_all_obj) > 0:
            self.confidence = max(self.confidence_of_all_obj)
        if len(self.confidence_of_all_obj) > 0:
            self.probability = max(self.probability_of_all_obj)
