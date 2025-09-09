from typing import Any
import logging

class DetectedObject:
    """Detected Object class."""
    def __init__(self, 
                 name: str,
                 is_detected: bool, 
                 confidence: float,
                 probability: float,
                 model_name: str | None = None, 
                 bounding_box_of_all_obj: list[Any] | None = None):
        self.name = name
        self.confidence = confidence
        self.probability = probability
        self.is_detected = is_detected
        self.model_name = model_name
        self.bounding_box_of_all_obj = bounding_box_of_all_obj

    def __str__(self) -> str:
        return f"Object: {self.name}, Detected: {self.is_detected}, Probability: {self.get_detected_probability()}"

    def get_detected_probability(self) -> float:
        if not self.is_detected:
            return 0
        if self.probability > 0:
            return self.probability
        if self.confidence > 0 and self.probability == 0:
            logging.info("Probability is not set, using confidence: %f", self.confidence)
            return self.confidence
        return self.probability
