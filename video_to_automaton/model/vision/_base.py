import abc


class ComputerVisionModel(abc.ABC):
    @abc.abstractmethod
    def load_model(self) -> any:
        """Load weight."""


class ComputerVisionDetector(ComputerVisionModel):
    """Computer Vision Detector."""

    def __init__(self, weight_path) -> None:
        """Computer Vision Detector.

        Args:
            weight_path (str): Path to weight file.
        """
        self._weight_path = weight_path
        self.load_model(weight_path)

    def load_model(self, weight_path):
        """Load weight."""
        self._weight = weight_path

    def get_weight(self):
        """Get weight."""
        return self._weight

    @abc.abstractmethod
    def detect(self, frame) -> any:
        """Detect object in frame."""
        pass
