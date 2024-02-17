import abc


class NeuralPerceptionModel(abc.ABC):
    """Base class for neural perception model."""

    def __init__(self, device: str) -> None:
        self.device = device

    @abc.abstractmethod
    def load_model(self) -> any:
        """Load weight."""
