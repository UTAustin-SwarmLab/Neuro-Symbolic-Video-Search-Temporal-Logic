import abc


class ComputerVisionModel(abc.ABC):
    @abc.abstractmethod
    def load_model(self) -> any:
        """Load weight."""
