import abc


class BasePercepter(abc.ABC):
    """Base class for percepters."""

    ...

    @abc.abstractmethod
    def perceive(self, *args, **kwargs):
        """Perceive the environment and return the perception."""
        raise NotImplementedError


class VisionPercepter(BasePercepter):
    """Base class for vision percepters."""

    def __init__(self):
        pass
