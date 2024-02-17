import abc

from ns_vfs.model._base import NeuralPerceptionModel


class VisionLanguageModelBase(NeuralPerceptionModel):
    """Vision Language Model Base class."""

    def __init__(self, device: str) -> None:
        super().__init__(device)

    @abc.abstractmethod
    def infer(self, vision_input: any, language_input: str) -> any: ...
