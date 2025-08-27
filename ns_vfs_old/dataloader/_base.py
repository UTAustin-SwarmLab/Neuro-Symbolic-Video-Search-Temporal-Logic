from typing import List, Dict, Union
import abc
import numpy as np

class DatasetLoader(abc.ABC):
    """Base class for loading dataset."""

    def __init__(self, video_path: str, subtitle_path: str) -> None:
        self.video_path = video_path
        self.subtitle_path = subtitle_path

    @abc.abstractmethod
    def load_all(self) -> List[Dict[str, Union[List[np.ndarray], None]]]:
        """Load video and subtitles."""

