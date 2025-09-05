from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import enum
import uuid


class VideoFormat(enum.Enum):
    MP4 = "mp4"
    LIST_OF_ARRAY = "list_of_array"

@dataclass
class VideoInfo:
    format: VideoFormat
    frame_width: int
    frame_height: int
    frame_count: int
    video_id: uuid.UUID = field(default_factory=uuid.uuid4)
    fps: float | None = None

class Reader(ABC):
    @abstractmethod
    def read_video(self) -> List[Dict[str, Any]]:
        pass

    def formatter(self, spec: str) -> str:
        spec = spec.replace("&", " and ")
        spec = spec.replace("|", " or ")
        spec = spec.replace("U", " until ")
        spec = spec.replace("F", " eventually ")
        spec = spec.replace("G", " always ")
        spec = spec.replace("X", " next ")
        spec = spec.replace('"', "")
        spec = spec.replace("'", "")
        spec = spec.replace("(", "")
        spec = spec.replace(")", "")
        while "  " in spec:
            spec = spec.replace("  ", " ")
        spec = spec.strip()
        return spec


