from __future__ import annotations

import abc


class BaseVideoProcessor(abc.ABC):
    """Video Processor."""

    @abc.abstractmethod
    def import_video(self, video_path) -> any:
        """Read video from video_path."""
