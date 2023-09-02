from __future__ import annotations

import abc

import cv2
import numpy as np


class VideoProcessor(abc.ABC):
    @abc.abstractmethod
    def import_video(self, video_path) -> any:
        """Read video from video_path."""
        pass


class VideoFrameProcessor(VideoProcessor):
    """Video Frame Processor."""

    def __init__(self, video_path) -> None:
        """Video Frame Processor.

        Args:
            video_path (str): Path to video file.
        """
        self._video_path = video_path
        self._processed_frames = None
        self.import_video(video_path)

    def import_video(self, video_path):
        """Read video from video_path."""
        self._cap = cv2.VideoCapture(video_path)
        self.original_video_height = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.original_video_width = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.original_vidoe_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.original_frame_count = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def get_video_by_frame(
        self,
        frame_scale: int = 5,
        second_per_frame: int = 2,
        return_format: str = "ndarray",
    ) -> np.ndarray | list:
        """Get video frames by frame_scale and second_per_frame.

        Args:
            frame_scale (int, optional): Scale of frame. Defaults to 5.
            second_per_frame (int, optional): Second per frame. Defaults to 2.
            return_format (str, optional): Return format. Defaults to "npndarray".

        Returns:
        any: Video frames.
        """
        frames = list()
        frame_counter = 0
        frame_per_sec = int(round(self.original_vidoe_fps)) * second_per_frame
        while self._cap.isOpened():
            ret, frame_img = self._cap.read()
            if not ret:
                break
            if frame_counter % frame_per_sec == 0:
                frame_img = cv2.resize(
                    frame_img,
                    (
                        int(self.original_video_width / frame_scale),
                        int(self.original_video_height / frame_scale),
                    ),
                )
                frames.append(frame_img)
            if cv2.waitKey(1) & 0xFF == ord("q"):  # on press of q break
                break
            frame_counter += 1
        self._cap.release()
        cv2.destroyAllWindows()
        if return_format == "npndarray":
            self._processed_frames = np.array(frames)
            return np.array(frames)
        else:
            self._processed_frames = frames
            return frames

    @property
    def number_of_frames(self):
        """Get number of frames in video."""
        if self._processed_frames is None:
            return self.original_frame_count
        else:
            return len(self._processed_frames)
