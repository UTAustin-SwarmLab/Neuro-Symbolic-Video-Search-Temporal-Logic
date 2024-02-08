from __future__ import annotations

import cv2
import numpy as np

from ns_vfs.processor._base import BaseVideoProcessor


class RealVideoProcessor(BaseVideoProcessor):
    """Real Video Processor."""

    def __init__(
        self,
        video_path: str,
        frame_duration_sec: int = 1,
        frame_scale: int | None = None,
    ) -> None:
        """Video Frame Processor.

        Args:
            video_path (str): Path to video file.
            frame_duration_sec (int, optional): Frame duration in seconds. Defaults to 1.
            frame_scale (int | None, optional): Frame scale. Defaults to None.
        """
        self._video_path = video_path
        self._frame_duration_sec = frame_duration_sec
        self._frame_scale = frame_scale
        self.current_frame_index = 0
        self.video_ended = False
        self.import_video(video_path)

    def _resize_frame(self, frame_img: np.ndarray, frame_scale: int) -> np.ndarray:
        """Resize frame image.

        Args:
            frame_img (np.ndarray): Frame image.
            frame_scale (int): Scale of frame.

        Returns:
            np.ndarray: Resized frame image.
        """
        return cv2.resize(
            frame_img,
            (
                int(self.original_video_width / frame_scale),
                int(self.original_video_height / frame_scale),
            ),
        )

    def import_video(self, video_path: str) -> None:
        """Read video from video_path.

        Args:
            video_path (str): Path to video file.
        """
        self._cap = cv2.VideoCapture(video_path)
        self.original_video_height = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.original_video_width = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.original_vidoe_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.original_frame_count = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def get_all_frames_of_video(
        self,
        return_format: str = "cv2",
    ) -> list:
        """Get video frames by frame_scale and second_per_frame.

        Args:
            return_format (str, optional): Return format. Defaults to "cv2".
                - [cv2, ndarray]
        """
        all_frames = list()
        frame_step = int(self.original_vidoe_fps * self._frame_duration_sec)
        for real_frame_idx in range(0, int(self.original_frame_count), int(frame_step)):
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, real_frame_idx)
            ret, frame_img = self._cap.read()
            if not ret:
                break
            if self._frame_scale is not None:
                frame_img = self._resize_frame(frame_img, self.frame_scale)
            frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            if return_format == "ndarray":
                frame_img = np.array(frame_img, dtype=np.uint8)
            all_frames.append(frame_img)
        self._cap.release()
        cv2.destroyAllWindows()
        return all_frames

    def get_next_frame(self, return_format: str = "ndarray") -> np.ndarray | None:
        """Get the next video frame based on frame step.

        Args:
            return_format (str, optional): Return format. Defaults to "ndarray".
                - [PIL.Image, ndarray]

        Returns:
            np.ndarray | None: The next frame as an ndarray, or None if no more frames are available or the video ended.
        """
        if self.video_ended:
            return None  # No more frames to process

        frame_step = int(self.original_vidoe_fps * self._frame_duration_sec)

        # Skip to the next frame based on frame_step
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)

        ret, frame_img = self._cap.read()

        if not ret:
            self.video_ended = True
            return None  # No more frames or error occurred

        # Update the current frame index for the next call
        self.current_frame_index += frame_step

        if self._frame_scale is not None:
            frame_img = self._resize_frame(frame_img, self._frame_scale)

        frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

        if return_format == "PIL.Image":
            # TODO: Convert to PIL.Image
            pass

        return frame_img
