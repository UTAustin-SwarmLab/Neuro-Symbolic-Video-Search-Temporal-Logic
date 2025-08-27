import enum
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class VideoFormat(enum.Enum):
    """Status Enum for the CV API."""

    MP4 = "mp4"
    LIST_OF_ARRAY = "list_of_array"


@dataclass
class VideoInfo:
    """Represents information about a video file."""

    format: VideoFormat
    frame_width: int
    frame_height: int
    original_frame_count: int
    video_id: uuid.UUID = field(default_factory=uuid.uuid4)
    video_path: str | None = None
    processed_fps: float | None = None
    processed_frame_count: int = 1
    original_fps: float | None = None


class Video:
    """vflow's Video Object."""

    def __init__(
        self,
        read_format: VideoFormat,
        video_path: str | Path | None = None,
        sequence_of_image: list[np.ndarray] | None = None,
    ) -> None:
        """Video Frame Processor.

        Args:
            video_path (str | Path): Path to video file.
            read_format (VideoFormat): Format to read the video in.
            sequence_of_image (list[np.ndarray] | None): List of image arrays
                for processing.
        """
        self._video_path = video_path
        self._read_format = read_format
        self.video_info = None
        if sequence_of_image:
            self.all_frames = sequence_of_image
            if isinstance(sequence_of_image[0], list):
                self.all_frames = sequence_of_image[0]
        self.import_video(str(video_path))
        self.current_frame_index = 0
        self.video_ended = False

    def __str__(self) -> str:
        """Return a concise string representation of the Video object."""
        return str(self.video_info)

    def __repr__(self) -> str:
        """Return a detailed string representation of the Video object."""
        return repr(self.video_info)

    def import_video(self, video_path: str | None) -> None:
        """Read video from video_path.

        Args:
            video_path (str): Path to video file.
        """
        logging.info(f"Video format: {self._read_format}")
        if self._read_format == VideoFormat.MP4:
            self._cap = cv2.VideoCapture(video_path)
            ret, _ = self._cap.read()
            if not ret:
                logging.error("Video path is invalid.")
            self.video_info = VideoInfo(
                video_path=str(self._video_path),
                format=self._read_format,
                frame_width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                frame_height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                original_fps=self._cap.get(cv2.CAP_PROP_FPS),
                original_frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            )
        elif self._read_format.LIST_OF_ARRAY:
            self.video_info = VideoInfo(
                format=self._read_format,
                frame_width=int(self.all_frames[0].shape[0]),
                frame_height=int(self.all_frames[0].shape[1]),
                original_frame_count=len(self.all_frames),
            )

    def _resize_frame_by_scale(self, frame_img: np.ndarray, frame_scale: int) -> np.ndarray:
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
                int(self.video_info.frame_width / frame_scale),
                int(self.video_info.frame_height / frame_scale),
            ),
        )

    def get_all_frames_of_video(
        self,
        return_format: str = "ndarray",
        frame_scale: int | None = None,
        desired_fps: int | None = None,
        desired_interval_in_sec: int | None = None,
    ) -> list:
        """Get video frames by frame_scale and second_per_frame.

        Args:
            return_format (str, optional): Return format. Defaults to "cv2".
                Options: [cv2, ndarray]
            frame_scale (int | None, optional): Frame scale. Defaults to None.
            desired_fps (int | None, optional): Desired FPS. Defaults to None.
            desired_interval_in_sec (int | None, optional): Interval between frames in seconds.
                If provided, frames will be extracted at this interval. Defaults to None.
        """
        if self._read_format == VideoFormat.LIST_OF_ARRAY:
            resize_func = lambda img: self.process_frame_image(
                frame_img=img,
                frame_scale=frame_scale,
                return_format=return_format,
            )
            all_frames = list(map(resize_func, self.all_frames))
            self.processed_frame_count = len(all_frames)
            return all_frames

        all_frames = []

        if self._read_format == VideoFormat.MP4 and desired_fps is None and desired_interval_in_sec is None:
            msg = (
                "Either desired_fps",
                "or desired_interval_in_sec must be provided.",
            )
            raise ValueError(msg)

        if self._read_format == VideoFormat.MP4:
            frame_step = self.get_frame_step(
                desired_fps=desired_fps,
                desired_interval_in_sec=desired_interval_in_sec,
            )

            for real_frame_idx in range(0, int(self.video_info.original_frame_count), int(frame_step)):
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, real_frame_idx)
                ret, frame_img = self._cap.read()
                frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                if not ret:
                    break
                frame_img = self.process_frame_image(
                    frame_img=frame_img,
                    frame_scale=frame_scale,
                    return_format=return_format,
                )
                all_frames.append(frame_img)
            self._cap.release()
            # cv2.destroyAllWindows()
            self.processed_frame_count = len(all_frames)
        return all_frames

    def get_next_frame(
        self,
        return_format: str = "ndarray",
        frame_scale: int | None = None,
        desired_fps: int | None = None,
        desired_interval_in_sec: int | None = None,
    ) -> np.ndarray | None:
        """Get the next video frame based on frame step.

        Args:
            return_format (str, optional): Return format. Defaults to "ndarray".
                - [cv2, ndarray, pil]
            frame_scale (int | None, optional): Frame scale. Defaults to None.
            desired_fps (int | None, optional): Desired FPS. Defaults to None.
            desired_interval_in_sec (int | None, optional): Desired interval.
                Defaults to None.

        Returns:
            np.ndarray | None: The next frame as an ndarray, or None if no more
                frames are available or the video ended.
        """
        if self._read_format == VideoFormat.MP4 and desired_fps is None and desired_interval_in_sec is None:
            msg = (
                "Either desired_fps or",
                "desired_interval_in_sec must be provided.",
            )
            raise ValueError(msg)

        if self.video_ended:
            logging.info("No frame available.")
            return None  # No more frames to process

        if self._read_format == VideoFormat.MP4:
            frame_step = self.get_frame_step(
                desired_fps=desired_fps,
                desired_interval_in_sec=desired_interval_in_sec,
            )
            # Skip to the next frame based on frame_step
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)

            ret, frame_img = self._cap.read()

            if not ret:
                self.video_ended = True
                return None  # No more frames or error occurred

            # Update the current frame index for the next call
            self.current_frame_index += frame_step

            frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

        if self._read_format == VideoFormat.LIST_OF_ARRAY:
            if self.current_frame_index < len(self.all_frames):
                frame_img = self.all_frames[self.current_frame_index]
                self.current_frame_index += 1
            else:
                # No more frames available.
                self.video_ended = True
                return None

        self.video_info.processed_frame_count += 1

        return self.process_frame_image(
            frame_img=frame_img,
            frame_scale=frame_scale,
            return_format=return_format,
        )

    def process_frame_image(
        self,
        frame_img: np.ndarray,
        return_format: str = "ndarray",
        frame_scale: int | None = None,
    ) -> np.ndarray:
        """Process a single frame image.

        Args:
            frame_img (np.ndarray): Input frame image.
            return_format (str, optional): Desired return format.
                Defaults to "ndarray".
            frame_scale (int | None, optional): Scale factor for resizing.
                Defaults to None.

        Returns:
            np.ndarray: Processed frame image.
        """
        if frame_scale is not None:
            frame_img = self._resize_frame_by_scale(frame_img, frame_scale)
        if return_format == "pil":
            frame_img = Image.fromarray(frame_img).convert("RGB")
        return frame_img

    def get_frame_step(
        self,
        desired_interval_in_sec: int | None = None,
        desired_fps: int | None = None,
    ) -> int:
        """Calculate the frame step based on desired interval or FPS.

        Args:
            desired_interval_in_sec (int | None): Desired interval between frames in seconds.
            desired_fps (int | None): Desired frames per second.

        Returns:
            int: Calculated frame step.
        """
        if desired_fps is not None:
            frame_step = int(round(self.video_info.original_fps / desired_fps))
            processed_fps = desired_fps
        if desired_interval_in_sec is not None:
            frame_step = int(round(self.video_info.original_fps * desired_interval_in_sec))
            processed_fps = round(1 / desired_interval_in_sec, 2)
        self.video_info.processed_fps = processed_fps

        return frame_step

    def insert_annotation_to_current_frame(self, annotations: list[str]) -> None:
        """Insert annotations to the current frame.

        Args:
            annotations (list[str]): List of annotations.
        """

    def get_video_info(self) -> VideoInfo:
        """Return the VideoInfo object containing video information."""
        return self.video_info
