from __future__ import annotations

import abc
import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ns_vfs.data.frame import Frame, FrameWindow
from ns_vfs.verification import check_automaton


class VideoProcessor(abc.ABC):
    @abc.abstractmethod
    def import_video(self, video_path) -> any:
        """Read video from video_path."""


class VideoFrameProcessor(VideoProcessor):
    """Video Frame Processor."""

    def __init__(self, video_path: str, artifact_dir: str) -> None:
        """Video Frame Processor.

        Args:
            video_path (str): Path to video file.
            artifact_dir (str): Path to artifact directory.
        """
        self._video_path = video_path
        self._artifact_dir = os.path.join(artifact_dir, "video_frame_processor")
        self._processed_frames = None
        self._frame_window = None
        self.import_video(video_path)

    def _resize_frame(
        self, frame_img: np.ndarray, frame_scale: int
    ) -> np.ndarray:
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

    def get_video_by_frame(
        self,
        frame_scale: int = 5,
        frame_duration_sec: int = 2,
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
        frame_per_sec = int(round(self.original_vidoe_fps) * frame_duration_sec)
        while self._cap.isOpened():
            ret, frame_img = self._cap.read()
            if not ret:
                break
            if frame_counter % frame_per_sec == 0:
                if frame_scale is not None:
                    frame_img = self._resize_frame(frame_img, frame_scale)

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


class VideoFrameWindowProcessor(VideoFrameProcessor):
    """Video Frame Window Processor.

    It processes video frame, build automaton and verification
    concurretnly by sliding window.
    """

    def __init__(self, video_path: str, artifact_dir: str) -> None:
        """Video Frame Processor.

        Args:
            video_path (str): Path to video file.
            artifact_dir (str): Path to artifact directory.
        """
        super().__init__(video_path, artifact_dir)

    def build_frame_window_synchronously(
        self,
        proposition_set: list,
        calculate_propositional_confidence: callable,
        build_automaton: callable,
        frame_duration_sec: int = 2,
        frame_scale: int | None = None,
        sliding_window_size: int = 5,
        save_image: bool = False,
    ) -> None:
        """Build frame window synchronously.

        It processes video frame, build automaton and verification
        concurrently by sliding window.

        Args:
            proposition_set (list): List of propositions.
            calculate_propositional_confidence (callable): Calculate propositional confidence.
            build_automaton (callable): Build automaton.
            frame_duration_sec (int, optional): Second per frame. Defaults to 2.
            frame_scale (int | None, optional): Scale of frame. Defaults to None.
            sliding_window_size (int, optional): Size of sliding window. Defaults to 5.
            save_image (bool, optional): Save image. Defaults to False.
        """
        frame_window_idx = 0
        frame_counter = 0
        frame_idx = 0

        # Calculate the frame skip rate
        frame_window = dict(frame_window_idx=0)
        temp_frame_set = list()

        while self._cap.isOpened():
            ret, frame_img = self._cap.read()
            if not ret:
                break
            if (
                frame_counter
                % int(frame_duration_sec * self.original_vidoe_fps)
                == 0
            ):
                if frame_scale is not None:
                    frame_img = self._resize_frame(frame_img, frame_scale)
                # --------------- frame image imported above --------------- #
                # Initialize frame
                frame: Frame = Frame(
                    frame_index=frame_idx,
                    frame_image=frame_img,
                )
                # Calculate propositional confidence
                for proposition in proposition_set:
                    propositional_confidence = (
                        calculate_propositional_confidence(
                            proposition=proposition,
                            frame_img=frame_img,
                            is_annotation=save_image,
                        )
                    )
                    frame.propositional_probability[
                        str(proposition)
                    ] = propositional_confidence
                temp_frame_set.append(frame)

                if len(temp_frame_set) == sliding_window_size:
                    if save_image:
                        self.replay_and_save(frames=temp_frame_set)

                    # Process frame window
                    frame_set = temp_frame_set.copy()
                    frame_window[frame_window_idx] = FrameWindow(
                        frame_window_idx=frame_window_idx,
                        frame_image_set=frame_set,
                    )

                    propositional_confidence = frame_window[
                        frame_window_idx
                    ].get_propositional_confidence()

                    # Build State & Compute Probability
                    states, transitions = build_automaton(
                        frame_set, propositional_confidence
                    )

                    # Verification - Build Transition Matrix
                    verification_result = check_automaton(
                        transitions=transitions,
                        states=states,
                        proposition_set=proposition_set,
                    )
                    frame_window[frame_window_idx].states = states
                    frame_window[frame_window_idx].transitions = transitions
                    frame_window[
                        frame_window_idx
                    ].verification_result = verification_result

                    frame_window_idx += 1
                    temp_frame_set.pop(0)

                frame_idx += 1
            frame_counter += 1

        self._cap.release()
        self._frame_window = frame_window
        return frame_window

    def replay_and_save(
        self,
        frames: list[Frame],
        frame_rate=2,
        is_imshow: bool = False,
        output_dir: str | None = None,
    ) -> None:
        """Replay and save frames.

        Args:
            frames (list[Frame]): List of frames.
            frame_rate (int, optional): Frame rate. Defaults to 2.
            is_imshow (bool, optional): Show image. Defaults to False.
            output_dir (str | None, optional): Output directory. Defaults to None.
        """
        if output_dir is None:
            output_dir = os.path.join(self._artifact_dir, "replay_frames")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        for idx, frame in enumerate(frames):
            plt.imshow(cv2.cvtColor(frame.frame_image, cv2.COLOR_BGR2RGB))
            plt.axis("off")  # hide the axis values
            plt.savefig(os.path.join(output_dir, f"frame_{idx}.png"))
            if is_imshow:
                plt.show()
            cv2.waitKey(
                int(1000 / frame_rate)
            )  # wait for specified milliseconds
        cv2.destroyAllWindows
