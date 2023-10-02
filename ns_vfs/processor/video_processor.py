from __future__ import annotations

import abc
import os

import cv2
import numpy as np

from ns_vfs.data.frame import Frame, FramesofInterest


class VideoProcessor(abc.ABC):
    """Video Processor."""

    @abc.abstractmethod
    def import_video(self, video_path) -> any:
        """Read video from video_path."""


class VideoFrameProcessor(VideoProcessor):
    """Video Frame Processor."""

    def __init__(
        self,
        video_path: str,
        artifact_dir: str,
        frame_duration_sec: int = 1,
        frame_scale: int | None = None,
        is_auto_import: bool = True,
    ) -> None:
        """Video Frame Processor.

        Args:
            video_path (str): Path to video file.
            artifact_dir (str): Path to artifact directory.
        """
        self._video_path = video_path
        self._artifact_dir = os.path.join(artifact_dir, "video_frame_processor")
        self._processed_frames = None
        self._frame_duration_sec = frame_duration_sec
        self._frame_scale = frame_scale
        if is_auto_import:
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

    def _reset_frame_set_and_confidence(self, proposition_set: list) -> tuple[list, list]:
        """Reset frame set and confidence.

        Args:
            proposition_set (list): List of propositions.

        Returns:
            tuple[list, list]: Frame set and confidence.
        """
        frame_set = list()
        interim_confidence_set = [[] for _ in range(len(proposition_set))]
        return frame_set, interim_confidence_set

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

    def process_and_get_frame_of_interest(
        self,
        ltl_formula: str = "",
        proposition_set: list = None,
        get_propositional_confidence_per_frame: callable = None,
        validate_propositional_confidence: callable = None,
        build_and_check_automaton: callable = None,
        update_frame_of_interest: callable = None,
    ) -> FramesofInterest:
        frame_idx = 0
        frame_set, interim_confidence_set = self._reset_frame_set_and_confidence(proposition_set)
        frame_of_interest = FramesofInterest(ltl_formula=ltl_formula)

        frame_step = int(self.original_vidoe_fps * self._frame_duration_sec)
        for real_frame_idx in range(0, int(self.original_frame_count), int(frame_step)):
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, real_frame_idx)
            ret, frame_img = self._cap.read()
            if not ret:
                break
            if self._frame_scale is not None:
                frame_img = self._resize_frame(frame_img, self.frame_scale)
            # --------------- frame image imported above --------------- #
            frame: Frame = Frame(frame_index=frame_idx, frame_image=frame_img, real_frame_idx=real_frame_idx)

            frame = get_propositional_confidence_per_frame(frame=frame, proposition_set=proposition_set)

            frame_set, interim_confidence_set = validate_propositional_confidence(
                frame_set=frame_set,
                frame=frame,
                proposition_set=proposition_set,
                interim_confidence_set=interim_confidence_set,
            )

            if len(frame_set) > 0:  # propositions in frame
                result = build_and_check_automaton(
                    frame_set=frame_set,
                    interim_confidence_set=interim_confidence_set,
                    include_initial_state=False,
                    proposition_set=proposition_set,
                    ltl_formula=ltl_formula,
                    verbose=False,
                )
                # if we uncomment below prop1 U porp2 won't work, need validation
                # if result == "False":  # propositions in frame but doesn't meet the initial ltl condition
                # frame_set, interim_confidence_set = self._reset_frame_set_and_confidence(proposition_set)
                # else:
                #   # Some Code
                if result:
                    # 2.1 Save result
                    frame_of_interest = update_frame_of_interest(
                        frame_set=frame_set, frame_of_interest=frame_of_interest
                    )
                    # 2.2 Reset frame set
                    frame_set, interim_confidence_set = self._reset_frame_set_and_confidence(proposition_set)
            frame_idx += 1
        frame_of_interest.reorder_frame_of_interest()
        return frame_of_interest

    # def get_frame_of_interest(
    #     self,
    #     proposition_set: list,
    #     calculate_propositional_confidence: callable,
    #     build_automaton: callable,
    #     ltl_formula: str,
    #     save_image: bool = False,
    #     is_annotation: bool = False,
    #     manual_confidence_probability: float | None = None,
    #     verbose: bool = False,
    # ) -> None:
    #     frame_idx = 0
    #     temp_frame_set = list()
    #     temp_confidence_set = [[] for _ in range(len(proposition_set))]
    #     frame_of_interest = list()
    #     reverse_search = False
    #     frame_of_interest_obj = FramesofInterest(ltl_formula=ltl_formula)
    #     if "!" in ltl_formula:
    #         reverse_search = True

    #     frame_step = int(self.original_vidoe_fps * frame_duration_sec)

    #     for real_frame_idx in range(0, int(self.original_frame_count), int(frame_step)):
    #         self._cap.set(cv2.CAP_PROP_POS_FRAMES, real_frame_idx)
    #         ret, frame_img = self._cap.read()
    #         if not ret:
    #             break

    #         if frame_scale is not None:
    #             frame_img = self._resize_frame(frame_img, frame_scale)
    #         # --------------- frame image imported above --------------- #
    #         # Initialize frame
    #         frame: Frame = Frame(frame_index=frame_idx, frame_image=frame_img, real_frame_idx=real_frame_idx)
    #         # Calculate propositional confidence
    #         for proposition in proposition_set:
    #             propositional_confidence, detected_obj, annotated_img = calculate_propositional_confidence(
    #                 proposition=proposition,
    #                 frame_img=frame_img,
    #                 is_annotation=is_annotation,
    #             )
    #             frame.object_detection[str(proposition)] = detected_obj
    #             frame.propositional_probability[str(proposition)] = propositional_confidence
    #             if annotated_img is not None:
    #                 frame.annotated_image.append(annotated_img)

    #         # 1. If Propositional Confidence is True
    #         propositional_confidence_of_frame = frame.propositional_confidence
    #         proposition_condition = sum(propositional_confidence_of_frame)
    #         if proposition_condition > 0:
    #             temp_frame_set.append(frame)
    #             for i in range(len(proposition_set)):
    #                 temp_confidence_set[i].append(propositional_confidence_of_frame[i])

    #             frame_set = temp_frame_set.copy()
    #             states, transitions = build_automaton(
    #                 frame_set, temp_confidence_set, include_initial_state=False
    #             )
    #             verification_result = check_automaton(
    #                 transitions=transitions,
    #                 states=states,
    #                 proposition_set=proposition_set,
    #                 ltl_formula=ltl_formula,
    #                 verbose=verbose,
    #             )
    #             verification_result_str = str(verification_result)
    #             string_result = verification_result_str.split("{")[-1].split("}")[0]
    #             if string_result[0] == "t":
    #                 result = True
    #             else:
    #                 result = False
    #                 temp_frame_set = list()
    #                 temp_confidence_set = [[] for _ in range(len(proposition_set))]
    #                 if reverse_search:
    #                     result = True

    #             # 2. If Verification Result is True
    #             if result:
    #                 # 2.1 Save result
    #                 if len(temp_frame_set) > 1:
    #                     for frame in temp_frame_set:
    #                         frame_of_interest.append([frame.frame_index])
    #                         frame_of_interest_obj.frame_idx_to_real_idx[
    #                             frame.frame_index
    #                         ] = frame.real_frame_idx
    #                         frame_of_interest_obj.frame_images.append(frame.frame_image)
    #                         for j in frame.annotated_image:
    #                             frame_of_interest_obj.annotated_image_images.append(j)

    #                 else:
    #                     for frame in temp_frame_set:
    #                         frame_of_interest.append([frame.frame_index])
    #                         frame_of_interest_obj.frame_idx_to_real_idx[
    #                             frame.frame_index
    #                         ] = frame.real_frame_idx
    #                         frame_of_interest_obj.frame_idx_to_real_idx
    #                         frame_of_interest_obj.frame_images.append(frame.frame_image)
    #                         for j in frame.annotated_image:
    #                             frame_of_interest_obj.annotated_image_images.append(j)
    #                 # 2.2 Reset frame set
    #                 temp_frame_set = list()
    #                 temp_confidence_set = [[] for _ in range(len(proposition_set))]

    #         frame_idx += 1
    #     if reverse_search:
    #         flattened_list = [item for sublist in frame_of_interest for item in sublist]
    #         frame_of_interest = [x for x in range(frame_idx) if x not in flattened_list]
    #     frame_of_interest = combine_consecutive_lists(frame_of_interest)
    #     frame_of_interest_obj.frame_of_interest = frame_of_interest
    #     return frame_of_interest_obj

    def get_video_by_frame(
        self,
        frame_scale: int = 5,
        frame_duration_sec: int = 2,
        return_format: str = "ndarray",
    ) -> np.ndarray | list:
        """Get video frames by frame_scale and second_per_frame.

        Args:
            frame_scale (int, optional): Scale of frame. Defaults to 5.
            frame_duration_sec (int, optional): Second per frame. Defaults to 2.
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
