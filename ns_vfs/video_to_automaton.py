from __future__ import annotations

import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import supervision as sv

from ns_vfs.common.utility import get_filename_with_datetime
from ns_vfs.data.frame import Frame
from ns_vfs.model.vision._base import ComputerVisionDetector
from ns_vfs.processor.video_processor import (
    VideoFrameWindowProcessor,
    VideoProcessor,
)
from ns_vfs.state import State


class VideotoAutomaton:
    def __init__(
        self,
        proposition_set: list[str],
        detector: ComputerVisionDetector,
        video_processor: VideoProcessor,
        artifact_dir: str,
        ltl_formula: str,
        is_annotation: bool = False,
        save_image: bool = False,
    ) -> None:
        """Initialize Video to Automaton.

        Args:
            proposition_set (list): List of propositions.
            detector (ComputerVisionDetector): Computer Vision Detector.
            video_processor (VideoProcessor): Video Processor.
            artifact_dir (str): Path to artifact directory.
            ltl_formula (str): LTL formula.
            is_annotation (bool, optional): Whether to annotate the frame. Defaults to False.
            save_image (bool, optional): Whether to save image. Defaults to False.
        """
        self.proposition_set = proposition_set
        self.proposition_combinations = self._create_proposition_status(len(proposition_set))
        self._detector = detector
        self._video_processor = video_processor
        self._artifact_dir = artifact_dir
        self._is_annotation = is_annotation
        self._ltl_formula = ltl_formula
        self._save_image = save_image

    def _sigmoid(self, x, k=1, x0=0) -> float:
        """Sigmoid function.

        Args:
            x (float): Input.
            k (int, optional): Steepness of the function. Defaults to 1.
            x0 (int, optional): Midpoint of the function. Defaults to 0.

        Returns:
            float: Sigmoid function.
        """
        return 1 / (1 + np.exp(-k * (x - x0)))

    def _annotate_frame(
        self,
        frame_img: np.ndarray,
        proposition: list,
        detected_obj: any,
        output_dir: str | None = None,
    ) -> None:
        """Annotate frame with bounding box.

        Args:
            frame_img (np.ndarray): Frame image.
            proposition (list): List of propositions.
            detected_obj (any): Detected object.
            output_dir (str | None, optional): Output directory. Defaults to None.
        """
        box_annotator = sv.BoxAnnotator()
        labels = [
            f"{proposition[class_id] if class_id is not None else None} {confidence:0.2f}"
            for _, _, confidence, class_id, _ in detected_obj
        ]

        annotated_frame = box_annotator.annotate(
            scene=frame_img.copy(), detections=detected_obj, labels=labels
        )

        sv.plot_image(annotated_frame, (16, 16))

        if output_dir is None:
            path = os.path.join(self._artifact_dir, "annotated_frame")

        if not os.path.exists(path):
            os.makedirs(path)

        filename = get_filename_with_datetime("annotated_frame.png")
        plt.savefig(os.path.join(path, filename))

    def _create_proposition_status(self, num_props):
        """Create all possible combinations of T and F for the number of propositions.

        Args:
            num_props (int): Number of propositions.


        Returns:
        list: List of all possible combinations of T and F.
        """
        label_list = []

        def add_labels(num_props, label, label_list):
            if len(label) == num_props:
                label_list.append(label)
                return
            add_labels(num_props, label + "T", label_list)
            add_labels(num_props, label + "F", label_list)

        add_labels(num_props, "", label_list)
        return label_list

    def _mapping_probability(
        self,
        confidence_per_video: float,
        true_threshold=0.66,
        false_threshold=0.38,
    ) -> float:
        """Mapping probability.

        Args:
            confidence_per_video (float): Confidence per video.
            true_threshold (float, optional): True threshold. Defaults to 0.66.
            false_threshold (float, optional): False threshold. Defaults to 0.38.

        Returns:
            float: Mapped probability.
        """
        if confidence_per_video >= true_threshold:
            return 1
        elif confidence_per_video <= false_threshold:
            return 0
        else:
            return round(self._sigmoid(confidence_per_video, k=50, x0=0.56), 2)

    def get_probabilistic_proposition_from_frame(
        self,
        proposition: str,
        frame_img: np.ndarray,
        is_annotation: bool = False,
    ) -> float:
        """Get probabilistic proposition from frame.

        Args:
            proposition (str): Proposition.
            frame_img (np.ndarray): Frame image.
            is_annotation (bool, optional): Whether to annotate the frame. Defaults to False.

        Returns:
            float: Probabilistic proposition from frame.
        """
        detected_obj = self._detector.detect(frame_img, [proposition])
        if len(detected_obj) > 0:
            if is_annotation:
                self._annotate_frame(
                    frame_img=frame_img,
                    detected_obj=detected_obj,
                    proposition=[proposition],
                )
            return self._mapping_probability(np.round(np.max(detected_obj.confidence), 2))
            # probability of the object in the frame
        else:
            return 0  # probability of the object in the frame is 0

    def calculate_confidence_of_proposition(
        self,
        proposition: str,
        frame_img: np.ndarray,
        is_annotation: bool = False,
    ) -> float:
        """Calculate confidence of proposition.

        Args:
            proposition (str): Proposition.
            frame_img (np.ndarray): Frame image.
            is_annotation (bool, optional): Whether to annotate the frame. Defaults to False.

        Returns:
            float: Confidence of proposition.
        """
        propositional_probability_on_frame = self.get_probabilistic_proposition_from_frame(
            proposition=proposition,
            frame_img=frame_img,
            is_annotation=is_annotation,
        )
        return propositional_probability_on_frame

    def build_automaton(
        self,
        frame_set: list[Frame],
        propositional_confidence: list[list[float]],
    ) -> (list, list):
        """Build automaton.

        Args:
            frame_set (list[Frame]): List of frames.
            propositional_confidence (list[list[float]]): List of propositional confidence.

        Returns:
            States: List of states.
            Transitions: List of transitions.
        """
        state_idx = 0
        states = list()
        transitions = list()

        state = State(state_idx, -1, "init", self.proposition_set)
        states.append(copy.deepcopy(state))

        prev_states = [copy.deepcopy(state)]

        for i in range(len(frame_set)):
            frame_set[i]
            current_state = list()
            for prop_comb in self.proposition_combinations:
                state.update(
                    frame_index=i,
                    proposition_combinations=prop_comb,
                )
                state.compute_probability(probabilities=propositional_confidence)
                if state.probability > 0:
                    state.state_index += 1
                    states.append(copy.deepcopy(state))
                    current_state.append(copy.deepcopy(state))

            # Transition
            for cur_state in current_state:
                for prev_state in prev_states:
                    transition = (
                        prev_state.state_index,
                        cur_state.state_index,
                        cur_state.probability,
                    )
                    transitions.append(transition)
            prev_states = current_state.copy()

        return states, transitions

    def build_frame_window_automaton(self) -> dict:
        """Build frame window automaton."""
        video_frames_automaton = self._video_processor.build_frame_window_synchronously(
            proposition_set=self.proposition_set,
            calculate_propositional_confidence=self.calculate_confidence_of_proposition,
            build_automaton=self.build_automaton,
            ltl_formula=self._ltl_formula,
            save_image=self._save_image,
            is_annotation=self._is_annotation,
        )
        path = os.path.join(self._artifact_dir, "frame_window_automata.npy")
        np.save(path, video_frames_automaton)
        return video_frames_automaton

    def build_frame_automaton(self) -> dict:
        """Build frame window automaton."""
        # TODO: If we need to build frame automaton differently, we can do it here.
        video_frames_automaton = None
        return video_frames_automaton

    def run(self) -> dict:
        """Run."""
        if isinstance(self._video_processor, VideoFrameWindowProcessor):
            return self.build_frame_window_automaton()
        else:
            return self.build_frame_automaton()
