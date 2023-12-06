from __future__ import annotations

import copy
import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from PIL import Image

from ns_vfs.common.utility import get_file_or_dir_with_datetime
from ns_vfs.data.frame import Frame
from ns_vfs.loader import LABEL_OF_INTEREST
from ns_vfs.model.vision._base import ComputerVisionDetector
from ns_vfs.processor.video_processor import (
    VideoProcessor,
)
from ns_vfs.state import State


class VideotoAutomaton:
    def __init__(
        self,
        proposition_set: list[str],
        detector: ComputerVisionDetector | dict,
        video_processor: VideoProcessor,
        artifact_dir: str,
        ltl_formula: str,
        save_annotation: bool = False,
        save_image: bool = False,
        mapping_threshold: tuple = (0.36, 0.58),
        mapping_param_a=1.00,
        mapping_param_x0=0.58,
        mapping_param_k=0.50,
        verbose: bool = False,
    ) -> None:
        """Initialize Video to Automaton.

        Args:
            proposition_set (list): List of propositions.
            detector (ComputerVisionDetector): Computer Vision Detector.
            video_processor (VideoProcessor): Video Processor.
            artifact_dir (str): Path to artifact directory.
            ltl_formula (str): LTL formula.
            save_annotation (bool, optional): Whether to annotate the frame. Defaults to False.
            save_image (bool, optional): Whether to save image. Defaults to False.
        """
        self.ltl_formula = ltl_formula
        self.proposition_set = proposition_set
        self.proposition_combinations = self._create_proposition_status(
            len(proposition_set)
        )
        self._detector = detector
        self._artifact_dir = artifact_dir
        self._save_annotation = save_annotation
        self._save_image = save_image
        self._mapping_threshold = mapping_threshold
        self._mapping_param_a = mapping_param_a
        self._mapping_param_x0 = mapping_param_x0
        self._mapping_param_k = mapping_param_k
        self._verbose = verbose
        if isinstance(detector, dict):
            self._double_model_mode = True
        else:
            self._double_model_mode = False

        if self._save_image:
            self._frame_img_path = os.path.join(
                self._artifact_dir, "frame_images"
            )
            if os.path.exists(self._frame_img_path):
                # If it exists, remove it (and all its contents)
                shutil.rmtree(self._frame_img_path)
            # Then, create the directory again
            os.makedirs(self._frame_img_path)

        if self._save_annotation:
            self._annotated_frame_path = os.path.join(
                self._artifact_dir, "annotated_frame"
            )
            if os.path.exists(self._annotated_frame_path):
                # If it exists, remove it (and all its contents)
                shutil.rmtree(self._annotated_frame_path)
            # Then, create the directory again
            os.makedirs(self._annotated_frame_path)

    def _annotate_frame(
        self,
        frame_img: np.ndarray,
        output_dir: str,
    ) -> None:
        """Annotate frame with bounding box.

        Args:
            frame_img (np.ndarray): Frame image.
            output_dir (str | None, optional): Output directory. Defaults to None.
        """
        box_annotator = sv.BoxAnnotator()
        if self._detector.get_detections() is not None:
            annotated_frame = box_annotator.annotate(
                scene=frame_img.copy(),
                detections=self._detector.get_detections(),
                labels=self._detector.get_labels(),
            )

            sv.plot_image(annotated_frame, (16, 16))

            filename = get_file_or_dir_with_datetime("annotated_frame", ".png")
            plt.savefig(os.path.join(output_dir, filename))

            image = Image.open(os.path.join(output_dir, filename))
            return np.array(image)
        else:
            return frame_img

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

    def get_probabilistic_proposition_from_frame(
        self,
        proposition: str,
        frame_img: np.ndarray,
        save_annotation: bool = False,
    ) -> float:
        """Get probabilistic proposition from frame.

        Args:
            proposition (str): Proposition.
            frame_img (np.ndarray): Frame image.
            save_annotation (bool, optional): Whether to annotate the frame. Defaults to False.

        Returns:
            float: Probabilistic proposition from frame.
            detected_obj (any): Detected object.
        """
        # if proposition not in yolo class label
        if self._double_model_mode:
            if proposition not in LABEL_OF_INTEREST:
                detector = self._detector["clip"]
                # detected_obj = detector.detect(frame_img, [proposition])
            else:
                detector = self._detector["yolo"]
                # detected_obj = detector.detect(frame_img, [proposition])
        else:
            detector = self._detector

        detected_obj = detector.detect(frame_img, [proposition])
        if self._save_image:
            filename = get_file_or_dir_with_datetime("frame", ".png")
            img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            Image.fromarray(img).save(
                os.path.join(self._frame_img_path, filename)
            )

        if detector.get_size() > 0:
            if save_annotation:
                if self._double_model_mode:
                    annotated_img = None
                else:
                    annotated_img = self._annotate_frame(
                        frame_img=frame_img,
                        output_dir=self._annotated_frame_path,
                    )
                confidence_after_mapping = detector._mapping_probability(
                    confidence_per_video=np.round(
                        np.max(detector.get_confidence()), 2
                    )
                )
                return (
                    confidence_after_mapping,
                    detected_obj,
                    annotated_img,
                )
            else:
                confidence_after_mapping = detector._mapping_probability(
                    confidence_per_video=np.round(
                        np.max(detector.get_confidence()), 2
                    )
                )
                # # # DEBUG # # #
                return (
                    confidence_after_mapping,
                    detected_obj,
                    None,
                )
            # probability of the object in the frame
        else:
            return 0, None, None  # probability of the object in the frame is 0

    def calculate_confidence_of_proposition(
        self,
        proposition: str,
        frame_img: np.ndarray,
        save_annotation: bool = False,
    ) -> float:
        """Calculate confidence of proposition.

        Args:
            proposition (str): Proposition.
            frame_img (np.ndarray): Frame image.
            save_annotation (bool, optional): Whether to annotate the frame. Defaults to False.

        Returns:
            float: Confidence of proposition.
        """
        (
            propositional_probability_on_frame,
            detected_obj,
            annotated_img,
        ) = self.get_probabilistic_proposition_from_frame(
            proposition=proposition,
            frame_img=frame_img,
            save_annotation=save_annotation,
        )
        return propositional_probability_on_frame, detected_obj, annotated_img

    def build_automaton(
        self,
        frame_set: list[Frame],
        propositional_confidence: list[list[float]],
        include_initial_state: bool = True,
    ) -> (list, list):
        """Build automaton.

        Args:
            frame_set (list[Frame]): List of frames.
            propositional_confidence (list[list[float]]): List of propositional confidence.

        Returns:
            States: List of states.
            Transitions: List of transitions.
        """
        # Initialize required variables
        state_idx = 0
        states = list()
        prev_states = list()
        transitions = list()

        if include_initial_state:
            state = State(state_idx, -1, "init", self.proposition_set)
            states.append(copy.deepcopy(state))
            prev_states.append(copy.deepcopy(state))

        for i in range(len(frame_set)):
            current_state = list()
            for prop_comb in self.proposition_combinations:
                if len(states) == 0:
                    state = State(
                        index=0,
                        frame_index=i,
                        proposition_status_set=prop_comb,
                        proposition_set=self.proposition_set,
                    )
                else:
                    state.update(
                        frame_index=i,
                        proposition_combinations=prop_comb,
                    )
                state.compute_probability(
                    probabilities=propositional_confidence
                )

                if state.probability > 0:
                    if len(prev_states) == 0:
                        # prev_states.append(copy.deepcopy(state))
                        states.append(copy.deepcopy(state))
                        current_state.append(copy.deepcopy(state))
                        state.state_index += 1
                    else:
                        if include_initial_state:
                            state.state_index += 1
                        states.append(copy.deepcopy(state))
                        current_state.append(copy.deepcopy(state))
                        if not include_initial_state:
                            state.state_index += 1

            if len(prev_states) == 0:
                prev_states = current_state.copy()
                for prev_state in prev_states:
                    transition = (
                        prev_state.state_index,
                        prev_state.state_index,
                        0,
                    )
                    transitions.append(transition)
            else:
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

    def build_frame_automaton(self) -> dict:
        """Build frame window automaton."""
        # TODO: If we need to build frame automaton differently, we can do it here.
        video_frames_automaton = None
        return video_frames_automaton
