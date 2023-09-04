from __future__ import annotations

import copy

import numpy as np
import supervision as sv

from ns_vfs.data.frame import Frame
from ns_vfs.model.vision._base import ComputerVisionDetector
from ns_vfs.processor.video_processor import (
    VideoProcessor,
)
from ns_vfs.state import State


class VideotoAutomaton:
    def __init__(
        self,
        proposition_set: list,
        detector: ComputerVisionDetector,
        video_processor: VideoProcessor,
    ):
        self.proposition_set = proposition_set
        self.proposition_status_combinations = self._create_proposition_status(
            len(proposition_set)
        )
        self._detector = detector
        self._video_processor = video_processor

    def _sigmoid(self, x, k=1, x0=0):
        return 1 / (1 + np.exp(-k * (x - x0)))

    def _annotate_frame(
        frame_img: np.ndarray, proposition: list, detected_obj: any
    ):
        box_annotator = sv.BoxAnnotator()
        labels = [
            f"{proposition[class_id] if class_id is not None else None} {confidence:0.2f}"
            for _, _, confidence, class_id, _ in detected_obj
        ]

        annotated_frame = box_annotator.annotate(
            scene=frame_img.copy(), detections=detected_obj, labels=labels
        )

        sv.plot_image(annotated_frame, (16, 16))

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
    ):
        if confidence_per_video >= true_threshold:
            return 1
        elif confidence_per_video <= false_threshold:
            return 0
        else:
            return round(self._sigmoid(confidence_per_video, k=50, x0=0.56), 2)

    def get_probabilistic_trajectory(self, propositional_confidence_map: list):
        probability = list()
        for conf_over_video in propositional_confidence_map:
            probability.append(
                self._mapping_probability(confidence_per_video=conf_over_video)
            )
        return probability

    def get_probabilistic_proposition_from_frame(
        self,
        proposition: str,
        frame_img: np.ndarray,
        is_annotation: bool = True,
    ):
        detected_obj = self._detector.detect(frame_img, [proposition])
        if len(detected_obj) > 0:
            if is_annotation:
                self._annotate_frame(
                    frame_img=frame_img,
                    detected_obj=detected_obj,
                    proposition=[proposition],
                    is_annotation=is_annotation,
                )
            # TODO: mapping here
            return self._mapping_probability(
                np.round(np.max(detected_obj.confidence), 2)
            )
            # probability of the object in the frame
        else:
            return 0  # probability of the object in the frame is 0

    def calculate_confidence_of_proposition(
        self,
        proposition: str,
        frame_img: np.ndarray,
        is_annotation: bool = False,
    ):
        propositional_probability_on_frame = (
            self.get_probabilistic_proposition_from_frame(
                proposition=proposition,
                frame_img=frame_img,
                is_annotation=is_annotation,
            )
        )

        return propositional_probability_on_frame

    def get_probabilistic_confidence_from_frame(
        self,
        proposition: str,
        video_frames: np.ndarray | dict,
        is_annotation: bool,
    ):
        trajectories = list()
        if isinstance(video_frames, dict):
            # Frames are in sliding window format
            for idx in range(len(video_frames.keys()) - 1):
                image_set = video_frames[idx].frame_image_set
                for i, frame in enumerate(image_set):
                    frame: Frame  # TODO: assert?
                    propositional_probability_on_frame = (
                        self.get_probabilistic_proposition_from_frame(
                            proposition=proposition,
                            frame_img=frame.frame_image,
                            is_annotation=is_annotation,
                        )
                    )
                    frame.propositional_probability[
                        str(proposition)
                    ] = propositional_probability_on_frame
                    image_set[i] = frame
                video_frames[idx].frame_image_set = image_set
            return video_frames
        else:
            for video_frame in video_frames:
                propositional_probability_on_frame = (
                    self.get_probabilistic_proposition_from_frame(
                        proposition=proposition,
                        frame_img=video_frame,
                        is_annotation=is_annotation,
                    )
                )
                trajectories.append(propositional_probability_on_frame)
            return trajectories

    def get_probability_of_trajectory(self, is_annotation: bool):
        video_frames = self._video_processor.get_synchronous_frame(
            proposition_set=self.proposition_set,
            calculate_propositional_confidence=self.calculate_confidence_of_proposition,
        )
        return video_frames

        # video_frames = self._video_processor.get_frame_by_sliding_window()
        # for proposition in self.proposition_set:
        #     video_frames = self.get_probabilistic_confidence_from_frame(
        #         proposition=proposition,
        #         video_frames=video_frames,
        #         is_annotation=is_annotation,
        #     )
        # return video_frames

        # video_frames = self._video_processor.get_frame()

    def buil

    def _build_state(self, video_traj_probability: list):
        states = list()
        state_idx = 0
        state = State(state_idx, -1, "initial", self.proposition_set)
        states.append(copy.deepcopy(state))
        # idx = 1
        for frame_idx in range(self._video_processor.number_of_frames):
            for proposition_status in self.proposition_status_combinations:
                state.update(
                    frame_index=frame_idx,
                    proposition_status_set=proposition_status,
                )
                state.compute_probability(probabilities=video_traj_probability)
                print(state)
                if state.probability > 0:
                    state.state_index += 1
                    # idx += 1
                    states.append(copy.deepcopy(state))
        return states

    def _build_transition(self, states: list):
        transitions = list()
        prev_states = [State(0, -1, "initial", self.proposition_set)]

        for frame_idx in range(self._video_processor.number_of_frames):
            current_states = list()
            for state in states:
                if state.frame_index == frame_idx:
                    current_states.append(state)
            for curr_state in current_states:
                for prev_state in prev_states:
                    transitions.append(
                        (
                            prev_state.state_index,
                            curr_state.state_index,
                            curr_state.probability,
                        )
                    )
            prev_states = current_states.copy()
        return transitions, prev_states

    def build_automaton(self, is_annotation: bool):
        video_traj_probability = self.get_probability_of_trajectory(
            is_annotation=is_annotation
        )

        states = self._build_state(
            video_traj_probability=video_traj_probability
        )
        transitions, prev_states = self._build_transition(states=states)
        return states, transitions, prev_states
