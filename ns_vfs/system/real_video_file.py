from __future__ import annotations

from ns_vfs.automaton.probabilistic import ProbabilisticAutomaton
from ns_vfs.data.frame import Frame, FramesofInterest
from ns_vfs.model_checker.stormpy import StormModelChecker
from ns_vfs.percepter._base import VisionPercepter
from ns_vfs.processor._base_video_processor import BaseVideoProcessor
from ns_vfs.processor.tlv_dataset.tlv_dataset_processor import (
    TLVDatasetProcessor,
)
from ns_vfs.system.node import Node
from ns_vfs.validator import FrameValidator


class NSVSNodeRealVideoFile(Node):
    def __init__(
        self,
        video_processor: BaseVideoProcessor,
        vision_percepter: VisionPercepter,
        ltl_formula: str,
        proposition_set: str,
    ) -> None:
        self.video_processor = video_processor
        if isinstance(video_processor, TLVDatasetProcessor):
            self.ltl_formula = f"P>=0.80 [{video_processor.ltl_formula}]"
            self.proposition_set = video_processor.proposition_set
        else:
            self.ltl_formula = ltl_formula
            self.proposition_set = proposition_set
        self.vision_percepter = vision_percepter

        self.frame_validator = FrameValidator(ltl_formula=self.ltl_formula)
        self.automaton = ProbabilisticAutomaton(
            include_initial_state=False, proposition_set=self.proposition_set
        )
        self.model_checker = StormModelChecker(
            proposition_set=self.proposition_set, ltl_formula=self.ltl_formula
        )
        self.frame_of_interest = FramesofInterest(ltl_formula=self.ltl_formula)

        self.frame_idx = 0

    def start(self) -> None:
        while True:
            frame_img = self.video_processor.get_next_frame()
            if frame_img is None:
                break  # No more frames or end of video
            ground_truth_object = self.video_processor.get_ground_truth_label(
                frame_idx=self.frame_idx
            )

            detected_objects: list = self.vision_percepter.perceive(
                image=frame_img,
                object_of_interest=self.proposition_set,
                ground_truth_object=ground_truth_object,
            )
            activity_of_interest = None

            frame = Frame(
                frame_idx=self.frame_idx,
                timestamp=self.video_processor.current_frame_index,
                frame_image=frame_img,
                object_of_interest=detected_objects,
                activity_of_interest=activity_of_interest,
            )
            self.frame_idx += 1

            # 1. frame validation
            if self.frame_validator.validate_frame(frame=frame):
                # 2. dynamic automaton construction
                self.automaton.add_frame_to_automaton(frame=frame)
                self.frame_of_interest.frame_buffer.append(frame)
                # 3. model checking
                model_checking_result = self.model_checker.check_automaton(
                    transitions=self.automaton.transitions,
                    states=self.automaton.states,
                    verbose=False,
                    is_filter=False,
                )
                if model_checking_result:
                    # specification satisfied
                    self.frame_of_interest.flush_frame_buffer()
                    self.automaton.reset()

        print(self.frame_of_interest.foi_list)
        print(self.video_processor.frames_of_interest)
        print(self.video_processor.labels_of_frames)

    def stop(self) -> None:
        print("NSVS Node stopped")
