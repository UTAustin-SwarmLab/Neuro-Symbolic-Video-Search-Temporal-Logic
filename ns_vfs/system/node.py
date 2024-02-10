from __future__ import annotations

import abc

from ns_vfs.automaton.probabilistic import ProbabilisticAutomaton
from ns_vfs.data.frame import FramesofInterest
from ns_vfs.model_checker.stormpy import StormModelChecker
from ns_vfs.percepter._base import VisionPercepter
from ns_vfs.processor._base_video_processor import BaseVideoProcessor
from ns_vfs.processor.tlv_dataset.tlv_dataset_processor import (
    TLVDatasetProcessor,
)
from ns_vfs.validator import FrameValidator


class Node(abc.ABC):
    @abc.abstractmethod
    def start(self) -> None: ...

    def stop(self) -> None: ...


class NSVSNode(Node):
    def __init__(
        self,
        video_processor: BaseVideoProcessor,
        vision_percepter: VisionPercepter,
        ltl_formula: str,
        proposition_set: str,
    ) -> None:
        self.video_processor = video_processor
        if isinstance(video_processor, TLVDatasetProcessor):
            self.ltl_formula = video_processor.ltl_formula
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
        print("NSVS Node started")
        while True:
            frame_img = self.video_processor.get_next_frame()
            if frame_img is None:
                break  # No more frames or end of video

            detected_objects: list = self.vision_percepter.perceive(
                image=frame_img, object_of_interest=self.proposition_set
            )
            activity_of_interest = None
            breakpoint()

    def stop(self) -> None:
        print("NSVS Node stopped")
