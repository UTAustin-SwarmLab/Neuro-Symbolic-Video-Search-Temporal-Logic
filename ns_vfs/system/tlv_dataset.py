from __future__ import annotations

import time

from omegaconf import DictConfig

from ns_vfs.automaton._base import Automaton
from ns_vfs.data.frame import Frame, FramesofInterest
from ns_vfs.model.vision.object_detection.dummy import DummyVisionModel
from ns_vfs.model_checker.stormpy import StormModelChecker
from ns_vfs.percepter._base import VisionPercepter
from ns_vfs.processor._base_video_processor import BaseVideoProcessor
from ns_vfs.processor.tlv_dataset.tlv_dataset_processor import (
    TLVDatasetProcessor,
)
from ns_vfs.system.node import Node
from ns_vfs.validator import FrameValidator


class NSVSNodeTLVDataset(Node):
    def __init__(
        self,
        video_processor: BaseVideoProcessor,
        vision_percepter: VisionPercepter,
        automaton: Automaton,
        ltl_formula: str,
        proposition_set: str,
        ns_vfs_system_cfg: DictConfig,
    ) -> None:
        self.video_processor = video_processor
        if isinstance(video_processor, TLVDatasetProcessor):
            self.ltl_formula = f"Pmin>=.80 [{video_processor.ltl_formula}]"
            self.proposition_set = video_processor.proposition_set
        else:
            self.ltl_formula = ltl_formula
            self.proposition_set = proposition_set
        self.vision_percepter = vision_percepter

        self.frame_validator = FrameValidator(ltl_formula=self.ltl_formula)
        self.automaton = automaton
        self.automaton.set_up(proposition_set=self.proposition_set)
        self.model_checker = StormModelChecker(
            proposition_set=self.proposition_set, ltl_formula=self.ltl_formula
        )
        self.frame_of_interest = FramesofInterest(ltl_formula=self.ltl_formula)

        self.ns_vfs_system_cfg = ns_vfs_system_cfg
        self.frame_idx = 0
        # Initialize latency measurement storage
        self.latency_measurement = {"yolo": [], "fv_ac": [], "mc": []}

    def start(self) -> None:
        while True:
            frame_img = self.video_processor.get_next_frame()
            if frame_img is None:
                break  # No more frames or end of video
            if isinstance(self.vision_percepter.cv_model, DummyVisionModel):
                ground_truth_object = (
                    self.video_processor.get_ground_truth_label(
                        frame_idx=self.frame_idx
                    )
                )
            else:
                ground_truth_object = None

            # Measure latency
            start_time_yolo = time.time()
            detected_objects: dict = self.vision_percepter.perceive(
                image=frame_img,
                object_of_interest=self.proposition_set,
                ground_truth_object=ground_truth_object,
            )
            end_time_yolo = time.time()
            latency_yolo = round((end_time_yolo - start_time_yolo) * 1000, 1)
            self.latency_measurement["yolo"].append(latency_yolo)
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
            start_time_fv_ac = time.time()
            if self.frame_validator.validate_frame(frame=frame):
                # 2. dynamic automaton construction
                self.automaton.add_frame_to_automaton(frame=frame)
                self.frame_of_interest.frame_buffer.append(frame)
                end_time_fv_ac = time.time()
                latency_fv_ac = round(
                    (end_time_fv_ac - start_time_fv_ac) * 1000, 1
                )
                self.latency_measurement["fv_ac"].append(latency_fv_ac)
                # 3. model checking
                start_time_mc = time.time()
                model_checking_result = self.model_checker.check_automaton(
                    transitions=self.automaton.transitions,
                    states=self.automaton.states,
                    verbose=self.ns_vfs_system_cfg.model_checker.verbose,
                    is_filter=self.ns_vfs_system_cfg.model_checker.is_filter,
                )
                end_time_mc = time.time()
                latency_mc = round((end_time_mc - start_time_mc) * 1000, 1)
                self.latency_measurement["mc"].append(latency_mc)

                if model_checking_result:
                    # specification satisfied
                    self.frame_of_interest.flush_frame_buffer()
                    self.automaton.reset()
            else:
                end_time_fv_ac = time.time()
                latency_fv_ac = round(
                    (end_time_fv_ac - start_time_fv_ac) * 1000, 1
                )
                self.latency_measurement["fv_ac"].append(latency_fv_ac)
                self.latency_measurement["mc"].append(0)

        print(self.ltl_formula)
        print(self.frame_of_interest.foi_list)
        print(self.video_processor.frames_of_interest)
        print(self.video_processor.labels_of_frames)
        print(self.latency_measurement)

    def stop(self) -> None:
        print("NSVS Node stopped")
