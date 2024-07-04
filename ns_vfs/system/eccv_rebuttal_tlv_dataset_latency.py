from __future__ import annotations
import pickle
import time
import sys
from pathlib import Path
from PIL import Image, ImageDraw

from omegaconf import DictConfig

from ns_vfs.automaton._base import Automaton
from ns_vfs.data.frame import Frame, FramesofInterest
from ns_vfs.model.vision.object_detection.dummy import DummyVisionModel
from ns_vfs.model.vision.activity_recognition.videomae import VideoMAE
from ns_vfs.model_checker.stormpy import StormModelChecker
from ns_vfs.percepter._base import VisionPercepter
from ns_vfs.percepter.activity_percepter import ActivityPercepter
from ns_vfs.processor._base_video_processor import BaseVideoProcessor
from ns_vfs.processor.tlv_dataset.tlv_dataset_processor import (
    TLVDatasetProcessor,
)
from ns_vfs.processor.video.real_video import RealVideoProcessor
from ns_vfs.system.node import Node
from ns_vfs.validator import FrameValidator

ACTIVITY_FRAME = 16


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
        self.activity_percepter = ActivityPercepter(VideoMAE())

        self.automaton = automaton

        self.ns_vfs_system_cfg = ns_vfs_system_cfg
        self.frame_idx = 0
        queue = []
        # Initialize latency measurement storage
        self.latency_measurement = {"yolo": [], "fv_ac": [], "mc": []}

    def start(self) -> None:
        data_path = Path(
            "/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_rebuttal/_validated_benchmark_video/coco/(prop1&prop2)Uprop3"
        )
        # List all .pkl files in the directory
        pkl_files = list(data_path.glob("*.pkl"))

        # Iterate through the list and read each .pkl file
        # for pkl_file in pkl_files:
        for i in range(1):
            # pkl_file = str(pkl_file)
            # try:
            # self.video_processor = TLVDatasetProcessor(pkl_file)
            self.video_processor = RealVideoProcessor(
                "/opt/Neuro-Symbolic-Video-Frame-Search/action_testing/-00nar1nEPc_000033_000043.mp4",
                frame_duration_sec=0.5,
            )

            self.ltl_formula = 'Pmin>=.80 [F "person" & "playing_harmonica"]'
            self.proposition_set = ["person", "playing_harmonica"]
            self.frame_validator = FrameValidator(ltl_formula=self.ltl_formula)

            self.automaton.set_up(proposition_set=self.proposition_set)
            self.automaton.reset()

            self.model_checker = StormModelChecker(
                proposition_set=self.proposition_set,
                ltl_formula=self.ltl_formula,
            )
            self.frame_of_interest = FramesofInterest(
                ltl_formula=self.ltl_formula
            )
            i = 0
            queue = []
            while True:
                frame_img = self.video_processor.get_next_frame()
                if frame_img is None:
                    break  # No more frames or end of video
                print(f"FRAME: {frame_img.shape}")
                queue.append(frame_img)
                if len(queue) < ACTIVITY_FRAME:
                    continue

                if isinstance(self.vision_percepter.cv_model, DummyVisionModel):
                    ground_truth_object = (
                        self.video_processor.get_ground_truth_label(
                            frame_idx=self.frame_idx
                        )
                    )
                else:
                    ground_truth_object = None
                print(type(self.vision_percepter.cv_model))
                print(list(self.proposition_set))

                # Measure latency
                start_time_yolo = time.time()
                detected_objects: dict = self.vision_percepter.perceive(
                    image=frame_img,
                    object_of_interest=[list(self.proposition_set)[0]],
                    ground_truth_object=ground_truth_object,
                )
                print(detected_objects)
                end_time_yolo = time.time()
                latency_yolo = round(
                    (end_time_yolo - start_time_yolo) * 1000, 1
                )
                self.latency_measurement["yolo"].append(latency_yolo)

                # output boudning boxes!
                boxes = detected_objects["person"].bounding_box_of_all_obj
                new_box = []
                for box in boxes:
                    rounded_boxes = [round(coord) for coord in box]
                    new_box.append(rounded_boxes)

                image = Image.fromarray(frame_img)
                draw = ImageDraw.Draw(image)
                for box in new_box:
                    draw.rectangle(
                        [(box[0], box[1]), (box[2], box[3])],
                        outline="red",
                        width=2,
                    )

                image.save(f"{i}.png")
                i += 1

                activity_of_interest = self.activity_percepter.perceive(
                    images=queue,
                    object_of_interest=[list(self.proposition_set)[1]],
                )
                detected_objects[list(self.proposition_set)[1]] = (
                    activity_of_interest
                )

                frame = Frame(
                    frame_idx=self.frame_idx,
                    timestamp=self.video_processor.current_frame_index,
                    frame_image=frame_img,
                    object_of_interest=detected_objects,
                    activity_of_interest=activity_of_interest,
                )
                self.frame_idx += 1
                queue.pop(0)

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

            print("\n\n\n\n\n")
            print(self.ltl_formula)
            print(self.frame_of_interest.foi_list)
            # print(self.video_processor.frames_of_interest)
            # print(self.video_processor.labels_of_frames)
            print(self.latency_measurement)

            sys.exit(0)

            # Data Cleaning
            import uuid

            num_frame = pkl_file.split('"_')[-1].split("_")[0]
            file_dir = Path(
                "/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_rebuttal/latency_experiment_results"
            )
            file_name = (
                f"latency_experiment_{num_frame}_{uuid.uuid4().hex[:4]}.pkl"
            )
            file_path = file_dir / file_name
            data = {
                "yolo": self.latency_measurement["yolo"][1:],
                "fv_ac": self.latency_measurement["fv_ac"][1:],
                "mc": self.latency_measurement["mc"][1:],
            }
            with open(file_path, "wb") as f:
                pickle.dump(data, f)

            # except pickle.UnpicklingError as e:
            #     print("data corrupted.")
            #     continue

    def stop(self) -> None:
        print("NSVS Node stopped")
