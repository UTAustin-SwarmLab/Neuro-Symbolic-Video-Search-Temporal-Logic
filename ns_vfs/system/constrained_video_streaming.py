from __future__ import annotations

import copy

import numpy as np
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFilter
from swarm_cv.image.annotator.text_annotator import TextAnnotator
from swarm_cv.image.filter.image_filter import ImageFilter

from ns_vfs.automaton._base import Automaton
from ns_vfs.data.frame import Frame, FramesofInterest
from ns_vfs.percepter._base import VisionPercepter
from ns_vfs.processor._base_video_processor import BaseVideoProcessor
from ns_vfs.system.node import Node

IS_BLUR = True

class ConstrainedVideoStreaming(Node):
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
        self.ltl_formula = ltl_formula
        self.proposition_set = proposition_set
        self.vision_percepter = vision_percepter
        self.automaton = automaton
        self.automaton.set_up(proposition_set=self.proposition_set)
        self.frame_of_interest = FramesofInterest(ltl_formula=self.ltl_formula)
        self.ns_vfs_system_cfg = ns_vfs_system_cfg
        self.text_annotator = TextAnnotator()
        self.frame_idx = 0

    def start(self) -> None:
        image_filter = ImageFilter()
        while True:
            frame_img = self.video_processor.get_next_frame()
            if frame_img is None:
                break  # No more frames or end of video
            detected_objects: dict = self.vision_percepter.perceive(
                image=frame_img,
                object_of_interest=self.proposition_set,
            )
            # # # >>> DEV >>> # # #
            data_processed = []
            bboxes_to_blur = []
            results = detected_objects["person"].all_obj_detected
            for row in results[0].boxes.data.cpu().numpy():
                bbox = row[:4].tolist()
                score = float(row[4])
                class_id = int(row[5])
                bboxes_to_blur.append(bbox)
                data_processed.append(
                    {
                        "bbox": bbox,
                        "conf": score,
                        "class_id": class_id,
                    }
                )
            blurred = image_filter.apply_filter_to_bbox(
                image=frame_img,
                bboxes=bboxes_to_blur,
                blur_radius=int(max(frame_img.shape[0], frame_img.shape[1])/40)
            )
            blurred.save(
                "/opt/Neuro-Symbolic-Video-Frame-Search/yolo_test_image_blurred.jpg"
            )
            np_blurred_image = np.array(blurred)
            # # # <<< DEV <<< # # #
            frame = Frame(
                frame_idx=self.frame_idx,
                timestamp=self.video_processor.current_frame_index,
                frame_image=frame_img,
                object_of_interest=detected_objects,
                activity_of_interest=None,
            )

            automaton_to_eval = copy.deepcopy(self.automaton)
            automaton_to_eval.add_frame_to_automaton(frame=frame)

            # Check and eval
            model_checking_result = automaton_to_eval.check_automaton()
            if not model_checking_result:
                # Blur the frame or do actions
                if IS_BLUR:
                    frame_img = np_blurred_image
                else:
                    frame_img = frame_img
                detected_objects: dict = self.vision_percepter.perceive(
                    image=frame_img,
                    object_of_interest=self.proposition_set,
                )
                frame = Frame(
                    frame_idx=self.frame_idx,
                    timestamp=self.video_processor.current_frame_index,
                    frame_image=frame_img,
                    object_of_interest=detected_objects,
                    activity_of_interest=None,
                )
                self.automaton.add_frame_to_automaton(frame=frame)
                # Insert text annotation
                if IS_BLUR:
                    text_annotations = [
                        "MODIFIED: Frame is blurred",
                        f"Probability of safety: {self.automaton.probability_of_safety}",
                        f"Probability of unsafety: {self.automaton.probability_of_unsafety}",
                    ]
                else:
                    text_annotations = [
                        f"Probability of safety: {self.automaton.probability_of_safety}",
                        f"Probability of unsafety: {self.automaton.probability_of_unsafety}",
                    ]
                self.text_annotator.insert_annotation(
                    image=frame_img,
                    text_annotations=text_annotations,
                    position="upper_right",
                    save_path=f"/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/sample/frame_{self.frame_idx}.png",
                )
            else:
                self.automaton = automaton_to_eval
                # Insert text annotation
                self.text_annotator.insert_annotation(
                    image=frame_img,
                    text_annotations=[
                        f"Probability of safety: {self.automaton.probability_of_safety}",
                        f"Probability of unsafety: {self.automaton.probability_of_unsafety}",
                    ],
                    position="upper_right",
                    save_path=f"/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/sample/frame_{self.frame_idx}.png",
                )
            self.frame_of_interest.frame_buffer.append(frame)
            self.frame_idx += 1

        self.frame_of_interest.flush_frame_buffer()
        # save result
        if self.ns_vfs_system_cfg.save_result_dir:
            self.frame_of_interest.save(
                path=self.ns_vfs_system_cfg.save_result_dir
            )
        print(self.frame_of_interest.foi_list)

    def stop(self) -> None:
        print("NSVS Node stopped")
