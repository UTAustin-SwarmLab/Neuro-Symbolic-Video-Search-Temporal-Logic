from __future__ import annotations

import copy

from omegaconf import DictConfig
from swarm_cv.image.annotator.bounding_box import visualize_bbox

# from PIL import ImageFilter
from swarm_cv.image.annotator.text_annotator import TextAnnotator
from swarm_cv.image.filter.image_filter import ImageFilter

from ns_vfs.automaton._base import Automaton
from ns_vfs.data.frame import Frame, FramesofInterest
from ns_vfs.percepter._base import VisionPercepter
from ns_vfs.processor._base_video_processor import BaseVideoProcessor
from ns_vfs.system.node import Node


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
        self.cvs_cfg = ns_vfs_system_cfg.constrained_video_streaming
        self.text_annotator = TextAnnotator()
        self.info = {}
        self.frame_idx = 0

    def recursive_model_checking(self, automaton, frame, filter_type):
        automaton_for_eval = copy.deepcopy(automaton)
        automaton_for_eval.add_frame_to_automaton(frame=frame)
        model_checking_result = automaton_for_eval.check_automaton()
        if not model_checking_result:
            # Perturb the image
            perturbed_image = ImageFilter().apply_filter_to_bbox(
                image=frame.frame_image,
                bboxes=frame.detected_bboxes,
                blur_radius=int(
                    max(frame.frame_image.shape[0], frame.frame_image.shape[1])
                    / 40
                ),
                filter_type=filter_type,
                bbox_border_thickness=10,
                bbox_border_color="#FF0000",
            )

            frame_img = perturbed_image

            # Run CV model with the perturbed image
            detected_objects: dict = self.vision_percepter.perceive(
                image=frame_img,
                object_of_interest=self.proposition_set,
            )

            # Recreate the frame
            frame = Frame(
                frame_idx=self.frame_idx,
                timestamp=self.video_processor.current_frame_index,
                frame_image=frame_img,
                object_of_interest=detected_objects,
                activity_of_interest=None,
            )
            automaton_for_eval = copy.deepcopy(automaton)
            automaton_for_eval.add_frame_to_automaton(frame=frame)
            model_checking_result = automaton_for_eval.check_automaton()
            if not model_checking_result:
                self.recursive_model_checking(automaton, frame, filter_type)
            else:
                automaton.add_frame_to_automaton(frame=frame)
                return automaton

    def start(self) -> None:
        while True:
            info = {}
            frame_img = self.video_processor.get_next_frame()
            if frame_img is None:
                break  # No more frames or end of video

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

            automaton_for_eval = copy.deepcopy(self.automaton)
            automaton_for_eval.add_frame_to_automaton(frame=frame)

            # Check and eval
            model_checking_result = automaton_for_eval.check_automaton()
            # if a violation is detected..
            if not model_checking_result:
                info["violation"] = True
                # image perturbation
                if self.cvs_cfg.property_constrained:
                    # >>> Need to get the bounding boxes caused the violation >>> #
                    # TODO: Need to get the bounding boxes caused the violation
                    info["detected_obj_without_perturbation"] = (
                        frame.detected_object_dict
                    )
                    info["probability_of_safety_before_perturbation"] = (
                        automaton_for_eval.probability_of_safety
                    )
                    # Perturb the image
                    perturbed_image = ImageFilter().apply_filter_to_bbox(
                        image=frame_img,
                        bboxes=frame.detected_bboxes,
                        blur_radius=int(
                            max(frame_img.shape[0], frame_img.shape[1]) / 40
                        ),
                        filter_type=self.cvs_cfg.filter_type,
                        bbox_border_thickness=10,
                        bbox_border_color="#FF0000",
                    )

                    frame_img = perturbed_image

                    # Run CV model with the perturbed image
                    detected_objects: dict = self.vision_percepter.perceive(
                        image=frame_img,
                        object_of_interest=self.proposition_set,
                    )

                    # Recreate the frame
                    frame = Frame(
                        frame_idx=self.frame_idx,
                        timestamp=self.video_processor.current_frame_index,
                        frame_image=frame_img,
                        object_of_interest=detected_objects,
                        activity_of_interest=None,
                    )
                    self.automaton.add_frame_to_automaton(frame=frame)

                    info["detected_obj_with_perturbation"] = (
                        frame.detected_object_dict
                    )
                    info["probability_of_safety_after_perturbation"] = (
                        self.automaton.probability_of_safety
                    )

                    # Get bboxes of detected objects after perturbation
                    frame_img = visualize_bbox(
                        image=frame_img,
                        bboxes=frame.detected_bboxes,
                        bbox_border_thickness=5,
                        bbox_border_color="#ee00ff",
                    )
                    # <<< DEV LOGGING <<< #
                    text_annotations = [
                        "UNSAFE: Violation detected -- perturbed image",
                        f"Probability of safety: {self.automaton.probability_of_safety}",
                        f"Probability of unsafety: {self.automaton.probability_of_unsafety}",
                    ]
                else:
                    # Don't perturb the image
                    self.automaton.add_frame_to_automaton(frame=frame)
                    text_annotations = [
                        "UNSAFE: Violation detected -- no image perturbation",
                        f"Probability of safety: {self.automaton.probability_of_safety}",
                        f"Probability of unsafety: {self.automaton.probability_of_unsafety}",
                    ]

                self.text_annotator.insert_annotation(
                    image=frame_img,
                    text_annotations=text_annotations,
                    position="upper_right",
                    save_path=f"/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/sample/frame_{self.frame_idx}.png",
                )
            # Violation not detected
            else:
                info["violation"] = False
                self.automaton = automaton_for_eval
                # Insert text annotation
                self.text_annotator.insert_annotation(
                    image=frame_img,
                    text_annotations=[
                        "SAFE: Violation is not detected",
                        f"Probability of safety: {self.automaton.probability_of_safety}",
                        f"Probability of unsafety: {self.automaton.probability_of_unsafety}",
                    ],
                    position="upper_right",
                    save_path=f"/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/sample/frame_{self.frame_idx}.png",
                )
            self.frame_of_interest.frame_buffer.append(frame)
            self.info[self.frame_idx] = info
            self.frame_idx += 1

        self.frame_of_interest.flush_frame_buffer()
        # save result
        if self.ns_vfs_system_cfg.save_result_dir:
            self.frame_of_interest.save(
                path=self.ns_vfs_system_cfg.save_result_dir
            )

        print(self.frame_of_interest.foi_list)
        print(self.info)

    def stop(self) -> None:
        print("NSVS Node stopped")
