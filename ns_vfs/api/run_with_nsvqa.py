from __future__ import annotations

import numpy as np
import json
from cvias.image.detection.vllm_detection import VLLMDetection

from ns_vfs.automaton.video_automaton import VideoAutomaton
from ns_vfs.data.frame import FramesofInterest, VideoFrame
from ns_vfs.model_checking.stormpy import StormModelChecker
from ns_vfs.percepter.single_vision_percepter import SingleVisionPercepter
from ns_vfs.validator import FrameValidator
from ns_vfs.dataloader.longvideobench import LongVideoBench


def run_nsvs_nsvqa(
    nsvqa_input_data: list[dict[str, list[np.ndarray] | None]],
    proposition_set: list[str],
    ltl_formula: str,
    output_path: str,
    api_key="EMPTY",
    api_base="http://localhost:8000/v1",
    model="OpenGVLab/InternVL2_5-8B",
    threshold_satisfaction_probability: float = 0.80,
    frame_scale: int | None = None,
    calibration_method: str = "temperature_scaling",
    desired_interval_in_sec: float | None = None,
    desired_fps: int | None = None,
    custom_prompt: str | None = None,
) -> None:
    # Yolo model initialization
    vllm_model = VLLMDetection(
        api_key=api_key,
        api_base=api_base,
        model=model,
        calibration_method=calibration_method,
    )
    # Video automaton initialization
    ltl_formula = f"P>={threshold_satisfaction_probability} [{ltl_formula}]"
    automaton = VideoAutomaton()
    automaton.set_up(proposition_set=proposition_set)
    # Model checker initialization
    model_checker = StormModelChecker(
        proposition_set=proposition_set, ltl_formula=ltl_formula
    )
    # Frame of interest initialization
    frame_of_interest = FramesofInterest(ltl_formula=ltl_formula)
    # Video processor initialization

    # Vision percepter initialization
    vision_percepter = SingleVisionPercepter(
        cv_models=vllm_model,
    )

    frame_validator = FrameValidator(ltl_formula=ltl_formula)
    frame_idx = 0
    model_checker_is_filter: bool = (False,)
    model_checker_type: str = ("sparse_ma",)
    for nsvqa_input in nsvqa_input_data:
        sequence_of_frames = nsvqa_input["frames"]
        detected_objects: dict = vision_percepter.perceive(
            image=sequence_of_frames,
            object_of_interest=proposition_set,
            extra_description_of_object=nsvqa_input["subtitle"],
        )
        activity_of_interest = None

        frame = VideoFrame(
            frame_idx=frame_idx,
            timestamp=frame_idx,
            frame_image=sequence_of_frames,
            object_of_interest=detected_objects,
            activity_of_interest=activity_of_interest,
        )
        frame_idx += 1

        # 1. frame validation
        if frame_validator.validate_frame(frame=frame):
            # 2. dynamic automaton construction
            automaton.add_frame(frame=frame)
            frame_of_interest.frame_buffer.append(frame)
            # 3. model checking
            model_checking_result = model_checker.check_automaton(
                transitions=automaton.transitions,
                states=automaton.states,
                model_type=model_checker_type,
                use_filter=model_checker_is_filter,
            )
            if model_checking_result:
                # specification satisfied
                frame_of_interest.flush_frame_buffer()
                automaton.reset()

    print("--------------------------------")
    print("Detected frames of interest:")
    print(frame_of_interest.foi_list)
    # save result
    if output_path:
        frame_of_interest.save(path=output_path)
        print(f"\nResults saved in {output_path}")

    return frame_of_interest.foi_list


if __name__ == "__main__":
    input_data_path = "/nas/mars/experiment_result/nsvqa/1_puls/longvideobench/longvideobench-outputs-updated.json"
    with open(input_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for sample in data:
        loader = LongVideoBench(sample["video_path"], sample["subtitle_path"])
        nsvqa_input = loader.load_all()
        extracted = sample["video_path"].split('/')[-1].split('.')[0]

        run_nsvs_nsvqa(
            nsvqa_input_data=nsvqa_input,
            desired_interval_in_sec=None,
            desired_fps=30,
            proposition_set=sample["proposition"],
            ltl_formula=sample["specification"],
            output_path=f"/nas/mars/experiment_result/nsvqa/2_nsvs/longvideobench/{extracted}/",
            threshold_satisfaction_probability=0.80,
            frame_scale=None,
            calibration_method="temperature_scaling",
        )
