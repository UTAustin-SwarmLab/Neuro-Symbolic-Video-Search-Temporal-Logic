from __future__ import annotations

from cvias.image.detection.vllm_detection import VLLMDetection

from ns_vfs.api.run_nsvs import run_nsvs
from ns_vfs.automaton.video_automaton import VideoAutomaton
from ns_vfs.data.frame import FramesofInterest
from ns_vfs.model_checking.stormpy import StormModelChecker
from ns_vfs.percepter.single_vision_percepter import SingleVisionPercepter


def run_nsvs_vlm(
    video_path: str,
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
    run_nsvs(
        video_path=video_path,
        vision_percepter=vision_percepter,
        automaton=automaton,
        frame_of_interest=frame_of_interest,
        model_checker=model_checker,
        proposition_set=proposition_set,
        ltl_formula=ltl_formula,
        output_path=output_path,
        desired_interval_in_sec=desired_interval_in_sec,
        desired_fps=desired_fps,
    )


if __name__ == "__main__":
    run_nsvs_vlm(
        video_path="/home/mc76728/repo/Coargus/Neuro-Symbolic-Video-Search-Temporal-Logic/sample_data/example_video.mp4",
        desired_interval_in_sec=None,
        desired_fps=30,
        proposition_set=["car", "truck"],
        ltl_formula='"car" U "truck"',
        output_path="/home/mc76728/repo/Coargus/Neuro-Symbolic-Video-Search-Temporal-Logic/_dev_",
        threshold_satisfaction_probability=0.80,
        frame_scale=None,
        calibration_method="temperature_scaling",
    )
