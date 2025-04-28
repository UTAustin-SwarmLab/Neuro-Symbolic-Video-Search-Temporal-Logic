from __future__ import annotations

from cvias.image.detection.object.yolo import Yolo

from ns_vfs.api.run_nsvs import run_nsvs
from ns_vfs.automaton.video_automaton import VideoAutomaton
from ns_vfs.data.frame import FramesofInterest
from ns_vfs.model_checking.stormpy import StormModelChecker
from ns_vfs.percepter.single_vision_percepter import SingleVisionPercepter


def run_nsvs_yolo(
    video_path: str,
    proposition_set: list[str],
    ltl_formula: str,
    output_path: str,
    yolo_model_name: str,
    yolo_model_path: str | None = None,
    gpu_number: int = 0,
    threshold_satisfaction_probability: float = 0.80,
    frame_scale: int | None = None,
    calibration_method: str = "temperature_scaling",
    desired_interval_in_sec: float | None = None,
    desired_fps: int | None = None,
) -> None:
    # Yolo model initialization
    yolo_model = Yolo(
        model_name="YOLOv8x",
        explicit_checkpoint_path=yolo_model_path,
        gpu_number=gpu_number,
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
        cv_models=yolo_model,
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
