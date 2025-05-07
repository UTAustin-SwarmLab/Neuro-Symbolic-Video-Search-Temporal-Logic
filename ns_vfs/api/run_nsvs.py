from __future__ import annotations

import vflow

from ns_vfs.automaton.video_automaton import VideoAutomaton
from ns_vfs.data.frame import FramesofInterest, VideoFrame
from ns_vfs.model_checking.stormpy import StormModelChecker
from ns_vfs.percepter._base import VisionPercepter
from ns_vfs.validator import FrameValidator


def run_nsvs(
    video_path: str,
    vision_percepter: VisionPercepter,
    automaton: VideoAutomaton,
    frame_of_interest: FramesofInterest,
    model_checker: StormModelChecker,
    proposition_set: list[str],
    ltl_formula: str,
    output_path: str,
    model_checker_is_filter: bool = False,
    model_checker_type: str = "sparse_ma",
    desired_interval_in_sec: float | None = None,
    desired_fps: int | None = None,
    **kwargs,
) -> list:
    frame_validator = FrameValidator(ltl_formula=ltl_formula)
    video_processor = vflow.read_video(video_path)
    frame_idx = 0
    while True:
        frame_img = video_processor.get_next_frame(
            return_format="ndarray",
            desired_interval_in_sec=desired_interval_in_sec,
            desired_fps=desired_fps,
        )
        if frame_img is None:
            break  # No more frames or end of video
        detected_objects: dict = vision_percepter.perceive(
            image=frame_img,
            object_of_interest=proposition_set,
        )
        activity_of_interest = None

        frame = VideoFrame(
            frame_idx=frame_idx,
            timestamp=video_processor.current_frame_index,
            frame_image=frame_img,
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
