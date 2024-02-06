from __future__ import annotations

import cv2

from ns_vfs.automaton.probabilistic import ProbabilisticAutomaton
from ns_vfs.config.loader import load_config
from ns_vfs.data.frame import Frame, FramesofInterest
from ns_vfs.model.vision.object_detection.clip_model import ClipPerception
from ns_vfs.model.vision.object_detection.yolo import Yolo
from ns_vfs.model_checker.stormpy import StormModelChecker
from ns_vfs.percepter.multi_vision_percepter import MultiVisionPercepter
from ns_vfs.processor.video.real_video import RealVideoProcessor
from ns_vfs.validator import FrameValidator


def main():
    # Given Propositions
    # LTL Formula
    config = load_config()
    video_path = "/opt/Neuro-Symbolic-Video-Frame-Search/sample_data/titanic_scene.mp4"  # Replace with your video path
    ltl_formula = 'P>=0.80 [F "person"]'
    proposition_set = ["person", "apple"]
    # Initialize the video processor
    processor = RealVideoProcessor(video_path, frame_duration_sec=1, frame_scale=2)
    cv_model = {
        "yolo": Yolo(
            weight_path=config.YOLO.YOLO_CHECKPOINT_PATH,
        ),
        "clip": ClipPerception(config=config),
    }
    percepter = MultiVisionPercepter(cv_models=cv_model)
    frame_validator = FrameValidator(ltl_formula=ltl_formula)
    automaton = ProbabilisticAutomaton(
        include_initial_state=False, proposition_set=proposition_set
    )
    model_checker = StormModelChecker(
        proposition_set=proposition_set, ltl_formula=ltl_formula
    )
    frame_of_interest = FramesofInterest(ltl_formula=ltl_formula)

    frame_idx = 0

    while True:
        # Get the next frame
        frame_img = processor.get_next_frame()
        if frame_img is None:
            break  # No more frames or end of video
        detected_objects: list = percepter.perceive(
            image=frame_img, object_of_interest=proposition_set
        )
        detected_activity = None

        frame = Frame(
            frame_idx=frame_idx,
            timestamp=processor.current_frame_index,
            frame_image=frame_img,
            object_detection=detected_objects,
            activity_detection=detected_activity,
        )
        # 1. frame validation
        if frame_validator.validate_frame(frame=frame):
            # 2. dynamic automaton construction
            automaton.add_frame_to_automaton(frame=frame)
            frame_of_interest.frame_buffer.append(frame)
            # 3. model checking
            model_checking_result = model_checker.check_automaton(
                transitions=automaton.transitions,
                states=automaton.states,
                verbose=False,
                is_filter=False,
            )
            if model_checking_result:
                # specification satisfied
                frame_of_interest.flush_frame_buffer()
                automaton.reset()

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    # Release resources
    cv2.destroyAllWindows()
    print(frame_of_interest.foi_list)


if __name__ == "__main__":
    main()
