from __future__ import annotations

from ns_vfs.automaton.probabilistic import ProbabilisticAutomaton
from ns_vfs.common.utility import save_image
from ns_vfs.config.loader import load_config
from ns_vfs.data.frame import Frame, FramesofInterest
from ns_vfs.model.vision.object_detection.clip_model import ClipPerception
from ns_vfs.model.vision.object_detection.yolo import Yolo
from ns_vfs.model_checker.stormpy import StormModelChecker
from ns_vfs.percepter.multi_vision_percepter import MultiVisionPercepter
from ns_vfs.processor.tlv_dataset.tlv_dataset_processor import (
    TLVDatasetProcessor,
)
from ns_vfs.validator import FrameValidator


def main():
    # Given Propositions
    # LTL Formula
    config = load_config()
    video_path = '/opt/Neuro-Symbolic-Video-Frame-Search/sample_data/ouputs/benchmark_ImageNet2017-1K_ltl_"stop_sign" U "sheep"_5_0.pkl'  # Replace with your video path

    # Initialize the video processor
    processor = TLVDatasetProcessor(video_path)
    ltl_formula = f"P>=0.80 [{processor.ltl_formula}]"
    proposition_set = processor.proposition_set
    cv_model = {
        "yolo": Yolo(
            weight_path=config.YOLO.YOLO_CHECKPOINT_PATH,
        ),
        "clip": ClipPerception(model_name="ViT-B/32"),
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

    print(processor.labels_of_frames)
    while True:
        # Get the next frame
        frame_img = processor.get_next_frame()
        if frame_img is None:
            break  # No more frames or end of video
        detected_objects: list = percepter.perceive(
            image=frame_img, object_of_interest=proposition_set
        )
        activity_of_interest = None

        frame = Frame(
            frame_idx=frame_idx,
            timestamp=processor.current_frame_index,
            frame_image=frame_img,
            object_of_interest=detected_objects,
            activity_of_interest=activity_of_interest,
        )
        save_image(image=frame_img, file_path="test1.png")

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

        frame_idx += 1

    print(frame_of_interest.foi_list)
    print(processor.frames_of_interest)
    print(processor.labels_of_frames)


if __name__ == "__main__":
    main()
