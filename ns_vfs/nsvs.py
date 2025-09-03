import numpy as np
import tqdm

from ns_vfs.model_checker.property_checker import PropertyChecker
from ns_vfs.model_checker.video_automaton import VideoAutomaton
from ns_vfs.video.frame import FramesofInterest
from ns_vfs.vlm.vllm_client import VLLMClient
from ns_vfs.video.frame import VideoFrame
from ns_vfs.vlm.internvl import InternVL

PRINT_ALL = False
DEVICE = 7

def run_nsvs(
    frames: list[np.ndarray],
    proposition: list,
    specification: str,
    model_type: str = "dtmc",
    num_of_frame_in_sequence = 3,
    satisfaction_threshold: float = 0.6,
    vlm_detection_threshold: float = 0.349,
):
    """Find relevant frames from a video that satisfy a specification"""

    # vlm = VLLMClient()
    vlm = InternVL(model_name="InternVL2-8B", device=DEVICE)

    automaton = VideoAutomaton(include_initial_state=True)
    automaton.set_up(proposition_set=proposition)

    checker = PropertyChecker(
        proposition=proposition,
        specification=specification,
        model_type=model_type,
        threshold=satisfaction_threshold,
    )

    frame_of_interest = FramesofInterest(num_of_frame_in_sequence)

    frame_windows = []
    for i in range(0, len(frames), num_of_frame_in_sequence):
        frame_windows.append(frames[i : i + num_of_frame_in_sequence])

    def process_frame(sequence_of_frames: list[np.ndarray], frame_count: int):
        object_of_interest = {}

        for prop in proposition:
            detected_object = vlm.detect(
                seq_of_frames=sequence_of_frames,
                scene_description=prop,
                threshold=vlm_detection_threshold
            )
            object_of_interest[prop] = detected_object
            if PRINT_ALL and detected_object.is_detected:
                print(f"\t{prop}: {detected_object.confidence}->{detected_object.probability}")

        frame = VideoFrame(
            frame_idx=frame_count,
            frame_images=sequence_of_frames,
            object_of_interest=object_of_interest,
        )
        return frame

    if PRINT_ALL:
        looper = enumerate(frame_windows)
    else:
        looper = tqdm.tqdm(enumerate(frame_windows), total=len(frame_windows))

    for i, sequence_of_frames in looper:
        if PRINT_ALL:
            print("\n" + "*"*50 + f" {i}/{len(frame_windows)-1} " + "*"*50)
            print("Detections:")
        frame = process_frame(sequence_of_frames, i)
        if PRINT_ALL:
            frame.save_frame_img(save_path=f"junk/{i}")

        if checker.validate_frame(frame_of_interest=frame):
            automaton.add_frame(frame=frame)
            frame_of_interest.frame_buffer.append(frame)
            model_check = checker.check_automaton(automaton=automaton)
            if model_check:
                automaton.reset()
                frame_of_interest.flush_frame_buffer()

    foi = frame_of_interest.foi_list

    if PRINT_ALL:
        print("\n" + "-"*107)
        print("Detected frames of interest:")
        print(foi)

    return foi

