from __future__ import annotations

from ns_vfs.config.loader import load_config
from ns_vfs.model.vision.grounding_dino import GroundingDino
from ns_vfs.model.vision.yolo import Yolo
from ns_vfs.model.vision.yolox import YoloX
from ns_vfs.model.vision.mmdetection import MMDetection
from ns_vfs.processor.video_processor import VideoFrameWindowProcessor
from ns_vfs.video_to_automaton import VideotoAutomaton

if __name__ == "__main__":
    sample_video_path = (
        "/opt/Neuro-Symbolic-Video-Frame-Search/VIRAT_S_050201_05_000890_000944.mp4"
    )

    config = load_config()

    frame2automaton = VideotoAutomaton(
        detector=YoloX(
            config=config.YOLOX,
            weight_path=config.YOLOX.YOLOX_CHECKPOINT_PATH,
        ),
        # detector=Yolo(
        #     config=config.YOLO,
        #     weight_path=config.YOLO.YOLO_CHECKPOINT_PATH,
        # ),
        # detector=MMDetection(
        #     config=config.MMDETECTION,
        #     config_path=config.MMDETECTION.MMDETECTION_CONFIG_PATH,
        #     weight_path=config.MMDETECTION.MMDETECTION_CHECKPOINT_PATH
        # ),
        # detector=GroundingDino(
        #     config=config.GROUNDING_DINO,
        #     weight_path=config.GROUNDING_DINO.GROUNDING_DINO_CHECKPOINT_PATH,
        #     config_path=config.GROUNDING_DINO.GROUNDING_DINO_CONFIG_PATH,
        # ),
        video_processor=VideoFrameWindowProcessor(
            video_path=sample_video_path,
            artifact_dir=config.VERSION_AND_PATH.ARTIFACTS_PATH,
        ),
        artifact_dir=config.VERSION_AND_PATH.ARTIFACTS_PATH,
        proposition_set=["car", "person"],
        is_annotation=True,  # TODO: Debug only
        save_image=True,  # TODO: Debug only
        ltl_formula='P>=0.99 [F "person"]',
    )

    frame_window_automata = frame2automaton.run()
    print(frame_window_automata)
