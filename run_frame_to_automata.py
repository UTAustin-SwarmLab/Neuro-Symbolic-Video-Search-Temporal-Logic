from __future__ import annotations

from ns_vfs.config.loader import load_config
from ns_vfs.model.vision.yolo import Yolo
from ns_vfs.processor.video_processor import (
    VideoFrameWindowProcessor,
)
from ns_vfs.video_to_automaton import VideotoAutomaton

if __name__ == "__main__":
    sample_video_path = (
        "/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/data/nyc_street/nyc_street_footage.mp4"
    )

    config = load_config()

    frame2automaton = VideotoAutomaton(
        detector=Yolo(
            config=config.YOLO,
            weight_path=config.YOLO.YOLO_CHECKPOINT_PATH,
        ),
        video_processor=VideoFrameWindowProcessor(
            video_path=sample_video_path,
            artifact_dir=config.VERSION_AND_PATH.ARTIFACTS_PATH,
        ),
        artifact_dir=config.VERSION_AND_PATH.ARTIFACTS_PATH,
        proposition_set=[0, 2, 9],
        is_annotation=True,  # TODO: Debug only
        save_image=True,  # TODO: Debug only
        ltl_formula='P>=0.99 [F "person"]',
    )

    frame_window_automata = frame2automaton.run()
    print(frame_window_automata)
