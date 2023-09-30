from __future__ import annotations

from ns_vfs.config.loader import load_config
from ns_vfs.model.vision.grounding_dino import GroundingDino
<<<<<<< HEAD:run_frame_to_automata.py
from ns_vfs.model.vision.yolo import Yolo
from ns_vfs.processor.video_processor import VideoFrameWindowProcessor
from ns_vfs.video_to_automaton import VideotoAutomaton

if __name__ == "__main__":
    sample_video_path = (
        "/opt/Neuro-Symbolic-Video-Frame-Search/VIRAT_S_050201_05_000890_000944.mp4"
    )
=======
from ns_vfs.processor.video_processor import (
    VideoFrameProcessor,
)
from ns_vfs.video_to_automaton import VideotoAutomaton

if __name__ == "__main__":
    sample_video_path = "/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/data/nyc_street/nyc_stree_53sec.mp4"
>>>>>>> pre-merge-0929:run_scripts/archieve/run_frame_to_automata.py

    config = load_config()

    frame_window_video_processor = VideoFrameProcessor(
        video_path=sample_video_path,
        artifact_dir=config.VERSION_AND_PATH.ARTIFACTS_PATH,
    )

    frame2automaton = VideotoAutomaton(
        # detector=Yolo(
        #     config=config.YOLO,
        #     weight_path=config.YOLO.YOLO_CHECKPOINT_PATH,
        # ),
        detector=GroundingDino(
            config=config.GROUNDING_DINO,
            weight_path=config.GROUNDING_DINO.GROUNDING_DINO_CHECKPOINT_PATH,
            config_path=config.GROUNDING_DINO.GROUNDING_DINO_CONFIG_PATH,
        ),
        video_processor=frame_window_video_processor,
        artifact_dir=config.VERSION_AND_PATH.ARTIFACTS_PATH,
<<<<<<< HEAD:run_frame_to_automata.py
        proposition_set=["person", "car"],
        is_annotation=False,  # TODO: Debug only
        save_image=False,  # TODO: Debug only
        ltl_formula='P>=0.99 [F "person"]',
    )

    frame_window_automata = frame2automaton.run()
    print(frame_window_automata)
=======
        proposition_set=["dog"],
        is_annotation=True,  # TODO: Debug only
        save_image=True,  # TODO: Debug only
        ltl_formula='P>=0.80 [F "dog"]',  # 'P>=0.99 [F "person"]' P>=0.99 [F ("person" U "car")] P>=0.99 [F "person" U "car"]
    )

    frame_of_interest_obj = frame2automaton.run()
>>>>>>> pre-merge-0929:run_scripts/archieve/run_frame_to_automata.py
