from __future__ import annotations

from ns_vfs.config.loader import load_config
from ns_vfs.model.vision.grounding_dino import GroundingDino
from ns_vfs.processor.video_processor import (
    VideoFrameProcessor,
)
from ns_vfs.video_to_automaton import VideotoAutomaton

if __name__ == "__main__":
    sample_video_path = "/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/data/summer_event/bali_beach_club_pt1_720p.mp4"

    config = load_config()

    print(config)

    frame2automaton = VideotoAutomaton(
        detector=GroundingDino(
            config=config.GROUNDING_DINO,
            weight_path=config.GROUNDING_DINO.GROUNDING_DINO_CHECKPOINT_PATH,
            config_path=config.GROUNDING_DINO.GROUNDING_DINO_CONFIG_PATH,
        ),
        video_processor=VideoFrameProcessor(
            video_path=sample_video_path,
            artifact_dir=config.VERSION_AND_PATH.ARTIFACTS_PATH,
        ),
        proposition_set=["face", "person", "car"],
        artifact_dir=config.VERSION_AND_PATH.ARTIFACTS_PATH,
    )

    frame_window_automata = frame2automaton.start()

    print("Development is in progress.")
