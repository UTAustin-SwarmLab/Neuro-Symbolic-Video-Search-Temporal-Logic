from __future__ import annotations

from ns_vfs.config.loader import load_config
from ns_vfs.frame_searcher import FrameSearcher
from ns_vfs.model.vision.grounding_dino import GroundingDino
from ns_vfs.processor.video_processor import (
    VideoFrameProcessor,
)
from ns_vfs.video_to_automaton import VideotoAutomaton

if __name__ == "__main__":
    sample_video_path = "/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/data/nyc_street/nyc_stree_53sec.mp4"

    config = load_config()

    video_processor = VideoFrameProcessor(
        video_path=sample_video_path,
        artifact_dir=config.VERSION_AND_PATH.ARTIFACTS_PATH,
    )

    video_automata_builder = VideotoAutomaton(
        detector=GroundingDino(
            config=config.GROUNDING_DINO,
            weight_path=config.GROUNDING_DINO.GROUNDING_DINO_CHECKPOINT_PATH,
            config_path=config.GROUNDING_DINO.GROUNDING_DINO_CONFIG_PATH,
        ),
        video_processor=video_processor,
        artifact_dir=config.VERSION_AND_PATH.ARTIFACTS_PATH,
        proposition_set=["dog"],
        save_annotation=True,  # TODO: Debug only
        save_image=False,  # TODO: Debug only
        ltl_formula='P>=0.90 [G "dog"]',  # 'P>=0.99 [F "person"]' P>=0.99 [F ("person" U "car")] P>=0.99 [F "person" U "car"]
    )

    frame_sercher = FrameSearcher(
        video_automata_builder=video_automata_builder, video_processor=video_processor
    )

    frame_of_interest = frame_sercher.search()
    frame_of_interest.save_frames("/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/tmp_1")
    # print(frame_of_interest)
    print("Done!")
