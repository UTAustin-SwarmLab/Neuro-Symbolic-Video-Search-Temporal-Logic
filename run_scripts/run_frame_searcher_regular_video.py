from __future__ import annotations

import argparse

from ns_vfs.config.loader import load_config
from ns_vfs.frame_searcher import FrameSearcher
from ns_vfs.model.vision.grounding_dino import GroundingDino
from ns_vfs.model.vision.yolo import Yolo
from ns_vfs.processor.video_processor import (
    VideoFrameProcessor,
)
from ns_vfs.video_to_automaton import VideotoAutomaton

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv_model", type=str, default="grounding_dino")
    parser.add_argument(
        "--video_path",
        type=str,
        default="/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/data/nyc_street/nyc_stree_53sec.mp4",
    )
    args = parser.parse_args()

    config = load_config()

    video_processor = VideoFrameProcessor(
        video_path=args.video_path,
        artifact_dir=config.VERSION_AND_PATH.ARTIFACTS_PATH,
    )

    if args.cv_model == "grounding_dino":
        cv_model = GroundingDino(
            config=config.GROUNDING_DINO,
            weight_path=config.GROUNDING_DINO.GROUNDING_DINO_CHECKPOINT_PATH,
            config_path=config.GROUNDING_DINO.GROUNDING_DINO_CONFIG_PATH,
        )
    else:
        cv_model = Yolo(
            config=config.YOLO,
            weight_path=config.YOLO.YOLO_CHECKPOINT_PATH,
        )

    video_automata_builder = VideotoAutomaton(
        detector=cv_model,
        video_processor=video_processor,
        artifact_dir=config.VERSION_AND_PATH.ARTIFACTS_PATH,
        proposition_set=["bicycle", "car"],
        save_annotation=True,  # TODO: Debug only
        save_image=False,  # TODO: Debug only
        ltl_formula='P>=0.90 ["car" U "bicycle"]',  # 'P>=0.99 [F "person"]' P>=0.99 [F ("person" U "car")] P>=0.99 [F "person" U "car"]
    )

    frame_sercher = FrameSearcher(
        video_automata_builder=video_automata_builder, video_processor=video_processor
    )

    frame_of_interest = frame_sercher.search()
    frame_of_interest.save_frames("/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/tmp_3")
    # print(frame_of_interest)
    print("Done!")
