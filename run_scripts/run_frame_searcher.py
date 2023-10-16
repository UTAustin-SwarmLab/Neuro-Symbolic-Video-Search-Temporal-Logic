from __future__ import annotations

import argparse

from ns_vfs.config.loader import load_config
from ns_vfs.frame_searcher import FrameSearcher
from ns_vfs.model.vision.grounding_dino import GroundingDino
from ns_vfs.model.vision.yolo import Yolo
from ns_vfs.model.vision.yolox import YoloX
from ns_vfs.processor.benchmark_video_processor import BenchmarkVideoFrameProcessor
from ns_vfs.processor.video_processor import (
    VideoFrameProcessor,
)
from ns_vfs.video_to_automaton import VideotoAutomaton

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv_model", type=str, default="yolo")
    parser.add_argument(
        "--video_processor", type=str, default="regular_video", choices=["regular_video", "benchmark_video"]
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="../VIRAT_S_050201_05_000890_000944.mp4",
    )
    parser.add_argument(
        "--proposition_set",
        type=str,
        default="bicycle,car",
        help="No space between propositions, separated by comma",
    )
    parser.add_argument(
        "--ltl_formula",
        type=str,
        default='P>=0.90 ["car" U "bicycle"]',
    )
    parser.add_argument(
        "--save_annotation",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/_result/frame_search_output",
    )
    parser.add_argument(
        "--manual_confidence_probability",
        type=str,
        default=1.0,
    )

    args = parser.parse_args()

    config = load_config()

    if args.video_processor == "regular_video":
        video_processor = VideoFrameProcessor(
            video_path=args.video_path,
            artifact_dir=config.VERSION_AND_PATH.ARTIFACTS_PATH,
        )
        proposition_set = args.proposition_set.split(",")
        ltl_formula = args.ltl_formula
    else:
        assert args.video_path.endswith(".pkl"), "Benchmark video path must be a .pkl file"
        manual_confidence_probability = args.manual_confidence_probability
        video_processor = BenchmarkVideoFrameProcessor(
            video_path=args.video_path,
            artifact_dir=config.VERSION_AND_PATH.ARTIFACTS_PATH,
            manual_confidence_probability=manual_confidence_probability,
        )
        benchmark_img_frame = video_processor.benchmark_image_frames
        proposition_set = benchmark_img_frame.proposition
        ltl_formula = f"P>=0.80 [{benchmark_img_frame.ltl_formula}]"

    if args.cv_model == "grounding_dino":
        cv_model = GroundingDino(
            config=config.GROUNDING_DINO,
            weight_path=config.GROUNDING_DINO.GROUNDING_DINO_CHECKPOINT_PATH,
            config_path=config.GROUNDING_DINO.GROUNDING_DINO_CONFIG_PATH,
        )
    elif args.cv_model == "yolo":
        cv_model = Yolo(
            config=config.YOLO,
            weight_path=config.YOLO.YOLO_CHECKPOINT_PATH,
        )
    else:
        cv_model = YoloX(
            config=config.YOLOX,
            weight_path=config.YOLOX.YOLOX_CHECKPOINT_PATH,
        )

    video_automata_builder = VideotoAutomaton(
        detector=cv_model,
        video_processor=video_processor,
        artifact_dir=config.VERSION_AND_PATH.ARTIFACTS_PATH,
        proposition_set=proposition_set,
        save_annotation=args.save_annotation,  # TODO: Debug only
        save_image=False,  # TODO: Debug only
        ltl_formula=ltl_formula,  # 'P>=0.99 [F "person"]' P>=0.99 [F ("person" U "car")] P>=0.99 [F "person" U "car"]
    )

    frame_sercher = FrameSearcher(
        video_automata_builder=video_automata_builder, video_processor=video_processor
    )

    frame_of_interest = frame_sercher.search()
    frame_of_interest.save_frames(args.output_dir)
    # print(frame_of_interest)
    print("Done!")
