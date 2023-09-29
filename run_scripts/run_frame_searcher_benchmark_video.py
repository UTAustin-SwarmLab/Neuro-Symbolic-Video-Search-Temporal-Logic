from __future__ import annotations

from ns_vfs.config.loader import load_config
from ns_vfs.frame_searcher import FrameSearcher
from ns_vfs.model.vision.grounding_dino import GroundingDino
from ns_vfs.processor.benchmark_video_processor import BenchmarkVideoFrameProcessor
from ns_vfs.video_to_automaton import VideotoAutomaton

if __name__ == "__main__":
    benchmark_frame_path = '/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/benchmark_frame_video/cifar10/Fprop1/benchmark_CIFAR10_ltl_F "deer"_50_0.pkl'

    config = load_config()

    video_processor = BenchmarkVideoFrameProcessor(
        video_path=benchmark_frame_path,
        artifact_dir=config.VERSION_AND_PATH.ARTIFACTS_PATH,
    )

    benchmark_img_frame = video_processor.benchmark_image_frames

    video_automata_builder = VideotoAutomaton(
        detector=GroundingDino(
            config=config.GROUNDING_DINO,
            weight_path=config.GROUNDING_DINO.GROUNDING_DINO_CHECKPOINT_PATH,
            config_path=config.GROUNDING_DINO.GROUNDING_DINO_CONFIG_PATH,
        ),
        video_processor=video_processor,
        artifact_dir=config.VERSION_AND_PATH.ARTIFACTS_PATH,
        proposition_set=benchmark_img_frame.proposition,
        save_annotation=False,  # TODO: Debug only
        save_image=False,  # TODO: Debug only
        ltl_formula=f"P>=0.90 [{benchmark_img_frame.ltl_formula}]",
    )

    frame_sercher = FrameSearcher(
        video_automata_builder=video_automata_builder, video_processor=video_processor
    )

    frame_of_interest = frame_sercher.search()
    # print(frame_of_interest)
    print("Done!")
