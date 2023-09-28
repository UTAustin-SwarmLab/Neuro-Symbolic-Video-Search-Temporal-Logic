from __future__ import annotations

from ns_vfs.config.loader import load_config
from ns_vfs.model.vision.grounding_dino import GroundingDino
from ns_vfs.processor.benchmark_video_processor import BenchmarkVideoFrameProcessor
from ns_vfs.video_to_automaton import VideotoAutomaton

if __name__ == "__main__":
    benchmark_fram_path = '/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/benchmark_frame_video/cifar10/G!prop1/benchmark_CIFAR10_ltl_G ! "deer"_75_2.pkl'

    config = load_config()

    benchmark_video_processor = BenchmarkVideoFrameProcessor(
        video_path=benchmark_fram_path,
        artifact_dir=config.VERSION_AND_PATH.ARTIFACTS_PATH,
    )

    benchmark_img_frame = benchmark_video_processor.benchmark_image_frames

    frame2automaton = VideotoAutomaton(
        detector=GroundingDino(
            config=config.GROUNDING_DINO,
            weight_path=config.GROUNDING_DINO.GROUNDING_DINO_CHECKPOINT_PATH,
            config_path=config.GROUNDING_DINO.GROUNDING_DINO_CONFIG_PATH,
        ),
        video_processor=benchmark_video_processor,
        artifact_dir=config.VERSION_AND_PATH.ARTIFACTS_PATH,
        proposition_set=benchmark_img_frame.proposition,
        is_annotation=True,  # TODO: Debug only
        save_image=True,  # TODO: Debug only
        ltl_formula=f"Pmin=? [{benchmark_img_frame.ltl_formula}]",
        manual_confidence_probability=None
        # 'P>=0.99 [F "person"]' P>=0.99 [F ("person" U "car")] P>=0.99 [F "person" U "car"]
    )

    frame_window_automata = frame2automaton.run()
