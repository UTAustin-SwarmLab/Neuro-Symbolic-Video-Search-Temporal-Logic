from __future__ import annotations

from video_to_automaton.config.loader import load_config
from video_to_automaton.model.vision.grounding_dino import GroundingDino
from video_to_automaton.processor.video_processor import (
    VideoFrameProcessor,
)
from video_to_automaton.video_to_automaton import VideotoAutomaton

if __name__ == "__main__":
    sample_video_path = "/opt/Video-to-Automoton/artifacts/data/hmdb51/clap/Applauding_Abby_clap_u_nm_np1_fr_med_1.avi"

    config = load_config()

    print(config)

    frame2automaton = VideotoAutomaton(
        detector=GroundingDino(
            config=config.GROUNDING_DINO,
            weight_path=config.GROUNDING_DINO.GROUNDING_DINO_CHECKPOINT_PATH,
            config_path=config.GROUNDING_DINO.GROUNDING_DINO_CONFIG_PATH,
        ),
        video_processor=VideoFrameProcessor(video_path=sample_video_path),
        proposition_set=["clap", "face", "baby"],
    )
    frame2automaton.build_automaton(is_annotation=False)
    print("Development is in progress.")
