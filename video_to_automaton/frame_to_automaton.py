import numpy as np

from video_to_automaton.config.loader import load_config
from video_to_automaton.model.vision._base import ComputerVisionDetector
from video_to_automaton.model.vision.grounding_dino import GroundingDino
from video_to_automaton.processor.video_processor import (
    VideoFrameProcessor,
    VideoProcessor,
)


class FrametoAutomaton:
    def __init__(
        self,
        proposition_set: list,
        detector: ComputerVisionDetector,
        video_processor: VideoProcessor,
    ):
        self.proposition_set = proposition_set
        self._detector = detector
        self._video_processor = video_processor

    def get_trajectory_from_frame(
        self, proposition: str, frame_img: np.ndarray
    ):
        trajectory = list()
        detected_obj = self._detector.detect(frame_img, [proposition])
        if len(detected_obj) > 0:
            trajectory.append(
                np.round(np.max(detected_obj.confidence), 2)
            )  # probability of the object in the frame
        else:
            trajectory.append(0)  # probability of the object in the frame is 0
        return trajectory

    def get_trajectory_from_video(
        self, proposition: str, video_frames: np.ndarray
    ):
        for video_frame in video_frames:
            self.get_trajectory_from_frame(
                proposition=proposition, frame_img=video_frame
            )

    def build_automaton(self):
        video_frames = self._video_processor.get_video_by_frame()
        for proposition in self.proposition_set:
            self.get_trajectory_from_video(
                proposition=proposition, video_frames=video_frames
            )


if __name__ == "__main__":
    sample_video_path = "/home/mc76728/repos/Video-to-Automaton/artifacts/data/hmdb51/clap/A_Round_of_Applause_clap_u_cm_np1_fr_med_0.avi"

    config = load_config()

    print(config)

    frame2automaton = FrametoAutomaton(
        detector=GroundingDino(
            config=config.GROUNDING_DINO,
            weight_path=config.GROUNDING_DINO.GROUNDING_DINO_CHECKPOINT_PATH,
            config_path=config.GROUNDING_DINO.GROUNDING_DINO_CONFIG_PATH,
        ),
        video_processor=VideoFrameProcessor(video_path=sample_video_path),
        proposition_set=["clap", "face"],
    )
    frame2automaton.build_automaton()
    print("Development is in progress.")
