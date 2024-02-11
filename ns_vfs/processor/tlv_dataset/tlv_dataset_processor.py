from __future__ import annotations

import numpy as np

from ns_vfs.processor._base_video_processor import BaseVideoProcessor


class TLVDatasetProcessor(BaseVideoProcessor):
    """TLV Dataset Processor."""

    def __init__(self, tlv_dataset_path: str) -> None:
        """TLV Dataset Processor.

        Args:
            video_path (str): Path to video file.
            artifact_dir (str): Path to artifact directory.
        """
        self.tlv_dataset = self.import_video(tlv_dataset_path)
        self.current_frame_index = 0

    def import_video(self, video_path: str) -> None:
        """Load and return an instance from a pickle file."""
        import pickle

        with open(video_path, "rb") as f:
            return pickle.load(f)

    def get_all_frames_of_video(self) -> list:
        """Get all frames of video."""
        return self.tlv_dataset["images_of_frames"]

    def get_next_frame(
        self, return_format: str = "ndarray"
    ) -> np.ndarray | None:
        """Get next frame of video."""
        if self.current_frame_index < len(self.tlv_dataset["images_of_frames"]):
            frame = self.tlv_dataset["images_of_frames"][
                self.current_frame_index
            ]
            self.current_frame_index += 1
            return frame
        else:
            return None

    def get_ground_truth_label(self, frame_idx: int) -> str:
        """Get ground truth label."""
        return self.tlv_dataset["labels_of_frames"][frame_idx]

    @property
    def ltl_formula(self) -> str:
        """Get LTL formula."""
        return self.tlv_dataset["ltl_formula"]

    @property
    def proposition_set(self) -> list:
        """Get proposition set."""
        return self.tlv_dataset["proposition"]

    @property
    def frames_of_interest(self) -> list:
        """Get frames of interest."""
        return self.tlv_dataset["frames_of_interest"]

    @property
    def labels_of_frames(self) -> list:
        """Get labels of frames."""
        return self.tlv_dataset["labels_of_frames"]
