from typing import List
import numpy as np
import cv2


class VideoFrame:
    """Frame class."""
    def __init__(
        self,
        frame_idx: int,
        frame_images: List[np.ndarray],
        object_of_interest: dict
    ):
        self.frame_idx = frame_idx
        self.frame_images = frame_images
        self.object_of_interest = object_of_interest

    def save_frame_img(self, save_path: str) -> None:
        """Save frame image."""
        if self.frame_images is not None:
            for idx, img in enumerate(self.frame_images):
                cv2.imwrite(f"{save_path}_{idx}.png", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def thresholded_detected_objects(self, threshold) -> dict:
        """Get all detected object."""

        detected_obj = {}
        for prop in self.object_of_interest.keys():
            probability = self.object_of_interest[prop].get_detected_probability()
            if probability > threshold:
                detected_obj[prop] = probability
        return detected_obj


class FramesofInterest:
    def __init__(self, num_of_frame_in_sequence):
        self.num_of_frame_in_sequence = num_of_frame_in_sequence
        self.foi_list = []
        self.frame_buffer = []

    def flush_frame_buffer(self):
        """Flush frame buffer to frame of interest."""
        if self.frame_buffer:
            frame_interval = [frame.frame_idx for frame in self.frame_buffer]
            self.foi_list.append([
                i*self.num_of_frame_in_sequence + j 
                for i in frame_interval 
                for j in range(self.num_of_frame_in_sequence)
            ])
            self.frame_buffer = []
