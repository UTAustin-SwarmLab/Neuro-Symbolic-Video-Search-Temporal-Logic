from __future__ import annotations

import dataclasses  # noqa: D100
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from ns_vfs.common.utility import (
    get_file_or_dir_with_datetime,
)


@dataclasses.dataclass
class Frame:
    """Frame class."""

    frame_idx: int
    timestamp: Optional[int] = None
    frame_image: Optional[np.ndarray] = None
    annotated_image: Dict[str, np.ndarray] = dataclasses.field(
        default_factory=dict
    )
    object_of_interest: Optional[dict] = None
    activity_of_interest: Optional[dict] = None

    def is_any_object_detected(self):
        """Check if object is detected."""
        if len(self.detected_object) == 0:
            return False
        else:
            return True

    @property
    def detected_object_list(self):
        """Get detected object."""
        detected_obj = []
        for obj_name, obj_value in self.object_of_interest.items():
            if obj_value.is_detected:
                detected_obj.append(obj_name)
        return detected_obj

    @property
    def detected_object_dict(self):
        """Get detected object info as dict."""
        detected_obj = {}
        for obj_name, obj_value in self.object_of_interest.items():
            if obj_value.is_detected:
                detected_obj[obj_name] = {}
                detected_obj[obj_name]["total_number_of_detection"] = (
                    obj_value.number_of_detection
                )
                detected_obj[obj_name]["maximum_probability"] = max(
                    obj_value.probability_of_all_obj
                )
                detected_obj[obj_name]["minimum_probability"] = min(
                    obj_value.probability_of_all_obj
                )
                detected_obj[obj_name]["maximum_confidence"] = max(
                    obj_value.confidence_of_all_obj
                )
                detected_obj[obj_name]["minimum_confidence"] = min(
                    obj_value.confidence_of_all_obj
                )

        return detected_obj

    @property
    def detected_bboxes(self):
        """Get detected object."""
        bboxes = []
        for obj_name, obj_value in self.object_of_interest.items():
            if obj_value.is_detected:
                for obj_prob in obj_value.probability_of_all_obj:
                    if obj_prob > 0:
                        bboxes += obj_value.bounding_box_of_all_obj
        return bboxes


@dataclasses.dataclass
class FramesofInterest:
    """Frame class."""

    ltl_formula: str
    foi_list: List[List[int]] = dataclasses.field(default_factory=list)
    frame_images: List[np.ndarray] = dataclasses.field(default_factory=list)
    annotated_images: List[np.ndarray] = dataclasses.field(default_factory=list)
    frame_idx_to_real_idx: dict = dataclasses.field(default_factory=dict)
    frame_buffer: List[Frame] = dataclasses.field(default_factory=list)

    def save_annotated_images(self, annotated_image: Dict[str, np.ndarray]):
        for a_img in list(annotated_image.values()):
            self.annotated_images.append(a_img)

    def save_frames_of_interest(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for idx, img in enumerate(self.frame_images):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            Image.fromarray(img_rgb).save(f"{path}/{idx}.png")
            try:
                if (
                    len(self.annotated_images) > 0
                    and self.annotated_images[idx] is not None
                ):
                    Image.fromarray(self.annotated_images[idx]).save(
                        f"{path}/{idx}_annotated.png"
                    )
            except:  # noqa: E722
                pass

    def flush_frame_buffer(self):
        """Flush frame buffer to frame of interest."""
        if len(self.frame_buffer) > 1:
            frame_interval = list()
            for frame in self.frame_buffer:
                frame_interval.append(frame.frame_idx)
                self.frame_idx_to_real_idx[frame.frame_idx] = frame.timestamp
                self.frame_images.append(frame.frame_image)
                self.save_annotated_images(frame.annotated_image)
            self.foi_list.append(frame_interval)
        else:
            for frame in self.frame_buffer:
                self.foi_list.append([frame.frame_idx])
                self.frame_idx_to_real_idx[frame.frame_idx] = frame.timestamp
                self.frame_images.append(frame.frame_image)
                self.save_annotated_images(frame.annotated_image)
        self.frame_buffer = list()

    def save(self, path: str | Path):
        from PIL import Image

        if isinstance(path, str):
            root_path = Path(path)
        else:
            root_path = path
        dir_name = get_file_or_dir_with_datetime("foi_result")
        frame_path = root_path / dir_name / "frame"
        annotation_path = root_path / dir_name / "annotation"

        frame_path.mkdir(parents=True, exist_ok=True)
        annotation_path.mkdir(parents=True, exist_ok=True)

        for idx, img in enumerate(self.frame_images):
            Image.fromarray(img).save(f"{frame_path}/{idx}.png")
            try:
                if (
                    len(self.annotated_images) > 0
                    and self.annotated_images[idx] is not None
                ):
                    Image.fromarray(self.annotated_images[idx]).save(
                        f"{annotation_path}/{idx}_annotated.png"
                    )
            except:  # noqa: E722
                pass
