from __future__ import annotations

import warnings

import clip
import cv2
import numpy as np
import torch
from PIL import Image

from ns_vfs.data.detected_object import DetectedObject
from ns_vfs.enums.status import Status
from ns_vfs.model.vision.object_detection._base import (
    ComputerVisionObjectDetector,
)

warnings.filterwarnings("ignore")


class ClipPerception(ComputerVisionObjectDetector):
    """CLIP Perception Model."""

    def __init__(self, model_name, gpu_device: int = 0) -> None:
        self.device = (
            f"cuda:{gpu_device}" if torch.cuda.is_available() else "cpu"
        )
        self.model, self.preprocess = self.load_model(
            model_name=model_name, device=self.device
        )

    def load_model(self, model_name, device) -> any:
        model, preprocess = clip.load(model_name, device=device)
        return model, preprocess

    def validate_object(self, object_name: str) -> bool:
        """Validate object (always true for CLIP).

        Returns:
            bool: True if object is valid.
        """
        return True

    def _parse_class_name(self, class_names: list[str]) -> list[str]:
        """Parse class name.

        Args:
            class_names (list[str]): List of class names.

        Returns:
            list[str]: List of class names.
        """
        result = []
        for class_name in class_names:
            c = class_name.replace("_", " ")
            result.append(f"all {c}s")
        return result

    def detect(self, frame_img: np.ndarray, classes: list) -> any:
        """Detect object in frame.

        Args:
            frame_img (np.ndarray): Frame image.
            classes (list[str]): List of class names.

        Returns:
            any: Detections.
        """

        class_name = classes[0].replace("_", " ")
        image_rgb = Image.fromarray(frame_img.astype("uint8"), "RGB")
        image = self.preprocess(image_rgb).unsqueeze(0).to(self.device)
        text = clip.tokenize(class_name).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)

            # Normalize features
            image_features_norm = image_features / image_features.norm(
                dim=1, keepdim=True
            )
            text_features_norm = text_features / text_features.norm(
                dim=1, keepdim=True
            )

            # Calculate similarity scores
            scores = (
                (image_features_norm @ text_features_norm.T)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )

            # logits_per_image, logits_per_text = self.model(image, text)
            # probs = logits_per_text.softmax(dim=-1).cpu().numpy()

        self._labels = []
        if len(scores) > 0:
            self._labels.append(f"{classes[0]} {scores[0]:0.4f}")
        else:
            self._labels.append({None}, {None})

        cllp_detection = {
            classes[0]: {
                "image_features": image_features,
                "text_features": text_features,
            }
        }

        self._detection = None  # Todo: Figure out what to do about it

        self._confidence = scores

        self._size = len(scores)

        probability = self._mapping_probability(int(scores))

        if probability > 0:
            is_detected = True
        else:
            is_detected = False

        return DetectedObject(
            name=class_name,
            model_name="clip",
            confidence_of_all_obj=list(self._confidence),
            probability_of_all_obj=[probability],
            all_obj_detected=cllp_detection,
            number_of_detection=Status.UNKNOWN,
            is_detected=is_detected,
        )

    def get_confidence_score(
        self, frame_img: np.ndarray, true_label: str
    ) -> any:
        # TODO: What is this about? It was not being called for YOLO
        self.detect(frame_img, true_label)
        return float(self._confidence[0])

    def _mapping_probability(
        self,
        confidence_per_video: float,
        true_threshold=0.30,
        false_threshold=0.230,
        a=1.00,
        k=56.546,
        x0=0.059,
    ) -> float:
        """Mapping probability.

        Args:
            confidence_per_video (float): Confidence per video.
            true_threshold (float, optional): True threshold. Defaults to 0.64.
            false_threshold (float, optional): False threshold. Defaults to 0.38.

        Returns:
            float: Mapped probability.
        """
        if confidence_per_video >= true_threshold:
            return 1
        elif confidence_per_video < false_threshold:
            return 0
        else:
            return round(
                self._sigmoid_mapping_estimation_function(
                    confidence_per_video,
                    a=a,
                    k=k,
                    x0=x0,
                ),
                2,
            )


def main():
    test = ClipPerception(None, None)
    img = cv2.imread(
        "/opt/Neuro-Symbolic-Video-Frame-Search/ship_on_the_sea.png"
    )
    test.detect(img, ["sunset", "dog", "person", "table"])
    print("Testing with 4 labels:")
    print(test._confidence)

    print("\nTesting with 1 label:")
    test.detect(img, ["ship_on_the_sea"])  # man backhug woman
    print(test._confidence)
    # breakpoint()


if __name__ == "__main__":
    main()
