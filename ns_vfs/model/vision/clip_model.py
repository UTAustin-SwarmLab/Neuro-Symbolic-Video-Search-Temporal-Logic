from __future__ import annotations

import warnings

import clip
import cv2
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image

from ns_vfs.model.vision._base import ComputerVisionDetector

warnings.filterwarnings("ignore")


class ClipPerception(ComputerVisionDetector):
    """Yolo."""

    def __init__(self, config: DictConfig, weight_path: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device) #ViT-L/14
        self._config = config
        # self._classes_reversed = {v: k for k, v in self.model.names.items()}

    def load_model(self, weight_path) -> None:
        """Not Needed For CLIP."""
        pass

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
        image = Image.fromarray(frame_img.astype("uint8"), "RGB")
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        text = clip.tokenize(self._parse_class_name(classes)).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)

            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            scores = image_features @ text_features.t()
            scores = scores[0].detach().cpu().numpy()

            # logits_per_image, logits_per_text = self.model(image, text)
            # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        self._labels = []
        if len(scores) > 0:
            self._labels.append(f"{classes[0]} {scores[0]:0.4f}")
        else:
            self._labels.append({None}, {None})

        self._detection = None  # Todo: Figure out what to do about it

        self._confidence = scores

        self._size = len(scores)

        return self._detection

    def get_confidence_score(self, frame_img: np.ndarray, true_label: str) -> any:
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
    img = cv2.imread("/opt/Neuro-Symbolic-Video-Frame-Search/ship_on_the_sea.png")
    test.detect(img, ["sunset", "dog", "person", "table"])
    print("Testing with 4 labels:")
    print(test._confidence)

    print("\nTesting with 1 label:")
    test.detect(img, ["ship_on_the_sea"])  # man backhug woman
    print(test._confidence)
    # breakpoint()


if __name__ == "__main__":
    main()
