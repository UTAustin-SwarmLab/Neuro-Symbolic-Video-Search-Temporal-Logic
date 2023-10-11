from __future__ import annotations

import warnings

import numpy as np
from omegaconf import DictConfig
import clip 
import torch
import cv2
from PIL import Image

from ns_vfs.model.vision._base import ComputerVisionDetector

warnings.filterwarnings("ignore")


class ClipPerception(ComputerVisionDetector):
    """Yolo."""

    def __init__(self, config: DictConfig, weight_path: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self._config = config
        # self._classes_reversed = {v: k for k, v in self.model.names.items()}

    def load_model(self, weight_path) -> None:
        """
        Not Needed For CLIP
        """
        pass

    def _parse_class_name(self, class_names: list[str]) -> list[str]:
        """Parse class name.

        Args:
            class_names (list[str]): List of class names.

        Returns:
            list[str]: List of class names.
        """
        return [f"all {class_name}s" for class_name in class_names]

    def detect(self, frame_img: np.ndarray, classes: list) -> any:
        """Detect object in frame.

        Args:
            frame_img (np.ndarray): Frame image.
            classes (list[str]): List of class names.

        Returns:
            any: Detections.
        """

        image = Image.fromarray(frame_img.astype('uint8'), 'RGB')
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        text = clip.tokenize(classes).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            
            logits_per_image, logits_per_text = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        self._detection = None # Todo: Figure out what to do about it

        self._confidence = probs

        self._size = len(probs)

        return self._detection

    def get_confidence_score(self, frame_img: np.ndarray, true_label: str) -> any:
        pass
        # TODO: What is this about? It was not being called for YOLO
        # max_conf = 0
        # class_ids = [self._classes_reversed[c.replace("_", " ")] for c in [true_label]]
        # detected_obj = self.model.predict(source=frame_img, classes=class_ids)[0]
        # all_detected_object_list = detected_obj.boxes.cls
        # all_detected_object_confidence = detected_obj.boxes.conf

        # for i in range(len(all_detected_object_list)):
        #     if all_detected_object_list[i] == class_ids[0]:
        #         if all_detected_object_confidence[i] > max_conf:
        #             max_conf = all_detected_object_confidence[i].cpu().item()
        # return max_conf


def main():
    test = ClipPerception(None, None)
    img = cv2.imread('calibration_plot_dino.png')
    test.detect(img, ['cat', 'dog'])
    breakpoint()



if __name__=='__main__':
    main()