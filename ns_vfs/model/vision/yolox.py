from __future__ import annotations

import warnings

import os
import cv2
import torch
import numpy as np
import supervision as sv
from loguru import logger
from omegaconf import DictConfig
from ultralytics import YOLO
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis


from ns_vfs.model.vision._base import ComputerVisionDetector

warnings.filterwarnings("ignore")


class YoloX(ComputerVisionDetector):
    """YoloX."""

    def __init__(self, config: DictConfig, weight_path: str) -> None:
        self.model = self.load_model(weight_path)
        self._config = config
        self._classes_reversed = {v: k for k, v in self.model.names.items()}

    def load_model(self, weight_path) -> YOLO:
        """Load weight.

        Args:
            weight_path (str): Path to weight file.

        Returns:
            None
        """
        exp = None
        exp.test_conf = 0.25
        exp.nmsthre = 0.45
        exp.test_size = (640, 640)

        model = exp.get_model()
        model.eval()
        ckpt = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])

        return Predictor(model, exp)

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
        class_ids = [self._classes_reversed[c.replace("_", " ")] for c in classes]
        detected_obj = self.model.predict(source=frame_img, classes=class_ids)

        self._labels = []
        for i in range(len(detected_obj[0].boxes)):
            class_id = int(detected_obj[0].boxes.cls[i])
            confidence = float(detected_obj[0].boxes.conf[i])
            self._labels.append(
                f"{detected_obj[0].names[class_id] if class_id is not None else None} {confidence:0.2f}"
            )

        self._detection = sv.Detections(xyxy=detected_obj[0].boxes.xyxy.cpu().detach().numpy())

        self._confidence = detected_obj[0].boxes.conf.cpu().detach().numpy()

        self._size = len(detected_obj[0].boxes)

        return detected_obj

    def get_confidence_score(self, frame_img: np.ndarray, true_label: str) -> any:
        max_conf = 0
        class_ids = [self._classes_reversed[c.replace("_", " ")] for c in [true_label]]
        detected_obj = self.model.predict(source=frame_img, classes=class_ids)[0]
        all_detected_object_list = detected_obj.boxes.cls
        all_detected_object_confidence = detected_obj.boxes.conf

        for i in range(len(all_detected_object_list)):
            if all_detected_object_list[i] == class_ids[0]:
                if all_detected_object_confidence[i] > max_conf:
                    max_conf = all_detected_object_confidence[i].cpu().item()
        return max_conf


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res