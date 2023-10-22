from __future__ import annotations

import warnings

import os
import cv2
import torch
import numpy as np
import supervision as sv
from omegaconf import DictConfig
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

    def load_model(self, weight_path) -> Predictor:
        """Load weight.
        Args:
            weight_path (str): Path to weight file.
        Returns:
            None
        """
        exp = get_exp(None, "yolox-x") # modify in case model changes
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
        class_reversed = COCO_CLASSES.index(classes[0])
        outputs, img_info = self.model.inference(frame_img)
        output = outputs[0].cpu()
        cls = (output[:, 6]).numpy()
        scores = (output[:, 4] * output[:, 5]).numpy()
        self._confidence = np.array([])
        self._labels = []
        bbox_total = (output[:, 0:4]/img_info["ratio"]).cpu().detach().numpy()
        # print(bbox_total)
        bbox = []
        for i in range(len(cls)):
            # print(cls[i])
            if cls[i] == class_reversed:
                self._confidence = np.append(self._confidence, scores[i])
                self._labels.append(f"{COCO_CLASSES[int(cls[i])]} {scores[i]}")
                bbox.append(bbox_total[i])

        # print(self._labels)
        
        # print(bbox)
        if len(bbox) == 0:
            self._detections = None
        else:
            self._detections = sv.Detections(xyxy=np.array(bbox))
        self._size = len(self._confidence)

        # result_image = self.model.visual(output, img_info, self.model.confthre)
        # file_name ="/opt/Neuro-Symbolic-Video-Frame-Search/ns_vfs/model/vision/test.png"
        # cv2.imwrite(file_name, result_image)
        # print(outputs)

        return outputs


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
        img_info = {}

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
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
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

        # print("CLS: %s" %cls)
        # print("cls_conf: %s" %cls_conf)
        # print("scores: %s" %scores)

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res