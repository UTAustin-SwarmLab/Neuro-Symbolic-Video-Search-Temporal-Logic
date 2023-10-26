from __future__ import annotations

from omegaconf import DictConfig
import supervision as sv
import numpy as np
import warnings
from PIL import Image


# from mmcv.runner import load_checkpoint
from mmcv.transforms import Compose
from mmengine.config import Config
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.runner import load_checkpoint
# from mmengine.registry import MODELS as MMENGINE_MODELS
# from mmengine.registry import Registry
import torch
from mmdetection.mmdet.registry import MODELS
# from mmrotate.registry import MODELS
# from ns_vfs.model.vision.registry import MODELS
from mmdetection.mmdet.utils.misc import get_test_pipeline_cfg

from ns_vfs.model.vision._base import ComputerVisionDetector
from mmdet.utils import register_all_modules
register_all_modules()

warnings.filterwarnings("ignore")


class MMDetection(ComputerVisionDetector):
    """MMDetection"""

    def __init__(
        self,
        config: DictConfig,
        config_path: str, # .py file
        weight_path: str  # .pth file
    ) -> None:
        self.model = self.load_model(weight_path, config_path)
        self._config = config

    def load_model(self, weight_path, config_path) -> DetInferencer:
        """Load weight.

        Args:
            weight_path (str): Path to weight file.

        Returns:
            None
        """
        # init_args = {'model': config_path, 'weights': weight_path, 'device': 'cpu', 'palette': 'none'}

        device='cuda:0'
        config = Config.fromfile(config_path)
        config.model.backbone.init_cfg = None
        # MODELS = Registry('model', parent=MMENGINE_MODELS, locations=['mmdet.models'])
        model = MODELS.build(config.model)
        model = revert_sync_batchnorm(model)
        checkpoint = load_checkpoint(model, weight_path, map_location=device)
        model.CLASSES = checkpoint['meta']['CLASSES']
        
        model.cfg = config_path
        model.to(device)
        model.eval()

        return model

        # return DetInferencer(**init_args)

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
        cfg = self.model.cfg
        test_pipeline = get_test_pipeline_cfg(cfg)
        test_pipeline[0].type = 'LoadImageFromNDArray'
        test_pipeline = Compose(test_pipeline)

        # prepare data
        data_ = dict(img=frame_img, img_id=0)
        data_ = test_pipeline(data_)

        data_['inputs'] = [data_['inputs']]
        data_['data_samples'] = [data_['data_samples']]

        # forward the model
        with torch.no_grad():
            result = self.model.test_step(data_)[0]
        
        labels_id = result.pred_instances.labels
        labels = []
        for l in labels_id:
            labels.append(self.model.CLASSES[l])
        if len(labels) == 0:
            self._labels = []
            self._confidence = np.array([])
            self._detections = None
            self._size = 0
            return None
        
        scores = result.pred_instances.scores.tolist()

        self._confidence = np.array([])
        self._labels = []
        bbox_total = result.pred_instances.bboxes.cpu().detach().numpy()

        bbox = []
        for i in range(len(labels)):
            if labels[i] == classes[0]:
                self._confidence = np.append(self._confidence, scores[i])
                self._labels.append(f"{labels[i]} {scores[i]}")
                bbox.append(bbox_total[i])
        if len(bbox) == 0:
            self._detections = None
        else:
            self._detections = sv.Detections(xyxy=np.array(bbox))
        self._size = len(self._confidence)


        return None

