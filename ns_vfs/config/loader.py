import os

import supervision as sv

from ns_vfs.config import config

assert config.VERSION_AND_PATH.SUPERVISION_VERSION == sv.__version__, "please install supervision==0.6.0"


def load_config():
    config.VERSION_AND_PATH.SAM_CHECKPOINT_PATH = os.path.join(
        config.VERSION_AND_PATH.ARTIFACTS_PATH,
        "weights",
        "sam_vit_h_4b8939.pth",
    )
    config.GROUNDING_DINO.GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(
        config.VERSION_AND_PATH.ARTIFACTS_PATH,
        "weights",
        "groundingdino_swinb_cogcoor.pth",
    )
    config.GROUNDING_DINO.GROUNDING_DINO_CONFIG_PATH = os.path.join(
        config.VERSION_AND_PATH.ARTIFACTS_PATH,
        "weights",
        "GroundingDINO_SwinB_cfg.py",
    )
    config.YOLO.YOLO_CHECKPOINT_PATH = os.path.join(
        config.VERSION_AND_PATH.ARTIFACTS_PATH,
        "weights",
        "yolov8x.pt",
    )
    return config
