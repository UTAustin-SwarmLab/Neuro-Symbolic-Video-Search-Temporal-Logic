import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ns_vfs.config.loader import load_config
from ns_vfs.loader import LABEL_OF_INTEREST
from ns_vfs.model.vision.grounding_dino import GroundingDino
from ns_vfs.model.vision.yolo import Yolo

IMAGE_LOADER = ["imagenet"]
NUM_SAMPLE = 3000
LABEL_OF_INTEREST = LABEL_OF_INTEREST
CV_MODEL_DES = "dino"
CONFIG = load_config()
CONF_RANGE = np.array(range(30)) / 50 + 0.2  # (0.2, 0.8)
ROOT_DIR = Path("/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/experiment_1.2_calibration")

TARGET_LABEL = ['plane', 'bird', 'car', 'cat', 'deer',
           'dog', 'horse', 'monkey', 'ship', 'truck']


def sigmoid(x, k=1, x0=0):
    return 1 / (1 + np.exp(-k * (x - x0)))


def get_accuracy_range(conf_range, conf_score, true_label, target_label, verbose):
    accuracy = []
    for low_threhold in conf_range:
        low = low_threhold
        high = low_threhold + 0.02
        correct = 0
        total = 0
        if verbose:
            print(f"current low: {round(low,2)}, high: {round(high,2)}")
        for label_key in conf_score.keys():
            if label_key in target_label:
                for s in range(len(conf_score[label_key])):
                    if conf_score[label_key][s] >= low and conf_score[label_key][s] < high:
                        if verbose:
                            # print(f"confidence score: {round(conf_score[label_key][s],2)}")
                            pass
                        total += 1  # counting confidence between low and high threshold
                        if true_label[label_key][s] is True:
                            correct += 1  # counting correct label between low and high threshold
        if total == 0:
            success_ratio = 1 if high > 0.5 else 0
        else:
            success_ratio = correct / total
        accuracy.append(success_ratio)
    return accuracy


if __name__ == "__main__":
    csv_result = {}
    if CV_MODEL_DES == "yolo":
        cv_model = Yolo(
            config=CONFIG.YOLO,
            weight_path="/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/weights/yolov8x.pt",  # CONFIG.YOLO.YOLO_CHECKPOINT_PATH,
        )
    else:
        cv_model = GroundingDino(
            config=CONFIG.GROUNDING_DINO,
            weight_path=CONFIG.GROUNDING_DINO.GROUNDING_DINO_CHECKPOINT_PATH,
            config_path=CONFIG.GROUNDING_DINO.GROUNDING_DINO_CONFIG_PATH,
        )
    csv_result["cv_model"] = CV_MODEL_DES
    conf_score = {}
    true_label = {}

    data_path = "/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/experiment_1.2_calibration/tfds_dataset.pkl"
    label_path = (
        "/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/experiment_1.2_calibration/tfds_label.pkl"
    )

    with open(label_path, "rb") as f:
        labels = pickle.load(f)

    with open(data_path, "rb") as f:
        images = pickle.load(f)
    df = pd.DataFrame()
    df["label"] = labels[:NUM_SAMPLE]

    # get confidence score
    for c in range(len(TARGET_LABEL)):
        print(f"current label: {TARGET_LABEL[c]}")
        scores = []
        for i in range(NUM_SAMPLE):
            if i % 1000 == 0:
                print(f"progress..{i}")
            conf = cv_model.get_confidence_score(images[i], TARGET_LABEL[c])
            # Plot a sample image
            # plt.imshow(images[i])
            # plt.axis("off")
            # plt.savefig(f"{i}_calibration_debug_img.png")
            scores.append(conf)
        df[TARGET_LABEL[c]] = scores

    df.to_csv(ROOT_DIR / "tsdf_conf_original_lbs.csv", index=False)
