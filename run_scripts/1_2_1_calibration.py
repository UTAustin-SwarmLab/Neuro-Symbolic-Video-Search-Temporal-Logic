import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ns_vfs.common.utility import get_file_or_dir_with_datetime, save_dict_to_pickle
from ns_vfs.config.loader import load_config
from ns_vfs.loader import LABEL_OF_INTEREST
from ns_vfs.loader.benchmark_coco import COCOImageLoader
from ns_vfs.loader.benchmark_imagenet import ImageNetDataloader
from ns_vfs.model.vision.clip_model import ClipPerception
from ns_vfs.model.vision.grounding_dino import GroundingDino
from ns_vfs.model.vision.yolo import Yolo

IMAGE_LOADER = ["coco"]
NUM_SAMPLE = 3000
LABEL_OF_INTEREST = LABEL_OF_INTEREST
CV_MODEL_DES = "clip"
CONFIG = load_config()
CONF_RANGE = np.array(range(40)) / 50 + 0.2
ROOT_DIR = Path("/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/experiment_1.2_calibration")

TARGET_LABEL = [
    "person",
    "bicycle",
    "car",
    "airplane",
    "truck",
    "cat",
    "dog",
    "horse",
    "laptop",
    "apple",
    "banana",
    "cup",
    "chair",
    "clock",
    "spoon",
]


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
            weight_path="/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/weights/yolov8n.pt",  # CONFIG.YOLO.YOLO_CHECKPOINT_PATH,
        )
    elif CV_MODEL_DES == "clip":
        cv_model = ClipPerception(config=CONFIG, weight_path=None)

    else:
        cv_model = GroundingDino(
            config=CONFIG.GROUNDING_DINO,
            weight_path=CONFIG.GROUNDING_DINO.GROUNDING_DINO_CHECKPOINT_PATH,
            config_path=CONFIG.GROUNDING_DINO.GROUNDING_DINO_CONFIG_PATH,
        )
    csv_result["cv_model"] = CV_MODEL_DES
    for img_loader in IMAGE_LOADER:
        conf_score = {}
        true_label = {}
        if img_loader == "coco":
            dataloader = COCOImageLoader(
                coco_dir_path="/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/data/benchmark_image_dataset/coco",
                annotation_file="annotations/instances_val2017.json",
                image_dir="val2017",
            )
        elif img_loader == "imagenet":
            image_dir = "/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/data/ILSVRC"
            dataloader = ImageNetDataloader(imagenet_dir_path=image_dir)
        data = dataloader.data
        images = data.images
        labels = data.labels
        print(f"Dataset: {img_loader}...Sampling {NUM_SAMPLE} images...")
        print(f"Number of images: {len(images)}, Number of labels: {len(labels)}")
        for input_label in LABEL_OF_INTEREST:
            if input_label in TARGET_LABEL:
                print(f"Current label: {input_label}")
                conf_score[input_label] = []
                true_label[input_label] = []
                for i in range(NUM_SAMPLE):
                    # random sampling
                    s = random.randint(0, len(images) - 1)
                    # true lable of random image data
                    if isinstance(images[s], tuple):
                        curr_img = images[s][0]
                    else:
                        curr_img = images[s]
                    curr_label = labels[s]
                    conf = cv_model.get_confidence_score(frame_img=curr_img, true_label=input_label)
                    print(conf)
                    conf_score[input_label].append(conf)
                    if input_label in curr_label:
                        true_label[input_label].append(True)
                        # plt.imshow(curr_img)
                        # plt.axis("off")
                        # plt.savefig(f"{i}_calibration_debug_img.png")
                        # print("debug ph")

                    else:
                        true_label[input_label].append(False)
                        # if conf > 0.8:
                        #     print(f"False positive: {input_label}")
                        #     plt.imshow(curr_img)
                        #     plt.axis("off")
                        #     plt.savefig(f"{s}_calibration_debug_img.png")
                        #     print("debug ph")

        save_dict_to_pickle(
            dict_obj=conf_score,
            path=ROOT_DIR,
            file_name=get_file_or_dir_with_datetime(
                f"conf_score_{img_loader}_{CV_MODEL_DES}_{NUM_SAMPLE}_{len(TARGET_LABEL)}",
                ext=".pkl",
            ),
        )
        save_dict_to_pickle(
            dict_obj=true_label,
            path=ROOT_DIR,
            file_name=get_file_or_dir_with_datetime(
                f"true_label_{img_loader}_{CV_MODEL_DES}_{NUM_SAMPLE}_{len(TARGET_LABEL)}",
                ext=".pkl",
            ),
        )
        # # # Getting Accuracy # # #
        accuracy = get_accuracy_range(
            conf_range=CONF_RANGE,
            conf_score=conf_score,
            true_label=true_label,
            target_label=TARGET_LABEL,
            verbose=False,
        )
        # # # Plotting # # #
        accuracy_series = pd.Series(accuracy)
        # Compute the moving average of the accuracy data
        window_size = 5  # Choose an appropriate window size
        smoothed_accuracy = accuracy_series.rolling(window=window_size).mean()
        smoothed_accuracy = accuracy[:window_size] + list(smoothed_accuracy[window_size:])

        xx = np.linspace(0, 1, len(accuracy))
        yy = sigmoid(np.array(xx), k=50, x0=0.56)
        ax = sns.lineplot(x=xx, y=yy, label="Mapping Estimation")
        sns.lineplot(x=xx, y=accuracy, ax=ax, label=f"{CV_MODEL_DES}", linestyle="dashed")
        sns.lineplot(
            x=xx,
            y=smoothed_accuracy,
            label=f"{CV_MODEL_DES}(smoothed)",
            linestyle="dashed",
        )
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        plt.savefig(
            ROOT_DIR / f"calibration_plot_{img_loader}_{CV_MODEL_DES}_{NUM_SAMPLE}_{len(TARGET_LABEL)}.png",
            dpi=300,
        )  # Adjust filename, format, and DPI as needed
