import glob
import pickle

# import _metrics
from _metrics import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ns_vfs.data.frame import BenchmarkLTLFrame
from ns_vfs.model.vision.clip_model import ClipPerception


def get_all_scores(gt, pred):
    accuracy = accuracy_score(gt, pred)
    precision = precision_score(gt, pred)
    recall = recall_score(gt, pred)
    f1 = f1_score(gt, pred)
    return accuracy, precision, recall, f1


model = ClipPerception(None, None)


# iterate over all files in gprop1 folder
threshold_list = [0.005, 0.0787, 0.152, 0.226, 0.3]

results = {}
for threshold in threshold_list:
    print("Running for threshold: ", threshold)
    accuracy_list, precision_list, recall_list, f1_list = [], [], [], []
    for file_name in glob.glob(
        "/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/_validated_benchmark_video/imagenet/Fprop1/*.pkl"
    ):
        # print('Running for file: ', file_name)
        preds = []
        # Open the file in binary read mode
        with open(file_name, "rb") as file:
            # Load the data from the file
            data: BenchmarkLTLFrame = pickle.load(file)
        for i in range(len(data.images_of_frames)):
            frame = data.images_of_frames[i]
            conf = model.get_confidence_score(frame, data.proposition[0])
            if conf > threshold:
                preds.append(True)
            else:
                preds.append(False)

        gt = [
            True if x == data.proposition[0] else False for x in data.labels_of_frames
        ]
        accuracy, precision, recall, f1 = get_all_scores(gt, preds)
        print(
            "Accuracy: ",
            accuracy,
            "Precision: ",
            precision,
            "Recall: ",
            recall,
            "F1: ",
            f1,
        )
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    results[threshold] = {}
    results[threshold]["accuracy"] = accuracy_list
    results[threshold]["precision"] = precision_list
    results[threshold]["recall"] = recall_list
    results[threshold]["f1"] = f1_list


# Plot accurave vs threshold box plot with error bars

# save results as pickle
with open(
    "/opt/Neuro-Symbolic-Video-Frame-Search/experiments/data/res_clipFprop1_v2.pkl",
    "wb",
) as file:
    pickle.dump(results, file)
