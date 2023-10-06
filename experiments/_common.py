import csv
from pathlib import Path

from _metrics import classification_metrics

from ns_vfs.common.utility import list_flatten
from ns_vfs.data.frame import BenchmarkLTLFrame, FramesofInterest


def write_to_csv_from_dict(dict_data: dict, csv_file_path: str, file_name="data.csv"):
    """Write dictionary to csv file."""
    csv_file_name = Path(csv_file_path) / file_name

    with open(csv_file_name, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=dict_data.keys())

        if not csv_file_name.exists():
            # If file does not exist, write header
            writer.writeheader()
        writer.writerow(dict_data)


def get_available_benchmark_video(path_to_directory: str):
    if isinstance(path_to_directory, str):
        directory_path = Path(path_to_directory)
        return list(directory_path.glob("*.pkl"))
    else:
        directory_path = path_to_directory
        return list(directory_path.rglob("*.pkl"))


def get_classification_score(search_result_per_video: dict):
    """Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2 * Precision * Recall / (Precision + Recall).
    """
    benchmark_video: BenchmarkLTLFrame = search_result_per_video["benchmark_video"]
    actual_result = set(list_flatten(benchmark_video.frames_of_interest))
    for key in list(search_result_per_video.keys()):
        if key in ["yolo", "grounding_dino"]:
            frame_of_interest: FramesofInterest = search_result_per_video[key]
            predictive_result = set(list_flatten(frame_of_interest.foi_list))

            accuracy, precision, recall, f1 = classification_metrics(
                actual_result=actual_result, predicted_result=predictive_result
            )
            result = dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1)

            search_result_per_video[f"{key}_classification_metrics"] = result
        else:
            pass
    return search_result_per_video
