from pathlib import Path

from ns_vfs.common.utility import list_flatten
from ns_vfs.data.frame import BenchmarkLTLFrame, FramesofInterest


def get_available_benchmark_video(path_to_directory: str):
    if isinstance(path_to_directory, str):
        directory_path = Path(path_to_directory)
        return list(directory_path.glob("*.pkl"))
    else:
        directory_path = path_to_directory
        return list(directory_path.rglob("*.pkl"))


def get_precision_recall_f1_score(search_result_per_video: dict):
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
            # True Positive (TP)
            TP = len(actual_result.intersection(predictive_result))

            # False Positive (FP)
            FP = len(predictive_result.difference(actual_result))

            # False Negative (FN)
            FN = len(actual_result.difference(predictive_result))

            # Calculating Precision and Recall
            precision = TP / (TP + FP) if TP + FP != 0 else 0
            recall = TP / (TP + FN) if TP + FN != 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
            result = dict(precision=precision, recall=recall, f1=f1)

            search_result_per_video[f"{key}_classification_metrics"] = result
        else:
            pass
    return search_result_per_video
