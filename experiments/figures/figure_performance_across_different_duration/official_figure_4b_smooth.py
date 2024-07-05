import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter
from swarm_visualizer.utility.general_utils import set_plot_properties

LLMModels = [
    "gpt-3.5-turbo-instruct",
    "gpt-3.5-turbo",
    "gpt-4",
    "NSVS-TL (Ours)",
]
LLMPlotColors = [
    "#a31621",
    "#2a7d2b",
    "#6a60d5",
    "#e7984a",
]

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = Path(current_dir) / "data"
data_loc_delta = {
    LLMModels[0]: data_dir
    / "yologpt-3.5-turbo-instruct_benchmark_search_result_vdelta1.csv",
    LLMModels[1]: data_dir
    / "yologpt-3.5-turbo-0613_benchmark_search_result_vdelta1.csv",
    LLMModels[2]: data_dir / "yologpt-4-0613_benchmark_search_result_vdelta1.csv",
    LLMModels[3]: data_dir / "yolo_benchmark_search_result_20231031_113154.csv",
}
result_dict_delta = {}

for k, v in data_loc_delta.items():
    if "NSVS" in k:
        df = pd.read_csv(v)
        header = [
            "dataset",
            "ltl_group",
            "number_of_frame",
            "ltl_formula",
            "video_length",
            "cv_model",
            "cv_model_weight",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "mapping_false_threshold",
            "mapping_true_threshold",
            "mapping_param_x0",
            "mapping_param_k",
        ]
        df.columns = header
        print(df)
        result_dict_delta[k] = df
        result_dict_delta[k]["model"] = k
        result_dict_delta[k] = result_dict_delta[k][
            result_dict_delta[k]["cv_model_weight"] == "yolov8x"
        ]
    else:
        df = pd.read_csv(v)
        header = [
            "dataset",
            "ltl_group",
            "ltl_formula",
            "number_of_frame",
            "cv_model",
            "cv_model_weight",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "mapping_false_threshold",
            "mapping_true_threshold",
            "mapping_param_x0",
            "mapping_param_k",
        ]
        df.columns = header
        print(df)
        result_dict_delta[k] = df
        result_dict_delta[k]["model"] = k

# Modify the plotting section
set_plot_properties()
plt.figure(figsize=(8, 6))

window = 25  # Increased window size for smoother rolling average
savgol_window = 30  # Window size for Savitzky-Golay filter
savgol_order = 3  # Polynomial order for Savitzky-Golay filter
set_plot_properties()
for j, (k, v) in enumerate(data_loc_delta.items()):
    df = result_dict_delta[k]
    df = df.sort_values(by="number_of_frame")
    if "yolo" in k:
        df = df[df["video_length"] % 10 == 0]

    # Apply rolling average
    df["f1_score_mean"] = (
        df["f1_score"].rolling(window=window, center=True, min_periods=1).mean()
    )
    df["f1_score_std"] = (
        df["f1_score"].rolling(window=window, center=True, min_periods=1).std()
    )
    df["mean_number_frames"] = (
        df["number_of_frame"].rolling(window=window, center=True, min_periods=1).mean()
    )

    # Apply Savitzky-Golay filter for additional smoothing
    df["f1_score_smooth"] = savgol_filter(
        df["f1_score_mean"], window_length=savgol_window, polyorder=savgol_order
    )

    sns.lineplot(
        x="mean_number_frames",
        y="f1_score_smooth",
        data=df,
        color=LLMPlotColors[j],
        label=k,
        alpha=0.8,
        linewidth=2,
    )
    plt.fill_between(
        df["mean_number_frames"],
        df["f1_score_smooth"] - 0.5 * df["f1_score_std"],
        df["f1_score_smooth"] + 0.5 * df["f1_score_std"],
        alpha=0.1,
        color=LLMPlotColors[j],
    )

plt.title("Performance across various video durations", fontsize=22)
plt.ylabel("Accuracy (F1 Score)")
plt.xlabel("Length of video clip (s)")
plt.legend(fontsize=9, loc="lower center", ncol=4)
# plt.tight_layout()

plt.savefig("figure4_b_smooth.png")
plt.show()
