import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
]  # ["#de1028", "#07610b", "#760af2", "#ff9100"]

data_loc_delta = {
    LLMModels[
        0
    ]: "/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/experiment_2.1_nsvs_ltl/yolo/yologpt-3.5-turbo-instruct_benchmark_search_result_vdelta1.csv",
    LLMModels[
        1
    ]: "/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/experiment_2.1_nsvs_ltl/yolo/yologpt-3.5-turbo-0613_benchmark_search_result_vdelta1.csv",
    LLMModels[
        2
    ]: "/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/experiment_2.1_nsvs_ltl/yolo/yologpt-4-0613_benchmark_search_result_vdelta1.csv",
    LLMModels[
        3
    ]: "/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/experiment_2.1_nsvs_ltl/yolo/yolo_benchmark_search_result_20231031_113154.csv",
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


# Create a line plot for accuracy over number of frames for yolo8x weight
set_plot_properties()
confidence_interval = 68
plt.figure(figsize=(8, 6))


window = 20
meaning = 1
set_plot_properties()
# Create a line plot for accur
for j, (k, v) in enumerate(data_loc_delta.items()):
    df = result_dict_delta[k]
    df = df.sort_values(by="number_of_frame")
    if "yolo" in k:
        df = df[df["video_length"] % 10 == 0]

    # Get mean and std_dev rolling for each series and plot a line plot
    df["f1_score_mean"] = df["f1_score"]
    df["f1_score_max"] = df["f1_score"]
    df["f1_score_std"] = df["f1_score"]
    df["f1_score_min"] = df["f1_score"]
    df["f1_score_bw"] = df["f1_score"]
    df["mean_number_frames"] = df["number_of_frame"]
    for _ in range(meaning):
        df["f1_score_mean"] = (
            df["f1_score_mean"]
            .rolling(window=window, min_periods=1, closed="both")
            .mean()
        )
        df["f1_score_max"] = (
            df["f1_score_max"]
            .rolling(window=window, min_periods=1, closed="both")
            .max()
        )
        df["f1_score_std"] = (
            df["f1_score_std"]
            .rolling(window=window, min_periods=1, closed="both")
            .std()
        )
        df["f1_score_min"] = (
            df["f1_score_min"]
            .rolling(window=window, min_periods=1, closed="both")
            .min()
        )
        df["mean_number_frames"] = (
            df["mean_number_frames"]
            .rolling(window=window, min_periods=1, closed="both")
            .mean()
        )
        df["f1_score_bw"] = 0.5 * (df["f1_score_max"] + df["f1_score_min"])
    print(df["f1_score_mean"])
    print(df["f1_score_std"])
    print(df["number_of_frame"])
    print(df["mean_number_frames"])

    sns.lineplot(
        x="mean_number_frames",
        y="f1_score_mean",
        data=df,
        # marker="o",
        color=LLMPlotColors[j],
        markersize=4,
        label=k,
        alpha=0.8,
    )
    plt.fill_between(
        df["mean_number_frames"],
        df["f1_score_mean"] - 0.5 * df["f1_score_std"],
        df["f1_score_mean"] + 0.5 * df["f1_score_std"],
        alpha=0.1,
        color=LLMPlotColors[j],
        step="pre",
    )
# sns.lineplot(x='number_of_frame', y='accuracy', data=df_yolo8n, marker="o", markersize=8, color="red", label='yolo8n', alpha=0.5, ci=20)

# plt.title("F1 Score vs Length of Video Clip satisfying A until B TL Specification", fontsize=25)
# plt.ylabel("F1 Score", fontsize=25)
# plt.xlabel("Length of Video Clip (s)", fontsize=25)
# ax.tick_params(axis="both", which="major", labelsize=20)

# desired_order = LLMModels

# # new_handles = ax.get_legend_handles_labels()[0]
# # new_labels = desired_order

# # fig.legend(new_handles, new_labels, loc="lower center", bbox_to_anchor=(0.5, 0.020), ncol=3, fontsize=15)
# # Show plot
# # plt.ylim(0.0, 1.0)
# # plt.tight_layout()
# plt.legend(fontsize=20, loc="lower right")
# # plt.grid(True)

# plt.savefig("f1_score_vs_video_length.jpg")

# plt.title("F1 Score vs Length of Video Clip satisfying A until B TL Specification", fontsize=25)
plt.title("Performance across various video durations")

plt.ylabel("F1 Score")
plt.xlabel("Length of video clip (s)")
# ax.tick_params(axis="both", which="major", labelsize=20)

# Adjust figure size if necessary
# fig.set_size_inches(8, 8)

# Place the legend below the plot
# plt.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)
plt.legend(fontsize=9.20, loc="lower center", ncol=4)
plt.tight_layout()
# plt.subplots_adjust(bottom=0.20)
# Adjust the margins and layout
# plt.tight_layout(rect=[0, 0, 1, 0.9])

plt.savefig("figure4_b.png", bbox_inches="tight")
