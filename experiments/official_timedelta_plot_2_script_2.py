# %% [markdown]
# # False Positive Threshold Experiment

# %%
# Import Required Libraries
import matplotlib.pyplot as plt
import seaborn as sns
from swarm_visualizer.utility.general_utils import set_plot_properties

LLMModels = ["gpt-3.5-turbo-instuct", "NSVS-TL", "gpt-3.5-turbo", "gpt-4"]
LLMPlotColors = ["blue", "red", "green", "orange", "brown"]

data_loc_delta = {
    LLMModels[
        0
    ]: "/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/experiment_2.1_nsvs_ltl/yolo/yologpt-3.5-turbo-instruct_benchmark_search_result_vdelta1.csv",
    LLMModels[
        1
    ]: "/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/experiment_2.1_nsvs_ltl/yolo/yolo_benchmark_search_result_20231031_113154.csv",
    LLMModels[
        2
    ]: "/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/experiment_2.1_nsvs_ltl/yolo/yologpt-3.5-turbo-0613_benchmark_search_result_vdelta1.csv",
    LLMModels[
        3
    ]: "/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/experiment_2.1_nsvs_ltl/yolo/yologpt-4-0613_benchmark_search_result_vns1.csv",
}
result_dict_delta = {}
fig, ax = plt.subplots(figsize=(12.5, 10))
# Set style and context to make the plot look more aesthetically pleasing
sns.set_style("whitegrid")
sns.set_context("paper")
window = 20
meaning = 1
set_plot_properties(
    font_size=25,
    legend_font_size=20,
    xtick_label_size=20,
    ytick_label_size=20,
    markersize=15,
    usetex=True,
)
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
        df["f1_score_mean"] = df["f1_score_mean"].rolling(window=window, min_periods=1, closed="both").mean()
        df["f1_score_max"] = df["f1_score_max"].rolling(window=window, min_periods=1, closed="both").max()
        df["f1_score_std"] = df["f1_score_std"].rolling(window=window, min_periods=1, closed="both").std()
        df["f1_score_min"] = df["f1_score_min"].rolling(window=window, min_periods=1, closed="both").min()
        df["mean_number_frames"] = (
            df["mean_number_frames"].rolling(window=window, min_periods=1, closed="both").mean()
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
        marker="o",
        color=LLMPlotColors[j],
        markersize=4,
        label=k,
        alpha=0.4,
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

plt.title("F1 Score vs Length of Video Clip satisfying A until B TL Specification", fontsize=25)
plt.ylabel("F1 Score", fontsize=25)
plt.xlabel("Length of Video Clip (s)", fontsize=25)
ax.tick_params(axis="both", which="major", labelsize=20)

desired_order = LLMModels

# new_handles = ax.get_legend_handles_labels()[0]
# new_labels = desired_order

# fig.legend(new_handles, new_labels, loc="lower center", bbox_to_anchor=(0.5, 0.020), ncol=3, fontsize=15)
# Show plot
# plt.ylim(0.0, 1.0)
# plt.tight_layout()
plt.legend(fontsize=20, loc="lower right")
# plt.grid(True)

plt.savefig("f1_score_vs_video_length.jpg")
