# %% [markdown]
# # False Positive Threshold Experiment

# %%
# Import Required Libraries
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from swarm_visualizer.utility.general_utils import set_plot_properties

# %%
#  Set Data Path
root_dir = Path("/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/experiment_2.1_nsvs_ltl/")
clip_csv_dir = root_dir / "clip/false_positive_threshold_experiment_0_to_1"
dino_csv_dir = root_dir / "dino/false_positive_threshold_experiment"
yolo_csv_dir = root_dir / "yolo/false_positive_threshold_experiment"
yolox_csv_dir = root_dir / "yolox/false_positive_threshold_experiment"
mrcnn_csv_dir = root_dir / "mrcnn/false_positive_threshold_experiment"


data_loc = {
    "clip": "/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/_result/benchmark_with_coco_and_imagenet/clip_benchmark_search_result_20231014_230610.csv",
    "yolo": "/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/_result/benchmark_with_coco_and_imagenet/yolo_benchmark_search_result_20231014_224708.csv",
    "dino": "/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/_result/benchmark_with_coco_and_imagenet/dino_benchmark_search_result_20231012_131832.csv",
    "mrcnn": "/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/_result/benchmark_with_coco_and_imagenet/mrcnn_benchmark_search_result_20231106_232921.csv",
}
result_dict = {}

for k, v in data_loc.items():
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
    result_dict[k] = df


# 1. YOLO DF
# Filter the DataFrame to include only rows where cv_model_weight is 'yolo8x'
yolo_df = result_dict["yolo"]
df_yolo8x = yolo_df[yolo_df["cv_model_weight"] == "yolov8x"]
sorted_yolo = df_yolo8x.sort_values(by="number_of_frame")
# 2. DINO DF
dino_df = result_dict["dino"]
sorted_dino = dino_df.sort_values(by="number_of_frame")
# 3. CLIP DF
clip_df = result_dict["clip"]
sorted_clip = clip_df.sort_values(by="number_of_frame")
# 4.MRCNN DF
mrcnn_df = result_dict["mrcnn"]
sorted_mrcnn = mrcnn_df.sort_values(by="number_of_frame")
# result = {
#     "yolov8": {
#         "xvec": sorted_yolo["number_of_frame"],
#         "ts_vector": sorted_yolo["f1_score"],
#         "lw": 3,
#         "linestyle": "-",
#         "color": "#1f77b4",
#     },
#     # "grounding-dino": {
#     #     "xvec": sorted_dino["number_of_frame"],
#     #     "ts_vector": sorted_dino["f1_score"],
#     #     "lw": 3,
#     #     "linestyle": "-",
#     #     "color": "#d62728",
#     # },
#     # "clip": {
#     #     "xvec": sorted_clip["number_of_frame"],
#     #     "ts_vector": sorted_clip["f1_score"],
#     #     "lw": 3,
#     #     "linestyle": "-",
#     #     "color": "#2ca02c",
#     # },
#     # "mrcnn": {
#     #     "xvec": sorted_mrcnn["number_of_frame"],
#     #     "ts_vector": sorted_mrcnn["f1_score"],
#     #     "lw": 3,
#     #     "linestyle": "-",
#     #     "color": "#2ca02c",
#     # },
# }
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# from swarm_visualizer.lineplot import plot_overlaid_ts
# # for title, data in result.items():
# plot_overlaid_ts(
#     normalized_ts_dict=result,
#     title_str=False,
#     ylabel="test",
#     xlabel="test",
#     fontsize=10,
#     legend_present=False,
#     ax=ax,
# )
# plt.show()
# plt.savefig("testtest.png")
# Set style and context to make the plot look more aesthetically pleasing
# sns.set_style("darkgrid")
# sns.set_context("paper")
set_plot_properties()
# Create a line plot for accuracy over number of frames for yolo8x weight
plt.figure(figsize=(8, 6))
sns.lineplot(
    x="number_of_frame",
    y="accuracy",
    data=df_yolo8x,
    marker="o",
    markersize=8,
    color="blue",
    label="YOLOv8",
    alpha=0.5,
    errorbar=("ci", 20),
)
sns.lineplot(
    x="number_of_frame",
    y="precision",
    data=dino_df,
    marker="o",
    markersize=8,
    color="red",
    label="DINO",
    alpha=0.5,
    errorbar=("ci", 20),
)
sns.lineplot(
    x="number_of_frame",
    y="precision",
    data=clip_df,
    marker="o",
    markersize=8,
    color="green",
    label="CLIP",
    alpha=0.5,
    errorbar=("ci", 20),
)
sns.lineplot(
    x="number_of_frame",
    y="precision",
    data=mrcnn_df,
    marker="o",
    markersize=8,
    color="black",
    label="MRCNN",
    alpha=0.5,
    errorbar=("ci", 20),
)
# sns.lineplot(x='number_of_frame', y='accuracy', data=df_yolo8n, marker="o", markersize=8, color="red", label='yolo8n', alpha=0.5, ci=20)
plt.title("LTL Frame Search Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Number of Frames")

# Show plot
plt.ylim(0.0, 1.0)
plt.tight_layout()
plt.legend()
plt.grid(True)
plt.savefig("test")
