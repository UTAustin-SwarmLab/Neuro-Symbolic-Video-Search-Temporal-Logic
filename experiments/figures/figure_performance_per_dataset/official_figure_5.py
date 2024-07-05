# Import Required Libraries
import os
from pathlib import Path

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from swarm_visualizer.boxplot import plot_paired_boxplot

TITLE_FONT_SIZE = 20
LABEL_FONT_SIZE = 20
sns.set_style(style="darkgrid")
sns.set_color_codes()
sns.set()
plt.rc("text", usetex=False)
# font = {"family": "normal", "weight": "bold", "size": 20}
# plt.rc("font", **font)
plt.rcParams["text.latex.preamble"] = r"\boldmath"

plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.weight"] = "bold"

params = {
    "legend.fontsize": 14,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.markersize": 10,
    "figure.autolayout": False,
}

pylab.rcParams.update(params)
LLMModels = ["gpt-3.5-turbo-instruct", "gpt-3.5-turbo", "gpt-4", "NSVS-TL (Ours)"]
LLMPlotColors = [
    "#a31621",
    "#2a7d2b",
    "#6a60d5",
    "#e7984a",
]  # ["#de1028", "#07610b", "#760af2", "#ff9100"]

current_dir = os.path.dirname(os.path.abspath(__file__))
coco_imagenet_dir = Path(current_dir) / "data" / "coco_and_imagenet"
data_loc = {
    LLMModels[0]: coco_imagenet_dir
    / "yologpt-3.5-turbo-instruct_benchmark_search_result_v4.csv",
    LLMModels[3]: coco_imagenet_dir
    / "yolo_benchmark_search_result_20231014_224708.csv",
    LLMModels[1]: coco_imagenet_dir
    / "yologpt-3.5-turbo-0613_benchmark_search_result_v4.csv",
    LLMModels[2]: coco_imagenet_dir / "yologpt-4-0613_benchmark_search_result_v4.csv",
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
    result_dict[k]["model"] = k
    if "NSVS" in k:
        result_dict[k] = result_dict[k][result_dict[k]["cv_model_weight"] == "yolov8x"]

# WAYMO
waymo_dir = Path(current_dir) / "data" / "waymo"
data_loc_waymo = {
    LLMModels[0]: waymo_dir
    / "mrcnngpt-3.5-turbo-instruct_benchmark_search_result_vwaymo.csv",
    LLMModels[3]: waymo_dir / "waymo_mrcnn_benchmark_search_result_20231108_234248.csv",
    LLMModels[1]: waymo_dir
    / "mrcnngpt-3.5-turbo-0613_benchmark_search_result_vwaymo.csv",
    LLMModels[2]: waymo_dir / "mrcnngpt-4-0613_benchmark_search_result_vwaymo.csv",
}
result_dict_waymo = {}

for k, v in data_loc_waymo.items():
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
    result_dict_waymo[k] = df
    result_dict_waymo[k]["model"] = k
all_data = pd.concat([dfs for dfs in result_dict_waymo.values()], ignore_index=True)
# set_plot_properties()
# Creating a box plot for accuracy of each ltl_group, differentiated by model
fig, ax = plt.subplots(figsize=(8, 6))
plot_paired_boxplot(
    all_data,
    x_var="ltl_group",
    y_var="f1_score",
    hue="model",
    pal=LLMPlotColors,
    title_str="F1 Score per LTL Group by Different Retrieval Models",
    ax=ax,
)
plt.title("F1 Score per TL Specification by Different Retrieval Algorithms")
plt.ylabel("F1 Score")
plt.xlabel("TL Specification")

# NUSCENES
nuscenes_dir = Path(current_dir) / "data" / "nuscenes"
data_loc_ns = {
    LLMModels[0]: nuscenes_dir
    / "mrcnngpt-3.5-turbo-instruct_benchmark_search_result_vnuscene.csv",
    LLMModels[3]: nuscenes_dir / "nuscene_mrcnn_benchmark_search_result_20231104.csv",
    LLMModels[1]: nuscenes_dir
    / "mrcnngpt-3.5-turbo-0613_benchmark_search_result_vnuscene.csv",
    LLMModels[2]: nuscenes_dir / "mrcnngpt-4-0613_benchmark_search_result_vnuscene.csv",
}
result_dict_ns = {}

for k, v in data_loc_ns.items():
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
    result_dict_ns[k] = df
    result_dict_ns[k]["model"] = k


all_data = pd.concat([dfs for dfs in result_dict_ns.values()], ignore_index=True)
# set_plot_properties()
# Creating a box plot for accuracy of each ltl_group, differentiated by model
fig, ax = plt.subplots(figsize=(8, 6))
plot_paired_boxplot(
    all_data,
    x_var="ltl_group",
    y_var="f1_score",
    hue="model",
    pal=LLMPlotColors,
    title_str="F1 Score per LTL Group by Different Retrieval Models",
    ax=ax,
)

for artist in ax.artists:
    artist.set_alpha(0.5)

plt.title("F1 Score per TL Specification by Different Retrieval Algorithms")
plt.ylabel("F1 Score")
plt.xlabel("TL Specification")


##########################################################################################################################################################
desired_order = ["gpt-3.5-turbo-instruct", "gpt-3.5-turbo", "gpt-4", "NSVS-TL (Ours)"]
all_data_syn = pd.concat([result_dict[key] for key in desired_order], ignore_index=True)
all_data_waymo = pd.concat(
    [result_dict_waymo[key] for key in desired_order], ignore_index=True
)
all_data_ns = pd.concat(
    [result_dict_ns[key] for key in desired_order], ignore_index=True
)
all_data_syn["ltl_group"] = all_data_syn["ltl_group"].replace(
    "Fprop1", "Eventually\nEvent A"
)
all_data_syn["ltl_group"] = all_data_syn["ltl_group"].replace(
    "Gprop1", "Always\nEvent A"
)
# all_data_syn["ltl_group"] = all_data_syn["ltl_group"].replace("prop1Uprop2", "A until B\nA U B")
all_data_syn["ltl_group"] = all_data_syn["ltl_group"].replace(
    "prop1Uprop2", "Event A\nuntil B"
)
all_data_syn["ltl_group"] = all_data_syn["ltl_group"].replace(
    "prop1&prop2", "Event A\nand B"
)
# all_data_syn["ltl_group"] = all_data_syn["ltl_group"].replace(
#     "(prop1&prop2)Uprop3", "A and B Until C\n(A & B) U C"
# )
all_data_syn["ltl_group"] = all_data_syn["ltl_group"].replace(
    "(prop1&prop2)Uprop3", "Event A and B\nUntil Event C"
)
all_data_ns["ltl_group"] = all_data_ns["ltl_group"].replace(
    "prop1Uprop2", "Event A\nuntil B"
)
all_data_ns["ltl_group"] = all_data_ns["ltl_group"].replace(
    "(prop1&prop2)Uprop3", "Event A and B\nUntil Event C"
)
all_data_waymo["ltl_group"] = all_data_waymo["ltl_group"].replace(
    "prop1Uprop2", "Event A\nuntil B"
)
all_data_waymo["ltl_group"] = all_data_waymo["ltl_group"].replace(
    "(prop1&prop2)Uprop3", "Event A and B\nUntil Event C"
)

# Data Visualization
# Set up plot properties for a consistent and professional look

# Recreate the figure with 2 rows and 2 columns

# Create subplots for each data category
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(45, 12))

# Dictionary to hold data for each subplot
data_for_subplots = {
    "Performance on COCO-TLV and ImageNet-TLV Dataset": all_data_syn,
    "Performance on NuScenes-TLV Dataset": all_data_ns,
    "Performance on Waymo-TLV Dataset": all_data_waymo,
}
# category_order = [
#     "Eventually\nF A",
#     "Always\nG A",
#     "A until B\nA U B",
#     "A and B\nA & B",
#     "A and B Until C\n(A & B) U C",
# ]
category_order = [
    "Eventually\nEvent A",
    "Always\nEvent A",
    "Event A\nand B",
    "Event A\nuntil B",
    "Event A and B\nUntil Event C",
]
# Loop through each subplot to plot the data
for i, (title, data) in enumerate(data_for_subplots.items()):
    if title == "Performance on COCO-TLV and ImageNet-TLV Dataset":
        order_list = category_order
    else:
        order_list = ["Event A\nuntil B", "Event A and B\nUntil Event C"]
    plot_paired_boxplot(
        data,
        x_var="ltl_group",
        y_var="f1_score",
        hue="model",
        pal=LLMPlotColors,
        title_str=title,
        ax=axes[i],
        order_list=order_list,
    )
    axes[i].set_xticklabels(
        axes[i].get_xticklabels(), horizontalalignment="center", fontsize=25
    )
    axes[i].set_title(title, fontsize=40)
    axes[i].set_xlabel("TL Specification", fontsize=25)
    axes[i].set_ylabel("F1 Score", fontsize=35)
    axes[i].legend().set_visible(False)

# Adjust legend to display model names
desired_order = ["gpt-3.5-turbo-instruct", "gpt-3.5-turbo", "gpt-4", "NSVS-TL (Ours)"]
handles, labels = axes[0].get_legend_handles_labels()
new_handles = [handles[labels.index(label)] for label in desired_order]
# Adjust legend to display model names
handles, labels = axes[0].get_legend_handles_labels()
legend_labels = [labels[handles.index(h)] for h in new_handles]

fig.legend(
    handles=new_handles,
    labels=legend_labels,
    loc="lower center",  # Align the center of the legend with the anchor
    bbox_to_anchor=(0.5, 0.0),  # Anchor at the bottom center of the figure
    ncol=len(LLMModels),
    fontsize=33,  # Adjust fontsize as needed
)

# Adjust the layout of the figure to make space for the legend
# fig.subplots_adjust(bottom=0.15)  # Increase the bottom margin
plt.tight_layout()
# Adjust the layout to make space for the bottom legend
plt.subplots_adjust(bottom=0.20)  # Increase bottom margin to make space for the legend.

# Save the figure
plt.savefig("TL_Models_F1_Score_Comparison.png")
plt.savefig("figure5.png")
