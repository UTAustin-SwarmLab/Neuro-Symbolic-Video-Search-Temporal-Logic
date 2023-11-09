# %% [markdown]
# # False Positive Threshold Experiment

# %%
# Import Required Libraries
import os
from pathlib import Path

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from swarm_visualizer.lineplot import plot_overlaid_ts

# %%
#  Set Data Path
root_dir = Path("/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/experiment_2.1_nsvs_ltl/")
clip_csv_dir = root_dir / "clip/false_positive_threshold_experiment_0_to_1"
dino_csv_dir = root_dir / "dino/false_positive_threshold_experiment"
yolo_csv_dir = root_dir / "yolo/false_positive_threshold_experiment"
yolox_csv_dir = root_dir / "yolox/false_positive_threshold_experiment"
mrcnn_csv_dir = root_dir / "mrcnn/false_positive_threshold_experiment"


# %%
# Get files from directory
clip_csv_files = sorted([f for f in os.listdir(clip_csv_dir) if f.startswith("conf_")])
dino_csv_files = sorted([f for f in os.listdir(dino_csv_dir) if f.startswith("conf_")])
yolo_csv_files = sorted([f for f in os.listdir(yolo_csv_dir) if f.startswith("conf_")])
yolox_csv_files = sorted([f for f in os.listdir(yolox_csv_dir) if f.startswith("conf_")])
mrcnn_csv_files = sorted([f for f in os.listdir(mrcnn_csv_dir) if f.startswith("conf_")])

# Header of the CSV file
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


# %%
# Concat Yolo Dataframe (YOLO)

# 1. read first csv file
yolo_df = pd.read_csv(yolo_csv_dir / yolo_csv_files[0])
yolo_df.columns = header

# 2. concat all csv files
for file in yolo_csv_files[1:]:
    df = pd.read_csv(yolo_csv_dir / file)
    df.columns = header
    df = df.reindex(columns=header)  # Align columns with the master DataFrame
    yolo_df = pd.concat([yolo_df, df], ignore_index=True)
    yolo_df = yolo_df.dropna()

yolo_df.head(3)

# %%
# Concat Yolo Dataframe (DINO)

# 1. read first csv file
dino_df = pd.read_csv(dino_csv_dir / dino_csv_files[0])
dino_df.columns = header

# 2. concat all csv files
for file in dino_csv_files[1:]:
    df = pd.read_csv(dino_csv_dir / file)
    df.columns = header
    df = df.reindex(columns=header)  # Align columns with the master DataFrame
    dino_df = pd.concat([dino_df, df], ignore_index=True)
    dino_df = dino_df.dropna()

dino_df.head(3)

# %%
# Concat Yolo Dataframe (CLIP)

# 1. read first csv file
clip_df = pd.read_csv(clip_csv_dir / clip_csv_files[0])
clip_df.columns = header

# 2. concat all csv files
for file in clip_csv_files[1:]:
    df = pd.read_csv(clip_csv_dir / file)
    df.columns = header
    df = df.reindex(columns=header)  # Align columns with the master DataFrame
    clip_df = pd.concat([clip_df, df], ignore_index=True)
    clip_df = clip_df.dropna()

clip_df.head(3)

# %%
# Concat Yolo X Dataframe (Yolo X)

# 1. read first csv file
yolo_x_df = pd.read_csv(yolox_csv_dir / yolox_csv_files[0])
yolo_x_df.columns = header

# 2. concat all csv files
for file in yolox_csv_files[1:]:
    df = pd.read_csv(yolox_csv_dir / file)
    df.columns = header
    df = df.reindex(columns=header)  # Align columns with the master DataFrame
    yolo_x_df = pd.concat([yolo_x_df, df], ignore_index=True)
    yolo_x_df = yolo_x_df.dropna()

yolo_x_df.head(3)

# %%
# Concat MRCNN Dataframe (MRCNN)

# 1. read first csv file
mrcnn_df = pd.read_csv(mrcnn_csv_dir / mrcnn_csv_files[0])
mrcnn_df.columns = header

# 2. concat all csv files
for file in mrcnn_csv_files[1:]:
    df = pd.read_csv(mrcnn_csv_dir / file)
    df.columns = header
    df = df.reindex(columns=header)  # Align columns with the master DataFrame
    mrcnn_df = pd.concat([mrcnn_df, df], ignore_index=True)
    mrcnn_df = mrcnn_df.dropna()

mrcnn_df.head(3)

# %% [markdown]
# ## Plotting

# %%
yolo_avg_recall = yolo_df.groupby("mapping_false_threshold")["recall"].mean()
dino_avg_recall = dino_df.groupby("mapping_false_threshold")["recall"].mean()
clip_avg_recall = clip_df.groupby("mapping_false_threshold")["recall"].mean()
yolox_avg_recall = yolo_x_df.groupby("mapping_false_threshold")["recall"].mean()
mrcnn_avg_recall = mrcnn_df.groupby("mapping_false_threshold")["recall"].mean()

yolo_avg_precision = yolo_df.groupby("mapping_false_threshold")["precision"].mean()
dino_avg_precision = dino_df.groupby("mapping_false_threshold")["precision"].mean()
clip_avg_precision = clip_df.groupby("mapping_false_threshold")["precision"].mean()
yolox_avg_precision = yolo_x_df.groupby("mapping_false_threshold")["precision"].mean()
mrcnn_avg_precision = mrcnn_df.groupby("mapping_false_threshold")["precision"].mean()

yolo_avg_f1 = yolo_df.groupby("mapping_false_threshold")["f1_score"].mean()
dino_avg_f1 = dino_df.groupby("mapping_false_threshold")["f1_score"].mean()
clip_avg_f1 = clip_df.groupby("mapping_false_threshold")["f1_score"].mean()
yolox_avg_f1 = yolo_x_df.groupby("mapping_false_threshold")["f1_score"].mean()
mrcnn_avg_f1 = mrcnn_df.groupby("mapping_false_threshold")["f1_score"].mean()


# %%
from scipy import stats


# Function to calculate confidence interval
def compute_confidence_interval(data, conf_interval=0.95):
    """Compute the confidence interval of the given data.

    Parameters:
    - data: A pandas Series or numpy array of the data

    Returns:
    - A tuple of (mean, lower bound of CI, upper bound of CI)
    """
    mean = np.mean(data)
    stderr = stats.sem(data)
    confidence_level = conf_interval
    interval = stderr * stats.t.ppf((1 + confidence_level) / 2.0, len(data) - 1)
    return mean, mean - interval, mean + interval


# %%

# # Calculating the confidence intervals for each model and mapping_false_threshold
# yolo_confidence_intervals = yolo_df.groupby('mapping_false_threshold')['f1_score'].apply(compute_confidence_interval)
# dino_confidence_intervals = dino_df.groupby('mapping_false_threshold')['f1_score'].apply(compute_confidence_interval)
# clip_confidence_intervals = clip_df.groupby('mapping_false_threshold')['f1_score'].apply(compute_confidence_interval)
# yolox_confidence_intervals = yolo_x_df.groupby('mapping_false_threshold')['f1_score'].apply(compute_confidence_interval)
# mrcnn_confidence_intervals = mrcnn_df.groupby('mapping_false_threshold')['f1_score'].apply(compute_confidence_interval)

# # Plotting the data with confidence intervals
# fig, ax = plt.subplots(figsize=(10,6))

# # Plotting for yolo with confidence intervals
# mean, lower, upper = zip(*yolo_confidence_intervals)
# ax.plot(yolo_avg_f1.index, mean, marker='o', linestyle='-', label='yolo-v8', linewidth=2, color='#1f77b4')
# ax.fill_between(yolo_avg_f1.index, lower, upper, color='#1f77b4', alpha=0.2)

# # Plotting for dino with confidence intervals
# mean, lower, upper = zip(*dino_confidence_intervals)
# ax.plot(dino_avg_f1.index, mean, marker='s', linestyle='--', label='dino', linewidth=2, color='#aec7e8')
# ax.fill_between(dino_avg_f1.index, lower, upper, color='#aec7e8', alpha=0.2)

# # Plotting for clip with confidence intervals
# mean, lower, upper = zip(*clip_confidence_intervals)
# ax.plot(clip_avg_f1.index, mean, marker='^', linestyle='-.', label='clip', linewidth=2, color='#c49c94')
# ax.fill_between(clip_avg_f1.index, lower, upper, color='#c49c94', alpha=0.2)

# # Plotting for yolox with confidence intervals
# mean, lower, upper = zip(*yolox_confidence_intervals)
# ax.plot(yolox_avg_f1.index, mean, marker='^', linestyle='-.', label='yolo-x', linewidth=2)
# ax.fill_between(yolox_avg_f1.index, lower, upper, alpha=0.2)

# # Plotting for mrcnn with confidence intervals
# mean, lower, upper = zip(*mrcnn_confidence_intervals)
# ax.plot(mrcnn_avg_f1.index, mean, marker='^', linestyle='-.', label='mrcnn', linewidth=2)
# ax.fill_between(mrcnn_avg_f1.index, lower, upper, alpha=0.2)

# # Adding labels and title with increased font size
# ax.set_xlabel('Mapping False Threshold', fontsize=15)
# ax.set_ylabel('Average F-1', fontsize=15)
# ax.set_title('Average F-1 with Confidence Intervals vs Mapping False Threshold', fontsize=17)

# # Increasing tick font size
# ax.tick_params(axis='both', which='major', labelsize=13)

# # Adding a legend to differentiate the lines with increased font size
# ax.legend(fontsize=13)

# # Displaying the grid with reduced alpha
# ax.grid(True, alpha=0.5)

# # Display the enhanced plot with confidence intervals
# plt.show()


# # %%

# # Calculating the confidence intervals for each model and mapping_false_threshold
# yolo_confidence_intervals = yolo_df.groupby('mapping_false_threshold')['precision'].apply(compute_confidence_interval)
# dino_confidence_intervals = dino_df.groupby('mapping_false_threshold')['precision'].apply(compute_confidence_interval)
# clip_confidence_intervals = clip_df.groupby('mapping_false_threshold')['precision'].apply(compute_confidence_interval)
# yolox_confidence_intervals = yolo_x_df.groupby('mapping_false_threshold')['precision'].apply(compute_confidence_interval)
# mrcnn_confidence_intervals = mrcnn_df.groupby('mapping_false_threshold')['precision'].apply(compute_confidence_interval)

# # Plotting the data with confidence intervals
# fig, ax = plt.subplots(figsize=(10,6))

# # Plotting for yolo with confidence intervals
# mean, lower, upper = zip(*yolo_confidence_intervals)
# ax.plot(yolo_avg_precision.index, mean, marker='o', linestyle='-', label='yolo-v8', linewidth=2, color='#1f77b4')
# ax.fill_between(yolo_avg_precision.index, lower, upper, color='#1f77b4', alpha=0.2)

# # Plotting for dino with confidence intervals
# mean, lower, upper = zip(*dino_confidence_intervals)
# ax.plot(dino_avg_precision.index, mean, marker='s', linestyle='--', label='dino', linewidth=2, color='#aec7e8')
# ax.fill_between(dino_avg_precision.index, lower, upper, color='#aec7e8', alpha=0.2)

# # Plotting for clip with confidence intervals
# mean, lower, upper = zip(*clip_confidence_intervals)
# ax.plot(clip_avg_precision.index, mean, marker='^', linestyle='-.', label='clip', linewidth=2, color='#c49c94')
# ax.fill_between(clip_avg_precision.index, lower, upper, color='#c49c94', alpha=0.2)

# # Plotting for yolox with confidence intervals
# mean, lower, upper = zip(*yolox_confidence_intervals)
# ax.plot(yolox_avg_precision.index, mean, marker='^', linestyle='-.', label='yolo-x', linewidth=2)
# ax.fill_between(yolox_avg_precision.index, lower, upper, alpha=0.2)

# # Plotting for mrcnn with confidence intervals
# mean, lower, upper = zip(*mrcnn_confidence_intervals)
# ax.plot(mrcnn_avg_precision.index, mean, marker='^', linestyle='-.', label='mrcnn', linewidth=2)
# ax.fill_between(mrcnn_avg_precision.index, lower, upper, alpha=0.2)

# # Adding labels and title with increased font size
# ax.set_xlabel('Mapping False Threshold', fontsize=15)
# ax.set_ylabel('Average Precision', fontsize=15)
# ax.set_title('Average Precision with Confidence Intervals vs Mapping False Threshold', fontsize=17)

# # Increasing tick font size
# ax.tick_params(axis='both', which='major', labelsize=13)

# # Adding a legend to differentiate the lines with increased font size
# ax.legend(fontsize=13)

# # Displaying the grid with reduced alpha
# ax.grid(True, alpha=0.5)

# # Display the enhanced plot with confidence intervals
# plt.show()


# # %%

# # Calculating the confidence intervals for each model and mapping_false_threshold
# yolo_confidence_intervals = yolo_df.groupby('mapping_false_threshold')['recall'].apply(compute_confidence_interval)
# dino_confidence_intervals = dino_df.groupby('mapping_false_threshold')['recall'].apply(compute_confidence_interval)
# clip_confidence_intervals = clip_df.groupby('mapping_false_threshold')['recall'].apply(compute_confidence_interval)
# yolox_confidence_intervals = yolo_x_df.groupby('mapping_false_threshold')['recall'].apply(compute_confidence_interval)
# mrcnn_confidence_intervals = mrcnn_df.groupby('mapping_false_threshold')['recall'].apply(compute_confidence_interval)

# # Plotting the data with confidence intervals
# fig, ax = plt.subplots(figsize=(10,6))

# # Plotting for yolo with confidence intervals
# mean, lower, upper = zip(*yolo_confidence_intervals)
# ax.plot(yolo_avg_recall.index, mean, marker='o', linestyle='-', label='Yolo', linewidth=2, color='#1f77b4')
# ax.fill_between(yolo_avg_recall.index, lower, upper, color='#1f77b4', alpha=0.2)

# # Plotting for dino with confidence intervals
# mean, lower, upper = zip(*dino_confidence_intervals)
# ax.plot(dino_avg_recall.index, mean, marker='s', linestyle='--', label='Dino', linewidth=2, color='#aec7e8')
# ax.fill_between(dino_avg_recall.index, lower, upper, color='#aec7e8', alpha=0.2)

# # Plotting for clip with confidence intervals
# mean, lower, upper = zip(*clip_confidence_intervals)
# ax.plot(clip_avg_recall.index, mean, marker='^', linestyle='-.', label='Clip', linewidth=2, color='#c49c94')
# ax.fill_between(clip_avg_recall.index, lower, upper, color='#c49c94', alpha=0.2)

# # Plotting for yolox with confidence intervals
# mean, lower, upper = zip(*yolox_confidence_intervals)
# ax.plot(yolox_avg_recall.index, mean, marker='^', linestyle='-.', label='yolo-x', linewidth=2)
# ax.fill_between(yolox_avg_recall.index, lower, upper, alpha=0.2)

# # Plotting for mrcnn with confidence intervals
# mean, lower, upper = zip(*mrcnn_confidence_intervals)
# ax.plot(mrcnn_avg_recall.index, mean, marker='^', linestyle='-.', label='mrcnn', linewidth=2)
# ax.fill_between(mrcnn_avg_recall.index, lower, upper, alpha=0.2)

# # Adding labels and title with increased font size
# ax.set_xlabel('Mapping False Threshold', fontsize=15)
# ax.set_ylabel('Average Recall', fontsize=15)
# ax.set_title('Average Recall with Confidence Intervals vs Mapping False Threshold', fontsize=17)

# # Increasing tick font size
# ax.tick_params(axis='both', which='major', labelsize=13)

# # Adding a legend to differentiate the lines with increased font size
# ax.legend(fontsize=13)

# # Displaying the grid with reduced alpha
# ax.grid(True, alpha=0.5)

# # Display the enhanced plot with confidence intervals
# plt.show()


# # %%

# # Calculating the confidence intervals for each model and mapping_false_threshold
# yolo_confidence_intervals = yolo_df.groupby('mapping_false_threshold')['f1_score'].apply(compute_confidence_interval)
# dino_confidence_intervals = dino_df.groupby('mapping_false_threshold')['f1_score'].apply(compute_confidence_interval)
# clip_confidence_intervals = clip_df.groupby('mapping_false_threshold')['f1_score'].apply(compute_confidence_interval)
# yolox_confidence_intervals = yolo_x_df.groupby('mapping_false_threshold')['f1_score'].apply(compute_confidence_interval)
# mrcnn_confidence_intervals = mrcnn_df.groupby('mapping_false_threshold')['f1_score'].apply(compute_confidence_interval)

# # Plotting the data with confidence intervals
# fig, ax = plt.subplots(figsize=(10,6))

# # Plotting for yolo with confidence intervals
# mean, lower, upper = zip(*yolo_confidence_intervals)
# ax.plot(yolo_avg_f1.index, mean, marker='o', linestyle='-', label='yolo-v8', linewidth=2, color='#1f77b4')
# ax.fill_between(yolo_avg_f1.index, lower, upper, color='#1f77b4', alpha=0.2)

# # Plotting for dino with confidence intervals
# mean, lower, upper = zip(*dino_confidence_intervals)
# ax.plot(dino_avg_f1.index, mean, marker='s', linestyle='--', label='dino', linewidth=2, color='#aec7e8')
# ax.fill_between(dino_avg_f1.index, lower, upper, color='#aec7e8', alpha=0.2)

# # Plotting for clip with confidence intervals
# mean, lower, upper = zip(*clip_confidence_intervals)
# ax.plot(clip_avg_f1.index, mean, marker='^', linestyle='-.', label='clip', linewidth=2, color='#c49c94')
# ax.fill_between(clip_avg_f1.index, lower, upper, color='#c49c94', alpha=0.2)

# # Plotting for yolox with confidence intervals
# mean, lower, upper = zip(*yolox_confidence_intervals)
# ax.plot(yolox_avg_f1.index, mean, marker='^', linestyle='-.', label='yolo-x', linewidth=2)
# ax.fill_between(yolox_avg_f1.index, lower, upper, alpha=0.2)

# # Plotting for mrcnn with confidence intervals
# mean, lower, upper = zip(*mrcnn_confidence_intervals)
# ax.plot(mrcnn_avg_f1.index, mean, marker='^', linestyle='-.', label='mrcnn', linewidth=2)
# ax.fill_between(mrcnn_avg_f1.index, lower, upper, alpha=0.2)

# # Adding labels and title with increased font size
# ax.set_xlabel('Mapping False Threshold', fontsize=15)
# ax.set_ylabel('Average F-1', fontsize=15)
# ax.set_title('Average F-1 with Confidence Intervals vs Mapping False Threshold', fontsize=17)

# # Increasing tick font size
# ax.tick_params(axis='both', which='major', labelsize=13)

# # Adding a legend to differentiate the lines with increased font size
# ax.legend(fontsize=13)

# # Displaying the grid with reduced alpha
# ax.grid(True, alpha=0.5)

# # Display the enhanced plot with confidence intervals
# plt.show()


# %% [markdown]
# # Swarm Visualizer

# %%
from swarm_visualizer.lineplot import plot_overlaid_ts

# %%
yolo_avg_f1 = yolo_df.groupby("mapping_false_threshold")["f1_score"].mean()
dino_avg_f1 = dino_df.groupby("mapping_false_threshold")["f1_score"].mean()
clip_avg_f1 = clip_df.groupby("mapping_false_threshold")["f1_score"].mean()
yolox_avg_f1 = yolo_x_df.groupby("mapping_false_threshold")["f1_score"].mean()
mrcnn_avg_f1 = mrcnn_df.groupby("mapping_false_threshold")["f1_score"].mean()
# Calculating the confidence intervals for each model and mapping_false_threshold
yolo_confidence_intervals = yolo_df.groupby("mapping_false_threshold")["f1_score"].apply(
    compute_confidence_interval
)
dino_confidence_intervals = dino_df.groupby("mapping_false_threshold")["f1_score"].apply(
    compute_confidence_interval
)
clip_confidence_intervals = clip_df.groupby("mapping_false_threshold")["f1_score"].apply(
    compute_confidence_interval
)
yolox_confidence_intervals = yolo_x_df.groupby("mapping_false_threshold")["f1_score"].apply(
    compute_confidence_interval
)
mrcnn_confidence_intervals = mrcnn_df.groupby("mapping_false_threshold")["f1_score"].apply(
    compute_confidence_interval
)
# Plotting for clip with confidence intervals
yolo_f1_mean, yolo_f1_lower, yolo_f1_upper = zip(*yolo_confidence_intervals)
dino_f1_mean, dino_f1_lower, dino_f1_upper = zip(*dino_confidence_intervals)
clip_f1_mean, clip_f1_lower, clip_f1_upper = zip(*clip_confidence_intervals)
yolox_f1_mean, yolox_f1_lower, yolox_f1_upper = zip(*yolox_confidence_intervals)
mrcnn_f1_mean, mrcnn_f1_lower, mrcnn_f1_upper = zip(*mrcnn_confidence_intervals)

f1_false_positive = {
    "yolov8": {
        "xvec": yolo_avg_f1.index,
        "ts_vector": yolo_f1_mean,
        "lw": 3,
        "linestyle": "-",
        "color": "#1f77b4",
    },
    "grounding-dino": {
        "xvec": dino_avg_f1.index,
        "ts_vector": dino_f1_mean,
        "lw": 3,
        "linestyle": "-",
        "color": "#d62728",
    },
    "clip": {
        "xvec": clip_avg_f1.index,
        "ts_vector": clip_f1_mean,
        "lw": 3,
        "linestyle": "-",
        "color": "#2ca02c",
    },
    # "yolox": {
    #     "xvec": yolox_avg_f1.index,
    #     "ts_vector": yolox_f1_mean,
    #     "lw": 3,
    #     "linestyle": "-",
    #     "color":"#ff7f0e"
    # },
    # "mrcnn": {
    #     "xvec": mrcnn_avg_f1.index,
    #     "ts_vector": mrcnn_f1_mean,
    #     "lw": 3,
    #     "linestyle": "-",
    #     "color":"#9467bd"
    # }
}


# %%
yolo_avg_precision = yolo_df.groupby("mapping_false_threshold")["precision"].mean()
dino_avg_precision = dino_df.groupby("mapping_false_threshold")["precision"].mean()
clip_avg_precision = clip_df.groupby("mapping_false_threshold")["precision"].mean()
yolox_avg_precision = yolo_x_df.groupby("mapping_false_threshold")["precision"].mean()
mrcnn_avg_precision = mrcnn_df.groupby("mapping_false_threshold")["precision"].mean()
# Calculating the confidence intervals for each model and mapping_false_threshold
yolo_confidence_intervals = yolo_df.groupby("mapping_false_threshold")["precision"].apply(
    compute_confidence_interval
)
dino_confidence_intervals = dino_df.groupby("mapping_false_threshold")["precision"].apply(
    compute_confidence_interval
)
clip_confidence_intervals = clip_df.groupby("mapping_false_threshold")["precision"].apply(
    compute_confidence_interval
)
yolox_confidence_intervals = yolo_x_df.groupby("mapping_false_threshold")["precision"].apply(
    compute_confidence_interval
)
mrcnn_confidence_intervals = mrcnn_df.groupby("mapping_false_threshold")["precision"].apply(
    compute_confidence_interval
)
# Plotting for clip with confidence intervals
yolo_precision_mean, yolo_precision_lower, yolo_precision_upper = zip(*yolo_confidence_intervals)
dino_precision_mean, dino_precision_lower, dino_precision_upper = zip(*dino_confidence_intervals)
clip_precision_mean, clip_precision_lower, clip_precision_upper = zip(*clip_confidence_intervals)
yolox_precision_mean, yolox_precision_lower, yolox_precision_upper = zip(*yolox_confidence_intervals)
mrcnn_precision_mean, mrcnn__precision_ower, mrcnn_precision_upper = zip(*mrcnn_confidence_intervals)

precision_false_positive = {
    "yolov8": {
        "xvec": yolo_avg_precision.index,
        "ts_vector": yolo_precision_mean,
        "lw": 3,
        "linestyle": "-",
        "color": "#1f77b4",
    },
    "grounding-dino": {
        "xvec": dino_avg_precision.index,
        "ts_vector": dino_precision_mean,
        "lw": 3,
        "linestyle": "-",
        "color": "#d62728",
    },
    "clip": {
        "xvec": clip_avg_precision.index,
        "ts_vector": clip_precision_mean,
        "lw": 3,
        "linestyle": "-",
        "color": "#2ca02c",
    },
    # "yolox": {
    #     "xvec": yolox_avg_precision.index,
    #     "ts_vector": yolox_precision_mean,
    #     "lw": 3,
    #     "linestyle": "-",
    #     "color":"#ff7f0e"
    # },
    # "mrcnn": {
    #     "xvec": mrcnn_avg_precision.index,
    #     "ts_vector": mrcnn_precision_mean,
    #     "lw": 3,
    #     "linestyle": "-",
    #     "color":"#9467bd"
    # }
}


# %%
yolo_avg_recall = yolo_df.groupby("mapping_false_threshold")["recall"].mean()
dino_avg_recall = dino_df.groupby("mapping_false_threshold")["recall"].mean()
clip_avg_recall = clip_df.groupby("mapping_false_threshold")["recall"].mean()
yolox_avg_recall = yolo_x_df.groupby("mapping_false_threshold")["recall"].mean()
mrcnn_avg_recall = mrcnn_df.groupby("mapping_false_threshold")["recall"].mean()

# Calculating the confidence intervals for each model and mapping_false_threshold
yolo_confidence_intervals = yolo_df.groupby("mapping_false_threshold")["recall"].apply(
    compute_confidence_interval
)
dino_confidence_intervals = dino_df.groupby("mapping_false_threshold")["recall"].apply(
    compute_confidence_interval
)
clip_confidence_intervals = clip_df.groupby("mapping_false_threshold")["recall"].apply(
    compute_confidence_interval
)
yolox_confidence_intervals = yolo_x_df.groupby("mapping_false_threshold")["recall"].apply(
    compute_confidence_interval
)
mrcnn_confidence_intervals = mrcnn_df.groupby("mapping_false_threshold")["recall"].apply(
    compute_confidence_interval
)

# Plotting for clip with confidence intervals
yolo_recall_mean, yolo_recall_lower, yolo_recall_upper = zip(*yolo_confidence_intervals)
dino_recall_mean, dino_recall_lower, dino_recall_upper = zip(*dino_confidence_intervals)
clip_recall_mean, clip_recall_lower, clip_recall_upper = zip(*clip_confidence_intervals)
yolox_recall_mean, yolox_recall_lower, yolox_recall_upper = zip(*yolox_confidence_intervals)
mrcnn_recall_mean, mrcnn_recall_lower, mrcnn_recall_upper = zip(*mrcnn_confidence_intervals)

recall_false_positive = {
    "yolov8": {
        "xvec": yolo_avg_recall.index,
        "ts_vector": yolo_recall_mean,
        "lw": 3,
        "linestyle": "-",
        "color": "#1f77b4",
    },
    "grounding-dino": {
        "xvec": dino_avg_recall.index,
        "ts_vector": dino_recall_mean,
        "lw": 3,
        "linestyle": "-",
        "color": "#d62728",
    },
    "clip": {
        "xvec": clip_avg_recall.index,
        "ts_vector": clip_recall_mean,
        "lw": 3,
        "linestyle": "-",
        "color": "#2ca02c",
    },
    # "yolox": {
    #     "xvec": yolox_avg_recall.index,
    #     "ts_vector": yolox_recall_mean,
    #     "lw": 3,
    #     "linestyle": "-",
    #     "color":"#ff7f0e"
    # },
    # "mrcnn": {
    #     "xvec": mrcnn_avg_recall.index,
    #     "ts_vector": mrcnn_recall_mean,
    #     "lw": 3,
    #     "linestyle": "-",
    #     "color":"#9467bd"
    # }
}

mrcnn_recall_mean

# %%

from ns_vfs.common.utility import load_pickle_to_dict

data = load_pickle_to_dict(
    "/opt/Neuro-Symbolic-Video-Frame-Search/experiments/mapping_estimiation_plot_data.pkl"
)

# Converting array to a Pandas Series
xvec = pd.Float64Index(np.linspace(1.0, 5.0, len(pd.Series(data["xx"]))))
xvec = tuple(data["xx"])
mapping_estimation = {
    "yolov8": {
        "xvec": xvec,
        "ts_vector": tuple(data["yolo_acc"]),
        "lw": 3,
        "linestyle": "-",
        "color": "#1f77b4",
    },
    "grounding-dino": {
        "xvec": xvec,
        "ts_vector": tuple(data["dino_acc"]),
        "lw": 3,
        "linestyle": "-",
        "color": "#d62728",
    },
    "clip": {
        "xvec": xvec,
        "ts_vector": tuple(data["clip_acc"]),
        "lw": 3,
        "linestyle": "-",
        "color": "#2ca02c",
    },
    "yolov8-est.": {
        "xvec": xvec,
        "ts_vector": tuple(data["yolo_mapping_fun"]),
        "lw": 3,
        "linestyle": "--",  # Dashed line
        "color": "#3147a8",  # Deep blue
    },
    "grounding-dino-est.": {
        "xvec": xvec,
        "ts_vector": tuple(data["dino_mapping_fun"]),
        "lw": 3,
        "linestyle": "--",  # Dashed line
        "color": "#cf704a",  # Crimson
    },
    "clip-est.": {
        "xvec": xvec,
        "ts_vector": tuple(data["clip_mapping_fun"]),
        "lw": 3,
        "linestyle": "--",  # Dashed line
        "color": "#8ccf4a",  # Forest green
    },
}

data

# %%
# Create a figure with 1 row and 3 columns
"""
    font_size: float = 20,
    legend_font_size: float = 14,
    xtick_label_size: float = 14,
    ytick_label_size: float = 14,
    markersize: float = 10,
"""
sns.set_style(style="darkgrid")
sns.set_color_codes()
sns.set()
# Recreate the figure with 2 rows and 2 columns
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
axes = axes.flatten()  # Flatten the axes array for easy indexing

handles, labels = [], []


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


# set_plot_properties()
# Data for the subplots - using the same data for simplicity, but you can change this
data_for_subplots = {
    "Precision": precision_false_positive,
    "Recall": recall_false_positive,
    "F1 Score": f1_false_positive,
    "Accuracy": mapping_estimation,
}

# Loop through each subplot and call the function
for i, (title, data) in enumerate(data_for_subplots.items()):
    print(title)
    if title == "Accuracy":
        x_label = "Confidence"
    else:
        x_label = "False Positive Threshold"
    plot_overlaid_ts(
        normalized_ts_dict=data,
        title_str=False,
        ylabel=title,
        xlabel=x_label,
        fontsize=10,
        legend_present=False,
        ax=axes[i],
    )
    # Get handles and labels from each subplot
    for handle, label in zip(*axes[i].get_legend_handles_labels()):
        if label not in labels:
            handles.append(handle)
            labels.append(label)

fig.set_constrained_layout(False)
fig.subplots_adjust(bottom=0.15, wspace=0.25, hspace=0.25)

# Create a unified legend below the subplots
# Manually setting the bbox_to_anchor to ensure the legend is outside the plots
# Reordering

desired_order = ["yolov8", "yolov8-est.", "clip", "clip-est.", "grounding-dino", "grounding-dino-est."]

new_handles = [handles[labels.index(label)] for label in desired_order]
new_labels = desired_order

fig.legend(new_handles, new_labels, loc="lower center", bbox_to_anchor=(0.5, 0.020), ncol=3, fontsize=15)

# Display the plot
plt.show()
plt.savefig("calibration_plot.png")

# %%

# Create a figure with 2 rows and 2 colu
