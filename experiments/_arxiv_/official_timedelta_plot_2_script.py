# %% [markdown]
# # False Positive Threshold Experiment

# %%
# Import Required Libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv(
    "/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/experiment_2.1_nsvs_ltl/yolo/yolo_benchmark_search_result_20231031_113154.csv"
)
# timedelta_yolo_benchmark_search_result_20231029_042913.csv
header = [
    "dataset",
    "ltl_group",
    "timedelta",
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

#
window_size = 200  # Example window size
df["f1_score_smoothed"] = df["f1_score"].rolling(window=window_size).mean()
#

# Set style and context to make the plot look more aesthetically pleasing
sns.set_style("darkgrid")
sns.set_context("paper")

plt.figure(figsize=(8, 6))
# sns.lineplot(
#     x="timedelta",
#     y="f1_score",
#     data=df,
#     markersize=8,
#     color="blue",
#     label="nsvs-ltl-yolo",
#     alpha=0.5,
#     errorbar=("ci", 20),
# )
sns.lineplot(
    x="timedelta",
    y="f1_score_smoothed",
    data=df,
    markersize=8,
    color="red",
    label="Smoothed",
    alpha=0.8,
)

plt.title("LTL Frame Search Accuracy")
plt.ylabel("F1 Score")
plt.xlabel("Time Delta in seconds")

# Show plot
plt.ylim(0.0, 1.0)
plt.tight_layout()
plt.legend()
plt.grid(True)
plt.savefig("fig_2_test")
