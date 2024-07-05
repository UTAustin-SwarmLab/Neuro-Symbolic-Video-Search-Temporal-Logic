# %% [markdown]
# # False Positive Threshold Experiment

# %%
# Import Required Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from swarm_visualizer.boxplot import plot_grouped_boxplot
from swarm_visualizer.utility.general_utils import set_plot_properties

df = pd.read_csv(
    "/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/experiment_2.1_nsvs_ltl/mrcnn/nuscenemrcnn_benchmark_search_result_20231104_022102.csv"
)
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

f1_scores_per_group = df.groupby("ltl_group")["f1_score"].apply(list)
_X_DATA = np.concatenate(
    [
        f1_scores_per_group["(prop1&prop2)Uprop3"],
        f1_scores_per_group["prop1Uprop2"],
    ]
)
_X_LABEL = np.concatenate(
    [
        np.repeat("(prop1&prop2)Uprop3", len(f1_scores_per_group["(prop1&prop2)Uprop3"])),
        np.repeat("prop1Uprop2", len(f1_scores_per_group["prop1Uprop2"])),
    ],
)

plot_dict = pd.DataFrame({"f1_score": _X_DATA, "ltl_group": _X_LABEL})


set_plot_properties()

fig, ax = plt.subplots(figsize=(10, 10))
# Create a grouped violinplot
# plot_grouped_violinplot(
#     df=plot_dict, x_var="ltl_group", y_var="f1_score", title_str="Grouped Violinplot", ax=ax
# )

plot_grouped_boxplot(df=plot_dict, x_var="ltl_group", y_var="f1_score", title_str="Grouped Violinplot", ax=ax)

# Save the plot
plt.savefig("_nuscene_grouped_violinplot.png", dpi=600)
# save_fig(fig, "grouped_violinplot.png", dpi=600)

# _X1_DATA = np.arange(0, 5, 0.05) + np.random.normal(0, 0.1, 100)
# _X2_DATA = np.arange(0, 10, 0.1) + np.random.normal(0, 0.1, 100)

# _X_DATA = np.concatenate([_X1_DATA, _X2_DATA])
# _X_LABEL = np.concatenate([np.repeat("$x_1$", 100), np.repeat("$x_2$", 100)])

# _GROUPS = np.concatenate(
#     [
#         np.repeat("a", 50),
#         np.repeat("b", 50),
#         np.repeat("a", 50),
#         np.repeat("b", 50),
#     ]
# )

# _DATA_FRAME = pd.DataFrame({r"$f_\theta(x)$": _X_DATA, "$x$": _X_LABEL, "Groups": _GROUPS})
