from scipy.interpolate import make_interp_spline
import matplotlib.colors as mcolors
from collections import defaultdict
import matplotlib.pyplot as plt

import numpy as np
import json
import os


folder1 = "/nas/mars/experiment_result/nsvs/nsvs2-prelims/nsvs"
folder2 = "/nas/mars/experiment_result/nsvs/nsvs2-prelims/internvl2"
out_path1 = "plots/plot_duration.png"
out_path2 = "plots/plot_complexity.png"


def compute_statistics():
    def process_folder(folder):
        TP = FP = FN = 0
        per_file_f1 = []

        for fname in os.listdir(folder):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(folder, fname), "r") as f:
                data = json.load(f)

            pred = set(map(int, data.get("frames_of_interest", [])))
            gt   = set(map(int, data.get("ground_truth", [])))

            tp = len(pred & gt)
            fp = len(pred - gt)
            fn = len(gt - pred)

            TP += tp; FP += fp; FN += fn

            precision_f = tp / (tp + fp) if (tp + fp) else 0.0
            recall_f    = tp / (tp + fn) if (tp + fn) else 0.0
            f1_file     = (2 * precision_f * recall_f / (precision_f + recall_f)
                           if (precision_f + recall_f) else 0.0)
            per_file_f1.append(float(f1_file))

        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall    = TP / (TP + FN) if (TP + FN) else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "TP": TP, "FP": FP, "FN": FN,
            "per_file_f1": per_file_f1,
        }

    stats1 = process_folder(folder1)
    stats2 = process_folder(folder2)

    # Print nicely
    for folder, stats in [(folder1, stats1), (folder2, stats2)]:
        print(f"[{folder}] Overall metrics:")
        print(f"  Precision: {stats['precision']:.4f}")
        print(f"  Recall:    {stats['recall']:.4f}")
        print(f"  F1:        {stats['f1']:.4f}")
        print()

    return {"folder1": stats1, "folder2": stats2}

def plot1():
    # Colors
    color_nsvs = "#1f77b4"
    color_ivl2 = "#b4421f"

    # Smoothing controls
    bandwidth = 20
    smooth_band   = 600
    smooth_center = 11

    def collect_points(folder):
        TP = FP = FN = 0
        xs, ys = [], []

        for fname in os.listdir(folder):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(folder, fname), "r") as f:
                data = json.load(f)

            pred = set(map(int, data.get("frames_of_interest", [])))
            gt   = set(map(int, data.get("ground_truth", [])))

            tp = len(pred & gt)
            fp = len(pred - gt)
            fn = len(gt - pred)

            TP += tp; FP += fp; FN += fn

            # per-file F1
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall    = tp / (tp + fn) if (tp + fn) else 0.0
            f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

            nframes = int(data.get("number_of_frames", 0))
            xs.append(int(nframes / 4))   # minutes
            ys.append(float(f1))

        return np.array(xs, dtype=int), np.array(ys, dtype=float)

    def compute_envelope(xs, ys):
        if len(xs) == 0:
            return None

        grouped = defaultdict(list)
        for x, y in zip(xs, ys):
            grouped[int(x)].append(float(y))

        durations = np.array(sorted(grouped.keys()))
        if len(durations) == 0:
            return None

        ymin_raw = np.array([min(grouped[d]) for d in durations])
        ymax_raw = np.array([max(grouped[d]) for d in durations])
        ymid_raw = (ymin_raw + ymax_raw) / 2.0

        # smooth envelope by local window
        ymin_s, ymax_s = [], []
        for d in durations:
            mask = np.abs(durations - d) <= bandwidth
            ymin_s.append(ymin_raw[mask].min())
            ymax_s.append(ymax_raw[mask].max())
        ymin_s, ymax_s = np.array(ymin_s), np.array(ymax_s)
        ymid_s = (ymin_s + ymax_s) / 2.0

        if len(durations) >= 4:
            x_band = np.linspace(durations.min(), durations.max(), smooth_band)
            ymin_smooth = make_interp_spline(durations, ymin_s, k=3)(x_band)
            ymax_smooth = make_interp_spline(durations, ymax_s, k=3)(x_band)

            x_center = np.linspace(durations.min(), durations.max(), smooth_center)
            ymid_smooth = make_interp_spline(durations, ymid_s, k=3)(x_center)
        else:
            # fallback: unsmoothed
            x_band, ymin_smooth, ymax_smooth = durations, ymin_s, ymax_s
            x_center, ymid_smooth = durations, ymid_s

        return x_band, ymin_smooth, ymax_smooth, x_center, ymid_smooth

    # Collect
    xs1, ys1 = collect_points(folder1)
    xs2, ys2 = collect_points(folder2)

    env1 = compute_envelope(xs1, ys1)
    env2 = compute_envelope(xs2, ys2)

    if env1 is None and env2 is None:
        print("Not enough data with valid 'number_of_frames' to plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    # Helper: shrink already-smoothed band by factor% around its own center line
    def shrink_band_on_spline(ymin_spline, ymax_spline, factor=0.5):
        center = (ymin_spline + ymax_spline) / 2.0
        ymin_new = center - factor * (center - ymin_spline)
        ymax_new = center + factor * (ymax_spline - center)
        return ymin_new, ymax_new, center

    if env1 is not None:
        x_band, ymin_s, ymax_s, x_center, ymid_s = env1
        ymin_plot, ymax_plot, _ = shrink_band_on_spline(ymin_s, ymax_s)
        base1 = mcolors.to_rgb(color_nsvs)
        darker1 = tuple(max(0.0, c * 0.75) for c in base1)
        ax.fill_between(x_band, ymin_plot, ymax_plot, color=base1, alpha=0.22)
        ax.plot(x_center, ymid_s, linewidth=2.5, color=darker1, label="NSVS")

    if env2 is not None:
        x_band, ymin_s, ymax_s, x_center, ymid_s = env2
        ymin_plot, ymax_plot, _ = shrink_band_on_spline(ymin_s, ymax_s)
        base2 = mcolors.to_rgb(color_ivl2)
        darker2 = tuple(max(0.0, c * 0.75) for c in base2)
        ax.fill_between(x_band, ymin_plot, ymax_plot, color=base2, alpha=0.22)
        ax.plot(x_center, ymid_s, linewidth=2.5, color=darker2, label="InternVL2")

    # Labels / styling (no title, larger fonts)
    ax.set_xlabel("Minutes", fontsize=17)
    ax.set_ylabel("F1 Score", fontsize=17)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=15)

    fig.tight_layout()
    fig.savefig(out_path1, dpi=200)
    plt.close(fig)


def plot2():
    bins = [1, 2, 3]

    def normalize_prop_names(prop_obj):
        if isinstance(prop_obj, dict):
            return {str(k) for k in prop_obj.keys() if str(k).strip()}
        elif isinstance(prop_obj, (list, tuple, set)):
            flat = []
            for item in prop_obj:
                if isinstance(item, (list, tuple, set)):
                    flat.extend(item)
                else:
                    flat.append(item)
            return {str(x) for x in flat if str(x).strip()}
        elif prop_obj:
            return {str(prop_obj)}
        return set()

    def compute_by_props(folder):
        by_props = defaultdict(list)
        for fname in os.listdir(folder):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(folder, fname), "r") as f:
                data = json.load(f)

            pred = set(map(int, data.get("frames_of_interest", [])))
            gt   = set(map(int, data.get("ground_truth", [])))

            tp = len(pred & gt)
            fp = len(pred - gt)
            fn = len(gt - pred)

            precision_f = tp / (tp + fp) if (tp + fp) else 0.0
            recall_f    = tp / (tp + fn) if (tp + fn) else 0.0
            f1_file     = (2 * precision_f * recall_f / (precision_f + recall_f)
                           if (precision_f + recall_f) else 0.0)

            prop_names = normalize_prop_names(data.get("propositions", []))
            n_props = len(prop_names)

            if n_props in bins:
                by_props[n_props].append(float(f1_file))
        return by_props

    by_props1 = compute_by_props(folder1)
    by_props2 = compute_by_props(folder2)

    # Paired layout: offset each dataset around integer ticks
    positions1 = [p - 0.15 for p in bins]
    positions2 = [p + 0.15 for p in bins]

    data1 = [by_props1.get(k, []) for k in bins]
    data2 = [by_props2.get(k, []) for k in bins]

    fig, ax = plt.subplots(figsize=(9, 6))  # bigger canvas

    bp1 = ax.boxplot(
        data1,
        positions=positions1,
        widths=0.3,
        patch_artist=True,
        showfliers=False,
    )
    for box in bp1['boxes']:
        box.set_facecolor("#1f77b4")
        box.set_alpha(0.35)
        box.set_edgecolor("#1f77b4")
        box.set_linewidth(1.5)
    for element in ['whiskers', 'caps', 'medians']:
        for artist in bp1[element]:
            artist.set_color("#1f77b4")
            artist.set_linewidth(1.5)

    bp2 = ax.boxplot(
        data2,
        positions=positions2,
        widths=0.3,
        patch_artist=True,
        showfliers=False,
    )
    for box in bp2['boxes']:
        box.set_facecolor("#b4421f")
        box.set_alpha(0.35)
        box.set_edgecolor("#b4421f")
        box.set_linewidth(1.5)
    for element in ['whiskers', 'caps', 'medians']:
        for artist in bp2[element]:
            artist.set_color("#b4421f")
            artist.set_linewidth(1.5)

    # Axes / labels
    ax.set_xticks(bins)
    ax.set_xticklabels([p for p in bins], fontsize=15)
    ax.set_xlabel("Number of Propositions", fontsize=17)
    ax.set_ylabel("F1 Score", fontsize=17)
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Legend
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["NSVS", "InternVL2"], fontsize=13)

    fig.tight_layout()
    fig.savefig(out_path2, dpi=200)
    plt.close(fig)

compute_statistics()
plot1()
plot2()
