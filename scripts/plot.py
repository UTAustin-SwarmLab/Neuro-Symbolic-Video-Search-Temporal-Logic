from scipy.interpolate import make_interp_spline
import matplotlib.colors as mcolors
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import json
import os


folder1 = "/nas/mars/experiment_result/nsvs/nsvs2-prelims/nsvs"
folder2 = "/nas/mars/experiment_result/nsvs/nsvs2-prelims/InternVL2-8B_frame_wise"
folder3 = "/nas/mars/experiment_result/nsvs/nsvs2-prelims/InternVL2-8B_sliding_window"

out_path_duration   = "scripts/plots/plot_duration.png"
out_path_complexity = "scripts/plots/plot_complexity.png"

labels = ["NSVS", "Frame-Wise", "Sliding-Window"]
colors = ["#1f77b4", "#b4421f", "#2ca02c"]

complexity_bins = [1, 2, 3]


def _safe_json_load(path):
    with open(path, "r") as f:
        return json.load(f)

def _per_file_stats(pred, gt):
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    precision_f = tp / (tp + fp) if (tp + fp) else 0.0
    recall_f    = tp / (tp + fn) if (tp + fn) else 0.0
    f1_file     = (2 * precision_f * recall_f / (precision_f + recall_f)
                   if (precision_f + recall_f) else 0.0)
    return tp, fp, fn, precision_f, recall_f, f1_file

def _iter_json(folder):
    for fname in os.listdir(folder):
        if fname.endswith(".json"):
            yield os.path.join(folder, fname)

def compute_statistics(folders):
    out = {}
    for folder in folders:
        TP = FP = FN = 0
        per_file_f1 = []

        for fpath in _iter_json(folder):
            data = _safe_json_load(fpath)
            pred = set(map(int, data.get("frames_of_interest", [])))
            gt   = set(map(int, data.get("ground_truth", [])))

            tp, fp, fn, _, _, f1_file = _per_file_stats(pred, gt)
            TP += tp; FP += fp; FN += fn
            per_file_f1.append(float(f1_file))

        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall    = TP / (TP + FN) if (TP + FN) else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        out[folder] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "TP": TP, "FP": FP, "FN": FN,
            "per_file_f1": per_file_f1,
        }

    # Pretty print
    for folder, stats in out.items():
        print(f"[{folder}] Overall metrics:")
        print(f"  Precision: {stats['precision']:.4f}")
        print(f"  Recall:    {stats['recall']:.4f}")
        print(f"  F1:        {stats['f1']:.4f}\n")

    return out


def _collect_duration_points(folder):
    xs, ys = [], []
    for fpath in _iter_json(folder):
        data = _safe_json_load(fpath)
        pred = set(map(int, data.get("frames_of_interest", [])))
        gt   = set(map(int, data.get("ground_truth", [])))
        tp, fp, fn, _, _, f1 = _per_file_stats(pred, gt)

        nframes = int(data.get("number_of_frames", 0))
        if nframes <= 0:
            continue
        minutes = int(nframes / 4)  # your original definition
        xs.append(minutes)
        ys.append(float(f1))
    return np.array(xs, dtype=int), np.array(ys, dtype=float)

def _compute_envelope(xs, ys, bandwidth=20, smooth_band=600, smooth_center=11):
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
        x_band, ymin_smooth, ymax_smooth = durations, ymin_s, ymax_s
        x_center, ymid_smooth = durations, ymid_s

    return x_band, ymin_smooth, ymax_smooth, x_center, ymid_smooth

def _shrink_band(ymin_spline, ymax_spline, factor=0.5):
    center = (ymin_spline + ymax_spline) / 2.0
    ymin_new = center - factor * (center - ymin_spline)
    ymax_new = center + factor * (ymax_spline - center)
    return ymin_new, ymax_new, center

def plot_duration(folders, labels, colors, out_path):
    envs = []
    for folder in folders:
        xs, ys = _collect_duration_points(folder)
        envs.append(_compute_envelope(xs, ys))

    if all(env is None for env in envs):
        print("Not enough data with valid 'number_of_frames' to plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    for env, lab, col in zip(envs, labels, colors):
        if env is None:
            continue
        x_band, ymin_s, ymax_s, x_center, ymid_s = env
        ymin_plot, ymax_plot, _ = _shrink_band(ymin_s, ymax_s, factor=0.5)

        base = mcolors.to_rgb(col)
        darker = tuple(max(0.0, c * 0.75) for c in base)

        ax.fill_between(x_band, ymin_plot, ymax_plot, color=base, alpha=0.22)
        ax.plot(x_center, ymid_s, linewidth=2.5, color=darker, label=lab)

    ax.set_xlabel("Minutes", fontsize=17)
    ax.set_ylabel("F1 Score", fontsize=17)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _normalize_prop_names(prop_obj):
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

def _complexity_by_props(folder, bins):
    by_props = defaultdict(list)
    for fpath in _iter_json(folder):
        data = _safe_json_load(fpath)
        pred = set(map(int, data.get("frames_of_interest", [])))
        gt   = set(map(int, data.get("ground_truth", [])))
        tp, fp, fn, _, _, f1_file = _per_file_stats(pred, gt)

        prop_names = _normalize_prop_names(data.get("propositions", []))
        n_props = len(prop_names)
        if n_props in bins:
            by_props[n_props].append(float(f1_file))
    return by_props

def plot_complexity(folders, labels, colors, bins, out_path):
    all_by_props = [ _complexity_by_props(f, bins) for f in folders ]

    width = 0.25
    offsets = [-(width), 0.0, width]  # for three models
    fig, ax = plt.subplots(figsize=(9, 6))

    handles = []
    for idx, (by_props, lab, col, off) in enumerate(zip(all_by_props, labels, colors, offsets)):
        positions = [p + off for p in bins]
        data = [by_props.get(k, []) for k in bins]

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.9,
            patch_artist=True,
            showfliers=False,
        )

        for box in bp['boxes']:
            box.set_facecolor(col)
            box.set_alpha(0.35)
            box.set_edgecolor(col)
            box.set_linewidth(1.5)
        for element in ['whiskers', 'caps', 'medians']:
            for artist in bp[element]:
                artist.set_color(col)
                artist.set_linewidth(1.5)

        handles.append(bp["boxes"][0])

    ax.set_xticks(bins)
    ax.set_xticklabels([p for p in bins], fontsize=15)
    ax.set_xlabel("Number of Propositions", fontsize=17)
    ax.set_ylabel("F1 Score", fontsize=17)
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(handles, labels, fontsize=13)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    folders = [folder1, folder2, folder3]
    compute_statistics(folders)
    plot_duration(folders, labels, colors, out_path_duration)
    plot_complexity(folders, labels, colors, complexity_bins, out_path_complexity)

