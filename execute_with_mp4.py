import tqdm
import itertools
import operator
import json
import time
import os

from ns_vfs.nsvs import run_nsvs
from ns_vfs.nsvs_yolo import *
from ns_vfs.video.read_mp4 import Mp4Reader


VIDEOS = [
    {
        "path": "demo_videos/car.mp4",
        "query": "car until truck"
    }
]
DEVICE = 7  # GPU device index
OPENAI_SAVE_PATH = ""
OUTPUT_DIR = "output"

import itertools

def fill_in_frame_count(arr, entry):
    scale = (entry["video_info"].fps) / (entry["metadata"]["sampling_rate_fps"])

    runs = []
    for _, grp in itertools.groupby(
        sorted(arr),
        key=lambda x, c=[0]: (x - (c.__setitem__(0, c[0] + 1) or c[0]))
    ):
        g = list(grp)
        runs.append((g[0], g[-1]))

    real = []
    for start_i, end_i in runs:
        a = int(round(start_i * scale))
        b = int(round(end_i * scale))
        if real and a <= real[-1]:
            a = real[-1] + 1
        real.extend(range(a, b + 1))
    return real


def _fill_in_frame_count_pairs(pairs, entry):
    if not pairs:
        return []
    scale = (entry["video_info"].fps) / (entry["metadata"]["sampling_rate_fps"])

    pairs = sorted(pairs, key=lambda t: int(t[0]))
    sampled_indices = [int(i) for i, _ in pairs]

    runs = []
    for _, grp in itertools.groupby(
        sampled_indices,
        key=lambda x, c=[0]: (x - (c.__setitem__(0, c[0] + 1) or c[0]))
    ):
        g = list(grp)
        runs.append((g[0], g[-1]))

    idx2bbox = {}
    for i, bbox in pairs:
        i = int(i)
        if i not in idx2bbox:
            idx2bbox[i] = bbox

    expanded: list[tuple[int, tuple[float, float, float, float]]] = []
    last_real = -1

    for start_i, end_i in runs:
        rep_bbox = idx2bbox.get(start_i)
        if rep_bbox is None:
            for k in range(start_i, end_i + 1):
                if k in idx2bbox:
                    rep_bbox = idx2bbox[k]
                    break
        if rep_bbox is None:
            continue

        a = int(round(start_i * scale))
        b = int(round(end_i * scale))
        if expanded and a <= last_real:
            a = last_real + 1
        for real_i in range(a, b + 1):
            expanded.append((real_i, rep_bbox))
        last_real = b

    return expanded


def process_entry(entry, run_with_yolo=False, cache_path=""):
    """
    VLM path (run_with_yolo=False):
        - Returns (foi, object_frame_dict_expanded)
          where object_frame_dict_expanded: Dict[str, List[int]] (real frame indices)

    YOLO path (run_with_yolo=True):
        - Expects run_nsvs_yolo to return (foi, object_frame_bounding_boxes)
          where object_frame_bounding_boxes: Dict[str, List[(sample_idx, bbox)]]
        - Returns (foi, object_frame_bounding_boxes_expanded)
          where each bbox is duplicated across the scaled span to real frames:
            Dict[str, List[(real_idx, bbox)]]
    """
    if run_with_yolo:
        foi, object_frame_bounding_boxes = run_nsvs_yolo(
            frames=entry["images"],
            proposition=entry['tl']['propositions'],
            specification=entry['tl']['specification'],
            yolo_cache_path=cache_path,
            vlm_detection_threshold=0.35,
        )
        foi = fill_in_frame_count([i for sub in foi for i in sub], entry)

        expanded_boxes = {}
        for key, pairs in (object_frame_bounding_boxes or {}).items():
            expanded_boxes[key] = _fill_in_frame_count_pairs(pairs, entry)
        return foi, expanded_boxes

    else:
        foi, object_frame_dict = run_nsvs(
            frames=entry['images'],
            proposition=entry['tl']['propositions'],
            specification=entry['tl']['specification'],
            model_name="InternVL2-8B",
            device=DEVICE
        )
        foi = fill_in_frame_count([i for sub in foi for i in sub], entry)
        object_frame_dict = {key: fill_in_frame_count(value, entry) for key, value in (object_frame_dict or {}).items()}
        return foi, object_frame_dict

def main():
    reader = Mp4Reader(VIDEOS, OPENAI_SAVE_PATH, sampling_rate_fps=1)
    data = reader.read_video()
    if not data:
        return
    
    # cache_path = preprocess_yolo(entry["images"], model_weights="yolov8n.pt",
    #                              device="cuda:0", out_path="yolo_cache.npz")

    with tqdm.tqdm(enumerate(data), total=len(data), desc="Processing entries") as pbar:
        for i, entry in pbar:
            start_time = time.time()
            foi = process_entry(entry, run_with_yolo=True)
            end_time = time.time()
            processing_time = round(end_time - start_time, 3)

            if foi:
                output = {
                    "tl": entry["tl"],
                    "metadata": entry["metadata"],
                    "video_info": entry["video_info"].to_dict(),
                    "frames_of_interest": foi,
                    "processting_time_seconds": processing_time
                }

                os.makedirs(OUTPUT_DIR, exist_ok=True)
                with open(os.path.join(OUTPUT_DIR, f"output_{i}.json"), "w") as f:
                    json.dump(output, f, indent=4)

if __name__ == "__main__":
    main()
