from tqdm import tqdm
import itertools
import operator
import json
import time
import os

from ns_vfs.nsvs import run_nsvs
from ns_vfs.video.read_mp4 import Mp4Reader


VIDEOS = [
    {
        "path": "demo_videos/blue_shirt.mp4",
        "query": "a woman is jumping and clapping until a candle is blown"
    }
]
DEVICE = 7  # GPU device index
OPENAI_SAVE_PATH = ""
OUTPUT_DIR = "output"

def fill_in_frame_count(arr, entry):
    scale = (entry["video_info"].fps) / (entry["metadata"]["sampling_rate_fps"])

    runs = []
    for _, grp in itertools.groupby(sorted(arr), key=lambda x, c=[0]: (x - (c.__setitem__(0, c[0]+1) or c[0]))):
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

def process_entry(entry):
    foi, object_frame_dict = run_nsvs(
        frames=entry['images'], 
        proposition=entry['tl']['propositions'],
        specification=entry['tl']['specification'],
        model_name="InternVL2-8B",
        device=DEVICE
    )

    foi = fill_in_frame_count([i for sub in foi for i in sub], entry)
    object_frame_dict = {key: fill_in_frame_count(value, entry) for key, value in object_frame_dict.items()}
    return foi, object_frame_dict

def main():
    reader = Mp4Reader(VIDEOS, OPENAI_SAVE_PATH, sampling_rate_fps=1)
    data = reader.read_video()
    if not data:
        return

    with tqdm(enumerate(data), total=len(data), desc="Processing entries") as pbar:
        for i, entry in pbar:
            start_time = time.time()
            foi = process_entry(entry)
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
