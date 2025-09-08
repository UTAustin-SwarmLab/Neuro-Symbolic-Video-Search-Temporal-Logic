from tqdm import tqdm
import json
import time
import os

from ns_vfs.nsvs import run_nsvs
from ns_vfs.video.read_mp4 import Mp4Reader


VIDEOS = [
    {
        "path": "/nas/mars/dataset/LongVideoBench/burn-subtitles/zVudr8cxHRE.mp4",
        "query": "a man is talking before getting up"
    }
]
DEVICE = 7  # GPU device index
OPENAI_SAVE_PATH = "/nas/mars/experiment_result/nsvs/openai_conversation_history/"
OUTPUT_DIR = "output"

def process_entry(entry):
    foi = run_nsvs(
        frames=entry['images'], 
        proposition=entry['tl']['propositions'],
        specification=entry['tl']['specification'],
        model_name="InternVL2-8B",
        device=DEVICE
    )
    return [i for sub in foi for i in sub]

def main():
    reader = Mp4Reader(VIDEOS, OPENAI_SAVE_PATH, sampling_rate_fps=0.1)
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
