from tqdm import tqdm
import json
import time

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

def process_entry(entry):
    try:
        foi = run_nsvs(
            frames=entry['images'], 
            proposition=entry['tl']['propositions'],
            specification=entry['tl']['specification'],
            model_name="InternVL2-8B",
            device=DEVICE
        )
    except Exception as _:
        foi = None

    return foi

def main():
    reader = Mp4Reader(VIDEOS, OPENAI_SAVE_PATH, sampling_rate_fps=0.1)
    data = reader.read_video()
    print(data[0]["tl"]["specification"])
    if not data:
        return

    with tqdm(enumerate(data), total=len(data), desc="Processing entries") as pbar:
        for i, entry in pbar:
            start_time = time.time()
            foi = process_entry(entry)
            end_time = time.time()
            entry["processing_time_seconds"] = round(end_time - start_time, 3)

            if foi:
                output = {
                    "propositions": entry['propositions'],
                    "specification": entry['specification'],
                    "ground_truth": entry['ground_truth'],
                    "frames_of_interest": foi,
                    "type": entry['type'],
                    "number_of_frames": entry['number_of_frames'],
                    "processting_time_seconds": entry['processing_time_seconds'],
                }

                with open(f"junk/output_{i}.json", "w") as f:
                    json.dump(output, f, indent=4)

if __name__ == "__main__":
    main()
