from tqdm import tqdm
import json
import time
import os

from ns_vfs.nsvs import run_nsvs
from ns_vfs.video.read_tlv import TLVReader


TLV_PATH = "/nas/dataset/tlv-dataset-v1"
DEVICE = 7  # GPU device index
OUTPUT_DIR = "output"

def process_entry(entry):
    foi = run_nsvs(
        frames=entry['images'], 
        proposition=entry['tl']['propositions'],
        specification=entry['tl']['specification'],
        model_name="InternVL2-8B",
        device=DEVICE
    )
    return foi

def main():
    reader = TLVReader(TLV_PATH)
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
