from enum import Enum
from tqdm import tqdm
import json
import time
import os

from ns_vfs.vlm.internvl import InternVL
from ns_vfs.video.read_tlv import TLVReader

class RunConfig(Enum):
    SLIDING_WINDOW = "sliding_window"
    FRAME_WISE = "frame_wise"
CURRENT_CONFIG = RunConfig.SLIDING_WINDOW
MODEL_NAME = "InternVL2-8B"
DEVICE = 7  # GPU device index

CALIBRATION_THRESHOLD = 0.349  # vllm threshold
THRESHOLD = 0.5  # detection threshold (fw)
STRIDE = 10  # slide stride (sw)
WINDOW = 20  # window length (sw)

def sliding_window(entry):  # answers "which sequence of `WINDOW` frames can best answer the query"
    query = entry["tl"]["query"]
    frames = entry["images"]

    model = InternVL(model_name=MODEL_NAME, device=DEVICE)
    best = {"prob": -1.0, "start": 1, "end": 1}
    foi = []

    t = 0
    windows = list(range(0, len(frames), STRIDE))
    with tqdm(windows, desc=f"Sliding window (stride={STRIDE}, window={WINDOW})") as pbar:
        for t in pbar:
            end_idx = min(t + WINDOW, len(frames))
            seq = frames[t:end_idx]

            detect = model.detect(seq, query, CALIBRATION_THRESHOLD)
            prob = detect.probability
            is_detected = detect.is_detected

            pbar.set_postfix( {"best_prob": f"{best['prob']:.3f}", "current_prob": f"{prob:.3f}", "detected": is_detected} )

            if prob > best["prob"] and is_detected:
                best.update({"prob": prob, "start": t, "end": end_idx})

    if best["prob"] != -1.0:
        foi = list(range(best["start"], best["end"] + 1))

    return foi

def frame_wise(entry):
    query = entry["tl"]["query"]
    frames = entry["images"]

    model = InternVL(model_name="InternVL2-8B", device=DEVICE)
    foi = []

    t = 0
    windows = range(len(frames))
    with tqdm(windows, desc=f"Framewise (threshold={THRESHOLD}") as pbar:
        for t in pbar:
            f = [frames[t]]

            detect = model.detect(f, query, CALIBRATION_THRESHOLD)
            prob = detect.probability
            is_detected = detect.is_detected

            pbar.set_postfix( {"current_prob": f"{prob:.3f}", "detected": is_detected} )

            if prob > THRESHOLD and is_detected:
                foi.append(t)

    return foi


def main():
    reader = TLVReader("/nas/dataset/tlv-dataset-v1")
    data = reader.read_video()
    if not data:
        return

    folder_name = f"{MODEL_NAME}_{CURRENT_CONFIG.value}"
    folder_name = os.path.join("/nas/mars/experiment_result/nsvs/nsvs2-prelims", folder_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    with tqdm(enumerate(data), total=len(data), desc="Processing entries") as pbar:
        for i, entry in pbar:
            start_time = time.time()
            if CURRENT_CONFIG == RunConfig.SLIDING_WINDOW:
                foi = sliding_window(entry)
            else:
                foi = frame_wise(entry)
            end_time = time.time()

            output = {
                "propositions": entry["tl"]["propositions"],
                "specification": entry["tl"]["specification"],
                "ground_truth": entry["metadata"]["ground_truth"],
                "frames_of_interest": foi,
                "type": entry["metadata"]["type"],
                "number_of_frames": entry["video_info"].frame_count,
                "processting_time_seconds": round(end_time - start_time, 3),
            }

            with open(f"{folder_name}/output_{i}.json", "w") as f:
                json.dump(output, f, indent=4)


if __name__ == "__main__":
    main()
