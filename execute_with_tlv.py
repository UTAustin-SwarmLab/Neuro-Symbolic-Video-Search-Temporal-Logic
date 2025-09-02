from tqdm import tqdm
import pickle
import json
import time
import os

from ns_vfs.nsvs import run_nsvs

def readTLV():
    base_path = "/nas/dataset/tlv-dataset-v1"

    data = []
    total_files = 0
    processed_files = 0

    # Count total files first for progress bar
    for dataset_dir in os.listdir(base_path):
        dataset_path = os.path.join(base_path, dataset_dir)
        if not os.path.isdir(dataset_path):
            continue
        for format_dir in os.listdir(dataset_path):
            format_path = os.path.join(dataset_path, format_dir)
            if not os.path.isdir(format_path):
                continue
            for filename in os.listdir(format_path):
                if filename.endswith(".pkl"):
                    total_files += 1

    def formatter(spec):
        spec = spec.replace("&", " and ")
        spec = spec.replace("|", " or ")
        spec = spec.replace("U", " until ")
        spec = spec.replace("F", " eventually ")
        spec = spec.replace("G", " always ")
        spec = spec.replace("X", " next ")
        spec = spec.replace('"', "")
        spec = spec.replace("'", "")
        spec = spec.replace("(", "")
        spec = spec.replace(")", "")
        while "  " in spec:
            spec = spec.replace("  ", " ")
        spec = spec.strip()
        return spec

    with tqdm(total=total_files, desc="Loading dataset files") as pbar:
        for dataset_dir in os.listdir(base_path):
            dataset_path = os.path.join(base_path, dataset_dir)
            if not os.path.isdir(dataset_path):
                continue

            for format_dir in os.listdir(dataset_path):
                format_path = os.path.join(dataset_path, format_dir)
                if not os.path.isdir(format_path):
                    continue

                for filename in os.listdir(format_path):
                    if not filename.endswith(".pkl"):
                        continue

                    file_path = os.path.join(format_path, filename)
                    try:
                        with open(file_path, "rb") as f:
                            raw = pickle.load(f)
                            entry = {
                                "propositions": raw["proposition"],
                                "specification": raw["ltl_formula"],
                                "query": formatter(raw["ltl_formula"]),
                                "ground_truth": [i for sub in raw["frames_of_interest"] for i in sub],
                                "images": raw["images_of_frames"],
                                "type": {"dataset": dataset_dir, "format": format_dir},
                                "number_of_frames": raw["number_of_frame"],
                            }
                            data.append(entry)
                            processed_files += 1
                            pbar.set_postfix({"loaded": processed_files})
                            return data
                    except Exception as e:
                        pass
                    finally:
                        pbar.update(1)
    return data

def process_entry(entry):
    try:
        foi = run_nsvs(
            frames=entry['images'], 
            proposition=entry['propositions'],
            specification=entry['specification']
        )
    except Exception as _:
        foi = None

    return foi

def main():
    data = readTLV()
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
