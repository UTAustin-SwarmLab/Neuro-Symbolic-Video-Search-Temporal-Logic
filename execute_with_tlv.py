from tqdm import tqdm
import pickle
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
                            return data
                            processed_files += 1
                            pbar.set_postfix({"loaded": processed_files})
                    except Exception as e:
                        pass
                    finally:
                        pbar.update(1)
    return data


def formatter(spec):
    # Replace logical operators with natural language
    spec = spec.replace("&", " and ")
    spec = spec.replace("|", " or ")
    spec = spec.replace("U", " until ")
    spec = spec.replace("F", " eventually ")
    spec = spec.replace("G", " always ")
    spec = spec.replace("X", " next ")

    # Remove quotes and parentheses
    spec = spec.replace('"', "")
    spec = spec.replace("'", "")
    spec = spec.replace("(", "")
    spec = spec.replace(")", "")

    # Remove extra spaces
    while "  " in spec:
        spec = spec.replace("  ", " ")
    spec = spec.strip()

    return spec

def process_entry(entry):
    foi = run_nsvs(
        frames=entry['images'], 
        proposition=entry['propositions'],
        specification=entry['specification']
    )

    print()
    print(f"Specification: {entry['specification']}")
    print(f"Ground Truth: {entry['ground_truth']}")
    print(f"NSVS Output: {foi}")
    import sys
    sys.exit(0)

def main():
    data = readTLV()
    if not data:
        return

    with tqdm(enumerate(data), total=len(data), desc="Processing entries") as pbar:
        for i, entry in pbar:
            process_entry(entry)

if __name__ == "__main__":
    main()
