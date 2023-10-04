from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path


def list_flatten(lst):
    flattened = []
    for item in lst:
        if isinstance(item, list):
            flattened.extend(list_flatten(item))
        else:
            flattened.append(item)
    return flattened


def get_file_or_dir_with_datetime(base_name, ext="."):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{current_time}{ext}"


def save_frames(
    frames: list, path="/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/result", file_label: str = ""
) -> None:
    """Save image to path.

    Args:
    path (str, optional): Path to save image.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    for idx, img in enumerate(frames):
        Image.fromarray(img).save(f"{path}/{file_label}_{idx}.png")


def save_dict_to_pickle(dict_obj: dict, path: str, file_name: str = "data.pkl"):
    # Decode the JSON data into a Python object
    # data_python = json.loads(dict_obj)
    full_path = Path(path) / file_name

    # Save the Python object using pickle
    with open(full_path, "wb") as file:
        pickle.dump(dict_obj, file)


def load_pickle_to_dict(path: str, file_name: str = "data.pkl") -> dict:
    full_path = Path(path) / file_name

    # Load the Python object using pickle
    with open(full_path, "rb") as file:
        dict_obj = pickle.load(file)

    return dict_obj
