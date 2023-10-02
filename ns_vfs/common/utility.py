from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path


def get_filename_with_datetime(base_name):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{current_time}.png"


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
