from typing import List, Dict, Any, Iterable
from tqdm import tqdm
import numpy as np
import pickle
import os

from ns_vfs.video.reader import VideoFormat, VideoInfo, Reader


class TLVReader(Reader):
    def __init__(self, tlv_path: str):
        self.tlv_path = tlv_path # /nas/dataset/tlv-dataset-v1

    def _iter_tlv(self) -> Iterable[tuple[str, str, str]]:
        for dataset_dir in os.listdir(self.tlv_path):
            dataset_path = os.path.join(self.tlv_path, dataset_dir)
            if not os.path.isdir(dataset_path):
                continue
            for format_dir in os.listdir(dataset_path):
                format_path = os.path.join(dataset_path, format_dir)
                if not os.path.isdir(format_path):
                    continue
                for fname in os.listdir(format_path):
                    if fname.endswith(".pkl"):
                        yield dataset_dir, format_dir, os.path.join(format_path, fname)


    def read_video(self) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []

        total = sum(1 for _ in self._iter_tlv())
        with tqdm(total=total, desc="Loading TLV files") as pbar:
            for dataset_dir, format_dir, file_path in self._iter_tlv():
                with open(file_path, "rb") as f:
                    raw = pickle.load(f)

                images: List[np.ndarray] = raw["images_of_frames"]
                if len(images) == 0:
                    pbar.update(1)
                    continue

                video_info = VideoInfo(
                    format=VideoFormat.LIST_OF_ARRAY,
                    frame_width=images[0].shape[1],
                    frame_height=images[0].shape[0],
                    frame_count=len(images),
                    fps=0.1, # 1 frame/10 sec
                )

                entry = {
                    "tl": {
                        "propositions": raw["proposition"],
                        "specification": raw["ltl_formula"],
                        "query": self.formatter(raw["ltl_formula"]),
                    },
                    "metadata": {
                        "type": {"dataset": dataset_dir, "format": format_dir},
                        "ground_truth": [i for sub in raw["frames_of_interest"] for i in sub],
                    },
                    "video_info": video_info,
                    "images": images,
                }
                entries.append(entry)
                pbar.update(1)

        return entries

