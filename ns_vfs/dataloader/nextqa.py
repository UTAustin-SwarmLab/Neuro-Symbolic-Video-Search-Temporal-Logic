from typing import List, Dict, Union
import numpy as np
import json
import cv2

from ns_vfs.dataloader._base import DatasetLoader

class NextQA(DatasetLoader):
    def _parse_timestamp(self, ts: str) -> float:
        """
        Parse a timestamp like "HH:MM:SS.mmm" into total seconds as float.
        """
        h, m, s = ts.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)

    def load_all(self, sample_fps: int = 2, chunk_size: int = 10) -> List[Dict[str, Union[List[np.ndarray], None]]]:
        """
        Load a video and subtitles, sample at `sample_fps` frames/sec, group every
        `chunk_size` frames into one dict, and attach subtitles overlapping each chunk.

        Returns:
            List of dicts of the form:
            [
              {'frames': [f1, f2, ..., f10], 'subtitle': None},
              {'frames': [f11, ..., f20], 'subtitle': None},
              ...
            ]
        """

        # --- 1) Open video and get duration ---
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_fps      = cap.get(cv2.CAP_PROP_FPS)
        duration_sec = total_frames / vid_fps

        # --- 2) Sample frames at regular intervals ---
        interval = 1.0 / sample_fps
        timestamps = np.arange(0, duration_sec, interval)

        sampled = []
        for t in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                break
            sampled.append((t, frame.copy()))
        cap.release()

        chunks: List[Dict[str, Union[List[np.ndarray], None]]] = []
        for i in range(0, len(sampled), chunk_size):
            chunk = sampled[i:i + chunk_size]
            if not chunk:
                continue

            frames = [f for (_, f) in chunk]

            chunks.append({
                'frames': frames,
                'subtitle': None
            })

        return chunks
