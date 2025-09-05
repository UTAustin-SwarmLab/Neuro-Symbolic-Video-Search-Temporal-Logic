from typing import List, Dict, Any
from tqdm import tqdm
import numpy as np
import cv2

from ns_vfs.video.reader import VideoFormat, VideoInfo, Reader


class Mp4Reader(Reader):
    def __init__(self, video_paths: str | List[str], sampling_rate_fps: float = 1.0):
        if isinstance(video_paths, str):
            self.video_paths = [video_paths]
        else:
            self.video_paths = video_paths
        if sampling_rate_fps is None or sampling_rate_fps <= 0:
            raise ValueError("sampling_rate_fps must be > 0")
        self.sampling_rate_fps = float(sampling_rate_fps)

    def _sampled_frame_indices(self, fps: float, frame_count: int) -> List[int]:
        if fps <= 0:
            fps = 1.0

        duration_sec = frame_count / fps if frame_count > 0 else 0.0
        step_sec = 1.0 / self.sampling_rate_fps

        times = [t for t in np.arange(0.0, duration_sec + 1e-9, step_sec)]
        idxs = sorted(set(int(round(t * fps)) for t in times if t * fps < frame_count))
        if not idxs and frame_count > 0:
            idxs = [0]
        return idxs

    def _read_one(self, path: str) -> Dict[str, Any] | None:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

            frame_idxs = self._sampled_frame_indices(fps, frame_count)

            images: List[np.ndarray] = []
            for idx in frame_idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                images.append(frame_rgb)

            if (width == 0 or height == 0) and images:
                height, width = images[0].shape[:2]

            video_info = VideoInfo(
                format=VideoFormat.MP4,
                frame_width=width,
                frame_height=height,
                frame_count=frame_count,
                fps=float(fps) if fps else None,
            )

            entry = {
                "tl": {
                    "propositions": {},
                    "specification": "",
                    "query": "",
                },
                "metadata": {
                    "file_path": path,
                    "sampling_rate_fps": self.sampling_rate_fps,
                },
                "video_info": video_info,
                "images": images,
            }
            return entry
        finally:
            cap.release()

    def read_video(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        with tqdm(total=len(self.video_paths), desc="Reading MP4s") as pbar:
            for p in self.video_paths:
                try:
                    entry = self._read_one(p)
                    if entry is not None:
                        results.append(entry)
                finally:
                    pbar.update(1)
        return results
