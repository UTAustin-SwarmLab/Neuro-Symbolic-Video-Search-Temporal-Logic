from typing import List, Dict, Any
from tqdm import tqdm
import numpy as np
import cv2

from ns_vfs.video.reader import VideoFormat, VideoInfo, Reader
from ns_vfs.puls.puls import PULS


class Mp4Reader(Reader):
    def __init__(self, videos: List[Dict[str, str]], openai_save_path: str, sampling_rate_fps: float = 1.0):
        self.videos = videos
        if sampling_rate_fps is None or sampling_rate_fps <= 0:
            raise ValueError("sampling_rate_fps must be > 0")
        self.openai_save_path = openai_save_path
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

    def _read_one(self, video_query: Dict[str, str]) -> Dict[str, Any] | None:
        path = video_query["path"]
        query = video_query["query"]

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None

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

        puls_output = PULS(query, self.openai_save_path)

        cap.release()
        entry = {
            "tl": {
                "propositions": puls_output["proposition"],
                "specification": puls_output["specification"],
                "query": query,
            },
            "metadata": {
                "video_path": path,
                "sampling_rate_fps": self.sampling_rate_fps,
                "puls_saved_path": puls_output["saved_path"],
            },
            "video_info": video_info,
            "images": images,
        }
        return entry

    def read_video(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        with tqdm(total=len(self.videos), desc="Reading MP4s") as pbar:
            for v in self.videos:
                entry = self._read_one(v)
                if entry is not None:
                    results.append(entry)
                pbar.update(1)
        return results
