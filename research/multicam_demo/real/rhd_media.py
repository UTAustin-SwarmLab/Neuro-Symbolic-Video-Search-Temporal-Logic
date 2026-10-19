"""Decode the four real 4K videos once: vehicle crops for appearance features, plus small demo videos."""
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RHD = ROOT / "real_data" / "roundabouthd" / "RoundaboutHD"
OUT = ROOT / "results" / "real"
K = 6  # crops per track


def encode_demo_video(c):
    dst = OUT / "video" / f"cam{c}.mp4"
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(RHD / f"imagesc00{c}/video.mp4"),
        "-vf", "scale=640:360:flags=area,fps=10", "-c:v", "h264_videotoolbox", "-b:v", "150k",
        "-maxrate", "210k", "-bufsize", "420k", "-g", "10", "-profile:v", "main", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-an", str(dst)], check=True)
    return str(dst), dst.stat().st_size


def crops_for_camera(c):
    tracks = [t for t in json.loads((OUT / "tracks.json").read_text())["tracks"] if t["cam"] == c]
    want = {}
    for t in tracks:
        bx = t["boxes"]
        n = len(bx)
        picks = sorted(set(int(round(q * (n - 1))) for q in np.linspace(0.15, 0.85, K)))
        for k, i in enumerate(picks):
            f, x1, y1, x2, y2 = bx[i]
            want.setdefault(int(f) - 1, []).append((t["lid"], k, x1, y1, x2, y2))
        f, x1, y1, x2, y2 = max(bx, key=lambda b: (b[3] - b[1]) * (b[4] - b[2]))
        want.setdefault(int(f) - 1, []).append((t["lid"], "best", x1, y1, x2, y2))
    d = OUT / "crops" / f"cam{c}"
    d.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(RHD / f"imagesc00{c}/video.mp4"))
    last = max(want)
    f = 0
    saved = 0
    while f <= last:
        ok, img = cap.read()
        if not ok:
            break
        for lid, k, x1, y1, x2, y2 in want.get(f, []):
            w, h = x2 - x1, y2 - y1
            px, py = 0.08 * w, 0.08 * h
            a, b = int(max(0, x1 - px)), int(max(0, y1 - py))
            e, g = int(min(img.shape[1], x2 + px)), int(min(img.shape[0], y2 + py))
            crop = img[b:g, a:e]
            if crop.size == 0:
                continue
            side = 160 if k == "best" else 224
            s = side / max(crop.shape[:2])
            if s < 1:
                crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(d / f"{lid}_{k}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved += 1
        f += 1
    return c, saved


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    with ProcessPoolExecutor(4) as ex:
        jobs = []
        if what in ("all", "video"):
            jobs += [ex.submit(encode_demo_video, c) for c in [1, 2, 3, 4]]
        if what in ("all", "crops"):
            jobs += [ex.submit(crops_for_camera, c) for c in [1, 2, 3, 4]]
        for j in jobs:
            print(j.result(), flush=True)
