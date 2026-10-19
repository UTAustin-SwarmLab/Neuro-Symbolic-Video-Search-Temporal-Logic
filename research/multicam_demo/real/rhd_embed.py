"""Appearance features for every single-camera track: off-the-shelf DINOv2 ViT-S/14, no training on this data."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "real"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load(p):
    im = Image.open(p).convert("RGB")
    w, h = im.size
    s = 224 / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    canvas = Image.new("RGB", (224, 224), (124, 116, 104))
    canvas.paste(im, ((224 - im.size[0]) // 2, (224 - im.size[1]) // 2))
    x = torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (x - MEAN) / STD, w * h


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True).eval().to(dev)
    tracks = json.loads((OUT / "tracks.json").read_text())["tracks"]
    t0 = time.time()
    feats, per_crop, n_img = {}, {}, 0
    for t in tracks:
        paths = sorted((OUT / "crops" / f"cam{t['cam']}").glob(f"{t['lid']}_[0-9].jpg"))
        if not paths:
            continue
        xs, ws = zip(*(load(p) for p in paths))
        with torch.no_grad():
            e = model(torch.stack(xs).to(dev)).float().cpu().numpy()
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        w = np.sqrt(np.array(ws, dtype=np.float64))  # larger, sharper crops count more
        v = (e * w[:, None]).sum(0)
        feats[f"{t['cam']}:{t['lid']}"] = (v / np.linalg.norm(v)).astype(np.float32).tolist()
        per_crop[f"{t['cam']}:{t['lid']}"] = dict(emb=e.astype(np.float16).tolist(), area=list(ws))
        n_img += len(paths)
    dt = time.time() - t0
    (OUT / "features_dinov2_vits14.json").write_text(json.dumps(dict(
        model="dinov2_vits14 (torch.hub facebookresearch/dinov2)", device=dev, images=n_img,
        seconds=round(dt, 2), ms_per_image=round(1000 * dt / max(1, n_img), 2), feats=feats)))
    (OUT / "features_dinov2_vits14_crops.json").write_text(json.dumps(per_crop))
    print("tracks", len(feats), "images", n_img, "seconds", round(dt, 1), "ms/img", round(1000 * dt / max(1, n_img), 2))


if __name__ == "__main__":
    main()
