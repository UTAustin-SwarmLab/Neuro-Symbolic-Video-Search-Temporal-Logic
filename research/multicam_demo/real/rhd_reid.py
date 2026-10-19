"""Vehicle re-ID features from a public open-source model (fast-reid, SBS R50-IBN trained on VeRi-776).

The model was trained on a different city's cameras; nothing here is trained on RoundaboutHD.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "real_data" / "reid" / "fast-reid-src"
OUT = ROOT / "results" / "real"
sys.path.insert(0, str(SRC))

from fastreid.config import get_cfg  # noqa: E402
from fastreid.modeling.meta_arch import build_model  # noqa: E402
from fastreid.utils.checkpoint import Checkpointer  # noqa: E402


def main(cfg_file="configs/VeRi/sbs_R50-ibn.yml", weights="veri_sbs_R50-ibn.pth", tag="veri_sbs_r50ibn"):
    cfg = get_cfg()
    cfg.merge_from_file(str(SRC / cfg_file))
    cfg.MODEL.BACKBONE.PRETRAIN = False
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg.MODEL.DEVICE = "cpu"
    model = build_model(cfg)
    ck = Checkpointer(model).load(str(ROOT / "real_data" / "reid" / weights))
    model = model.eval().to(dev)
    model.device_override = dev
    tracks = json.loads((OUT / "tracks.json").read_text())["tracks"]
    H, W = cfg.INPUT.SIZE_TEST
    t0, n, feats = time.time(), 0, {}
    for t in tracks:
        paths = sorted((OUT / "crops" / f"cam{t['cam']}").glob(f"{t['lid']}_[0-9].jpg"))
        if not paths:
            continue
        ims, areas = [], []
        for p in paths:
            im = cv2.imread(str(p))[:, :, ::-1]  # RGB, as in fast-reid's predictor
            areas.append(im.shape[0] * im.shape[1])
            ims.append(cv2.resize(im, (W, H), interpolation=cv2.INTER_CUBIC))
        x = torch.from_numpy(np.stack(ims).astype(np.float32)).permute(0, 3, 1, 2).to(dev)
        with torch.no_grad():
            e = model(x).float().cpu().numpy()
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        feats[f"{t['cam']}:{t['lid']}"] = dict(emb=e.astype(np.float16).tolist(), area=areas)
        n += len(paths)
    dt = time.time() - t0
    (OUT / f"features_{tag}_crops.json").write_text(json.dumps(dict(
        model=f"fast-reid {cfg_file} + {weights}", device=dev, images=n, seconds=round(dt, 2),
        ms_per_image=round(1000 * dt / max(1, n), 2), feats=feats)))
    print(tag, "tracks", len(feats), "images", n, "s", round(dt, 1), "ms/img", round(1000 * dt / max(1, n), 2))


if __name__ == "__main__":
    main(*sys.argv[1:])
