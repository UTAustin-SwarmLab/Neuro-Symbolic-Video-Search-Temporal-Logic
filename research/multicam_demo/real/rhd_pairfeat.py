"""Pairwise appearance evidence for a departure/arrival pair, and a site-calibrated pair gate."""
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent.parent / "results" / "real"


def attach_descriptors(tracks):
    crops = json.loads((OUT / "features_dinov2_vits14_crops.json").read_text())
    reid = json.loads((OUT / "features_veri_sbs_r50ibn_crops.json").read_text())["feats"]
    for t in tracks:
        k = f"{t['cam']}:{t['lid']}"
        c = crops.get(k)
        t["cemb"] = np.array(c["emb"], dtype=np.float32) if c else None
        r = reid.get(k)
        t["remb"] = np.array(r["emb"], dtype=np.float32) if r else None
        if r:
            w = np.sqrt(np.array(r["area"], dtype=np.float64))
            v = (t["remb"] * w[:, None]).sum(0)
            t["rmean"] = v / np.linalg.norm(v)
        hs, labs = [], []
        for p in sorted((OUT / "crops" / f"cam{t['cam']}").glob(f"{t['lid']}_[0-9].jpg")):
            im = cv2.imread(str(p))
            h, w = im.shape[:2]
            core = im[int(.2 * h):int(.8 * h), int(.2 * w):int(.8 * w)]
            hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
            H = cv2.calcHist([hsv], [0, 1, 2], None, [8, 3, 3], [0, 180, 0, 256, 0, 256]).ravel()
            hs.append(H / max(H.sum(), 1))
            labs.append(cv2.cvtColor(core, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0))
        t["hists"] = np.array(hs) if hs else None
        t["hist"] = np.mean(hs, 0) if hs else None
        t["lab"] = np.median(labs, 0) if labs else None


FEATURES = ["reid_mean", "reid_best", "reid_avg", "dino_mean", "dino_best", "dino_avg",
            "hsv_mean", "hsv_best", "hsv_avg", "dL", "da", "db"]


def pair_features(d, r):
    q = d["remb"] @ r["remb"].T
    e = d["cemb"] @ r["cemb"].T
    bh = 1 - np.sqrt(d["hists"][:, None, :] * r["hists"][None, :, :]).sum(-1)
    return np.array([
        1 - float(d["rmean"] @ r["rmean"]), 1 - float(q.max()), 1 - float(q.mean()),
        1 - float(d["emb"] @ r["emb"]), 1 - float(e.max()), 1 - float(e.mean()),
        1 - float(np.sqrt(d["hist"] * r["hist"]).sum()), float(bh.min()), float(bh.mean()),
        *np.abs(d["lab"] - r["lab"]),
    ])


class PairGate:
    """Logistic pair score, thresholded so that (almost) every labelled true pair passes."""

    def __init__(self, keep=0.995, margin=0.5):
        self.keep, self.margin = keep, margin

    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)
        self.lr = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000).fit(self.sc.transform(X), y)
        s_true = self.score(X[y == 1])
        # threshold below the (1-keep) quantile of true-pair scores, then a further margin in logit units
        self.th = float(np.quantile(s_true, 1 - self.keep)) - self.margin
        return self

    def score(self, X):
        return self.lr.decision_function(self.sc.transform(np.atleast_2d(X)))

    def ok(self, x):
        return float(self.score(x)[0]) >= self.th
