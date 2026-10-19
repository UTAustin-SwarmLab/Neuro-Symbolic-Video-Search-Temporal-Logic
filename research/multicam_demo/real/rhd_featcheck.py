"""Diagnostic: how well does each appearance descriptor separate true handoff pairs from same-window impostors?"""
import json, sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rhd_monitor import load, link_sets, true_pairs, LINKS, key, OUT
from collections import defaultdict

data = load(); T = data["tracks"]
bycam = defaultdict(list)
for t in T: bycam[t["cam"]].append(t)

def color_desc(t):
    ps = sorted((OUT / "crops" / f"cam{t['cam']}").glob(f"{t['lid']}_[0-9].jpg"))
    hs, labs = [], []
    for p in ps:
        im = cv2.imread(str(p)); h, w = im.shape[:2]
        c = im[int(.2*h):int(.8*h), int(.2*w):int(.8*w)]
        hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
        H = cv2.calcHist([hsv], [0, 1, 2], None, [8, 3, 3], [0, 180, 0, 256, 0, 256]).ravel()
        hs.append(H / max(H.sum(), 1)); labs.append(cv2.cvtColor(c, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0))
    return np.mean(hs, 0), np.median(labs, 0)

for t in T:
    t["hist"], t["lab"] = color_desc(t)

def d_dino(a, b): return 1 - float(a["emb"] @ b["emb"])
def d_hist(a, b): return float(np.sum(np.sqrt(a["hist"] * b["hist"]))) * -1 + 1  # 1 - Bhattacharyya coeff
def d_lab(a, b): return float(np.linalg.norm((a["lab"] - b["lab"]) * np.array([0.5, 1, 1])))

res = defaultdict(lambda: dict(true=[], imp=[]))
for link in LINKS:
    deps, _, arrs, maybe_arrs = link_sets(T, link)
    tp = true_pairs(deps, bycam[link[2]])
    for d in deps:
        r_true = tp.get(key(d))
        for r in maybe_arrs:
            dt = r["t_entry"] - d["t_exit"]
            if not (15 <= dt <= 90): continue
            is_true = r_true is not None and key(r) == key(r_true)
            for nm, f in [("dino", d_dino), ("hist", d_hist), ("lab", d_lab)]:
                res[nm]["true" if is_true else "imp"].append(f(d, r))
for nm, v in res.items():
    tr, im = np.array(v["true"]), np.array(v["imp"])
    out = [nm, "n_true", len(tr), "n_imp", len(im)]
    for q in (95, 99, 100):
        th = np.percentile(tr, q)
        out += [f"reject@{q}%", round(float((im > th).mean()), 3)]
    print(*out)
# combined AND-gates: each gate at the given true-pair acceptance
tr = np.c_[res["dino"]["true"], res["hist"]["true"], res["lab"]["true"]]
im = np.c_[res["dino"]["imp"], res["hist"]["imp"], res["lab"]["imp"]]
for q in (97.5, 99, 99.5, 100):
    th = np.percentile(tr, q, axis=0)
    acc_true = (tr <= th).all(1).mean()
    rej = (im > th).any(1).mean()
    print("AND all three at", q, "true kept", round(float(acc_true), 3), "impostors rejected", round(float(rej), 3))
