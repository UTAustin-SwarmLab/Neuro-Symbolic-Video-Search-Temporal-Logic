"""RoundaboutHD: per-camera tracks with GPS positions, roundabout legs and (evaluation-only) global IDs.

Inputs are the dataset's own single-camera track annotations (SCT_GT) and camera models. The
multi-camera ground truth is joined in only as an evaluation label (`gid`); nothing downstream
that makes a decision reads it.
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cameratransform as ct
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RHD = ROOT / "real_data" / "roundabouthd" / "RoundaboutHD"
OSM = ROOT / "real_data" / "osm" / "osm_96th.json"
OUT = ROOT / "results" / "real"
FPS = 15.0
CAMS = [1, 2, 3, 4]  # west to east along E 96th St


def local_frame():
    """Metric east/north frame centred between the four cameras."""
    pos = [json.loads((RHD / f"imagesc00{c}/geo-mapping/cam0{c}_fitted_cam.json").read_text()) for c in CAMS]
    lat0 = float(np.mean([p["gps_lat"] for p in pos]))
    lon0 = float(np.mean([p["gps_lon"] for p in pos]))
    kx = 111320.0 * math.cos(math.radians(lat0))
    ky = 110540.0

    def to_xy(lat, lon):
        return (np.asarray(lon) - lon0) * kx, (np.asarray(lat) - lat0) * ky

    return lat0, lon0, to_xy


def roundabout_centres(to_xy, cam_xy):
    els = json.loads(OSM.read_text())["elements"]
    pts = defaultdict(list)
    for e in els:
        if e.get("tags", {}).get("junction") not in ("roundabout", "circular"):
            continue
        for p in e["geometry"]:
            x, y = to_xy(p["lat"], p["lon"])
            c = min(CAMS, key=lambda k: math.hypot(x - cam_xy[k][0], y - cam_xy[k][1]))
            if math.hypot(x - cam_xy[c][0], y - cam_xy[c][1]) < 120:
                pts[c].append((float(x), float(y)))
    out = {}
    for c in CAMS:
        a = np.array(pts[c])
        # algebraic circle fit (Kasa) on the ring nodes
        A = np.c_[2 * a[:, 0], 2 * a[:, 1], np.ones(len(a))]
        b = (a ** 2).sum(1)
        cx, cy, k = np.linalg.lstsq(A, b, rcond=None)[0]
        out[c] = dict(x=float(cx), y=float(cy), r=float(math.sqrt(k + cx * cx + cy * cy)), n=len(a))
    return out


def load_sct(c):
    rows = np.loadtxt(RHD / f"imagesc00{c}/SCT/SCT_GT.txt")
    return rows  # frame, id, x1, y1, x2, y2, cls


def load_mcvt():
    rows = np.loadtxt(RHD / "Multi_CAM_Ground_Turth.txt")
    by = defaultdict(list)
    for cam, gid, f, x, y, w, h, _, _ in rows:
        by[(int(cam), int(f))].append((int(gid), x, y, x + w, y + h))
    return by


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


# Where each roundabout leg leaves each camera's image (full-resolution px, foot point of the box).
# Read off the start/end-point plots of the single-camera tracks; legs toward the neighbouring
# cameras are W (toward the camera to the west) and E (toward the camera to the east).
NEAR = {1: "E", 2: "E", 3: "N", 4: "W"}  # the leg running under each camera, which leaves via the bottom edge
GATES = {
    1: [(30, 1150, "E"), (2400, 120, "W"), (200, 430, "S"), (3650, 540, "N")],
    2: [(30, 1250, "E"), (2300, 420, "W"), (1800, 500, "W"), (60, 640, "S"), (3760, 930, "N")],
    3: [(100, 560, "E"), (3780, 700, "W"), (3100, 520, "W"), (30, 1780, "N"), (30, 1140, "N"), (2250, 250, "S")],
    4: [(30, 1300, "W"), (2300, 480, "E"), (220, 670, "N"), (2620, 640, "S")],
}
for _c, _leg in NEAR.items():
    GATES[_c] += [(x, 2150, _leg) for x in range(0, 1700, 300)]
GATE_R = 450


def gate_of(c, x, y):
    best, leg = GATE_R, "X"
    for gx, gy, g in GATES[c]:
        d = math.hypot(x - gx, y - gy)
        if d < best:
            best, leg = d, g
    return leg


def bearing(dx, dy):
    return (math.degrees(math.atan2(dx, dy)) + 360) % 360  # 0 = north, 90 = east


def leg_of(b):
    if 45 <= b < 135:
        return "E"
    if 135 <= b < 225:
        return "S"
    if 225 <= b < 315:
        return "W"
    return "N"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lat0, lon0, to_xy = local_frame()
    cams = {}
    for c in CAMS:
        p = RHD / f"imagesc00{c}/geo-mapping/cam0{c}_fitted_cam.json"
        d = json.loads(p.read_text())
        m = ct.load_camera(str(p))
        m.setGPSpos(d["gps_lat"], d["gps_lon"])
        x, y = to_xy(d["gps_lat"], d["gps_lon"])
        cams[c] = dict(model=m, x=float(x), y=float(y), lat=d["gps_lat"], lon=d["gps_lon"], heading=d["heading_deg"])
    cam_xy = {c: (cams[c]["x"], cams[c]["y"]) for c in CAMS}
    rb = roundabout_centres(to_xy, cam_xy)
    mcvt = load_mcvt()

    tracks = []
    for c in CAMS:
        rows = load_sct(c)
        foot = np.c_[(rows[:, 2] + rows[:, 4]) / 2, rows[:, 5]]
        gps = cams[c]["model"].gpsFromImage(foot)
        X, Y = to_xy(gps[:, 0], gps[:, 1])
        per = defaultdict(list)
        for i, r in enumerate(rows):
            per[int(r[1])].append(i)
        for lid, idx in per.items():
            idx = sorted(idx, key=lambda i: rows[i, 0])
            fr = rows[idx, 0].astype(int)
            votes = Counter()
            for i in idx:
                f = int(rows[i, 0])
                box = rows[i, 2:6]
                best, bi = 0.5, None
                for g, *gb in mcvt.get((c, f), []):
                    v = iou(box, gb)
                    if v > best:
                        best, bi = v, g
                if bi is not None:
                    votes[bi] += 1
            gid, nv = votes.most_common(1)[0] if votes else (None, 0)
            if nv < min(10, 0.5 * len(idx)):  # MCVT labels often cover only part of a single-camera track
                gid = None
            xs, ys = X[idx], Y[idx]
            ok = np.isfinite(xs) & np.isfinite(ys) & (np.hypot(xs - rb[c]["x"], ys - rb[c]["y"]) < 250)
            if ok.sum() < 2:
                continue
            fr_all = fr
            xs, ys, fr = xs[ok], ys[ok], fr[ok]
            cx, cy = rb[c]["x"], rb[c]["y"]
            dist = np.hypot(xs - cx, ys - cy)
            # entry/exit legs from the first/last points outside the ring's inner area
            far = np.where(dist > 0.8 * rb[c]["r"])[0]
            i0 = far[0] if len(far) else 0
            i1 = far[-1] if len(far) else len(xs) - 1
            b_in = bearing(xs[i0] - cx, ys[i0] - cy)
            b_out = bearing(xs[i1] - cx, ys[i1] - cy)
            step = max(1, len(xs) // 40)
            path = [[round(float(xs[k]), 1), round(float(ys[k]), 1), int(fr[k])] for k in range(0, len(xs), step)]
            if path[-1][2] != int(fr[-1]):
                path.append([round(float(xs[-1]), 1), round(float(ys[-1]), 1), int(fr[-1])])
            disp = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
            tracks.append(dict(
                cam=c, lid=lid, gid=gid, f0=int(fr_all[0]), f1=int(fr_all[-1]), n=len(idx),
                t_entry=fr_all[0] / FPS, t_exit=fr_all[-1] / FPS,
                leg_in=gate_of(c, (rows[idx[0], 2] + rows[idx[0], 4]) / 2, rows[idx[0], 5]),
                leg_out=gate_of(c, (rows[idx[-1], 2] + rows[idx[-1], 4]) / 2, rows[idx[-1], 5]),
                geo_in=leg_of(b_in), geo_out=leg_of(b_out),
                d_in=round(float(dist[0]), 1), d_out=round(float(dist[-1]), 1), disp=round(disp, 1),
                path=path,
                boxes=[[int(rows[i, 0]), *[round(float(v), 1) for v in rows[i, 2:6]]] for i in idx],
            ))
    meta = dict(lat0=lat0, lon0=lon0, fps=FPS, n_frames=9000,
                cams={c: {k: v for k, v in cams[c].items() if k != "model"} for c in CAMS},
                roundabouts=rb)
    (OUT / "tracks.json").write_text(json.dumps(dict(meta=meta, tracks=tracks)))
    print("tracks", len(tracks), "with gid", sum(t["gid"] is not None for t in tracks))
    for c in CAMS:
        ts = [t for t in tracks if t["cam"] == c]
        print(c, "rb", {k: round(v, 1) for k, v in rb[c].items()}, "cam", round(cams[c]["x"], 1), round(cams[c]["y"], 1))
        print("   legs in", Counter(t["leg_in"] for t in ts), "out", Counter(t["leg_out"] for t in ts))
        print("   in->out", Counter(t["leg_in"] + t["leg_out"] for t in ts).most_common(12))


if __name__ == "__main__":
    main()
