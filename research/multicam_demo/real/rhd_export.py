"""Bundle the real-footage replay for the tactical page: map layers, tracks, boxes, alerts, crops, results."""
import base64
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "real"
OSM = ROOT / "real_data" / "osm" / "osm_96th.json"
EXTENT = dict(x0=-2000, x1=2000, y0=-1100, y1=1100)
CROSS = {1: "PRIORITY WAY W DR", 2: "DELEGATES ROW", 3: "N GRAY RD", 4: "HAZEL DELL PKWY"}
VIEW_HALF = 75.0  # metres of corridor each camera covers on either side of its roundabout
EXIT_LEN = 700.0
FPS = 15.0


def utm(lat, lon, zone=16):
    """WGS84 -> UTM (northern hemisphere), Snyder's series; sub-metre over a few km."""
    a, f, k0 = 6378137.0, 1 / 298.257223563, 0.9996
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    lam0 = math.radians(-183 + 6 * zone)
    phi, lam = math.radians(lat), math.radians(lon)
    N = a / math.sqrt(1 - e2 * math.sin(phi) ** 2)
    T = math.tan(phi) ** 2
    C = ep2 * math.cos(phi) ** 2
    A = math.cos(phi) * (lam - lam0)
    M = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * phi
             - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * math.sin(2 * phi)
             + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * phi)
             - (35 * e2 ** 3 / 3072) * math.sin(6 * phi))
    E = k0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T * T + 72 * C - 58 * ep2) * A ** 5 / 120) + 500000
    Nn = k0 * (M + N * math.tan(phi) * (A * A / 2 + (5 - T + 9 * C + 4 * C * C) * A ** 4 / 24
                                       + (61 - 58 * T + T * T + 600 * C - 330 * ep2) * A ** 6 / 720))
    return E, Nn


def mgrs_square(E, N, zone=16):
    cols = ["ABCDEFGH", "JKLMNPQR", "STUVWXYZ"][(zone - 1) % 3]
    rows = "ABCDEFGHJKLMNPQRSTUV"
    off = 0 if zone % 2 else 5
    return cols[int(E // 100000) - 1] + rows[(int(N // 100000) + off) % 20]


def main():
    tr = json.loads((OUT / "tracks.json").read_text())
    meta, tracks = tr["meta"], tr["tracks"]
    lat0, lon0 = meta["lat0"], meta["lon0"]
    kx, ky = 111320.0 * math.cos(math.radians(lat0)), 110540.0
    to_xy = lambda lat, lon: ((lon - lon0) * kx, (lat - lat0) * ky)  # noqa: E731
    to_ll = lambda x, y: (lat0 + y / ky, lon0 + x / kx)  # noqa: E731
    # local metres -> UTM as an affine fit (exact to well under a metre over the map)
    P = [(x, y) for x in (-2000, 0, 2000) for y in (-1100, 0, 1100)]
    U = np.array([utm(*to_ll(x, y)) for x, y in P])
    A = np.c_[np.array(P), np.ones(len(P))]
    aff = np.linalg.lstsq(A, U, rcond=None)[0]
    E0, N0 = utm(lat0, lon0)
    grid = dict(zone="16S", square=mgrs_square(E0, N0), aff=np.round(aff, 6).tolist())

    els = json.loads(OSM.read_text())["elements"]
    inside = lambda x, y: EXTENT["x0"] - 300 <= x <= EXTENT["x1"] + 300 and EXTENT["y0"] - 300 <= y <= EXTENT["y1"] + 300  # noqa: E731
    roads, bld, water, green = [], [], [], []
    corridor = []
    for e in els:
        t = e.get("tags", {})
        pts = [to_xy(p["lat"], p["lon"]) for p in e.get("geometry", [])]
        if not pts or not any(inside(*p) for p in pts):
            continue
        q = [[round(x, 1), round(y, 1)] for x, y in pts]
        if "highway" in t:
            cls = t["highway"]
            if cls in ("proposed", "construction"):
                continue
            roads.append([cls, t.get("name", ""), q, t.get("junction", "")])
            if t.get("name") == "East 96th Street" and t.get("junction") != "roundabout":
                corridor.append(q)
        elif "building" in t:
            bld.append(q)
        elif t.get("natural") == "water" or "waterway" in t:
            water.append([("line" if "waterway" in t else "area"), q])
        elif t.get("landuse") in ("grass", "meadow", "recreation_ground", "forest", "village_green", "cemetery"):
            green.append(q)

    cams = []
    for c in (1, 2, 3, 4):
        m, rb = meta["cams"][str(c)], meta["roundabouts"][str(c)]
        cams.append(dict(id=f"CAM-{c}", n=c, x=round(m["x"], 1), y=round(m["y"], 1), heading=round(m["heading"], 1),
                         fov=72, rb=[round(rb["x"], 1), round(rb["y"], 1), round(rb["r"], 1)],
                         name=f"96TH ST & {CROSS[c]}", lat=round(m["lat"], 6), lon=round(m["lon"], 6)))
    # gray zones: the 96th St corridor between neighbouring cameras' views
    seg = sorted((p for q in corridor for p in q), key=lambda p: p[0])

    def corridor_between(xa, xb):
        pts = [p for p in seg if xa <= p[0] <= xb]
        yat = lambda x: float(np.interp(x, [p[0] for p in seg], [p[1] for p in seg]))  # noqa: E731
        return [[round(xa, 1), round(yat(xa), 1)]] + [p for p in pts] + [[round(xb, 1), round(yat(xb), 1)]]

    zones = []
    for i, (a, b) in enumerate(((1, 2), (2, 3), (3, 4))):
        xa = meta["roundabouts"][str(a)]["x"] + VIEW_HALF
        xb = meta["roundabouts"][str(b)]["x"] - VIEW_HALF
        line = corridor_between(xa, xb)
        # exits: other roads that touch the corridor inside the zone, followed outward
        exits = []
        for cls, name, q, junc in roads:
            if name == "East 96th Street" or cls in ("footway", "cycleway", "path", "steps", "pedestrian", "motorway", "motorway_link") or junc:
                continue
            for end in (0, -1):
                x, y = q[end]
                if not (xa + 5 <= x <= xb - 5):
                    continue
                d = min(math.hypot(x - p[0], y - p[1]) for p in line)
                if d < 14:
                    poly = q if end == 0 else q[::-1]
                    L, out = 0.0, [poly[0]]
                    for p0, p1 in zip(poly, poly[1:]):
                        L += math.hypot(p1[0] - p0[0], p1[1] - p0[1])
                        out.append(p1)
                        if L >= EXIT_LEN:
                            break
                    if L > 25:
                        exits.append(dict(cls=cls, name=name, line=out, len=round(L, 1)))
        rank = {"primary": 0, "secondary": 1, "tertiary": 2, "residential": 3, "unclassified": 4, "service": 5}
        exits.sort(key=lambda e: (rank.get(e["cls"], 6), -e["len"]))
        zones.append(dict(id=f"G{i + 1}", a=f"CAM-{a}", b=f"CAM-{b}", line=line, length=round(sum(
            math.hypot(q1[0] - q0[0], q1[1] - q0[1]) for q0, q1 in zip(line, line[1:])), 1), exits=exits[:10]))

    rep = json.loads((OUT / "monitor_report.json").read_text())
    pair = rep["pair"]
    events = pair["events"]
    used = set()
    for e in events:
        used.update(e["candidates"])
        used.update(e.get("arrivals", []))
    tby = {f"{t['cam']}:{t['lid']}": t for t in tracks}
    out_tracks = {}
    for k, t in tby.items():
        step = max(1, len(t["path"]) // 24)
        path = [[p[0], p[1], round((p[2] - 1) / FPS, 2)] for p in t["path"][::step]]
        if path[-1][2] != round((t["path"][-1][2] - 1) / FPS, 2):
            p = t["path"][-1]
            path.append([p[0], p[1], round((p[2] - 1) / FPS, 2)])
        boxes = [[round((b[0] - 1) / FPS, 2), *[int(round(v / 6)) for v in b[1:]]] for b in t["boxes"][::3]]
        rec = dict(cam=t["cam"], lid=t["lid"], t0=round((t["f0"] - 1) / FPS, 2), t1=round((t["f1"] - 1) / FPS, 2),
                   li=t["leg_in"], lo=t["leg_out"], path=path, boxes=boxes, gid=t["gid"])
        if k in used:
            p = OUT / "crops" / f"cam{t['cam']}" / f"{t['lid']}_best.jpg"
            if p.exists():
                import cv2
                im = cv2.imread(str(p))
                s = 96 / max(im.shape[:2])
                if s < 1:
                    im = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 78])
                rec["crop"] = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        out_tracks[k] = rec
    for e in events:
        e["t"] = round(e["t"] - 1 / FPS, 2)

    vlm = json.loads((OUT / "vlm_qwen4b_real" / "summary.json").read_text()) if (OUT / "vlm_qwen4b_real" / "summary.json").exists() else None
    data = dict(world=dict(extent=EXTENT, grid=grid, cams=cams, zones=zones, origin=[lat0, lon0],
                           osm=dict(roads=[r[:3] for r in roads], buildings=bld, water=water, green=green)),
                duration=600.0, tracks=out_tracks, events=events,
                results=dict(gates={g: rep[g]["total"] for g in rep}, links=pair["summary"],
                             calibration=pair["calibration"], vlm=vlm))
    s = json.dumps(data, separators=(",", ":"))
    (OUT / "demo_real.json").write_text(s)
    print("demo_real.json %.2f MB" % (len(s) / 1e6), "tracks", len(out_tracks), "with crops", sum("crop" in v for v in out_tracks.values()),
          "events", len(events), Counter(e["kind"] for e in events), "roads", len(roads), "buildings", len(bld),
          "grid", grid["zone"], grid["square"], "zones", [(z["id"], z["length"], len(z["exits"])) for z in zones])


if __name__ == "__main__":
    main()
