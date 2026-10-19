"""Export demo scenarios: simulation, perception output, monitor timeline, baselines, VLM answer.

Each scenario is a held-out evaluation episode, so the VLM answer shown in the demo is the
actual answer recorded by vlm_eval.py for the same episode.
"""
import argparse
import base64
import json
from pathlib import Path
import cv2
import numpy as np
from episode import run, truth_map
from monitor import Calibration, check_link, committed_baseline, count_baseline, describe
from world import CAMERAS, CAM_ORDER, HALF_VIEW, LINKS, NW_CORNER_S, ROUTE, SIDE_TRACK, STOP_S, cam_interval, gray_interval

ROOT = Path(__file__).resolve().parent


def jpeg(img, scale=2):
    img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def timeline(summaries, quality, cal, duration):
    """Monitor state changes over time for every link, plus departure tip-offs."""
    events, last = [], {}
    for a, b, g in LINKS:
        for d in summaries[a]:
            if d["exits_right"]:
                lo, hi = cal.window(d, (a, b))
                events.append(dict(t=round(d["t_exit"], 1), type="tipoff", link=f"{a}->{b}", tid=d["tid"],
                                   text=f"{a} -> {b}: expect {describe(d)} between T+{lo:.0f}s and T+{hi:.0f}s",
                                   window=[round(lo, 1), round(hi, 1)]))
    for step in np.arange(0, duration + 0.5, 0.5):
        for a, b, g in LINKS:
            arrs = [r for r in summaries[b] if r["t_exit"] <= step or r["t_entry"] <= step]
            deps = [d for d in summaries[a] if d["t_exit"] <= step]
            res = check_link((a, b), deps, arrs, quality, cal, step)
            key = (res["status"], res["certified_missing"], tuple(res["missing_candidates"]),
                   tuple(res["unexplained_arrivals"]), res["unverified_departures"], tuple(res["unverified_arrivals"]))
            if last.get((a, b)) != key and step > 0 and res["n_departures"] > 0:
                if True:
                    events.append(dict(t=float(step), type=res["status"].lower().replace(" ", "_"), link=res["link"],
                                       gray=g, certified_missing=res["certified_missing"],
                                       candidates=res["missing_candidates"], unexplained=res["unexplained_arrivals"],
                                       unverified=res["unverified_departures"] + len(res["unverified_arrivals"]),
                                       actions=res["actions"]))
            last[(a, b)] = key
    return sorted(events, key=lambda e: (e["t"], e["type"] != "tipoff"))


def export(seed, kind, cal, vlm_rows, label, blurb):
    r = run(seed, kind)
    ep, S, Q = r["episode"], r["summaries"], r["quality"]
    tm = truth_map(ep, r["tracks"])
    final = {f"{a}->{b}": check_link((a, b), S[a], S[b], Q, cal, ep.duration) for a, b, _ in LINKS}
    committed = {f"{a}->{b}": committed_baseline((a, b), S[a], S[b], cal, ep.duration) for a, b, _ in LINKS}
    counts = {f"{a}->{b}": count_baseline((a, b), S[a], S[b], cal, ep.duration) for a, b, _ in LINKS}
    tracks = {}
    for cam, trs in r["tracks"].items():
        tracks[cam] = []
        for tr in trs:
            idx = list(range(0, len(tr.times), 2))
            tracks[cam].append(dict(tid=tr.tid, t=[round(tr.times[i], 2) for i in idx], u=[round(tr.us[i], 2) for i in idx],
                                    box_h=[int(tr.boxes[i][3]) for i in idx], desc=describe(tr.summary()),
                                    crop=jpeg(tr.crop) if tr.crop is not None else None,
                                    truth=tm.get(tr.tid)))
    vlm = next((v for v in vlm_rows if v["key"] == f"{kind}_{seed}"), None)
    return dict(
        id=f"{kind}_{seed}", label=label, blurb=blurb, kind=kind, seed=seed, duration=ep.duration,
        degraded={k: list(v) for k, v in ep.degraded.items()}, truth=ep.truth,
        vehicles=[dict(vid=v.vid, vtype=v.vtype, color=v.color, bgr=list(v.bgr()), length=v.length, width=v.width,
                       fate=v.fate, decoy=v.decoy, knots=[[round(t, 2), round(s, 2)] for t, s in v.knots],
                       side=[[round(t, 2), round(d, 2)] for t, d in v.side_knots]) for v in ep.vehicles],
        tracks=tracks, events=timeline(S, Q, cal, ep.duration),
        final=final, committed=committed, counts=counts,
        quality={c: [[round(t, 1), round(q, 1)] for t, q in qs[::4]] for c, qs in Q.items()},
        vlm=None if vlm is None else dict(verdict=vlm.get("verdict"), parsed=vlm.get("parsed"),
                                          seconds=round(vlm["seconds"], 1), images=vlm["images"],
                                          expected=vlm["expected"], correct=vlm["correct"]),
        compute=dict(frames=r["timing"]["frames"], perceive_s=round(r["timing"]["perceive_s"], 2)))


def main(args):
    cal_d = json.loads((ROOT / "results" / args.eval / "calibration.json").read_text())
    cal = Calibration(cal_d["r_lo"], cal_d["r_hi"], cal_d["tau"], cal_d["q_min"])
    vlm_rows = json.loads((ROOT / "results" / args.vlm / "rows.json").read_text()) if args.vlm else []
    picks = json.loads(args.picks)
    scenarios = [export(p["seed"], p["kind"], cal, vlm_rows, p["label"], p["blurb"]) for p in picks]
    world = dict(route=ROUTE.tolist(), side=SIDE_TRACK.tolist(), cameras={c: cam_interval(c) for c in CAM_ORDER},
                 cam_order=CAM_ORDER, half_view=HALF_VIEW, nw_corner_s=NW_CORNER_S, stop_s=STOP_S,
                 links=[dict(a=a, b=b, gray=g, interval=gray_interval(i)) for i, (a, b, g) in enumerate(LINKS)],
                 calibration=cal.__dict__)
    out = ROOT / "results" / args.out
    out.write_text(json.dumps(dict(world=world, scenarios=scenarios)))
    print("wrote", out, "%.1f MB" % (out.stat().st_size / 1e6))
    for s in scenarios:
        f = s["final"]["CAM-W->CAM-N"]
        print(s["id"], s["label"], "| ours:", f["status"], f["certified_missing"], f["missing_candidates"], f["unexplained_arrivals"],
              "| committed:", s["committed"]["CAM-W->CAM-N"], "| count:", s["counts"]["CAM-W->CAM-N"]["status"],
              "| vlm:", s["vlm"] and s["vlm"]["verdict"], "| events:", len(s["events"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eval", default="heldout_v1")
    p.add_argument("--vlm", default="")
    p.add_argument("--picks", required=True)
    p.add_argument("--out", default="demo_data.json")
    main(p.parse_args())
