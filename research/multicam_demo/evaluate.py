"""Calibrate on normal traffic, then evaluate held-out episodes against baselines.

Calibration seeds and evaluation seeds are disjoint. Thresholds are fixed before any
evaluation episode is processed. Results are written to results/<name>/.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
from episode import run, truth_map
from monitor import Calibration, LINK_DIST, appearance_distance, check_link, committed_baseline, count_baseline
from world import LINKS

ROOT = Path(__file__).resolve().parent
KINDS = ("none", "stop", "divert", "swap", "degraded")


def process(args):
    seed, kind = args
    r = run(seed, kind)
    ep = r["episode"]
    tm = truth_map(ep, r["tracks"])
    return dict(seed=seed, kind=kind, truth=ep.truth, degraded=ep.degraded, summaries=r["summaries"],
                quality={c: [(round(t, 2), round(q, 2)) for t, q in qs] for c, qs in r["quality"].items()},
                truth_map=tm, duration=ep.duration, timing=r["timing"],
                vehicles=[dict(vid=v.vid, vtype=v.vtype, color=v.color, fate=v.fate, decoy=v.decoy) for v in ep.vehicles])


def calibrate(records):
    ratios, dists, quals = [], [], []
    for rec in records:
        tm = rec["truth_map"]
        for a, b, _ in LINKS:
            for d in rec["summaries"][a]:
                for r in rec["summaries"][b]:
                    if tm[d["tid"]] is not None and tm[d["tid"]] == tm[r["tid"]] and d["speed"]:
                        ratios.append((r["t_entry"] - d["t_exit"]) * d["speed"] / LINK_DIST[(a, b)])
                        dists.append(appearance_distance(d, r))
        for qs in rec["quality"].values():
            quals.extend(q for _, q in qs)
    ratios, dists, quals = np.array(ratios), np.array(dists), np.array(quals)
    cal = Calibration(r_lo=float(np.quantile(ratios, 0.0) * 0.97), r_hi=float(np.quantile(ratios, 1.0) * 1.03),
                      tau=float(np.quantile(dists, 0.995) * 1.15), q_min=float(np.quantile(quals, 0.001) * 0.7))
    stats = dict(pairs=len(ratios), ratio_min=float(ratios.min()), ratio_max=float(ratios.max()),
                 dist_p50=float(np.median(dists)), dist_max=float(dists.max()),
                 quality_p001=float(np.quantile(quals, 0.001)))
    return cal, stats


def score(rec, cal):
    t_now = rec["duration"]
    S, Q, tm, kind = rec["summaries"], rec["quality"], rec["truth_map"], rec["kind"]
    rows = []
    for a, b, g in LINKS:
        link = (a, b)
        ours = check_link(link, S[a], S[b], Q, cal, t_now)
        base = committed_baseline(link, S[a], S[b], cal, t_now)
        cnt = count_baseline(link, S[a], S[b], cal, t_now)
        anomaly_link = (a, b) == ("CAM-W", "CAM-N")
        if kind in ("stop", "divert", "swap") and anomaly_link:
            expected = "ALERT"
        elif kind == "degraded" and (a == "CAM-N" or b == "CAM-N"):
            expected = "UNVERIFIED"
        else:
            expected = "ALL ACCOUNTED"
        miss_tids = [d["tid"] for d in S[a] if tm.get(d["tid"]) in rec["truth"]["missing"]]
        rows.append(dict(seed=rec["seed"], kind=kind, link=f"{a}->{b}", expected=expected,
                         ours=ours["status"], committed=base["status"], count=cnt["status"],
                         ours_candidates=ours["missing_candidates"], ours_missing=ours["certified_missing"],
                         ours_unexplained=ours["unexplained_arrivals"], committed_named=base["named_missing"],
                         true_missing_tids=miss_tids,
                         candidates_contain_truth=bool(miss_tids) and all(t in ours["missing_candidates"] for t in miss_tids),
                         committed_named_correct=bool(miss_tids) and base["named_missing"] == miss_tids,
                         n_candidates=len(ours["missing_candidates"])))
    return rows


def summarize(rows):
    out = {}
    for method in ("ours", "committed", "count"):
        by = defaultdict(list)
        for r in rows:
            by[r["expected"]].append(r[method] == r["expected"])
        false_clear = [r[method] == "ALL ACCOUNTED" for r in rows if r["expected"] != "ALL ACCOUNTED"]
        false_alarm = [r[method] == "ALERT" for r in rows if r["expected"] != "ALERT"]
        out[method] = dict(accuracy=float(np.mean([r[method] == r["expected"] for r in rows])),
                           by_expected={k: [int(sum(v)), len(v)] for k, v in by.items()},
                           false_all_clear=[int(sum(false_clear)), len(false_clear)],
                           false_alarm=[int(sum(false_alarm)), len(false_alarm)])
    miss_rows = [r for r in rows if r["true_missing_tids"] and r["kind"] in ("stop", "divert") and r["link"] == "CAM-W->CAM-N"]
    out["identity"] = dict(
        episodes=len(miss_rows),
        ours_candidate_set_contains_truth=int(sum(r["candidates_contain_truth"] for r in miss_rows)),
        ours_mean_candidates=float(np.mean([r["n_candidates"] for r in miss_rows])) if miss_rows else None,
        ours_singleton=int(sum(r["n_candidates"] == 1 for r in miss_rows)),
        committed_named_correct=int(sum(r["committed_named_correct"] for r in miss_rows)),
        committed_named_any=int(sum(bool(r["committed_named"]) for r in miss_rows)))
    return out


def main(args):
    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.perf_counter()
    cal_jobs = [(s, "none") for s in range(1000, 1000 + args.n_cal)]
    eval_jobs = [(s, k) for k in KINDS for s in range(args.start + 1000 * KINDS.index(k),
                                                         args.start + 1000 * KINDS.index(k) + args.n_eval)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        cal_recs = list(pool.map(process, cal_jobs))
        cal, stats = calibrate(cal_recs)
        (out / "calibration.json").write_text(json.dumps(dict(cal.__dict__, **stats, seeds=[s for s, _ in cal_jobs]), indent=2))
        print("calibration", cal, stats, "%.0fs" % (time.perf_counter() - t0), flush=True)
        eval_recs = list(pool.map(process, eval_jobs))
    with (out / "episodes.jsonl").open("w") as f:
        for rec in eval_recs:
            f.write(json.dumps(rec) + "\n")
    rows = [row for rec in eval_recs for row in score(rec, cal)]
    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    summary = summarize(rows)
    timing = dict(frames=int(sum(r["timing"]["frames"] for r in eval_recs)),
                  perceive_s=float(sum(r["timing"]["perceive_s"] for r in eval_recs)),
                  render_s=float(sum(r["timing"]["render_s"] for r in eval_recs)))
    summary["timing"] = timing
    summary["config"] = dict(n_cal=args.n_cal, n_eval_per_kind=args.n_eval, start=args.start, kinds=list(KINDS))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))
    print("done %.0fs" % (time.perf_counter() - t0))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--n-cal", type=int, default=24)
    p.add_argument("--n-eval", type=int, default=20)
    p.add_argument("--start", type=int, default=20000)
    p.add_argument("--workers", type=int, default=8)
    main(p.parse_args())
