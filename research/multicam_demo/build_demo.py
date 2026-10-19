"""Inline demo data and held-out numbers into the page template."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def frac(pair):
    return f"{pair[0]}/{pair[1]}"


def main(args):
    data = json.loads((ROOT / "results" / args.data).read_text())
    summ = json.loads((ROOT / "results" / args.eval / "summary.json").read_text())
    cal = json.loads((ROOT / "results" / args.eval / "calibration.json").read_text())
    vlm = json.loads((ROOT / "results" / args.vlm / "summary.json").read_text()) if args.vlm else None
    vrows = json.loads((ROOT / "results" / args.vlm / "rows.json").read_text()) if args.vlm else []
    n = sum(v[1] for v in summ["ours"]["by_expected"].values())
    def correct(m):
        return f"{sum(v[0] for v in summ[m]['by_expected'].values())}/{n}"
    per_frame_ms = 1000 * summ["timing"]["perceive_s"] / summ["timing"]["frames"]
    rows = [
        dict(name="Count-only handoff", acc=correct("count"), false_clear=frac(summ["count"]["false_all_clear"]),
             false_alarm=frac(summ["count"]["false_alarm"]), compute=f"{per_frame_ms:.1f} ms/frame CPU"),
        dict(name="Tracker that commits to one re-ID match", acc=correct("committed"),
             false_clear=frac(summ["committed"]["false_all_clear"]), false_alarm=frac(summ["committed"]["false_alarm"]),
             compute=f"{per_frame_ms:.1f} ms/frame CPU"),
        dict(name="This monitor", acc=correct("ours"), false_clear=frac(summ["ours"]["false_all_clear"]),
             false_alarm=frac(summ["ours"]["false_alarm"]), compute=f"{per_frame_ms:.1f} ms/frame CPU"),
    ]
    if vlm:
        exp_alert = [r for r in vrows if r["expected"] != "all_arrived"]
        exp_clear = [r for r in vrows if r["expected"] == "all_arrived"]
        fc = sum(r.get("verdict") == "all_arrived" for r in exp_alert)
        fa = sum(r.get("verdict") in ("vehicle_missing", "unexpected_vehicle") for r in vrows if "vehicle" not in r["expected"])
        rows.insert(2, dict(name=f"Qwen3-VL-4B on {vlm['episodes']} of the episodes, W→N link", acc=f"{vlm['correct']}/{vlm['episodes']}",
                            false_clear=f"{fc}/{len(exp_alert)}", false_alarm=f"{fa}/{sum('vehicle' not in r['expected'] for r in vrows)}",
                            compute=f"{vlm['mean_seconds']:.0f} s per query, {vlm['mean_images']:.0f} frames"))
    ident = summ["identity"]
    note = (f"{summ['config']['n_eval_per_kind']} held-out episodes for each of five scenario types (normal, stop, divert, swap, "
            f"degraded camera), scored on all three handoff links ({n} link verdicts). A false all-clear is reporting every vehicle "
            f"accounted for when a vehicle vanished or a camera could not see. On stop and divert episodes the certified candidate set "
            f"contained the vanished vehicle in {ident['ours_candidate_set_contains_truth']}/{ident['episodes']}; the committed tracker "
            f"named the right vehicle in {ident['committed_named_correct']}/{ident['episodes']}.")
    cal_note = (f"Calibrated on {len(cal['seeds'])} separate normal-traffic episodes: travel-time ratio "
                f"{cal['r_lo']:.2f}–{cal['r_hi']:.2f} of the speed-based estimate, appearance gate {cal['tau']:.1f} "
                f"(largest same-vehicle distance {cal['dist_max']:.1f}), and a minimum image-contrast level for counting a "
                f"frame as covered.")
    html = (ROOT / args.template).read_text()
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    html = html.replace("/*__EVAL__*/null", json.dumps(dict(rows=rows, note=note, cal=cal_note, vlm_default=(vlm or {}).get("verdict_counts", {}).get("cannot_confirm"), vlm_n=(vlm or {}).get("episodes"))))
    out = ROOT / args.out
    out.write_text(html)
    print("wrote", out, "%.2f MB" % (out.stat().st_size / 1e6))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="demo_data.json")
    p.add_argument("--eval", default="heldout_v1")
    p.add_argument("--vlm", default="")
    p.add_argument("--out", default="perimeter_handoff_monitor.html")
    p.add_argument("--template", default="tactical_template.html")
    main(p.parse_args())
