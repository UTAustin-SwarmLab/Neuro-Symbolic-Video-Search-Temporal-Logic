"""Off-the-shelf open-source VLM baseline on the real RoundaboutHD footage.

prepare (in .venv-real): pick 60 s departure bins, dump frames from both cameras, write truth.
run (in .venv-vlm): ask Qwen3-VL the handoff question for each bin.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "real"
MODELS = ROOT.parent / "active_verification" / "data" / "models"
STEP = 4.0

VIEW = {
    1: dict(E_out="vehicles heading east toward camera 2 leave the roundabout and drive out of view at the lower-left corner of the image",
            E_in="vehicles arriving from camera 2 (westbound) enter at the lower-left corner and drive up into the roundabout"),
    2: dict(E_out="vehicles heading east toward camera 3 leave the roundabout and drive out of view at the lower-left corner of the image",
            E_in="vehicles arriving from camera 3 (westbound) enter at the lower-left corner and drive up into the roundabout",
            W_out="vehicles heading west toward camera 1 leave the roundabout by the far road at the top centre of the image",
            W_in="vehicles arriving from camera 1 (eastbound) enter from the far road at the top centre of the image"),
    3: dict(W_out="vehicles heading west toward camera 2 leave by the road at the right edge of the image",
            W_in="vehicles arriving from camera 2 (eastbound) enter from the right edge of the image",
            E_out="vehicles heading east toward camera 4 leave by the road at the left edge of the image",
            E_in="vehicles arriving from camera 4 (westbound) enter from the left edge of the image"),
    4: dict(W_out="vehicles heading west toward camera 3 leave the roundabout and drive out of view at the lower-left corner of the image",
            W_in="vehicles arriving from camera 3 (eastbound) enter at the bottom of the image and drive up into the roundabout"),
}
VIEW[1]["W_in"] = "vehicles arriving from the west enter from the far road at the top centre of the image"
GAP_M = {(1, 2): 620, (2, 3): 920, (3, 4): 800}
VERDICTS = ("all_arrived", "vehicle_missing", "cannot_confirm")


def instructions(item):
    a, b, la, lb = item["a"], item["b"], item["leg_a"], item["leg_b"]
    gap = GAP_M[tuple(sorted((a, b)))]
    return f"""You are the monitoring assistant for a chain of fixed traffic cameras along one arterial road.
Camera {a} and camera {b} each watch a roundabout on the road. Between them is about {gap} m of road that no camera sees; it has side streets and parking-lot entrances.
At camera {a}, {VIEW[a][la + '_out']}.
At camera {b}, {VIEW[b][lb + '_in']}.
A vehicle that drives straight through normally reaches camera {b} {item['T_lo']:.0f} to {item['T_hi']:.0f} seconds after it leaves camera {a}.
You get still frames sampled every {STEP:.0f} s. Camera {a} frames cover t={item['t0']:.0f}-{item['t1']:.0f} s; camera {b} frames cover t={item['b0']:.0f}-{item['b1']:.0f} s. Each image is labeled with its camera and time.

Question: did every vehicle that left camera {a} toward camera {b} between t={item['t0']:.0f} s and t={item['t1']:.0f} s arrive at camera {b}?
Use exactly one verdict:
- "all_arrived": every such vehicle later arrived at camera {b}.
- "vehicle_missing": at least one such vehicle never arrived at camera {b} (it turned off or stopped between the cameras).
- "cannot_confirm": the frames are not enough to decide.
Return only JSON: {{"left_a": <number of vehicles that left camera {a} toward camera {b}>, "arrived_b": <number of vehicles that arrived at camera {b} from camera {a}'s direction>, "verdict": "<one verdict>", "which": "<short description of the missing vehicle(s), or none>"}}
"""


def prepare(n_each=15, seed=7):
    import random
    import cv2
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rhd_monitor import Evaluator, HALVES, LINKS, VIDEO_END, certificate, key, lname

    ev = Evaluator()
    bins = []
    for fold, (cal_half, ev_half) in enumerate([(HALVES[0], HALVES[1]), (HALVES[1], HALVES[0])]):
        cal, gates, _ = ev.calibrate(cal_half)
        for link in LINKS:
            nm = lname(link)
            c = cal[nm]
            lo, hi = c["T_lo"], c["T_hi"]
            deps, _, arrs, R = ev.sets[nm]
            tp = ev.truth[nm]
            D = [d for d in deps if ev_half[0] <= d["t_exit"] < ev_half[1] and d["t_exit"] + hi <= VIDEO_END]
            adj = [[j for j, r in enumerate(R) if lo <= r["t_entry"] - d["t_exit"] <= hi
                    and ev.gate_ok(gates, "pair", ev.feat(d, r))] for d in D]
            flagged = set(key(D[i]) for cc in certificate(len(D), len(R), adj) if cc["k"] > 0 for i in cc["cand"])
            pinned = set(key(D[cc["cand"][0]]) for cc in certificate(len(D), len(R), adj) if cc["k"] > 0 and len(cc["cand"]) == cc["k"])
            for t0 in range(int(ev_half[0]), int(ev_half[1]), 60):
                t1 = t0 + 60
                if t1 + hi > VIDEO_END:
                    continue
                Db = [d for d in D if t0 <= d["t_exit"] < t1]
                if not Db:
                    continue
                miss = [key(d) for d in Db if key(d) not in tp or tp[key(d)]["t_entry"] - d["t_exit"] > hi]
                n_arr = sum(1 for r in arrs if t0 + lo <= r["t_entry"] <= t1 + hi)
                bins.append(dict(key=f"{nm}_f{fold}_{t0}", link=nm, a=link[0], b=link[2], leg_a=link[1], leg_b=link[3],
                                 t0=t0, t1=t1, b0=t0 + lo, b1=t1 + hi, T_lo=lo, T_hi=hi,
                                 n_dep=len(Db), n_missing=len(miss), n_arr_window=n_arr,
                                 expected="vehicle_missing" if miss else "all_arrived",
                                 monitor="vehicle_missing" if any(key(d) in flagged for d in Db) else "all_arrived",
                                 monitor_pinned=sum(key(d) in pinned for d in Db)))
    rnd = random.Random(seed)
    pos = [b for b in bins if b["n_missing"] > 0]
    neg = [b for b in bins if b["n_missing"] == 0]
    pick = rnd.sample(pos, min(n_each, len(pos))) + rnd.sample(neg, min(n_each, len(neg)))
    dst = OUT / "vlm_inputs"
    dst.mkdir(parents=True, exist_ok=True)
    caps = {c: cv2.VideoCapture(str(OUT / "video" / f"cam{c}.mp4")) for c in (1, 2, 3, 4)}
    manifest = []
    for item in pick:
        frames = []
        for cam, s0, s1 in ((item["a"], item["t0"], item["t1"]), (item["b"], item["b0"], item["b1"])):
            t = s0
            while t <= s1 + 1e-6:
                caps[cam].set(cv2.CAP_PROP_POS_FRAMES, int(round(t * 10)))
                ok, img = caps[cam].read()
                if ok:
                    f = dst / f"{item['key']}_cam{cam}_{t:07.1f}.jpg"
                    cv2.imwrite(str(f), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    frames.append(dict(cam=f"camera {cam}", t=round(t, 1), file=str(f.relative_to(ROOT))))
                t += STEP
        manifest.append(dict(item, frames=frames))
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("bins", len(bins), "pos", len(pos), "neg", len(neg), "picked", len(manifest),
          "mean frames", sum(len(m["frames"]) for m in manifest) / len(manifest))
    mon_ok = sum(m["monitor"] == m["expected"] for m in bins)
    print("monitor bin accuracy on all bins", mon_ok, "/", len(bins), "on picked",
          sum(m["monitor"] == m["expected"] for m in manifest), "/", len(manifest))


def run(name="vlm_qwen4b_real", model_name="Qwen3-VL-4B-Instruct-4bit"):
    import mlx.core as mx
    from mlx_vlm import generate, load
    from PIL import Image

    manifest = json.loads((OUT / "vlm_inputs" / "manifest.json").read_text())
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    mx.set_memory_limit(16 * 1024**3)
    model, processor = load(str(MODELS / model_name), trust_remote_code=False, local_files_only=True)
    rows = []
    for item in manifest:
        content = [{"type": "text", "text": instructions(item)}]
        images = []
        for f in item["frames"]:
            content.append({"type": "text", "text": f"\n{f['cam']} at t={f['t']:.0f} s:\n"})
            content.append({"type": "image"})
            images.append(Image.open(ROOT / f["file"]).convert("RGB"))
        content.append({"type": "text", "text": "\nReturn the JSON now.\n"})
        prompt = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        t0 = time.perf_counter()
        mx.random.seed(0)
        res = generate(model, processor, prompt, image=images, max_tokens=200, temperature=0.0, seed=0,
                       prefill_step_size=512, verbose=False)
        dt = time.perf_counter() - t0
        row = dict(key=item["key"], link=item["link"], images=len(images), seconds=dt, text=res.text,
                   prompt_tokens=res.prompt_tokens, expected=item["expected"], monitor=item["monitor"],
                   n_dep=item["n_dep"], n_missing=item["n_missing"], n_arr_window=item["n_arr_window"])
        try:
            clean = res.text.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
            p = json.loads(clean)
            assert p["verdict"] in VERDICTS
            row.update(parsed=p, parse_ok=True, verdict=p["verdict"], correct=p["verdict"] == item["expected"],
                       count_a_ok=p.get("left_a") == item["n_dep"])
        except Exception as e:  # noqa: BLE001
            row.update(parse_ok=False, error=repr(e), verdict=None, correct=False, count_a_ok=False)
        rows.append(row)
        print(item["key"], row["verdict"], "expected", item["expected"], "monitor", item["monitor"], "%.1fs" % dt, flush=True)
        (out / "rows.json").write_text(json.dumps(rows, indent=1))
    pos = [r for r in rows if r["expected"] == "vehicle_missing"]
    neg = [r for r in rows if r["expected"] == "all_arrived"]
    summary = dict(model=model_name, queries=len(rows), correct=sum(r["correct"] for r in rows),
                   missing_caught=sum(r["verdict"] == "vehicle_missing" for r in pos), missing_total=len(pos),
                   false_all_clear=sum(r["verdict"] == "all_arrived" for r in pos),
                   false_alarm=sum(r["verdict"] == "vehicle_missing" for r in neg), clear_total=len(neg),
                   cannot_confirm=sum(r["verdict"] == "cannot_confirm" for r in rows),
                   parse_failures=sum(not r["parse_ok"] for r in rows),
                   count_a_ok=sum(bool(r["count_a_ok"]) for r in rows),
                   monitor_correct=sum(r["monitor"] == r["expected"] for r in rows),
                   monitor_missing_caught=sum(r["monitor"] == "vehicle_missing" for r in pos),
                   monitor_false_alarm=sum(r["monitor"] == "vehicle_missing" for r in neg),
                   mean_seconds=sum(r["seconds"] for r in rows) / len(rows),
                   mean_images=sum(r["images"] for r in rows) / len(rows),
                   mean_prompt_tokens=sum(r["prompt_tokens"] for r in rows) / len(rows))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    {"prepare": prepare, "run": run}[sys.argv[1]]()
