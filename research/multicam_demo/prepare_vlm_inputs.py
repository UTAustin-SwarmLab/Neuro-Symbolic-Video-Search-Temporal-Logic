"""Render the frames an off-the-shelf VLM sees for the CAM-W -> CAM-N handoff question.

The VLM is given a favourable setup: frames are sampled every 4 s, cropped to the road and enlarged, only from the spans in which the
convoy passes CAM-W (ground-truth span) and from the generous window in which it should
reach CAM-N. Truth goes to a separate file that the VLM runner reads only for scoring.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from render import background, noise_bank, render_camera
from world import cam_interval, gray_interval, make_episode

ROOT = Path(__file__).resolve().parent
KINDS = ("none", "stop", "divert", "swap", "degraded")


def crossing(v, s_target):
    for (t0, s0), (t1, s1) in zip(v.knots[:-1], v.knots[1:]):
        if s0 <= s_target <= s1 and s1 > s0:
            return t0 + (s_target - s0) / (s1 - s0) * (t1 - t0)
    return None


def main(args):
    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=False)
    w_lo, w_hi = cam_interval("CAM-W")
    n_lo, n_hi = cam_interval("CAM-N")
    d = gray_interval(1)[1] - gray_interval(1)[0]
    manifest, truth = [], {}
    for kind in KINDS:
        for k in range(args.per_kind):
            seed = args.start + 1000 * KINDS.index(kind) + k
            ep = make_episode(seed, kind)
            enters = [t for v in ep.vehicles if (t := crossing(v, w_lo)) is not None]
            exits = [t for v in ep.vehicles if (t := crossing(v, w_hi)) is not None]
            w_span = (min(enters) - 2.0, max(exits) + 2.0)
            n_span = (min(exits) + d / 12.5 * 0.85 - 4.0, max(exits) + d / 9.5 * 1.18 + 16.0)
            key = f"{kind}_{seed}"
            folder = out / key
            folder.mkdir()
            frames = []
            for cam, span in (("CAM-W", w_span), ("CAM-N", n_span)):
                bg, noise = background(cam), noise_bank(cam)
                # Every vehicle stays in view ~13-16 s, so a 4 s step shows each one at least twice.
                for i, t in enumerate(np.arange(span[0], span[1] + 1e-6, args.step)):
                    img = render_camera(cam, float(t), ep.vehicles, bg, noise, int(t * 4), ep.degraded.get(cam))
                    # Crop to the road band and enlarge 2x so vehicles are large enough for the VLM.
                    img = cv2.resize(img[95:175], (960, 160), interpolation=cv2.INTER_CUBIC)
                    path = folder / f"{cam}_{i:02d}.png"
                    cv2.imwrite(str(path), img)
                    frames.append(dict(cam=cam, t=round(float(t), 1), file=str(path.relative_to(ROOT))))
            passed_w = sum(1 for v in ep.vehicles if crossing(v, w_hi) is not None)
            passed_n = sum(1 for v in ep.vehicles if crossing(v, n_hi) is not None)
            expected = {"none": "all_arrived", "stop": "vehicle_missing", "divert": "vehicle_missing",
                        "swap": "vehicle_missing|unexpected_vehicle", "degraded": "cannot_confirm"}[kind]
            truth[key] = dict(kind=kind, seed=seed, passed_w=passed_w, passed_n=passed_n, expected=expected,
                              degraded=ep.degraded)
            manifest.append(dict(key=key, frames=frames))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    print("prepared", len(manifest), "episodes in", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="vlm_inputs_v1")
    p.add_argument("--start", type=int, default=20000)
    p.add_argument("--per-kind", type=int, default=8)
    p.add_argument("--step", type=float, default=4.0)
    main(p.parse_args())
