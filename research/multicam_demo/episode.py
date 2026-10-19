"""Run one episode end to end: render every camera frame, perceive, return tracklets.

The evaluator-only ground-truth mapping lives in `truth_map` and is never passed to the
monitor or baselines.
"""
import time
import numpy as np
from perception import CameraPerception
from render import background, noise_bank, render_camera
from world import CAM_ORDER, cam_interval, make_episode

FPS = 4.0


def estimate_background(cam, bg, noise, n=15):
    frames = [np.clip(bg.astype(np.int16) + noise[k], 0, 255).astype(np.uint8) for k in range(n)]
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def run(seed, kind, keep_frames_for=None, **kw):
    ep = make_episode(seed, kind, **kw)
    t0 = time.perf_counter()
    tracks, quality, frames = {}, {}, {}
    render_s = perceive_s = 0.0
    for cam in CAM_ORDER:
        bg, noise = background(cam), noise_bank(cam)
        perc = CameraPerception(cam, estimate_background(cam, bg, noise))
        keep = keep_frames_for and cam in keep_frames_for
        frames[cam] = []
        n_frames = int(ep.duration * FPS)
        for k in range(n_frames):
            t = k / FPS
            a = time.perf_counter()
            img = render_camera(cam, t, ep.vehicles, bg, noise, k, ep.degraded.get(cam))
            b = time.perf_counter()
            perc.step(t, img)
            c = time.perf_counter()
            render_s += b - a
            perceive_s += c - b
            if keep:
                frames[cam].append((t, img))
        tracks[cam] = perc.finish()
        quality[cam] = perc.quality
    summaries = {cam: [tr.summary() for tr in trs] for cam, trs in tracks.items()}
    return dict(episode=ep, tracks=tracks, summaries=summaries, quality=quality, frames=frames,
                timing=dict(render_s=render_s, perceive_s=perceive_s, wall_s=time.perf_counter() - t0,
                            frames=int(ep.duration * FPS) * len(CAM_ORDER)))


def truth_map(ep, tracks):
    """Evaluator only: tracklet id -> ground-truth vehicle id by position agreement."""
    out = {}
    for cam, trs in tracks.items():
        lo, _ = cam_interval(cam)
        for tr in trs:
            mid = len(tr.times) // 2
            t, u = tr.times[mid], tr.us[mid]
            best, best_d = None, 6.0
            for v in ep.vehicles:
                s = v.s_at(t)
                if s is None:
                    continue
                d = abs((s - lo) - u)
                if d < best_d:
                    best, best_d = v.vid, d
            out[tr.tid] = best
    return out
