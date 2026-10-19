"""Rectified overhead view for each fixed camera: road along x, traffic left to right.

Synthetic pixels, not photorealistic video. Each camera has its own fixed background,
sensor noise, and optional dust degradation. Vehicles are drawn top-down with shadow,
cab and windshield so that type and colour are visible but similar vehicles stay similar.
"""
import zlib
import cv2
import numpy as np
from world import CAMERAS, HALF_VIEW, cam_interval

W, H = 480, 270
PX_PER_M = W / (2 * HALF_VIEW)          # 3.2 px per meter
LANE_V = 2.0                            # meters right of the road centre


def to_px(u, v):
    return u * PX_PER_M, H / 2 + v * PX_PER_M


def background(cam):
    rng = np.random.default_rng(zlib.crc32(cam.encode()))
    base = np.array([150, 186, 206], dtype=np.float32)          # desert sand (BGR)
    img = np.ones((H, W, 3), np.float32) * base
    low = cv2.resize(rng.normal(0, 1, (18, 32)).astype(np.float32), (W, H))
    img += low[..., None] * 9
    img += rng.normal(0, 3, (H, W, 1)).astype(np.float32)
    for _ in range(40):                                          # rocks and scrub
        x, y = int(rng.integers(0, W)), int(rng.choice([rng.integers(0, 90), rng.integers(180, H)]))
        r = int(rng.integers(2, 6))
        cv2.circle(img, (x, y), r, tuple(float(c) for c in base * rng.uniform(0.62, 0.8)), -1)
    y0, y1 = int(H / 2 - 4.6 * PX_PER_M), int(H / 2 + 4.6 * PX_PER_M)
    img[y0:y1] = img[y0:y1] * 0.0 + np.array([96, 98, 100], np.float32) + rng.normal(0, 2.5, (y1 - y0, W, 1))
    for x in range(0, W, 26):                                    # dashed centre line
        cv2.line(img, (x, H // 2), (x + 12, H // 2), (205, 210, 214), 1)
    cv2.line(img, (0, y0), (W, y0), (170, 176, 180), 1)
    cv2.line(img, (0, y1), (W, y1), (170, 176, 180), 1)
    if cam in ("CAM-W", "CAM-N"):                                # perimeter fence posts
        for x in range(8, W, 34):
            cv2.line(img, (x, 22), (x, 30), (70, 80, 90), 1)
        cv2.line(img, (0, 26), (W, 26), (90, 98, 106), 1)
    return np.clip(img, 0, 255).astype(np.uint8)


def camera_optics(cam):
    """Fixed per-camera colour gains and scale error, plus a slow exposure drift phase."""
    rng = np.random.default_rng(zlib.crc32(cam.encode()) + 11)
    return dict(gain=rng.uniform(0.9, 1.1, 3), scale=rng.uniform(0.95, 1.05), phase=rng.uniform(0, 6.28))


def noise_bank(cam, k=24, sigma=4.0):
    rng = np.random.default_rng(zlib.crc32(cam.encode()) + 7)
    return [rng.normal(0, sigma, (H, W, 3)).astype(np.int16) for _ in range(k)]


def draw_vehicle(img, u, v, vehicle, scale=1.0):
    L, Wd = vehicle.length * PX_PER_M * scale, vehicle.width * PX_PER_M * scale
    cx, cy = to_px(u, v)
    col = np.array(vehicle.bgr(), dtype=float)
    def rect(x0, y0, x1, y1, c):
        cv2.rectangle(img, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))),
                      tuple(int(t) for t in np.clip(c, 0, 255)), -1)
    rect(cx - L / 2 + 3, cy - Wd / 2 + 3, cx + L / 2 + 3, cy + Wd / 2 + 3, (60, 66, 72))   # shadow
    rect(cx - L / 2, cy - Wd / 2, cx + L / 2, cy + Wd / 2, col)
    if vehicle.vtype == "truck":
        rect(cx + L / 2 - 0.26 * L, cy - Wd / 2, cx + L / 2, cy + Wd / 2, col * 0.86)        # cab
        rect(cx + L / 2 - 0.30 * L, cy - Wd / 2 + 1, cx + L / 2 - 0.27 * L, cy + Wd / 2 - 1, col * 0.6)
    elif vehicle.vtype == "pickup":
        rect(cx - L / 2 + 2, cy - Wd / 2 + 2, cx - L / 2 + 0.42 * L, cy + Wd / 2 - 2, col * 0.72)  # bed
    elif vehicle.vtype == "van":
        rect(cx - L / 2 + 2, cy - Wd / 2 + 1, cx + L / 2 - 4, cy + Wd / 2 - 1, col * 1.06)
    rect(cx + L / 2 - 0.22 * L, cy - Wd / 2 + 1, cx + L / 2 - 0.12 * L, cy + Wd / 2 - 1, (40, 46, 52))  # windshield


def render_camera(cam, t, vehicles, bg, noise, frame_index, degraded=None):
    lo, hi = cam_interval(cam)
    o = OPTICS[cam]
    exposure = 1.0 + 0.022 * np.sin(2 * np.pi * t / 170.0 + o["phase"])
    out = (_bg_float(cam, bg) * exposure).astype(np.int16)
    drawn = [v for v in vehicles if (s := v.s_at(t)) is not None and lo - 6 <= s <= hi + 6]
    if drawn:
        img = bg.copy()
        for v in drawn:
            draw_vehicle(img, v.s_at(t) - lo, LANE_V, v, o["scale"])
        m = np.abs(img.astype(np.int16) - bg.astype(np.int16)).sum(axis=2) > 0
        # Vehicles carry the camera's white balance; untouched pixels keep the background.
        out[m] = (img[m].astype(np.float32) * o["gain"][None, :] * exposure).astype(np.int16)
    out = out + noise[frame_index % len(noise)]
    if degraded and degraded[0] <= t <= degraded[1]:
        dust = np.array([170, 190, 205], np.float32)
        ramp = min(1.0, (t - degraded[0]) / 6.0, (degraded[1] - t) / 6.0)
        a = 0.86 * ramp
        out = (out.astype(np.float32) * (1 - a) + dust * a).astype(np.int16)
        out += noise[(frame_index * 7 + 3) % len(noise)] * 2
    out = np.clip(out, 0, 255).astype(np.uint8)
    stamp = f"{cam}  T+{t:06.1f}s"
    cv2.putText(out, stamp, (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


_BG_CACHE = {}


def _bg_float(cam, bg):
    if cam not in _BG_CACHE:
        _BG_CACHE[cam] = bg.astype(np.float32)
    return _BG_CACHE[cam]

OPTICS = {cam: camera_optics(cam) for cam in CAMERAS}

