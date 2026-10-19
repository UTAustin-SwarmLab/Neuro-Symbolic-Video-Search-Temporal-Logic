"""Per-camera perception from pixels only: background subtraction, blobs, tracklets.

Output per camera is a list of tracklets with entry/exit times, speed, appearance, and a
best crop, plus a per-frame image-quality signal used for coverage reasoning. This is a
classical CV stand-in for an open-source detector + re-ID embedding on real cameras.
"""
from dataclasses import dataclass, field
import cv2
import numpy as np
from render import H, W, PX_PER_M

# The timestamp overlay occupies the bottom strip; ignore it for detection.
ROI_BOTTOM = H - 22


@dataclass
class Tracklet:
    cam: str
    tid: str
    times: list = field(default_factory=list)
    us: list = field(default_factory=list)          # along-road position, meters
    colors: list = field(default_factory=list)      # Lab colour of the blob core
    lengths: list = field(default_factory=list)     # blob length, meters
    boxes: list = field(default_factory=list)
    crop: object = None
    crop_score: float = -1.0
    missed: int = 0

    @property
    def t_entry(self):
        return self.times[0]

    @property
    def t_exit(self):
        return self.times[-1]

    def speed(self):
        if len(self.times) < 3:
            return None
        A = np.vstack([self.times, np.ones(len(self.times))]).T
        slope, _ = np.linalg.lstsq(A, np.array(self.us), rcond=None)[0]
        return float(slope)

    def appearance(self):
        c = np.median(np.array(self.colors), axis=0)
        L = float(np.median(self.lengths))
        return np.array([c[0], c[1], c[2], L], dtype=float)

    def summary(self):
        a = self.appearance()
        return dict(cam=self.cam, tid=self.tid, t_entry=round(self.t_entry, 2),
                    t_exit=round(self.t_exit, 2), speed=self.speed(), lab=[float(x) for x in a[:3]],
                    length_m=float(a[3]), n_frames=len(self.times),
                    enters_left=bool(self.us[0] < 12.0), exits_right=bool(self.us[-1] > 2 * 75.0 - 12.0))


class CameraPerception:
    def __init__(self, cam, background, diff_threshold=34, min_area=40):
        self.cam = cam
        self.bg = background.astype(np.int16)
        self.diff_threshold = diff_threshold
        self.min_area = min_area
        self.active, self.done = [], []
        self.quality = []           # (t, gray std) per frame
        self._n = 0
        self.kernel = np.ones((3, 3), np.uint8)

    def detect(self, frame):
        diff = np.abs(frame.astype(np.int16) - self.bg).sum(axis=2)
        mask = (diff > self.diff_threshold).astype(np.uint8)
        mask[ROI_BOTTOM:] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        dets = []
        for k in range(1, n):
            x, y, w, h, area = stats[k]
            if area < self.min_area or w < 8:
                continue
            # Colour from the blob core, excluding the shadow-offset edge.
            core = labels[y:y + h, x:x + w] == k
            core[-4:, :] = False
            core[:, -4:] = False
            px = lab[y:y + h, x:x + w][core]
            if len(px) < 10:
                continue
            col = np.median(px.reshape(-1, 3), axis=0).astype(float)
            u_center = (x + (w - 3) / 2) / PX_PER_M
            touches_edge = x <= 1 or x + w >= W - 1
            dets.append(dict(u=u_center, color=col, length=(w - 3) / PX_PER_M, box=(int(x), int(y), int(w), int(h)),
                             full=not touches_edge))
        return dets

    def step(self, t, frame):
        gray = cv2.cvtColor(frame[:ROI_BOTTOM], cv2.COLOR_BGR2GRAY)
        self.quality.append((t, float(gray.std())))
        dets = self.detect(frame)
        used = set()
        for tr in sorted(self.active, key=lambda r: -r.us[-1]):
            best, best_d = None, 7.0
            for i, d in enumerate(dets):
                if i in used:
                    continue
                dist = d["u"] - tr.us[-1]
                if -1.5 <= dist <= best_d:
                    best, best_d = i, dist
            if best is None:
                tr.missed += 1
                continue
            used.add(best)
            d = dets[best]
            tr.missed = 0
            tr.times.append(t)
            tr.us.append(d["u"])
            tr.boxes.append(d["box"])
            if d["full"]:
                tr.colors.append(d["color"])
                tr.lengths.append(d["length"])
                score = -abs(d["u"] - 75.0)
                if score > tr.crop_score:
                    x, y, w, h = d["box"]
                    x0, y0 = max(0, x - 10), max(0, y - 14)
                    tr.crop = frame[y0:min(ROI_BOTTOM, y + h + 14), x0:min(W, x + w + 10)].copy()
                    tr.crop_score = score
        for i, d in enumerate(dets):
            if i in used or d["u"] > 30.0:
                continue        # new tracks only near the entry edge
            self._n += 1
            tr = Tracklet(self.cam, f"{self.cam[-1]}{self._n}", [t], [d["u"]], [], [], [d["box"]])
            if d["full"]:
                tr.colors.append(d["color"])
                tr.lengths.append(d["length"])
            self.active.append(tr)
        still = []
        for tr in self.active:
            if tr.missed > 3:
                if len(tr.times) >= 4 and tr.colors:
                    self.done.append(tr)
            else:
                still.append(tr)
        self.active = still

    def finish(self):
        for tr in self.active:
            if len(tr.times) >= 4 and tr.colors:
                self.done.append(tr)
        self.active = []
        return self.done
