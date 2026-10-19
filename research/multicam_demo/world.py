"""Perimeter world: a ring road around a base, four fixed cameras, unobserved gray zones.

Distances are meters and times are seconds. Every vehicle follows one route polyline;
a camera sees a fixed arc-length interval of that route. Anomalies happen only inside
gray zones, where no camera can see them. Ground truth stays in `Episode.truth`; the
perception and monitoring code never reads it.
"""
from dataclasses import dataclass, field
import math
import numpy as np

ROUTE = np.array([(400, -900), (400, -500), (-700, -500), (-700, 500),
                  (700, 500), (700, -200), (450, -200)], dtype=float)
SEG = np.diff(ROUTE, axis=0)
SEG_LEN = np.hypot(SEG[:, 0], SEG[:, 1])
CUM = np.concatenate([[0.0], np.cumsum(SEG_LEN)])
ROUTE_LEN = float(CUM[-1])

HALF_VIEW = 75.0          # each camera sees +/-75 m of road along its axis
CAMERAS = {               # name -> arc-length at the camera's center
    "CAM-S": 800.0,
    "CAM-W": 2000.0,
    "CAM-N": 3200.0,
    "CAM-E": 4250.0,
}
CAM_ORDER = ["CAM-S", "CAM-W", "CAM-N", "CAM-E"]
LINKS = [("CAM-S", "CAM-W", "G1"), ("CAM-W", "CAM-N", "G2"), ("CAM-N", "CAM-E", "G3")]
NW_CORNER_S = float(CUM[3])     # side track branches off here, inside G2
STOP_S = 2300.0                 # a pull-off inside G2, out of every camera's view
SIDE_TRACK = np.array([(-700, 500), (-860, 660), (-1000, 820)], dtype=float)

TYPES = {"truck": (8.6, 2.5), "pickup": (5.4, 2.0), "suv": (4.8, 2.0), "van": (5.3, 2.1)}
COLORS = {  # BGR
    "tan": (122, 170, 202), "olive": (62, 112, 100), "white": (226, 228, 226),
    "black": (42, 42, 44), "grey": (132, 132, 130), "red": (44, 46, 176),
    "blue": (164, 98, 44),
}


def cam_interval(cam):
    c = CAMERAS[cam]
    return c - HALF_VIEW, c + HALF_VIEW


def gray_interval(link_index):
    a, b, _ = LINKS[link_index]
    return cam_interval(a)[1], cam_interval(b)[0]


def route_xy(s):
    s = float(np.clip(s, 0.0, ROUTE_LEN))
    i = int(np.searchsorted(CUM, s, side="right") - 1)
    i = min(i, len(SEG) - 1)
    f = (s - CUM[i]) / SEG_LEN[i]
    return ROUTE[i] + f * SEG[i]


def side_xy(d):
    """Position d meters along the side track from the NW corner."""
    seg = np.diff(SIDE_TRACK, axis=0)
    lens = np.hypot(seg[:, 0], seg[:, 1])
    for p, v, L in zip(SIDE_TRACK[:-1], seg, lens):
        if d <= L:
            return p + v * (d / L)
        d -= L
    return SIDE_TRACK[-1].copy()


@dataclass
class Vehicle:
    vid: str
    vtype: str
    color: str
    rgb_jitter: tuple
    # piecewise-linear arc-length schedule: list of (t, s); after the last knot the
    # vehicle holds still (stop) or is gone (left the map) depending on `fate`.
    knots: list = field(default_factory=list)
    fate: str = "route"         # route | stop | divert | decoy
    side_knots: list = field(default_factory=list)   # (t, d) on the side track
    decoy: bool = False

    @property
    def length(self):
        return TYPES[self.vtype][0]

    @property
    def width(self):
        return TYPES[self.vtype][1]

    def bgr(self):
        base = np.array(COLORS[self.color], dtype=float) + np.array(self.rgb_jitter)
        return tuple(int(x) for x in np.clip(base, 0, 255))

    def s_at(self, t):
        """Arc length on the main route at time t, or None when off the route."""
        if not self.knots or t < self.knots[0][0]:
            return None
        for (t0, s0), (t1, s1) in zip(self.knots[:-1], self.knots[1:]):
            if t0 <= t <= t1:
                return s0 + (s1 - s0) * (t - t0) / max(t1 - t0, 1e-9)
        t_last, s_last = self.knots[-1]
        if self.fate == "stop":
            return s_last
        if self.fate == "route" and s_last >= ROUTE_LEN - 1e-6:
            return None       # entered the base
        return None           # diverted onto the side track

    def world_xy(self, t):
        s = self.s_at(t)
        if s is not None:
            return route_xy(s), "route"
        if self.side_knots and self.side_knots[0][0] <= t <= self.side_knots[-1][0]:
            for (t0, d0), (t1, d1) in zip(self.side_knots[:-1], self.side_knots[1:]):
                if t0 <= t <= t1:
                    return side_xy(d0 + (d1 - d0) * (t - t0) / max(t1 - t0, 1e-9)), "side"
        return None, None


@dataclass
class Episode:
    seed: int
    kind: str                   # none | stop | divert | swap | degraded
    vehicles: list
    duration: float
    degraded: dict              # cam -> (t_start, t_end) of dust degradation
    truth: dict                 # evaluator-only facts


def _schedule(t_start, v_base, rng, gray_mult=None):
    """Knots (t, s) for the full route with per-zone speed variation."""
    bounds = sorted({0.0, ROUTE_LEN, *[x for c in CAM_ORDER for x in cam_interval(c)]})
    knots, t = [(t_start, 0.0)], t_start
    for a, b in zip(bounds[:-1], bounds[1:]):
        in_cam = any(cam_interval(c)[0] <= a < cam_interval(c)[1] for c in CAM_ORDER)
        if in_cam:
            m = rng.uniform(0.97, 1.03)
        else:
            key = round(a, 1)
            m = gray_mult.get(key, rng.uniform(0.85, 1.15)) if gray_mult else rng.uniform(0.85, 1.15)
        t += (b - a) / (v_base * m)
        knots.append((t, b))
    return knots


def _enforce_gaps(vehicles, min_gap=2.6, cams=CAM_ORDER):
    """Queue vehicles that would overlap inside a camera by delaying the later one."""
    for cam in cams:
        enter = cam_interval(cam)[0]
        def t_enter(v):
            for (t0, s0), (t1, s1) in zip(v.knots[:-1], v.knots[1:]):
                if s0 <= enter <= s1 and s1 > s0:
                    return t0 + (enter - s0) / (s1 - s0) * (t1 - t0)
            return None
        timed = sorted(((t_enter(v), v) for v in vehicles if t_enter(v) is not None),
                       key=lambda x: x[0])
        last = -1e9
        for t_e, v in timed:
            if t_e - last < min_gap:
                delay = min_gap - (t_e - last) + 0.05
                v.knots = [(t + (delay if s >= enter - 1e-6 else 0.0), s) for t, s in v.knots]
                # The knot just before the camera absorbs the delay (slower in the gray zone).
                t_e += delay
            last = t_e


def _no_overtaking_in_view(vehicles, min_gap=2.6, cams=CAM_ORDER):
    """One lane: inside a camera view a follower cannot pass or merge into its leader."""
    def crossing(v, target):
        for (t0, s0), (t1, s1) in zip(v.knots[:-1], v.knots[1:]):
            if s0 <= target <= s1 and s1 > s0:
                return t0 + (target - s0) / (s1 - s0) * (t1 - t0)
        return None
    for cam in cams:
        enter, leave = cam_interval(cam)
        order = sorted((v for v in vehicles if crossing(v, enter) is not None and crossing(v, leave) is not None),
                       key=lambda v: crossing(v, enter))
        prev_exit = -1e9
        for v in order:
            t_exit = crossing(v, leave)
            if t_exit < prev_exit + min_gap:
                delay = prev_exit + min_gap - t_exit + 0.05
                v.knots = [(t + (delay if s >= leave - 1e-6 else 0.0), s) for t, s in v.knots]
                t_exit += delay
            prev_exit = t_exit


def _appearance(rng, avoid=None):
    vtype = rng.choice(list(TYPES))
    color = rng.choice(list(COLORS))
    if avoid is not None:
        for _ in range(100):
            if all(vtype != a.vtype and color != a.color for a in avoid):
                break
            vtype, color = rng.choice(list(TYPES)), rng.choice(list(COLORS))
    jitter = tuple(float(x) for x in rng.normal(0, 4, 3))
    return str(vtype), str(color), jitter


def make_episode(seed, kind, n_vehicles=None, similar_pair=None, target=None):
    rng = np.random.default_rng(seed)
    n = int(n_vehicles or rng.integers(3, 6))
    similar_pair = bool(rng.random() < 0.7) if similar_pair is None else similar_pair
    vehicles = []
    for i in range(n):
        vtype, color, jitter = _appearance(rng)
        vehicles.append(Vehicle(f"V{i + 1}", vtype, color, jitter))
    if similar_pair and n >= 2:
        i, j = sorted(rng.choice(n, 2, replace=False))
        vehicles[j].vtype, vehicles[j].color = vehicles[i].vtype, vehicles[i].color
        vehicles[j].rgb_jitter = tuple(float(x) for x in rng.normal(0, 4, 3))
    t = 5.0
    for v in vehicles:
        v_base = rng.uniform(9.5, 12.5)
        v.knots = _schedule(t, v_base, rng)
        t += rng.uniform(4.0, 9.0)
    _enforce_gaps(vehicles)

    truth = {"kind": kind, "n": n, "missing": [], "decoy": None, "link": "CAM-W->CAM-N"}
    degraded = {}
    if kind in ("stop", "divert", "swap"):
        if target is None:
            target = int(rng.integers(n))
        v = vehicles[target]
        truth["missing"] = [v.vid]
        cut_s = STOP_S if kind == "stop" else NW_CORNER_S
        t_cut = None
        new = []
        for (t0, s0), (t1, s1) in zip(v.knots[:-1], v.knots[1:]):
            new.append((t0, s0))
            if s0 <= cut_s <= s1:
                t_cut = t0 + (cut_s - s0) / (s1 - s0) * (t1 - t0)
                new.append((t_cut, cut_s))
                break
        v.knots = new
        v.fate = "stop" if kind == "stop" else "divert"
        if kind != "stop":
            speed = rng.uniform(6, 9)
            v.side_knots = [(t_cut, 0.0), (t_cut + 460.0 / speed, 460.0)]
        if kind == "swap":
            vtype, color, jitter = _appearance(rng, avoid=vehicles)
            d = Vehicle(f"X{n + 1}", vtype, color, jitter, decoy=True, fate="route")
            t_join = t_cut + rng.uniform(-4, 6)
            speed = rng.uniform(9.5, 12.5)
            d.side_knots = [(t_join - 460.0 / 7.5, 460.0), (t_join, 0.0)]
            sched = _schedule(0.0, speed, rng)
            # Re-time the decoy's schedule so it is at the NW corner at t_join.
            t_at_corner = np.interp(NW_CORNER_S, [s for _, s in sched], [t for t, _ in sched])
            d.knots = [(t - t_at_corner + t_join, s) for t, s in sched if s >= NW_CORNER_S - 1e-6]
            d.knots.insert(0, (t_join, NW_CORNER_S))
            d.knots = sorted({(round(t, 6), s) for t, s in d.knots}, key=lambda x: x[0])
            vehicles.append(d)
            truth["decoy"] = d.vid
    if kind == "degraded":
        # Dust over CAM-N for the whole time any vehicle could be arriving there.
        entries = []
        for v in vehicles:
            for (t0, s0), (t1, s1) in zip(v.knots[:-1], v.knots[1:]):
                a = cam_interval("CAM-N")[0]
                if s0 <= a <= s1:
                    entries.append(t0 + (a - s0) / (s1 - s0) * (t1 - t0))
        degraded["CAM-N"] = (min(entries) - 30.0, max(entries) + 40.0)
    # Camera by camera along the road: entry spacing, then no overtaking while in view.
    # Delays only shift knots downstream, so earlier cameras stay valid.
    for cam in CAM_ORDER:
        _enforce_gaps(vehicles, cams=[cam])
        _no_overtaking_in_view(vehicles, cams=[cam])
    end = 0.0
    for v in vehicles:
        end = max(end, v.knots[-1][0], v.side_knots[-1][0] if v.side_knots else 0.0)
    return Episode(seed, kind, vehicles, min(end + 10.0, 900.0), degraded, truth)
