"""Camera-network handoff monitor with identity-free certificates.

Specification for every link A -> B with gray zone G between them:

    every vehicle that leaves A toward B appears at B within its predicted window,
    and every vehicle that appears at B is explained by some departure from A.

Departures and arrivals form a bipartite graph whose edges pass two calibrated gates:
arrival time inside the departure's window, and appearance distance below tau. No
identity is ever committed. By Hall's theorem, if the closed departures cannot all be
matched to distinct arrivals, at least |D| - M of them did not arrive, whatever the
true identities are. The certificate is the deficient set itself.

Absence is only asserted where the downstream camera was healthy for the whole window.
Otherwise the verdict is UNVERIFIED, never "all clear".
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from world import COLORS, LINKS, cam_interval, gray_interval

LINK_DIST = {(a, b): gray_interval(i)[1] - gray_interval(i)[0] for i, (a, b, _) in enumerate(LINKS)}
GRAY_NAME = {(a, b): g for a, b, g in LINKS}
SIDE_TRACKS = {"G2": "NW side track"}


@dataclass
class Calibration:
    r_lo: float          # travel-time ratio bounds: T * v_hat / d
    r_hi: float
    tau: float           # appearance gate
    q_min: float         # minimum image-quality signal for a frame to count as covered

    def window(self, dep, link):
        v = max(dep["speed"] or 1.0, 1.0)
        d = LINK_DIST[link]
        return dep["t_exit"] + self.r_lo * d / v, dep["t_exit"] + self.r_hi * d / v

    def reverse_window(self, arr, link):
        v = max(arr["speed"] or 1.0, 1.0)
        d = LINK_DIST[link]
        return arr["t_entry"] - self.r_hi * d / v, arr["t_entry"] - self.r_lo * d / v


def appearance_distance(a, b):
    la, lb = np.array(a["lab"]), np.array(b["lab"])
    w = np.array([0.6, 1.0, 1.0])
    return float(np.sqrt(np.sum(w * (la - lb) ** 2)) + 2.5 * abs(a["length_m"] - b["length_m"]))


def describe(tr):
    """Human-readable appearance from measured colour and length only."""
    import cv2
    best, best_d = None, 1e9
    for name, bgr in COLORS.items():
        lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
        d = np.linalg.norm(lab - np.array(tr["lab"]))
        if d < best_d:
            best, best_d = name, d
    L = tr["length_m"]
    kind = "truck" if L > 7.0 else ("light vehicle" if L < 5.1 else "pickup/van")
    return f"{best} {kind} (~{L:.1f} m)"


def max_matching(n_left, n_right, edges):
    adj = [[] for _ in range(n_left)]
    for i, j in edges:
        adj[i].append(j)
    match_r = [-1] * n_right
    def augment(i, seen):
        for j in adj[i]:
            if not seen[j]:
                seen[j] = True
                if match_r[j] < 0 or augment(match_r[j], seen):
                    match_r[j] = i
                    return True
        return False
    size = sum(augment(i, [False] * n_right) for i in range(n_left))
    return size, match_r


def covered(quality, a, b, q_min):
    qs = [q for t, q in quality if a <= t <= b]
    return bool(qs) and min(qs) >= q_min


def gaps(quality, q_min):
    out, start = [], None
    for t, q in quality:
        if q < q_min and start is None:
            start = t
        if q >= q_min and start is not None:
            out.append((start, t))
            start = None
    if start is not None:
        out.append((start, quality[-1][0]))
    return out


def check_link(link, departures, arrivals, quality, cal, t_now):
    """Verdict for one link at time t_now. departures/arrivals are tracklet summaries."""
    a_cam, b_cam = link
    deps = [d for d in departures if d["exits_right"]]
    arrs = [r for r in arrivals if r["enters_left"] and r["t_entry"] <= t_now]
    win = [cal.window(d, link) for d in deps]
    def ok(i, j):
        return (win[i][0] <= arrs[j]["t_entry"] <= win[i][1]
                and appearance_distance(deps[i], arrs[j]) <= cal.tau)
    edges = [(i, j) for i in range(len(deps)) for j in range(len(arrs)) if ok(i, j)]
    closed = [i for i in range(len(deps)) if win[i][1] < t_now]
    # A departure counts as evidence only if its own camera saw it clearly, and its absence
    # downstream only if the downstream camera was clear for the whole window.
    dep_ok = [covered(quality[a_cam], d["t_entry"], d["t_exit"], cal.q_min) for d in deps]
    arr_ok = [covered(quality[b_cam], r["t_entry"], r["t_exit"], cal.q_min) for r in arrs]
    covered_closed = [i for i in closed if dep_ok[i] and covered(quality[b_cam], *win[i], cal.q_min)]
    uncovered_closed = [i for i in closed if i not in covered_closed]

    def sub(rows):
        index = {r: k for k, r in enumerate(rows)}
        return [(index[i], j) for i, j in edges if i in index]

    # Certified missing: Hall deficiency of closed, fully observed departures.
    m_cov, _ = max_matching(len(covered_closed), len(arrs), sub(covered_closed))
    certified_missing = len(covered_closed) - m_cov
    candidates = []
    if certified_missing > 0:
        for k, i in enumerate(covered_closed):
            rest = [r for r in covered_closed if r != i]
            m_rest, _ = max_matching(len(rest), len(arrs), sub(rest))
            if m_rest == m_cov:
                candidates.append(i)
    m_all_closed, _ = max_matching(len(closed), len(arrs), sub(closed))
    unverified = max(0, (len(closed) - m_all_closed) - certified_missing)

    # Unexplained arrivals: arrival-side Hall deficiency against all departures.
    edges_rev = [(j, i) for i, j in edges]
    m_arr, _ = max_matching(len(arrs), len(deps), edges_rev)
    deficit_arr = len(arrs) - m_arr
    unexplained, unverified_arr = [], []
    if deficit_arr > 0:
        for j in range(len(arrs)):
            others = [k for k in range(len(arrs)) if k != j]
            index = {k: n for n, k in enumerate(others)}
            m_rest, _ = max_matching(len(others), len(deps), [(index[jj], i) for jj, i in edges_rev if jj in index])
            if m_rest == m_arr:
                lo, hi = cal.reverse_window(arrs[j], link)
                admissible = arr_ok[j] and covered(quality[a_cam], lo, hi, cal.q_min)
                (unexplained if admissible else unverified_arr).append(j)
    if certified_missing > 0 or unexplained:
        status = "ALERT"
    elif unverified > 0 or unverified_arr:
        status = "UNVERIFIED"
    elif len(closed) < len(deps):
        status = "PENDING"
    else:
        status = "ALL ACCOUNTED"
    gray = GRAY_NAME[link]
    actions = []
    if certified_missing > 0:
        where = gray + (f" and the {SIDE_TRACKS[gray]}" if gray in SIDE_TRACKS else "")
        actions.append(f"Dispatch ISR asset to {where}: {certified_missing} vehicle(s) left {a_cam} and never reached {b_cam}.")
    if unexplained:
        actions.append(f"Track and identify {len(unexplained)} vehicle(s) at {b_cam} that no {a_cam} departure explains; they entered inside {gray}.")
    if unverified or unverified_arr:
        g = gaps(quality[b_cam], cal.q_min) or gaps(quality[a_cam], cal.q_min)
        span = f" ({g[0][0]:.0f}-{g[0][1]:.0f} s)" if g else ""
        actions.append(f"Coverage gap{span}: cannot confirm {unverified + len(unverified_arr)} handoff(s); task an alternate sensor or slew a PTZ.")
    return dict(link=f"{a_cam}->{b_cam}", gray=gray, status=status, n_departures=len(deps),
                n_arrivals=len(arrs), closed=len(closed), certified_missing=certified_missing,
                missing_candidates=[deps[i]["tid"] for i in candidates],
                unverified_departures=unverified,
                unexplained_arrivals=[arrs[j]["tid"] for j in unexplained],
                unverified_arrivals=[arrs[j]["tid"] for j in unverified_arr],
                windows={deps[i]["tid"]: [round(w, 1) for w in win[i]] for i in range(len(deps))},
                edges=[[deps[i]["tid"], arrs[j]["tid"]] for i, j in edges], actions=actions)


def committed_baseline(link, departures, arrivals, cal, t_now):
    """Conventional tracker: commit to one re-ID assignment, then check the rule.

    Same gates as the monitor, but a single Hungarian assignment and no coverage model.
    It names one specific vehicle as missing.
    """
    deps = [d for d in departures if d["exits_right"]]
    arrs = [r for r in arrivals if r["enters_left"] and r["t_entry"] <= t_now]
    if not deps:
        unknown = [r["tid"] for r in arrs]
        return dict(status="ALERT" if unknown else "ALL ACCOUNTED", named_missing=[], named_unknown=unknown)
    BIG = 1e6
    cost = np.full((len(deps), max(len(arrs), 1)), BIG)
    for i, d in enumerate(deps):
        lo, hi = cal.window(d, link)
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        for j, r in enumerate(arrs):
            dist = appearance_distance(d, r)
            if lo <= r["t_entry"] <= hi and dist <= cal.tau:
                cost[i, j] = dist / cal.tau + abs(r["t_entry"] - mid) / half
    rows, cols = linear_sum_assignment(cost)
    matched = {i: j for i, j in zip(rows, cols) if cost[i, j] < BIG and j < len(arrs)}
    named_missing = [deps[i]["tid"] for i in range(len(deps))
                     if i not in matched and cal.window(deps[i], link)[1] < t_now]
    named_unknown = [arrs[j]["tid"] for j in range(len(arrs)) if j not in matched.values()]
    status = "ALERT" if named_missing or named_unknown else "ALL ACCOUNTED"
    return dict(status=status, named_missing=named_missing, named_unknown=named_unknown)


def count_baseline(link, departures, arrivals, cal, t_now):
    """Count-only handoff: compares how many left with how many arrived in time."""
    deps = [d for d in departures if d["exits_right"] and cal.window(d, link)[1] < t_now]
    if not deps:
        return dict(status="ALL ACCOUNTED")
    lo = min(cal.window(d, link)[0] for d in deps)
    hi = max(cal.window(d, link)[1] for d in deps)
    arrs = [r for r in arrivals if r["enters_left"] and lo <= r["t_entry"] <= hi]
    return dict(status="ALERT" if len(arrs) < len(deps) else "ALL ACCOUNTED",
                departures=len(deps), arrivals=len(arrs))
