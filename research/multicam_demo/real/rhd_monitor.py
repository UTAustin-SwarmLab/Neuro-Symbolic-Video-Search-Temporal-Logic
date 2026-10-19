"""Identity-free handoff certificates on real traffic (RoundaboutHD, E 96th St, four cameras).

For every directed link A -> B the monitor sees only single-camera tracks: when a vehicle left A
toward B, when vehicles entered B from A's side, and appearance vectors per track. A departure and
an arrival are joined by an edge when the travel time lies in the calibrated window and the pair
passes the calibrated appearance gate. No identity is committed.

In each connected component of that graph, Hall's deficiency k_c = |D_c| - maximum matching is a
lower bound on the number of departures that did not reach B within the window, whatever the true
identities. The departures left unmatched by some maximum matching form the candidate set, which
holds at least k_c truly missing vehicles whenever every true pair is an edge. The same argument on
the arrival side certifies arrivals that no departure explains (they entered inside the gray zone).

Two-fold time split: windows and gates are calibrated with labelled pairs from one half of the
recording and evaluated on the other half. Global IDs are read only for calibration and scoring.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rhd_pairfeat import FEATURES, PairGate, attach_descriptors, pair_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "real"
VIDEO_END = 600.0
HALVES = [(0.0, 300.0), (300.0, 600.0)]
LINKS = [  # A, exit leg at A, B, entry leg at B, gray zone
    (1, "E", 2, "W", "G1"), (2, "W", 1, "E", "G1"),
    (2, "E", 3, "W", "G2"), (3, "W", 2, "E", "G2"),
    (3, "E", 4, "W", "G3"), (4, "W", 3, "E", "G3"),
]
GATES = ["pair", "reid", "dino", "none"]


def key(t):
    return f"{t['cam']}:{t['lid']}"


def lname(link):
    return f"CAM{link[0]}->CAM{link[2]}"


def load():
    tracks = json.loads((OUT / "tracks.json").read_text())
    feats = json.loads((OUT / "features_dinov2_vits14.json").read_text())["feats"]
    for t in tracks["tracks"]:
        t["emb"] = np.array(feats[key(t)]) if key(t) in feats else None
    return tracks


def link_sets(tracks, link):
    a, la, b, lb, _ = link
    deps = [t for t in tracks if t["cam"] == a and t["leg_out"] == la]
    maybe_deps = [t for t in tracks if t["cam"] == a and t["leg_out"] in (la, "X")]
    arrs = [t for t in tracks if t["cam"] == b and t["leg_in"] == lb]
    maybe_arrs = [t for t in tracks if t["cam"] == b and t["leg_in"] in (lb, "X")]
    return deps, maybe_deps, arrs, maybe_arrs


def true_pairs(deps, b_tracks):
    """Labelled departure -> first later sighting of the same vehicle at B (any leg)."""
    by = defaultdict(list)
    for r in b_tracks:
        if r["gid"] is not None:
            by[r["gid"]].append(r)
    out = {}
    for d in deps:
        if d["gid"] is None:
            continue
        later = [r for r in by[d["gid"]] if r["t_entry"] > d["t_exit"]]
        if later:
            out[key(d)] = min(later, key=lambda r: r["t_entry"])
    return out


def max_matching(n_left, adj, n_right, order=None):
    match_r = [-1] * n_right
    fails = []

    def augment(i, seen):
        for j in adj[i]:
            if not seen[j]:
                seen[j] = True
                if match_r[j] < 0 or augment(match_r[j], seen):
                    match_r[j] = i
                    return True
        return False

    for i in (order if order is not None else range(n_left)):
        if not augment(i, [False] * n_right):
            fails.append(i)
    match_l = [-1] * n_left
    for j, i in enumerate(match_r):
        if i >= 0:
            match_l[i] = j
    return match_l, match_r, fails


def certificate(n_left, n_right, adj, active=None):
    """Components, deficiency per component and Gallai-Edmonds candidate sets on the left side."""
    active = list(range(n_left)) if active is None else active
    sub = {i: adj[i] for i in active}
    adj_full = [sub.get(i, []) for i in range(n_left)]
    match_l, match_r, _ = max_matching(n_left, adj_full, n_right, order=active)
    miss = set(i for i in active if match_l[i] < 0)
    stack = list(miss)
    while stack:
        i = stack.pop()
        for j in adj_full[i]:
            i2 = match_r[j]
            if i2 >= 0 and i2 not in miss:
                miss.add(i2)
                stack.append(i2)
    parent = list(range(n_left + n_right))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in active:
        for j in adj_full[i]:
            parent[find(i)] = find(n_left + j)
    comps = defaultdict(lambda: dict(left=[], right=[]))
    for i in active:
        comps[find(i)]["left"].append(i)
    for j in range(n_right):
        r = find(n_left + j)
        if r in comps:
            comps[r]["right"].append(j)
    out = []
    for c in comps.values():
        k = sum(1 for i in c["left"] if match_l[i] < 0)
        out.append(dict(left=c["left"], right=c["right"], k=k, cand=[i for i in c["left"] if i in miss]))
    return out


def committed(n_left, n_right, adj, cost):
    """Conventional re-ID: one min-cost assignment under the same gates; unmatched = named missing."""
    if n_left == 0:
        return []
    BIG = 1e6
    C = np.full((n_left, max(1, n_right)), BIG)
    for i in range(n_left):
        for j in adj[i]:
            C[i, j] = cost(i, j)
    rows, cols = linear_sum_assignment(C)
    matched = {i for i, j in zip(rows, cols) if C[i, j] < BIG}
    return [i for i in range(n_left) if i not in matched]


class Evaluator:
    def __init__(self):
        self.T = load()["tracks"]
        attach_descriptors(self.T)
        self.bycam = defaultdict(list)
        for t in self.T:
            self.bycam[t["cam"]].append(t)
        self.sets = {lname(l): link_sets(self.T, l) for l in LINKS}
        self.truth = {lname(l): true_pairs(self.sets[lname(l)][0], self.bycam[l[2]]) for l in LINKS}
        self.pf = {}

    def feat(self, d, r):
        k = (key(d), key(r))
        if k not in self.pf:
            self.pf[k] = pair_features(d, r)
        return self.pf[k]

    def calibrate(self, half):
        cal, Xs, ys = {}, [], []
        for link in LINKS:
            deps, _, _, maybe_arrs = self.sets[lname(link)]
            tp = self.truth[lname(link)]
            in_half = [d for d in deps if half[0] <= d["t_exit"] < half[1]]
            T = np.array([tp[key(d)]["t_entry"] - d["t_exit"] for d in in_half if key(d) in tp])
            med = float(np.median(T))
            through = T[T <= 2.5 * med]  # long stops are what the monitor should flag, not calibrate on
            c = dict(T_lo=0.8 * float(np.percentile(through, 5)), T_hi=1.3 * float(np.percentile(through, 95)),
                     T_med=med, n_pairs=int(len(T)))
            cal[lname(link)] = c
            for d in in_half:
                rt = tp.get(key(d))
                for r in maybe_arrs:
                    if c["T_lo"] <= r["t_entry"] - d["t_exit"] <= c["T_hi"]:
                        Xs.append(self.feat(d, r))
                        ys.append(int(rt is not None and key(rt) == key(r)))
        X, y = np.array(Xs), np.array(ys)
        gates = {"pair": PairGate(0.995, 0.5).fit(X, y)}
        for name, col in (("reid", FEATURES.index("reid_mean")), ("dino", FEATURES.index("dino_mean"))):
            gates[name] = float(np.quantile(X[y == 1, col], 0.995) * 1.1)
        return cal, gates, dict(pairs=int(y.sum()), impostors=int((1 - y).sum()))

    def gate_ok(self, gates, name, f):
        if name == "none":
            return True
        if name == "pair":
            return gates["pair"].ok(f)
        return f[FEATURES.index(f"{name}_mean")] <= gates[name]

    def run(self, gate_name="pair", keep_events=False):
        summary = defaultdict(lambda: defaultdict(int))
        events, cal_log = [], []
        for fold, (cal_half, ev_half) in enumerate([(HALVES[0], HALVES[1]), (HALVES[1], HALVES[0])]):
            cal, gates, cal_n = self.calibrate(cal_half)
            cal_log.append(dict(fold=fold, cal_half=cal_half, **cal_n, links=cal,
                                gate={k: (v if isinstance(v, float) else "logistic pair score") for k, v in gates.items()}))
            for link in LINKS:
                nm = lname(link)
                c = cal[nm]
                lo, hi, med = c["T_lo"], c["T_hi"], c["T_med"]
                deps, maybe_deps, arrs, maybe_arrs = self.sets[nm]
                tp = self.truth[nm]
                D = [d for d in deps if ev_half[0] <= d["t_exit"] < ev_half[1] and d["t_exit"] + hi <= VIDEO_END]
                pending = sum(1 for d in deps if ev_half[0] <= d["t_exit"] < ev_half[1] and d["t_exit"] + hi > VIDEO_END)
                R = maybe_arrs
                adj = [[j for j, r in enumerate(R) if lo <= r["t_entry"] - d["t_exit"] <= hi
                        and self.gate_ok(gates, gate_name, self.feat(d, r))] for d in D]
                miss = [key(d) not in tp or tp[key(d)]["t_entry"] - d["t_exit"] > hi for d in D]
                s = summary[nm]
                s["departures"] += len(D)
                s["pending_at_end"] += pending
                s["truly_missing"] += sum(miss)
                s["edges"] += sum(len(a) for a in adj)
                # streaming: departures close in window-end order; deficiency can only grow
                order = sorted(range(len(D)), key=lambda i: D[i]["t_exit"])
                _, _, fails = max_matching(len(D), adj, len(R), order=order)
                comps = certificate(len(D), len(R), adj)
                comp_of = {i: ci for ci, cc in enumerate(comps) for i in cc["left"]}
                for cc in comps:
                    if cc["k"] == 0:
                        continue
                    m_in = sum(miss[i] for i in cc["cand"])
                    s["mon_claimed"] += cc["k"]
                    s["mon_found"] += min(cc["k"], m_in)
                    s["mon_wrong"] += max(0, cc["k"] - m_in)
                    s["mon_set"] += len(cc["cand"])
                    s["mon_alerts"] += 1
                    s["mon_pinned"] += int(len(cc["cand"]) == cc["k"])
                covered = set(i for cc in comps if cc["k"] > 0 for i in cc["cand"] if miss[i])
                s["mon_missed"] += sum(1 for i in range(len(D)) if miss[i] and i not in covered)
                if keep_events:
                    seen_k = defaultdict(int)
                    for i in fails:
                        # the prefix certificate at the moment departure i's window closed
                        t_close = D[i]["t_exit"] + hi
                        act = [x for x in order if D[x]["t_exit"] + hi <= t_close]
                        pc = certificate(len(D), len(R), adj, active=act)
                        cc = next(c2 for c2 in pc if i in c2["left"])
                        events.append(dict(
                            kind="missing", link=nm, gray=link[4], fold=fold, t=round(t_close, 2), k=cc["k"],
                            dep=key(D[i]), candidates=[key(D[x]) for x in cc["cand"]],
                            arrivals=[key(R[j]) for j in cc["right"]],
                            truth_missing=[key(D[x]) for x in cc["cand"] if miss[x]],
                            window=[round(D[i]["t_exit"] + lo, 2), round(t_close, 2)]))
                # committed re-ID baseline
                def cost(i, j):
                    return (1 - gates["pair"].score(self.feat(D[i], R[j]))[0] / 10) + abs((R[j]["t_entry"] - D[i]["t_exit"]) - med) / (hi - lo)
                named = committed(len(D), len(R), adj, cost)
                s["com_claimed"] += len(named)
                s["com_found"] += sum(miss[i] for i in named)
                s["com_wrong"] += sum(not miss[i] for i in named)
                s["com_missed"] += sum(1 for i in range(len(D)) if miss[i] and i not in set(named))
                # count-only baseline: 60 s departure bins vs arrivals shifted back by the median transit
                for b0 in np.arange(ev_half[0], ev_half[1], 60.0):
                    Db = [i for i, d in enumerate(D) if b0 <= d["t_exit"] < b0 + 60]
                    nb = sum(1 for r in arrs if b0 <= r["t_entry"] - med < b0 + 60)
                    k = max(0, len(Db) - nb)
                    mb = sum(miss[i] for i in Db)
                    s["cnt_claimed"] += k
                    s["cnt_found"] += min(k, mb)
                    s["cnt_wrong"] += max(0, k - mb)
                    s["cnt_set"] += len(Db) if k else 0
                    s["cnt_missed"] += mb - min(k, mb)
                # arrival side: arrivals in the evaluation half that no departure explains
                A = [r for r in arrs if ev_half[0] <= r["t_entry"] < ev_half[1] and r["t_entry"] - hi >= 0]
                Dall = maybe_deps
                adj_r = [[i for i, d in enumerate(Dall) if lo <= r["t_entry"] - d["t_exit"] <= hi
                          and self.gate_ok(gates, gate_name, self.feat(d, r))] for r in A]
                dep_gids = defaultdict(list)
                for d in deps:
                    if d["gid"] is not None:
                        dep_gids[d["gid"]].append(d)
                joined = [not any(0 < r["t_entry"] - d["t_exit"] <= hi for d in dep_gids.get(r["gid"], [])) for r in A]
                s["arrivals"] += len(A)
                s["truly_joined"] += sum(joined)
                comps_r = certificate(len(A), len(Dall), adj_r)
                for cc in comps_r:
                    if cc["k"] == 0:
                        continue
                    m_in = sum(joined[j] for j in cc["cand"])
                    s["join_claimed"] += cc["k"]
                    s["join_found"] += min(cc["k"], m_in)
                    s["join_wrong"] += max(0, cc["k"] - m_in)
                    s["join_set"] += len(cc["cand"])
                    if keep_events:
                        j0 = max(cc["cand"], key=lambda j: A[j]["t_entry"])
                        events.append(dict(
                            kind="joined", link=nm, gray=link[4], fold=fold, t=round(max(A[j]["t_entry"] for j in cc["left"]), 2),
                            k=cc["k"], arr=key(A[j0]), candidates=[key(A[j]) for j in cc["cand"]],
                            truth_joined=[key(A[j]) for j in cc["cand"] if joined[j]]))
        agg = defaultdict(int)
        for v in summary.values():
            for k2, x in v.items():
                agg[k2] += x
        return dict(gate=gate_name, summary={k: dict(v) for k, v in summary.items()}, total=dict(agg),
                    calibration=cal_log, events=sorted(events, key=lambda e: e["t"]))


def strip(cal_log):
    return [{k: v for k, v in c.items()} for c in cal_log]


if __name__ == "__main__":
    ev = Evaluator()
    reports = {}
    for g in GATES:
        r = ev.run(g, keep_events=(g == "pair"))
        reports[g] = r
        t = r["total"]
        print(f"[{g}] deps {t['departures']} missing {t['truly_missing']} | monitor claimed {t.get('mon_claimed',0)} "
              f"found {t.get('mon_found',0)} wrong {t.get('mon_wrong',0)} set {t.get('mon_set',0)} pinned {t.get('mon_pinned',0)} "
              f"| committed claimed {t.get('com_claimed',0)} wrong {t.get('com_wrong',0)} | count claimed {t.get('cnt_claimed',0)} "
              f"wrong {t.get('cnt_wrong',0)} | joiners {t['truly_joined']}/{t['arrivals']} claimed {t.get('join_claimed',0)} wrong {t.get('join_wrong',0)}")
    (OUT / "monitor_report.json").write_text(json.dumps(reports, indent=1, default=float))
