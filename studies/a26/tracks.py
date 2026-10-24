"""Track and prediction comparisons for A26's identity gates.

`serialize_tracks` and `compare_tracks` are copies of D2's (d2/recorders.py, d2/baseline_tracks.py),
so that no D2 code runs on the machine (A25c's rule for the test annotations there).
"""
from __future__ import annotations

import math


def serialize_tracks(trajs) -> list[dict]:
    """Tracks as `model.traj` holds them -> plain lists, in the form of session 3's cache."""
    out = []
    for t in trajs:
        fs = sorted(int(f) for f in t["trajectory"])
        out.append(dict(tid=int(t["tid"]), category=t["category"], score=float(t["score"]),
                        begin_fid=int(t["begin_fid"]), end_fid=int(t["end_fid"]),
                        boxes=[[float(v) for v in t["trajectory"][f]] for f in fs]))
    return out


def _same_float(a, b) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == b


def compare_tracks(rec: list, base: list) -> list:
    """Same tracks (ids in order), classes, begin/end, every box exactly, and the score exactly
    (NaN equals NaN). Returns the differences; empty means identical."""
    diffs = []
    if len(rec) != len(base):
        diffs.append(f"track count {len(rec)} vs baseline {len(base)}")
    for r, b in zip(rec, base):
        for k in ("tid", "category", "begin_fid", "end_fid"):
            if r[k] != b[k]:
                diffs.append(f"tid {b['tid']}: {k} {r[k]!r} vs baseline {b[k]!r}")
        if not _same_float(float(r["score"]), float(b["score"])):
            diffs.append(f"tid {b['tid']}: score {r['score']!r} vs baseline {b['score']!r}")
        if r["boxes"] != b["boxes"]:
            n = sum(x != y for x, y in zip(r["boxes"], b["boxes"])) + abs(len(r["boxes"]) - len(b["boxes"]))
            diffs.append(f"tid {b['tid']}: {n} boxes differ")
    return diffs


def compare_predictions(ours: list, ref: list, tol: float) -> list:
    """Formatted predictions (one video, one split): same order, triplets, durations and
    trajectories exactly; scores within `tol`. Returns the differences; empty means identical."""
    diffs = []
    if len(ours) != len(ref):
        diffs.append(f"instance count {len(ours)} vs reference {len(ref)}")
    for i, (a, b) in enumerate(zip(ours, ref)):
        for k in ("triplet", "duration", "sub_traj", "obj_traj"):
            if list(a[k]) != list(b[k]):
                diffs.append(f"#{i}: {k} differs")
        if not abs(float(a["score"]) - float(b["score"])) <= tol:
            diffs.append(f"#{i}: score {a['score']!r} vs reference {b['score']!r}")
    return diffs
