"""Slot-level measurement at the relation classifier's own time scale.

Builds the classifier's scoring grid from WHOLE annotation tracks, cuts the
measurement windows W0-W3 around each slot, and defines the two relative-motion
candidates of stage 1 session 2:

  D1 distance -- session 1's rule (measured_axes.relative_motion), unchanged;
  D2 heading  -- the subject's camera-compensated velocity projected onto the
                 unit subject->object direction.

Grid (interface_audit.md B2; models/gen_labels.py:655-667): ordered pairs of
distinct tracks overlapping >= 10 frames; the overlap cut into 30-frame slots;
a final partial slot kept if >= 10 frames, otherwise trimmed.

Every decision is a pure function of per-window statistics (`d1_stats`,
`d2_stats`), so the thresholds fitted from cached statistics and the functions
applied to boxes are the same code. Time reversal is exact by construction:
per-step quantities are reversed and negated, and every sum is math.fsum.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage1"))
import measured_axes as M  # noqa: E402

CLIP_LEN = 30
MIN_OVERLAP = 10
MIN_TAIL = 10
LABEL_MIN_COVER = 15
WINDOWS = ("W0", "W1", "W2", "W3")
_HALF = {"W0": 0, "W1": 1, "W2": 2}


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
def whole_tracks(trajectories: list[list[dict]]) -> dict[int, tuple[int, int]]:
    """tid -> [first annotated frame, last + 1), from an annotation's per-frame
    `trajectories`. Gaps inside a span are left as gaps (build_pair skips them)."""
    span: dict[int, list[int]] = {}
    for fid, frame in enumerate(trajectories):
        for e in frame:
            s = span.setdefault(e["tid"], [fid, fid + 1])
            s[1] = fid + 1
    return {t: (b, e) for t, (b, e) in span.items()}


def slot_grid(tracks: dict[int, tuple[int, int]]):
    """-> list of (sub_tid, obj_tid, (b, e), [(s0, s1), ...]) over ordered pairs."""
    out = []
    for s, (sb, se) in tracks.items():
        for o, (ob, oe) in tracks.items():
            if s == o:
                continue
            b, e = max(sb, ob), min(se, oe)
            if e - b < MIN_OVERLAP:
                continue
            n, tail = (e - b) // CLIP_LEN, (e - b) % CLIP_LEN
            if tail >= MIN_TAIL:
                n += 1
            elif tail > 0:
                e -= tail
            slots = [(b + k * CLIP_LEN, e if k == n - 1 else b + (k + 1) * CLIP_LEN)
                     for k in range(n)]
            out.append((s, o, (b, e), slots))
    return out


def window(slot: tuple[int, int], overlap: tuple[int, int], w: str) -> tuple[int, int]:
    """W0 = the slot; W1/W2 = +/- 1/2 slots; W3 = the whole overlap; all clipped
    to the pair overlap."""
    b, e = overlap
    if w == "W3":
        return b, e
    k = _HALF[w] * CLIP_LEN
    return max(b, slot[0] - k), min(e, slot[1] + k)


def slot_labels(slot, sub_tid, obj_tid, rels, min_cover: int = LABEL_MIN_COVER) -> set[str]:
    """Predicates of `rels` (already restricted to BASE) whose annotated extent
    covers >= min_cover frames of the slot for this ordered pair."""
    s0, s1 = slot
    return {r["predicate"] for r in rels
            if r["subject_tid"] == sub_tid and r["object_tid"] == obj_tid
            and min(s1, r["end_fid"]) - max(s0, r["begin_fid"]) >= min_cover}


# ---------------------------------------------------------------------------
# D2 heading
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Heading:
    v: np.ndarray   # (k, 2) compensated subject steps
    u: np.ndarray   # (k, 2) unit subject->object direction at each step's midpoint


def build_heading(frames, sub_tid, obj_tid, begin, end, ecc_video) -> Heading:
    """One entry per step t-1 -> t in [begin, end) where both boxes exist at
    both frames and a warp exists. The direction is the normalised sum of the
    unit directions at t-1 and t, so it is identical under time reversal."""
    def unit(s, o):
        d = M._centre(o) - M._centre(s)
        n = math.hypot(d[0], d[1])
        return d / n if n > 0 else None

    vs, us = [], []
    end = min(end, len(frames))
    for t in range(max(begin, 0) + 1, end):
        s0, s1 = frames[t - 1].get(sub_tid), frames[t].get(sub_tid)
        o0, o1 = frames[t - 1].get(obj_tid), frames[t].get(obj_tid)
        if s0 is None or s1 is None or o0 is None or o1 is None:
            continue
        w = M._warp(ecc_video, t)
        if w is None:
            continue
        a, b = unit(s0, o0), unit(s1, o1)
        if a is None or b is None:
            continue
        m = a + b
        nm = math.hypot(m[0], m[1])
        if nm == 0:
            continue
        q = w @ np.array([*M._centre(s0), 1.0])
        vs.append(M._centre(s1) - q[:2])
        us.append(m / nm)
    return Heading(np.asarray(vs, float).reshape(-1, 2), np.asarray(us, float).reshape(-1, 2))


def reverse_heading(h: Heading) -> Heading:
    return replace(h, v=-h.v[::-1], u=h.u[::-1])


def _weighted_cos(v: np.ndarray, u: np.ndarray) -> float:
    """sum(v . u) / sum(|v|): the mean cosine, weighted by step length, so
    near-zero jitter steps contribute almost nothing."""
    if not len(v):
        return float("nan")
    num = math.fsum(float(x) for x in (v * u).sum(1))
    den = math.fsum(float(x) for x in np.linalg.norm(v, axis=1))
    return num / den if den > 0 else float("nan")


def _thirds_idx(k: int):
    t = max(k // 3, 1)
    return slice(0, t), slice(k - t, k)


def d2_stats(pair: M.Pair, h: Heading) -> dict:
    first, last = _thirds_idx(len(h.v))
    return dict(
        mean_cos=_weighted_cos(h.v, h.u),
        first_cos=_weighted_cos(h.v[first], h.u[first]) if len(h.v) >= 3 else float("nan"),
        last_cos=_weighted_cos(h.v[last], h.u[last]) if len(h.v) >= 3 else float("nan"),
        sub_motion=M.subject_motion(pair), obj_motion=M.object_motion(pair),
        dir_cos=M._direction_cos(pair))


def d2_decide(st: dict, c: float, motion_threshold: float) -> str:
    """pass: subject moving, first-third cosine > c and last-third < -c (the sign
    flips + to - inside the window); toward / away: subject moving, mean cosine
    > c / < -c; co_move: both moving, net velocities agree (cosine > 0), |mean
    cosine| < c; else none. Checked in that order."""
    sm = st["sub_motion"] > motion_threshold
    if sm and st["first_cos"] > c and st["last_cos"] < -c:
        return "pass"
    if sm and st["mean_cos"] > c:
        return "toward"
    if sm and st["mean_cos"] < -c:
        return "away"
    if (sm and st["obj_motion"] > motion_threshold and st["dir_cos"] > 0
            and abs(st["mean_cos"]) < c):
        return "co_move"
    return "none"


def relative_motion_heading(pair: M.Pair, h: Heading, c: float, motion_threshold: float) -> str:
    return d2_decide(d2_stats(pair, h), c, motion_threshold)


# ---------------------------------------------------------------------------
# D1 distance, as statistics (identical to measured_axes.relative_motion)
# ---------------------------------------------------------------------------
def d1_stats(pair: M.Pair) -> dict:
    d = M.distance_series(pair)
    first, mid, last = M._thirds(d)
    return dict(n=len(d), first=M._mean(first), last=M._mean(last),
                midmin=float(mid.min()) if len(mid) else float("nan"),
                sub_motion=M.subject_motion(pair), obj_motion=M.object_motion(pair),
                dir_cos=M._direction_cos(pair))


def d1_decide(st: dict, dead_zone: float, motion_threshold: float) -> str:
    if st["n"] >= 3 and st["midmin"] == st["midmin"] and \
            st["first"] - st["midmin"] > dead_zone and st["last"] - st["midmin"] > dead_zone:
        return "pass"
    dd = st["last"] - st["first"]
    if dd < -dead_zone:
        return "approach"
    if dd > dead_zone:
        return "recede"
    if st["sub_motion"] > motion_threshold and st["obj_motion"] > motion_threshold and st["dir_cos"] > 0:
        return "co_move"
    return "none"
