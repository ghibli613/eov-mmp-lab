"""MEASURED axes of the frozen schema, as functions of two box sequences.

Every function takes a `Pair`: the subject and object GT boxes over a relation's
annotated extent (frames where both are annotated), plus camera-compensated
per-step displacements of the two box centres. Build one with `build_pair`.

Frozen schema: pilot_analysis/stage0e/phi_frozen.json. Thresholds that cannot be
avoided (moving speed, adjacency gap, relative-motion dead zone) are fitted by
pilot_analysis/stage1/axis_check.py and read with `load_thresholds()`.

Symmetry is exact by construction, and axis_check.py tests it:
  * every signed axis is the sign of a mean difference or of a mean LOG ratio,
    so swapping subject and object flips it exactly;
  * displacements are stored per step, so time reversal is "reverse the order
    and negate the steps";
  * relative distance is measured within each frame, where camera motion
    cancels, and split into symmetric thirds.

Camera compensation: VidVRD_ECC_train.json holds one 3x3 warp per frame keyed by
the 1-based frame index; for 0-based frame t, ecc[str(t + 1)] maps frame t-1
into frame t (models/tracking/deep_sort/track.py:141; checked in stage 0c).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THRESHOLDS_FILE = os.path.join(HERE, "thresholds.json")

HORIZONTAL = ("left", "right")
VERTICAL = ("above", "beneath")
DEPTH = ("front", "behind")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Pair:
    sub: np.ndarray        # (n, 4) x1, y1, x2, y2 per frame where both are annotated
    obj: np.ndarray        # (n, 4)
    sub_steps: np.ndarray  # (k, 2) compensated centre displacement / frame diagonal
    obj_steps: np.ndarray  # (k, 2)
    width: float
    height: float

    @property
    def n(self) -> int:
        return len(self.sub)


def _warp(ecc_video: dict, t: int) -> np.ndarray | None:
    m = ecc_video.get(str(t + 1))
    if m is None:
        return None
    m = np.asarray(m, float)
    # same guard as the tracker: a degenerate warp is treated as identity
    return m if np.linalg.norm(np.eye(3) - m) < 100 else np.eye(3)


def _mean(x) -> float:
    """Order-independent mean (math.fsum is exactly rounded), so reversing a
    sequence cannot change a result in the last bit."""
    x = list(np.ravel(x))
    return math.fsum(x) / len(x) if x else float("nan")


def _centre(b) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])


def build_pair(frames: list[dict], sub_tid: int, obj_tid: int, begin: int, end: int,
               ecc_video: dict, width: float, height: float) -> Pair | None:
    """frames[t] maps tid -> (x1, y1, x2, y2). Returns None when fewer than two
    frames have both boxes. Steps are taken only between consecutive frames
    where the entity is annotated in both and a warp exists."""
    end = min(end, len(frames))
    diag = math.hypot(width, height)
    sub, obj = [], []
    for t in range(max(begin, 0), end):
        s, o = frames[t].get(sub_tid), frames[t].get(obj_tid)
        if s is not None and o is not None:
            sub.append(s)
            obj.append(o)
    if len(sub) < 2:
        return None

    def steps(tid):
        out = []
        for t in range(max(begin, 0) + 1, end):
            b0, b1 = frames[t - 1].get(tid), frames[t].get(tid)
            if b0 is None or b1 is None:
                continue
            w = _warp(ecc_video, t)
            if w is None:
                continue
            q = w @ np.array([*_centre(b0), 1.0])
            out.append((_centre(b1) - q[:2]) / diag)
        return np.asarray(out, float).reshape(-1, 2)

    return Pair(np.asarray(sub, float), np.asarray(obj, float),
                steps(sub_tid), steps(obj_tid), float(width), float(height))


def swap(p: Pair) -> Pair:
    """Subject and object exchanged."""
    return replace(p, sub=p.obj, obj=p.sub, sub_steps=p.obj_steps, obj_steps=p.sub_steps)


def time_reverse(p: Pair) -> Pair:
    """The same pair played backwards."""
    return replace(p, sub=p.sub[::-1], obj=p.obj[::-1],
                   sub_steps=-p.sub_steps[::-1], obj_steps=-p.obj_steps[::-1])


# ---------------------------------------------------------------------------
# Continuous scores (sign = the axis value; magnitude = confidence, for AUC)
# ---------------------------------------------------------------------------
def horizontal_score(p: Pair) -> float:
    """mean(subject centre x - object centre x) / frame width. < 0 = left."""
    return float(np.mean((p.sub[:, 0] + p.sub[:, 2]) - (p.obj[:, 0] + p.obj[:, 2])) / 2 / p.width)


def vertical_score(p: Pair) -> float:
    """mean(subject centre y - object centre y) / frame height. < 0 = above."""
    return float(np.mean((p.sub[:, 1] + p.sub[:, 3]) - (p.obj[:, 1] + p.obj[:, 3])) / 2 / p.height)


def depth_score(p: Pair) -> float:
    """mean(subject bottom y - object bottom y) / frame height. > 0 = front."""
    return float(np.mean(p.sub[:, 3] - p.obj[:, 3]) / p.height)


def _area(b: np.ndarray) -> np.ndarray:
    return np.maximum(b[:, 2] - b[:, 0], 1.0) * np.maximum(b[:, 3] - b[:, 1], 1.0)


def _diag(b: np.ndarray) -> np.ndarray:
    return np.hypot(np.maximum(b[:, 2] - b[:, 0], 1.0), np.maximum(b[:, 3] - b[:, 1], 1.0))


def larger_score(p: Pair) -> float:
    """mean log(subject area / object area). > 0 = larger."""
    return _mean(np.log(_area(p.sub)) - np.log(_area(p.obj)))


def taller_score(p: Pair) -> float:
    """mean log(subject height / object height). > 0 = taller."""
    hs = np.maximum(p.sub[:, 3] - p.sub[:, 1], 1.0)
    ho = np.maximum(p.obj[:, 3] - p.obj[:, 1], 1.0)
    return _mean(np.log(hs) - np.log(ho))


def speed(steps: np.ndarray) -> float:
    """Mean compensated per-frame displacement, as a fraction of the frame diagonal."""
    return _mean(np.linalg.norm(steps, axis=1)) if len(steps) else float("nan")


def subject_speed(p: Pair) -> float:
    return speed(p.sub_steps)


def object_speed(p: Pair) -> float:
    return speed(p.obj_steps)


def motion_rate(steps: np.ndarray, boxes: np.ndarray, width: float, height: float) -> float:
    """NET compensated displacement over the extent, in the entity's own box
    diagonals per frame. Replaces `speed` for moving/stationary decisions: box
    jitter cancels in a net displacement but accumulates in a mean step, and
    dividing by the box rather than the frame stops distant subjects looking
    stationary (stage 1 axis_check: subject_state FIT AUC 0.725 -> 0.812)."""
    if not len(steps):
        return float("nan")
    net = math.hypot(math.fsum(steps[:, 0]), math.fsum(steps[:, 1])) * math.hypot(width, height)
    return net / _mean(_diag(boxes)) / len(steps)


def subject_motion(p: Pair) -> float:
    return motion_rate(p.sub_steps, p.sub, p.width, p.height)


def object_motion(p: Pair) -> float:
    return motion_rate(p.obj_steps, p.obj, p.width, p.height)


def faster_score(p: Pair) -> float:
    """log(subject speed / object speed). > 0 = faster; nan without motion data."""
    a, b = subject_speed(p), object_speed(p)
    if not (a > 0 and b > 0):
        return float("nan")
    return math.log(a) - math.log(b)


def gap_score(p: Pair) -> float:
    """Mean Euclidean gap between the two rectangles (0 when they touch or
    overlap), divided by the mean of the two box diagonals."""
    dx = np.maximum(0.0, np.maximum(p.obj[:, 0] - p.sub[:, 2], p.sub[:, 0] - p.obj[:, 2]))
    dy = np.maximum(0.0, np.maximum(p.obj[:, 1] - p.sub[:, 3], p.sub[:, 1] - p.obj[:, 3]))
    return float(np.mean(np.hypot(dx, dy) / ((_diag(p.sub) + _diag(p.obj)) / 2)))


def distance_series(p: Pair) -> np.ndarray:
    """Per-frame centre distance divided by the mean of the two box diagonals.
    Within-frame, so camera translation cancels; box-scaled, so zoom cancels."""
    cs = np.stack([(p.sub[:, 0] + p.sub[:, 2]) / 2, (p.sub[:, 1] + p.sub[:, 3]) / 2], 1)
    co = np.stack([(p.obj[:, 0] + p.obj[:, 2]) / 2, (p.obj[:, 1] + p.obj[:, 3]) / 2], 1)
    return np.linalg.norm(cs - co, axis=1) / ((_diag(p.sub) + _diag(p.obj)) / 2)


def _thirds(d: np.ndarray):
    k = max(len(d) // 3, 1)
    return d[:k], d[k:len(d) - k], d[len(d) - k:]


def delta_d(p: Pair) -> float:
    """mean(last third of d) - mean(first third). < 0 = closing."""
    first, _, last = _thirds(distance_series(p))
    return _mean(last) - _mean(first)


def _direction_cos(p: Pair) -> float:
    a = np.array([math.fsum(p.sub_steps[:, 0]), math.fsum(p.sub_steps[:, 1])])
    b = np.array([math.fsum(p.obj_steps[:, 0]), math.fsum(p.obj_steps[:, 1])])
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


# ---------------------------------------------------------------------------
# Axis values
# ---------------------------------------------------------------------------
def _sign_value(score: float, neg: str, pos: str) -> str | None:
    if score != score or score == 0:
        return None
    return neg if score < 0 else pos


def horizontal(p: Pair) -> str | None:
    return _sign_value(horizontal_score(p), "left", "right")


def vertical(p: Pair) -> str | None:
    return _sign_value(vertical_score(p), "above", "beneath")


def depth(p: Pair) -> str | None:
    return _sign_value(depth_score(p), "behind", "front")


def contained(p: Pair) -> bool:
    """Frozen definition (1): in a majority of frames, the fraction of the
    subject box inside the object box exceeds the fraction of the object box
    inside the subject box, AND the subject's top edge is not above the object's."""
    ix = np.maximum(0.0, np.minimum(p.sub[:, 2], p.obj[:, 2]) - np.maximum(p.sub[:, 0], p.obj[:, 0]))
    iy = np.maximum(0.0, np.minimum(p.sub[:, 3], p.obj[:, 3]) - np.maximum(p.sub[:, 1], p.obj[:, 1]))
    inter = ix * iy
    frame = (inter / _area(p.sub) > inter / _area(p.obj)) & (p.sub[:, 1] >= p.obj[:, 1])
    return bool(frame.sum() > p.n / 2)


def proximity(p: Pair, gap_threshold: float) -> str:
    """contained, else adjacent (normalised gap below threshold), else separated.
    The schema's `overlapping` value is assigned to no predicate and is not emitted."""
    if contained(p):
        return "contained"
    return "adjacent" if gap_score(p) < gap_threshold else "separated"


def _state(v: float, threshold: float) -> str | None:
    if v != v:
        return None
    return "moving" if v > threshold else "stationary"


def subject_state(p: Pair, motion_threshold: float) -> str | None:
    return _state(subject_motion(p), motion_threshold)


def object_state(p: Pair, motion_threshold: float) -> str | None:
    return _state(object_motion(p), motion_threshold)


def relative_motion(p: Pair, dead_zone: float, motion_threshold: float,
                    _motion=(subject_motion, object_motion)) -> str:
    """pass: the middle third of d dips below BOTH end thirds by more than the
    dead zone (so the minimum is strictly inside the extent); approach / recede:
    the last third's mean differs from the first third's by more than the dead
    zone; co_move: both moving (motion_rate above threshold), net directions
    agree (cosine > 0), |change in d| within the dead zone; else none. Checked in
    that order. `_motion` exists only so axis_check.py can reproduce the
    as-specified (rejected) speed-based variant."""
    d = distance_series(p)
    if len(d) >= 3:
        first, mid, last = _thirds(d)
        if len(mid) and (_mean(first) - mid.min() > dead_zone) and (_mean(last) - mid.min() > dead_zone):
            return "pass"
    dd = delta_d(p)
    if dd < -dead_zone:
        return "approach"
    if dd > dead_zone:
        return "recede"
    if (_motion[0](p) > motion_threshold and _motion[1](p) > motion_threshold
            and _direction_cos(p) > 0):
        return "co_move"
    return "none"


def comparative(p: Pair) -> dict[str, bool | None]:
    """Each comparative as True (subject is larger/taller/faster), False (the
    object is), or None (tie or no motion data)."""
    out = {}
    for name, s in (("larger", larger_score(p)), ("taller", taller_score(p)),
                    ("faster", faster_score(p))):
        out[name] = None if (s != s or s == 0) else s > 0
    return out


# ---------------------------------------------------------------------------
def load_thresholds(path: str = THRESHOLDS_FILE) -> dict:
    """{'motion': ..., 'gap': ..., 'dead_zone': ...}, as frozen by axis_check.py."""
    with open(path) as f:
        return json.load(f)["thresholds"]
