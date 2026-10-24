"""Read-only probe: how often does a track end up with no scored detection, and
what class does it get?

It changes no behaviour. It wraps the two track-formatting functions in
`models.end2end_model` and `End2End_Model.forward`, only to count.

  training  `format_trajectories`        gen_labels.py:430-482
            a track none of whose lines has score > 0 gets class id 1
            (antelope, NOVEL) and score 0                       [lines 464-474]
  inference `format_trajectories_test`   gen_labels.py:586-660
            a track none of whose lines carries a positive score vector has
            `traj_scores[track] == []`, so f = 0, `max(sssss)/f` is 0.0/0 = nan
            (a RuntimeWarning, not an error) and `np.argmax(zeros)` = 0, i.e.
            class id 0 (airplane, a base class) with score NaN   [lines 638-649]

`python studies/rb1/scripts/fallback_probe.py` runs a synthetic demonstration of both
with the repo's real functions (CPU, no model). `fallback_count_gpu.py` uses
`install()` on real videos.
"""
from __future__ import annotations

import json
import math
import os
import sys
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))   # studies/rb1/scripts -> METOR (A25b)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

COUNTS = {}          # video -> {"train": {...}, "test": {...}}
_current = ["<none>"]
_installed = []


def _slot(mode):
    return COUNTS.setdefault(_current[0], {}).setdefault(
        mode, {"calls": 0, "tracks": 0, "unscored_tracks": 0, "unscored_survive_formatting": 0,
               "fallback_class": None})


def _unscored_ids(temp, mode):
    """Track ids none of whose lines is scored, by the same test the function uses."""
    seen, scored = set(), set()
    for line in temp:
        tid = line[1]
        seen.add(tid)
        if mode == "train":
            if line[7] > 0:
                scored.add(tid)
        else:
            if max(line[10:]) > 0:
                scored.add(tid)
    return seen, seen - scored


def _wrap(fn, mode):
    def wrapped(temp):
        seen, unscored = _unscored_ids(temp, mode)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")        # the nan division is the thing being measured
            out = fn(temp)
        s = _slot(mode)
        s["calls"] += 1
        s["tracks"] += len(seen)
        s["unscored_tracks"] += len(unscored)
        if mode == "train":
            surv = [t for t in out if t["score"] == 0]
        else:
            surv = [t for t in out if isinstance(t["score"], float) and math.isnan(t["score"])]
        s["unscored_survive_formatting"] += len(surv)
        if surv:
            s["fallback_class"] = sorted({t["category"] for t in surv})
        return out
    wrapped.__wrapped__ = fn
    return wrapped


def install():
    """Patch the names `models.end2end_model` calls, and record the video name."""
    import models.end2end_model as m
    if _installed:
        return
    m.format_trajectories = _wrap(m.format_trajectories, "train")
    m.format_trajectories_test = _wrap(m.format_trajectories_test, "test")
    orig_forward = m.End2End_Model.forward

    def forward(self, input, sort_model):
        _current[0] = input["video_name"][0]
        return orig_forward(self, input, sort_model)
    m.End2End_Model.forward = forward
    _installed.append((m, orig_forward))


def uninstall():
    while _installed:
        m, orig_forward = _installed.pop()
        m.format_trajectories = m.format_trajectories.__wrapped__
        m.format_trajectories_test = m.format_trajectories_test.__wrapped__
        m.End2End_Model.forward = orig_forward


def dump(path):
    with open(path, "w") as f:
        json.dump(COUNTS, f, indent=2)


# ------------------------------------------------------------ synthetic demo
def demo():
    """The repo's real functions on hand-made track lines: one scored track and
    one unscored track, in each mode. No model, no data beyond the split file."""
    import models.gen_labels as gl
    warnings.simplefilter("ignore")

    def train_lines(tid, scored):
        # (frame, track, x1, y1, x2, y2, category, score, _, _)
        return [[f, tid, 10, 10, 50, 50, 5 if scored else 0, 0.9 if scored else 0, 0, 0]
                for f in range(20)]

    train = train_lines(0, True) + train_lines(1, False)
    out_train = gl.format_trajectories(train)

    def test_lines(tid, scored):
        rows = []
        for f in range(20):
            r = np.zeros(45)
            r[:6] = [f, tid, 10, 10, 50, 50]
            if scored:
                r[10 + 5] = 0.8                       # class id 5 scored
            rows.append(r)
        return rows

    test = test_lines(0, True) + test_lines(1, False)
    out_test = gl.format_trajectories_test(test)
    res = {
        "train_format_trajectories": [{"track": t["tid"], "category": t["category"], "score": t["score"]} for t in out_train],
        "test_format_trajectories_test": [{"track": t["tid"], "category": t["category"],
                                           "score": ("nan" if math.isnan(t["score"]) else float(t["score"]))}
                                          for t in out_test],
    }
    split = json.load(open(__import__("utils.paths", fromlist=["x"]).OBJ_SPLIT_INFO))
    res["split_of_fallback_classes"] = {"train fallback (id 1)": [split["id2cls"]["1"], split["cls2split"][split["id2cls"]["1"]]],
                                        "test fallback (id 0)": [split["id2cls"]["0"], split["cls2split"][split["id2cls"]["0"]]]}
    return res


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2))
