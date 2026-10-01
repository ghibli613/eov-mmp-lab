#!/usr/bin/env python
"""Free partial check on motion blindness: does AP vary with ACTUAL motion?

    python pilot_analysis/scripts/displacement_check.py --preds <dir>

If the model is motion-blind, its accuracy on dynamic predicates should not
depend on how much the objects actually moved -- it would be reading geometry
either way. If accuracy rises with displacement, it is using motion.

Weaker than a frame-shuffling probe, because displacement correlates with other
things (long fast-moving objects are also easier to detect), but it costs
nothing and runs on the dumps already here.

Displacement is measured on the GROUND-TRUTH trajectories, so it is a property of
the instance rather than of the model's output.
"""
from __future__ import annotations
import argparse, json, math, os, sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from inference.video_relation_detection_openvoc import viou
from utils import paths

PART = "pilot_analysis/predicate_partition.json"


def centre(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def rel_displacement(r):
    """How much the subject moves RELATIVE to the object, normalised by box size.
    Relative, because a predicate like walk_left is about relative motion, and
    normalised so a large near object is not counted as moving more than a small
    far one."""
    s, o = r["sub_traj"], r["obj_traj"]
    n = min(len(s), len(o))
    if n < 2:
        return None
    d = []
    for i in range(n):
        sc, oc = centre(s[i]), centre(o[i])
        d.append((sc[0] - oc[0], sc[1] - oc[1]))
    scale = max(1.0, sum(max(b[2] - b[0], b[3] - b[1]) for b in s[:n]) / n)
    return math.hypot(d[-1][0] - d[0][0], d[-1][1] - d[0][1]) / scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="pilot_analysis/preds_from_pod/full")
    ap.add_argument("--split", default="all", choices=["all", "novel"])
    a = ap.parse_args()
    part = json.load(open(PART))
    gt = json.load(open(paths.TEST_RELATION_GT))
    segs = json.load(open(f"{a.preds}/segments_raw_{a.split}.json"))

    rows = []          # (displacement, is_dynamic, correct)
    for vid, rels in gt.items():
        pairs = segs.get(vid, [])
        for r in rels:
            p = r["triplet"][1]
            if a.split == "novel" and part[p]["ov_split"] != "novel":
                continue
            disp = rel_displacement(r)
            if disp is None:
                continue
            b, e = r["duration"]
            best, best_ov = None, 0.0
            for pr in pairs:
                if pr["sbj_cls"] != r["triplet"][0] or pr["obj_cls"] != r["triplet"][2]:
                    continue
                ov = min(viou(pr["sbj_traj"], pr["duration"], r["sub_traj"], r["duration"]),
                         viou(pr["obj_traj"], pr["duration"], r["obj_traj"], r["duration"]))
                if ov > best_ov:
                    best, best_ov = pr, ov
            if best is None or best_ov < 0.5:
                continue
            votes = Counter()
            for sg in best["segments"]:
                fs, fe = sg["frame_range"]
                if fe <= b or fs >= e or not sg["preds"]:
                    continue
                votes[sg["preds"][0][0]] += 1
            if not votes:
                continue
            rows.append((disp, part[p]["time"] == "dynamic",
                         votes.most_common(1)[0][0] == p))

    print(f"'{a.split}' pass: {len(rows)} GT instances with a matched pair and "
          f"measurable displacement\n")
    for name, want_dyn in (("DYNAMIC predicates", True), ("static predicates", False)):
        sub = [(d, ok) for d, dyn, ok in rows if dyn == want_dyn]
        if len(sub) < 20:
            print(f"{name}: only {len(sub)} instances, skipped"); continue
        sub.sort()
        q = len(sub) // 4
        print(f"{name}  (n={len(sub)}, top-1 accuracy by relative displacement)")
        print(f"  {'quartile':10} {'n':>5} {'median disp':>12} {'accuracy':>9}")
        for i, lab in enumerate(("Q1 least", "Q2", "Q3", "Q4 most")):
            chunk = sub[i * q:(i + 1) * q] if i < 3 else sub[3 * q:]
            acc = sum(1 for _, ok in chunk if ok) / len(chunk)
            med = chunk[len(chunk) // 2][0]
            print(f"  {lab:10} {len(chunk):5d} {med:12.3f} {acc:8.1%}")
        lo = sum(1 for _, ok in sub[:q] if ok) / q
        hi = sum(1 for _, ok in sub[3 * q:] if ok) / len(sub[3 * q:])
        print(f"  Q4 - Q1: {hi - lo:+.1%}"
              f"   -> {'accuracy RISES with motion' if hi - lo > 0.05 else 'FLAT in motion' if abs(hi-lo) <= 0.05 else 'accuracy FALLS with motion'}")
        print()
    print("Reading: motion blindness predicts FLAT for dynamic predicates -- the")
    print("model would be reading geometry regardless of how much things moved.")
    print("A rise means motion is being used. The static row is the control: it")
    print("should be flat under either hypothesis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
