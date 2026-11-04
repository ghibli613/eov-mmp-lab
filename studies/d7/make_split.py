#!/usr/bin/env python3
"""A31 §1: split the 160 HELD videos into HELD-A (80) and HELD-B (80).

    python studies/d7/make_split.py            # from the repo root; writes studies/d7/held_split.json

Flags: novel = the video's annotation lists a novel-class object (subject/objects, cls2split);
long = frame-folder count strictly above numpy.median of the 160. Strata in the fixed order
(novel, long), (novel, not long), (not novel, long), (not novel, not long); one generator
default_rng(20261102); per stratum sorted IDs then rng.permutation; concatenated; A, B, A, B...
along the whole list. The test-label guard is installed first.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "studies", "rpt"))
sys.path.insert(0, REPO)
import no_test_labels  # noqa: E402
no_test_labels.install()

import numpy as np  # noqa: E402

SPLIT = "pilot_analysis/stage1/train_split.json"
SPLIT_SHA = "095dd89ced7260945e76f020a68dcee5793bdd5bca3a6f375f7537dc7c267f1a"
CLS_SPLIT = "data/vidvrd/data/openvoc_obj_class_spilt_info.json"
SEED = 20261102
NO_SAM3 = ["ILSVRC2015_train_00280000", "ILSVRC2015_train_00225001", "ILSVRC2015_train_00971000"]
STRATA = [(True, True), (True, False), (False, True), (False, False)]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split(held, novel, frames, seed=SEED):
    """The A31 procedure on given flags and counts (pure; tested). -> dict."""
    median = float(np.median([frames[v] for v in held]))
    long_ = {v: frames[v] > median for v in held}
    rng = np.random.default_rng(seed)
    strata, order = [], []
    for nv, lg in STRATA:
        members = sorted(v for v in held if novel[v] == nv and long_[v] == lg)
        perm = [str(x) for x in rng.permutation(members)] if members else []
        strata.append(dict(novel=nv, long=lg, members_sorted=members, permuted=perm))
        order += perm
    half = {v: "A" if i % 2 == 0 else "B" for i, v in enumerate(order)}
    for s in strata:
        s["count"] = len(s["members_sorted"])
        s["A"] = sum(half[v] == "A" for v in s["permuted"])
        s["B"] = sum(half[v] == "B" for v in s["permuted"])
    return dict(median_frames=median, long=long_, strata=strata, order=order, half=half,
                held_a=sorted(v for v in held if half[v] == "A"), held_b=sorted(v for v in held if half[v] == "B"))


def main():
    os.chdir(REPO)
    from utils import paths
    if sha256(SPLIT) != SPLIT_SHA:
        raise SystemExit(f"!! STOP: {SPLIT} differs from its pin")
    held = sorted(json.load(open(SPLIT))["held"])
    assert len(held) == 160
    cls2split = json.load(open(CLS_SPLIT))["cls2split"]
    novel, novel_objects, frames = {}, {}, {}
    for v in held:
        anno = json.load(open(os.path.join(paths.ANNO_TRAIN_DIR, f"{v}.json")))
        objs = [o["category"] for o in anno["subject/objects"]]
        novel_objects[v] = sorted({c for c in objs if cls2split[c] == "novel"})
        novel[v] = bool(novel_objects[v])
        frames[v] = len(os.listdir(os.path.join(paths.FRAME_DIR, v)))
    s = split(held, novel, frames)
    assert len(s["held_a"]) == len(s["held_b"]) == 80 and not set(s["held_a"]) & set(s["held_b"])
    out = dict(
        label="A31 §1: the HELD split", seed=SEED,
        method=("strata (novel, long), (novel, not long), (not novel, long), (not novel, not long); "
                "per stratum sorted IDs then default_rng(20261102).permutation, one generator; "
                "concatenated; A, B, A, B... starting with A"),
        inputs={SPLIT: SPLIT_SHA, CLS_SPLIT: sha256(CLS_SPLIT)},
        median_frames=s["median_frames"], held_a=s["held_a"], held_b=s["held_b"],
        strata=[{k: st[k] for k in ("novel", "long", "count", "A", "B", "members_sorted", "permuted")} for st in s["strata"]],
        no_sam3_files={v: s["half"][v] for v in NO_SAM3},
        per_video={v: dict(half=s["half"][v], novel=novel[v], novel_objects=novel_objects[v], frames=frames[v],
                           long=s["long"][v]) for v in held})
    path = "studies/d7/held_split.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    digest = sha256(path)
    with open(path + ".sha256", "w") as f:
        f.write(f"{digest}  held_split.json\n")
    print(json.dumps(dict(median_frames=s["median_frames"], held_a=len(s["held_a"]), held_b=len(s["held_b"]),
                          strata=[(st["novel"], st["long"], st["count"], st["A"], st["B"]) for st in s["strata"]],
                          no_sam3_files=out["no_sam3_files"], sha256=digest), indent=1))


if __name__ == "__main__":
    main()
