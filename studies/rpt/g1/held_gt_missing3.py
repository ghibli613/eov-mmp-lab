#!/usr/bin/env python3
"""RPT G1 (A29, Definitions item 4): the base GT relations of the 3 HELD videos that stage 2 left out
of held_gt.json (no ECC), built by stage 2's own construction (step1_parts.py:40-53). CPU only.

The construction is first re-run on the 157 videos already in held_gt.json and must reproduce it
exactly; then it is applied to the 3. Training annotations only, base relations only; no test file
is opened (the test-label guard is installed first).

    python studies/rpt/g1/held_gt_missing3.py      # writes studies/rpt/g1/held_gt_missing3.json(.sha256)
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RPT = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(RPT, "..", ".."))
sys.path.insert(0, RPT)
sys.path.insert(0, os.path.join(REPO, "pilot_analysis", "stage2"))

import no_test_labels  # noqa: E402
no_test_labels.install()

import common as C                      # noqa: E402  (stage 2)
from step1_parts import gt_relation_insts  # noqa: E402  (stage 2's construction, lines 40-53)

HELD_GT = os.path.join(REPO, "pilot_analysis", "stage2", "data", "held_gt.json")
HELD_GT_SHA256 = "78d55c69f9aa2473f4584ae998592dcb67d95ce02a67ab0cd9a8bd8446184dee"
MISSING = ["ILSVRC2015_train_00265063", "ILSVRC2015_train_00679000", "ILSVRC2015_val_00123004"]
OUT = os.path.join(HERE, "held_gt_missing3.json")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def build(video, base):
    _, _, _, rels, _, d = C.load_train_video(os.path.join(C.ANNO_TRAIN, f"{video}.json"), base)
    return gt_relation_insts(d, rels)


def main():
    assert sha256(HELD_GT) == HELD_GT_SHA256
    held = json.load(open(C.SPLIT))["held"]
    gt = json.load(open(HELD_GT))
    assert sorted(set(held) - set(gt)) == MISSING and set(gt) <= set(held), "HELD vs held_gt.json"
    _, split = C.load_phi()
    base = {p for p in split if split[p] == "base"}

    for v in sorted(gt):                                  # the construction reproduces stage 2's file
        assert json.loads(json.dumps(build(v, base))) == gt[v], v
    out = {v: build(v, base) for v in MISSING}
    assert all(r["triplet"][1] in base for rs in out.values() for r in rs)
    json.dump(out, open(OUT, "w"))
    h = sha256(OUT)
    open(OUT + ".sha256", "w").write(f"{h}  {os.path.basename(OUT)}\n")
    print(f"reproduced held_gt.json on its {len(gt)} videos; "
          + ", ".join(f"{v}: {len(out[v])}" for v in MISSING)
          + f" base relations; total {sum(map(len, out.values()))}; sha256 {h}")


if __name__ == "__main__":
    main()
