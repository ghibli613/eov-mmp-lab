#!/usr/bin/env python3
"""Stage 2 Batch A, step 4 -- descriptive only (A6). Runs only if G2 passed.

    python pilot_analysis/stage2/step4_mix.py [--data DIR]

HELD, base predicates compete (the repo's own split file, target split "base"):
EOV-MMP alone, recipe alone (full learned part), and the mix
EOV^(1-w) * recipe^w at w = 0.25, 0.5, 0.75. mAP, R@50, R@100. Nothing is
chosen from these numbers.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, json, os, time

import numpy as np

import common as C
import evaluation as EV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(C.OUT, "data"))
    ap.add_argument("--force-small", action="store_true", help="plumbing run on the small data only")
    a = ap.parse_args()
    t0 = time.time()
    g2 = json.load(open(os.path.join(a.data, "g2.json")))
    if g2["verdict"] != "PASS" and not a.force_small:
        sys.exit("G2 did not pass: step 4 is not run (A6).")
    C.verify_frozen()
    phi, split = C.load_phi()
    names = EV.names()
    base_cols = [i for i, p in enumerate(names) if split[p] == "base"]
    d = dict(np.load(os.path.join(a.data, "slots.npz")))
    held_pairs = json.load(open(os.path.join(a.data, "held_pairs.json")))
    gt_path = os.path.join(a.data, "held_gt.json")
    H = np.where(d["subset"] == 2)[0]
    Rh = np.load(os.path.join(a.data, "recipe_full_held.npy")).astype(np.float64)
    R = np.zeros((len(d["subset"]), 132))
    R[H] = Rh
    E = EV.eov(d)
    rows = {}
    for label, S_ in [("EOV-MMP alone", E), ("recipe alone", R)] + \
                     [(f"mix w={w}", E ** (1 - w) * R ** w) for w in C.MIX_WEIGHTS]:
        rows[label] = EV.evaluate(EV.predict(held_pairs, EV.only(S_, base_cols)), gt_path, "base")
        print(f"  {label:14s} mAP {rows[label]['mAP']:6.2f}  R@50 {rows[label]['R50']:6.2f}  R@100 {rows[label]['R100']:6.2f}",
              flush=True)
    out = os.path.join(a.data, "step4.json")
    json.dump(dict(split="base (HELD)", weights_on_recipe=C.MIX_WEIGHTS, fixed_mix=0.5, rows=rows), open(out, "w"), indent=1)
    C.write_log("step4_mix", dict(weights=C.MIX_WEIGHTS, post=C.POST),
                [os.path.join(a.data, f) for f in ("slots.npz", "held_pairs.json", "held_gt.json", "recipe_full_held.npy")],
                [out], extra=dict(seconds=round(time.time() - t0, 1)))


if __name__ == "__main__":
    main()
