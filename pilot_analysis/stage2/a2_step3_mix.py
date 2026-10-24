#!/usr/bin/env python3
"""Stage 2 Batch A2, step 3 -- descriptive only (A7). Runs only if G2-LO2 passed.

    python pilot_analysis/stage2/a2_step3_mix.py [--small]

The full learned part (all base labels), c(p) recomputed with it over FIT slots,
the chosen alpha. HELD, base predicates compete (the repo's split file, target
"base"): EOV-MMP alone, r' alone, and EOV^(1-w) * r'^w at w = 0.25, 0.5, 0.75.
mAP, R@50, R@100. Nothing is chosen from these numbers.
--small: plumbing run on the first 5 HELD videos.

Writes data/a2_step3.json, a2_coverage_full.json, a2_rprime_full_held.npy (with --small: *_small).
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, json, os, time

import numpy as np

import common as C
import evaluation as EV
from a2_step1_alpha import raw_recipes, recipes_after_g1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    data = os.path.join(C.OUT, "data")
    sfx = "_small" if a.small else ""
    if json.load(open(os.path.join(data, "a2_g2.json")))["verdict"] != "PASS":
        sys.exit("G2-LO2 did not pass: step 3 is not run (A7).")
    C.verify_frozen()
    phi, split = C.load_phi()
    names = EV.names()
    base_cols = [i for i, p in enumerate(names) if split[p] == "base"]
    alpha = json.load(open(os.path.join(data, "a2_alpha.json")))["chosen_alpha"]
    rec, dropped = recipes_after_g1(phi, data)
    d = dict(np.load(os.path.join(data, "slots.npz")))
    held_pairs = json.load(open(os.path.join(data, "held_pairs.json")))
    gt_path = os.path.join(data, "held_gt.json")
    if a.small:
        keep = sorted(held_pairs)[:5]
        held_pairs = {v: held_pairs[v] for v in keep}
        gt_path = os.path.join(data, "held_gt_small5.json")
    H = np.concatenate([np.arange(*p["rows"]) for ps in held_pairs.values() for p in ps])
    N = len(d["subset"])
    P = np.load(os.path.join(data, "learned_full.npy"))
    c = C.coverage(raw_recipes(d, np.where(d["subset"] != 2)[0], P, rec, names, dropped))
    Rp = C.corrected(raw_recipes(d, H, P, rec, names, dropped), c, alpha)
    R = np.zeros((N, 132))
    R[H] = Rp
    E = EV.eov(d)
    rows = {}
    for label, S_ in [("EOV-MMP alone", E), ("r' alone", R)] + \
                     [(f"mix w={w}", E ** (1 - w) * R ** w) for w in C.MIX_WEIGHTS]:
        rows[label] = EV.evaluate(EV.predict(held_pairs, EV.only(S_, base_cols)), gt_path, "base")
        print(f"  {label:14s} mAP {rows[label]['mAP']:6.2f}  R@50 {rows[label]['R50']:6.2f}  R@100 {rows[label]['R100']:6.2f}",
              flush=True)
    paths = [os.path.join(data, f"a2_step3{sfx}.json"), os.path.join(data, f"a2_coverage_full{sfx}.json"),
             os.path.join(data, f"a2_rprime_full_held{sfx}.npy")]
    json.dump(dict(split="base (HELD)", alpha=alpha, learned_part="full", weights_on_recipe=C.MIX_WEIGHTS,
                   fixed_mix=0.5, rows=rows), open(paths[0], "w"), indent=1)
    json.dump({p: (None if np.isnan(c[j]) else float(c[j])) for j, p in enumerate(names)}, open(paths[1], "w"), indent=1)
    np.save(paths[2], Rp.astype(np.float32))
    C.write_log("a2_step3_mix" + sfx, dict(alpha=alpha, weights=C.MIX_WEIGHTS, post=C.POST),
                [os.path.join(data, f) for f in ("slots.npz", "held_pairs.json", "held_gt.json", "learned_full.npy",
                                                 "a2_alpha.json", "a2_g2.json", "g1.json")],
                paths, extra=dict(seconds=round(time.time() - t0, 1)))


if __name__ == "__main__":
    main()
