#!/usr/bin/env python3
"""Stage 2 Batch A2, step 1 -- coverage c(p) and the choice of alpha (A7).

    python pilot_analysis/stage2/a2_step1_alpha.py [--small]

c(p): mean recipe score over all FIT slots (labelled or not), with the LO2 learned
part. alpha from {0, .25, .5, .75, 1}: recipe-only r' = r / c^alpha on the 63 FIT
early-stop videos, all base predicates except LO2 competing, through the repo's
post-processing and evaluator; highest mAP wins, ties (2 decimals) to the smaller.
--small: the plumbing run on the first 5 early-stop videos (es_*_small.json).

Writes data/a2_coverage_lo2.json and data/a2_alpha.json (with --small: *_small.json).
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, json, os, time

import numpy as np

import common as C
import evaluation as EV

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def recipes_after_g1(phi, data):
    dropped = {tuple(x) for x in json.load(open(os.path.join(data, "g1.json")))["dropped"]}
    return {p: C.effective_recipe(p, phi, dropped) for p in phi}, dropped


def raw_recipes(d, rows, P, rec, names, dropped):
    L = P[rows].astype(np.float64).copy()
    for ax, v in dropped:
        L[:, C.OUT_IDX[(ax, v)]] = np.nan
    return C.recipe_matrix(rec, names, d["meas"][rows], d["vehicle"][rows], L, fill=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    C.verify_frozen()
    data = os.path.join(C.OUT, "data")
    sfx = "_small" if a.small else ""
    phi, split = C.load_phi()
    names = EV.names()
    lo2 = json.load(open(os.path.join(C.OUT, "a2_step0_selection.json")))["chosen"]
    compete = sorted(p for p in names if split[p] == "base" and p not in lo2)
    assert len(compete) == 61
    rec, dropped = recipes_after_g1(phi, data)
    d = dict(np.load(os.path.join(data, "slots.npz")))
    P = np.load(os.path.join(data, "learned_lo2.npy"))
    fit_rows = np.where(d["subset"] != 2)[0]
    c = C.coverage(raw_recipes(d, fit_rows, P, rec, names, dropped))
    es_pairs = json.load(open(os.path.join(data, f"es_pairs{sfx}.json")))
    es_gt = os.path.join(data, f"es_gt{sfx}.json")
    es_rows = np.concatenate([np.arange(*p["rows"]) for ps in es_pairs.values() for p in ps])
    assert (d["subset"][es_rows] == 1).all()
    R_es = raw_recipes(d, es_rows, P, rec, names, dropped)
    cols = [names.index(p) for p in compete]
    sp_path = EV.split_file(os.path.join(data, "split_a2_alpha.json"), set(compete), "compete")
    grid = []
    for al in ALPHAS:
        S = np.zeros((len(d["subset"]), 132))
        S[es_rows] = C.corrected(R_es, c, al)
        r = EV.evaluate(EV.predict(es_pairs, EV.only(S, cols)), es_gt, "compete", sp_path)
        grid.append(dict(alpha=al, **r))
        print(f"  alpha {al:4.2f}: mAP {r['mAP']:6.2f}  R@50 {r['R50']:6.2f}  R@100 {r['R100']:6.2f}", flush=True)
    best = max(round(g["mAP"], 2) for g in grid)
    chosen = min(g["alpha"] for g in grid if round(g["mAP"], 2) == best)
    cov = {p: (None if np.isnan(c[j]) else float(c[j])) for j, p in enumerate(names)}
    paths = [os.path.join(data, f"a2_coverage_lo2{sfx}.json"), os.path.join(data, f"a2_alpha{sfx}.json")]
    json.dump(dict(learned_part="lo2", fit_slots=len(fit_rows), coverage=cov), open(paths[0], "w"), indent=1)
    json.dump(dict(alphas=ALPHAS, competing=compete, early_stop_videos=len(es_pairs), early_stop_slots=len(es_rows),
                   grid=grid, chosen_alpha=chosen, tie_rule="highest mAP; equal to 2 decimals -> smaller alpha"),
              open(paths[1], "w"), indent=1)
    C.write_log("a2_step1_alpha" + sfx, dict(alphas=ALPHAS, post=C.POST),
                [os.path.join(data, f) for f in ("slots.npz", "learned_lo2.npy", "g1.json", f"es_pairs{sfx}.json",
                                                 f"es_gt{sfx}.json")] + [os.path.join(C.OUT, "a2_step0_selection.json")],
                paths + [sp_path], extra=dict(seconds=round(time.time() - t0, 1)))
    top = sorted(((v, p) for p, v in cov.items() if v is not None and split[p] == "base"), reverse=True)
    print("highest coverage (base):", [(p, round(v, 3)) for v, p in top[:6]])
    print("lowest coverage (base):", [(p, round(v, 3)) for v, p in top[-4:]])
    print(f"chosen alpha: {chosen} | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
