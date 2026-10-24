#!/usr/bin/env python3
"""Stage 2 Batch A, step 3 -- recipe scores for all 132 predicates on every HELD
slot, and gate G2 (A6).

    python pilot_analysis/stage2/step3_g2.py [--data DIR]

Recipes are the effective ones after G1 (g1.json drops). G2: the leave-out
learned part, HELD, PredCls on GT tracks, only the 10 leave-out predicates
compete; recipe-only vs random U(0,1) (mean of 20 seeds). PASS iff recipe-only
mAP >= 3 x random and >= random + 5. EOV-MMP alone is an upper reference; the
full learned part's recipe-only mAP is a non-zero-shot reference.

Writes g2.json, recipe_{full,leaveout}_held.npy, recipes_after_g1.json, split_leaveout.json.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, json, os, time

import numpy as np

import common as C
import evaluation as EV


def learned_matrix(P, dropped):
    L = P.astype(np.float64).copy()
    for ax, v in dropped:
        L[:, C.OUT_IDX[(ax, v)]] = np.nan
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(C.OUT, "data"))
    a = ap.parse_args()
    t0 = time.time()
    C.verify_frozen()
    phi, split = C.load_phi()
    sel = json.load(open(os.path.join(C.OUT, "step0_selection.json")))
    leave = sel["leave_out"]["chosen"]
    g1 = json.load(open(os.path.join(a.data, "g1.json")))
    dropped = {tuple(x) for x in g1["dropped"]}
    names = EV.names()
    assert sorted(names) == sorted(phi)
    d = dict(np.load(os.path.join(a.data, "slots.npz")))
    held_pairs = json.load(open(os.path.join(a.data, "held_pairs.json")))
    gt_path = os.path.join(a.data, "held_gt.json")
    H = np.where(d["subset"] == 2)[0]

    rec = {p: C.effective_recipe(p, phi, dropped) for p in names}
    rec0 = {p: C.effective_recipe(p, phi) for p in names}
    coll0, coll1 = C.collision_groups(rec0), C.collision_groups(rec)
    new_coll = [g for g in coll1 if g not in coll0]
    no_recipe = sorted(p for p in names if not rec[p])
    json.dump(dict(dropped=sorted(map(list, dropped)), recipes={p: [list(x) for x in rec[p]] for p in names},
                   no_recipe=no_recipe, collisions_before_g1=coll0, collisions_after_g1=coll1,
                   new_collisions=new_coll), open(os.path.join(a.data, "recipes_after_g1.json"), "w"), indent=1)

    R = {}
    for ver in ("full", "leaveout"):
        P = np.load(os.path.join(a.data, f"learned_{ver}.npy"))
        r = C.recipe_matrix(rec, names, d["meas"][H], d["vehicle"][H], learned_matrix(P[H], dropped))
        assert r.shape == (len(H), 132) and not np.isnan(r).any()
        full = np.zeros((len(d["subset"]), 132))
        full[H] = r
        R[ver] = full
        np.save(os.path.join(a.data, f"recipe_{ver}_held.npy"), r.astype(np.float32))
    print(f"recipe scores: {len(H)} HELD slots x 132 | {time.time() - t0:.0f}s", flush=True)

    cols = [names.index(p) for p in leave]
    sp_path = EV.split_file(os.path.join(a.data, "split_leaveout.json"), set(leave), "leaveout")
    ev = lambda S_: EV.evaluate(EV.predict(held_pairs, EV.only(S_, cols)), gt_path, "leaveout", sp_path)
    res = dict(recipe_leaveout=ev(R["leaveout"]), recipe_full_reference=ev(R["full"]), eov_upper_reference=ev(EV.eov(d)))
    rand = []
    for s in range(C.G2_RANDOM_SEEDS):
        U = np.zeros((len(d["subset"]), 132))
        U[H] = np.random.default_rng(s).uniform(size=(len(H), 132))
        rand.append(ev(U))
        print(f"  random seed {s}: mAP {rand[-1]['mAP']:.2f}", flush=True)
    rm = {k: float(np.mean([r[k] for r in rand])) for k in ("mAP", "R50", "R100")}
    res["random_mean"] = rm
    res["random_sd_mAP"] = float(np.std([r["mAP"] for r in rand]))
    res["random_per_seed_mAP"] = [r["mAP"] for r in rand]
    x = res["recipe_leaveout"]["mAP"]
    ok_ratio, ok_margin = x >= C.G2_RATIO * rm["mAP"], x >= rm["mAP"] + C.G2_MARGIN
    verdict = "PASS" if (ok_ratio and ok_margin) else "FAIL"
    g2 = dict(leave_out=leave, recipes={p: [list(f) for f in rec[p]] for p in leave},
              recipes_distinct=len({rec[p] for p in leave}) == len(leave),
              n_gt=sum(1 for v in json.load(open(gt_path)).values() for r in v if r["triplet"][1] in leave),
              results=res, test=dict(ratio=dict(need=f">= {C.G2_RATIO} x random", value=x / rm["mAP"] if rm["mAP"] else None,
                                                ok=bool(ok_ratio)),
                                     margin=dict(need=f">= random + {C.G2_MARGIN}", value=x - rm["mAP"], ok=bool(ok_margin))),
              verdict=verdict)
    out = os.path.join(a.data, "g2.json")
    json.dump(g2, open(out, "w"), indent=1)
    C.write_log("step3_g2", dict(random_seeds=C.G2_RANDOM_SEEDS, ratio=C.G2_RATIO, margin=C.G2_MARGIN, post=C.POST),
                [os.path.join(a.data, f) for f in ("slots.npz", "held_pairs.json", "held_gt.json", "g1.json",
                                                   "learned_full.npy", "learned_leaveout.npy")],
                [out, sp_path, os.path.join(a.data, "recipes_after_g1.json"),
                 os.path.join(a.data, "recipe_full_held.npy"), os.path.join(a.data, "recipe_leaveout_held.npy")],
                extra=dict(seconds=round(time.time() - t0, 1)))
    print(json.dumps({k: v for k, v in res.items() if k != "random_per_seed_mAP"}, indent=1))
    print("new collisions after G1:", new_coll, "| no recipe:", no_recipe)
    print(f"G2: recipe-only {x:.2f} vs random {rm['mAP']:.2f} (ratio {x / max(rm['mAP'], 1e-9):.2f}, "
          f"margin {x - rm['mAP']:+.2f}) -> {verdict}")


if __name__ == "__main__":
    main()
