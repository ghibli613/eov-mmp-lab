#!/usr/bin/env python3
"""Stage 2 Batch A2, step 2 -- gate G2 on LO2 with the coverage-corrected recipe
score, and the secondary results (A7).

    python pilot_analysis/stage2/a2_step2_g2.py [--small]

Primary: r' = r / c^alpha (alpha from a2_alpha.json, c from the LO2 learned part
over FIT slots), HELD, PredCls on GT tracks, only LO2 competes. Random U(0,1),
mean of seeds 0-19. PASS iff r' mAP >= 3 x random and >= random + 5 (A6's thresholds).
Secondary: LO2 at alpha = 0; the A6 set with A6's leave-out learned part at
alpha = 0 and at the chosen alpha ("seen during diagnosis, not a test"); per LO2
predicate, how often it tops the other nine on its own labelled HELD slots.
--small: plumbing run on the first 5 HELD videos.

Writes data/a2_g2.json, a2_recipe_lo2_held.npy (r, unfilled), a2_rprime_lo2_held.npy,
a2_coverage_a6leaveout.json (with --small: *_small).
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, json, os, time

import numpy as np

import common as C
import evaluation as EV
from a2_step1_alpha import raw_recipes, recipes_after_g1


def top1(Rs, rows_labels, names, group):
    """share of p's own labelled slots where p's score >= the max of the rest of `group`."""
    cols = [names.index(p) for p in group]
    out = {}
    for j, p in zip(cols, group):
        own = [i for i, l in rows_labels if p in l]
        oth = [c for c in cols if c != j]
        out[p] = dict(n=len(own), share=float(np.mean([Rs[i, j] >= Rs[i, oth].max() for i in own])) if own else None)
    return out


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
    a6 = json.load(open(os.path.join(C.OUT, "step0_selection.json")))["leave_out"]["chosen"]
    alpha = json.load(open(os.path.join(data, "a2_alpha.json")))["chosen_alpha"]
    cov = json.load(open(os.path.join(data, "a2_coverage_lo2.json")))["coverage"]
    c_lo2 = np.array([np.nan if cov[p] is None else cov[p] for p in names])
    rec, dropped = recipes_after_g1(phi, data)
    d = dict(np.load(os.path.join(data, "slots.npz")))
    meta = json.load(open(os.path.join(data, "slots_meta.json")))
    held_pairs = json.load(open(os.path.join(data, "held_pairs.json")))
    gt_path = os.path.join(data, "held_gt.json")
    if a.small:
        keep = sorted(held_pairs)[:5]
        held_pairs = {v: held_pairs[v] for v in keep}
        gt = json.load(open(gt_path))
        gt_path = os.path.join(data, "held_gt_small5.json")
        json.dump({v: gt[v] for v in keep}, open(gt_path, "w"))
    H = np.concatenate([np.arange(*p["rows"]) for ps in held_pairs.values() for p in ps])
    assert (d["subset"][H] == 2).all()
    N = len(d["subset"])
    fit_rows = np.where(d["subset"] != 2)[0]

    def full(M):
        S = np.zeros((N, 132))
        S[H] = M
        return S

    def run(S, group, split_path):
        cols = [names.index(p) for p in group]
        return EV.evaluate(EV.predict(held_pairs, EV.only(S, cols)), gt_path, "compete", split_path)

    def random_baseline(group, split_path):
        rs = []
        for s in range(C.G2_RANDOM_SEEDS):
            U = np.zeros((N, 132))
            U[H] = np.random.default_rng(s).uniform(size=(len(H), 132))
            rs.append(run(U, group, split_path))
        return dict(mean={k: float(np.mean([r[k] for r in rs])) for k in ("mAP", "R50", "R100")},
                    sd_mAP=float(np.std([r["mAP"] for r in rs])), per_seed_mAP=[r["mAP"] for r in rs])

    # ---- LO2 (the test)
    P_lo2 = np.load(os.path.join(data, "learned_lo2.npy"))
    R_lo2 = raw_recipes(d, H, P_lo2, rec, names, dropped)
    sp_lo2 = EV.split_file(os.path.join(data, f"split_lo2{sfx}.json"), set(lo2), "compete")
    Rp = C.corrected(R_lo2, c_lo2, alpha)
    R0 = C.corrected(R_lo2, c_lo2, 0.0)
    prim = run(full(Rp), lo2, sp_lo2)
    rnd = random_baseline(lo2, sp_lo2)
    x, rm = prim["mAP"], rnd["mean"]["mAP"]
    ok_ratio, ok_margin = x >= C.G2_RATIO * rm, x >= rm + C.G2_MARGIN
    verdict = "PASS" if (ok_ratio and ok_margin) else "FAIL"
    sec = dict(lo2_alpha0=run(full(R0), lo2, sp_lo2))

    # ---- A6 set, seen during diagnosis (A6's leave-out learned part, its own coverage)
    P_a6 = np.load(os.path.join(data, "learned_leaveout.npy"))
    c_a6 = C.coverage(raw_recipes(d, fit_rows, P_a6, rec, names, dropped))
    R_a6 = raw_recipes(d, H, P_a6, rec, names, dropped)
    sp_a6 = EV.split_file(os.path.join(data, f"split_a6set{sfx}.json"), set(a6), "compete")
    sec["a6set_alpha0"] = run(full(C.corrected(R_a6, c_a6, 0.0)), a6, sp_a6)
    sec["a6set_chosen_alpha"] = run(full(C.corrected(R_a6, c_a6, alpha)), a6, sp_a6)
    sec["a6set_note"] = "seen during diagnosis, not a test"

    # ---- top-1 among LO2 on own labelled HELD slots
    labs = [(k, set(meta["labels"][i])) for k, i in enumerate(H)]
    t1 = dict(alpha0=top1(R0, labs, names, lo2), chosen=top1(Rp, labs, names, lo2))

    out = dict(lo2=lo2, alpha=alpha, coverage_lo2={p: cov[p] for p in lo2},
               recipes={p: [list(f) for f in rec[p]] for p in lo2},
               recipes_distinct=len({rec[p] for p in lo2}) == len(lo2),
               n_gt=sum(1 for g in json.load(open(gt_path)).values() for r in g if r["triplet"][1] in lo2),
               primary=prim, random=rnd,
               test=dict(ratio=dict(need=f">= {C.G2_RATIO} x random", value=x / rm if rm else None, ok=bool(ok_ratio)),
                         margin=dict(need=f">= random + {C.G2_MARGIN}", value=x - rm, ok=bool(ok_margin))),
               verdict=verdict, secondary=sec, top1_own_slots=t1,
               coverage_a6set={p: float(c_a6[names.index(p)]) for p in a6})
    paths = [os.path.join(data, f"a2_g2{sfx}.json"), os.path.join(data, f"a2_recipe_lo2_held{sfx}.npy"),
             os.path.join(data, f"a2_rprime_lo2_held{sfx}.npy"), os.path.join(data, f"a2_coverage_a6leaveout{sfx}.json")]
    json.dump(out, open(paths[0], "w"), indent=1)
    np.save(paths[1], R_lo2.astype(np.float32))
    np.save(paths[2], Rp.astype(np.float32))
    json.dump({p: (None if np.isnan(c_a6[j]) else float(c_a6[j])) for j, p in enumerate(names)}, open(paths[3], "w"), indent=1)
    C.write_log("a2_step2_g2" + sfx, dict(alpha=alpha, random_seeds=C.G2_RANDOM_SEEDS, ratio=C.G2_RATIO,
                                          margin=C.G2_MARGIN, post=C.POST),
                [os.path.join(data, f) for f in ("slots.npz", "slots_meta.json", "held_pairs.json", "held_gt.json",
                                                 "learned_lo2.npy", "learned_leaveout.npy", "a2_alpha.json",
                                                 "a2_coverage_lo2.json", "g1.json")],
                paths + [sp_lo2, sp_a6], extra=dict(seconds=round(time.time() - t0, 1)))
    print(f"LO2, alpha {alpha}: r' mAP {x:.2f} (R@50 {prim['R50']:.2f}, R@100 {prim['R100']:.2f}) | random {rm:.2f} "
          f"(sd {rnd['sd_mAP']:.2f}) | ratio {x / max(rm, 1e-9):.2f}, margin {x - rm:+.2f} -> {verdict}")
    print(f"LO2 alpha 0: {sec['lo2_alpha0']['mAP']:.2f} | A6 set alpha 0: {sec['a6set_alpha0']['mAP']:.2f}, "
          f"alpha {alpha}: {sec['a6set_chosen_alpha']['mAP']:.2f}")
    for p in lo2:
        b, c_ = t1["alpha0"][p], t1["chosen"][p]
        print(f"  {p:14s} n={b['n']:4d} top-1 own slots: alpha 0 {b['share'] if b['share'] is None else round(100 * b['share'], 1)}% "
              f"-> {c_['share'] if c_['share'] is None else round(100 * c_['share'], 1)}%  (c = {cov[p]:.3f})")


if __name__ == "__main__":
    main()
