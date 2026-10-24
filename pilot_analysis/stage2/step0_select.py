#!/usr/bin/env python3
"""Stage 2 Batch A, step 0 -- the inputs amendment A6 freezes. No training, no scoring.

    python3 pilot_analysis/stage2/step0_select.py

1. Verifies every frozen artefact against preregistration.md (stops on mismatch).
2. Counts, from TRAINING annotations (base only; novel dropped at parse):
   FIT support of every learned output, HELD instances per base predicate.
3. Picks the G2 leave-out set by the brief's fixed rule.
4. Draws the Batch-B shuffled-recipe permutation (novel predicates, same number
   of parts, seed 0, no fixed point).
Writes step0_selection.json; A6 quotes it.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import glob, json, os
from collections import Counter

import numpy as np

import common as C

N_LEAVE, MIN_LEARNED = 10, 4


def main():
    if os.path.exists(os.path.join(C.OUT, "step0_selection.json")):
        sys.exit("step0_selection.json exists and is pinned by A6; not overwriting it")
    frozen = C.verify_frozen()
    phi, split = C.load_phi()
    base = sorted(p for p in phi if split[p] == "base")
    novel = sorted(p for p in phi if split[p] == "novel")
    sp = C.splits()
    fit, held = set(sp["fit"]), set(sp["held"])

    fit_inst, held_inst, novel_dropped = Counter(), Counter(), 0
    for f in sorted(glob.glob(os.path.join(C.ANNO_TRAIN, "*.json"))):
        v, W, H, rels, n_nov, _ = C.load_train_video(f, set(base))
        novel_dropped += n_nov
        for r in rels:
            if v in fit:
                fit_inst[r["predicate"]] += 1
            elif v in held:
                held_inst[r["predicate"]] += 1
    assert novel_dropped == 3301, novel_dropped

    def support(removed=()):
        s = Counter()
        for p, n in fit_inst.items():
            if p in removed:
                continue
            for out in C.instance_values(p, phi):
                s[out] += n
        return s

    sup_all = support()
    rec0 = {p: C.effective_recipe(p, phi) for p in phi}

    # ---- G2 leave-out set: the brief's rule, applied greedily in HELD order
    eligible = [p for p in base if rec0[p] and held_inst[p] > 0]
    order = sorted(eligible, key=lambda p: (-held_inst[p], p))
    chosen, log = [], []
    for p in order:
        if len(chosen) == N_LEAVE:
            break
        if any(rec0[p] == rec0[q] for q in chosen):
            log.append((p, "shares a recipe with a chosen predicate"))
            continue
        trial = chosen + [p]
        s = support(set(trial))
        short = sorted({(q, f"{a}={v}", s[(a, v)]) for q in trial
                        for a, v in C.consulted_outputs(rec0[q]) if s[(a, v)] < C.SUPPORT_MIN})
        if short:
            log.append((p, f"support would fall below {C.SUPPORT_MIN}: {short}"))
            continue
        n_learned = sum(C.needs_learned_part(rec0[q]) for q in trial)
        if MIN_LEARNED - n_learned > N_LEAVE - len(trial):
            log.append((p, "no learned part, and the remaining places are needed for learned ones"))
            continue
        chosen.append(p)
        log.append((p, "chosen"))
    assert len(chosen) == N_LEAVE
    s_after = support(set(chosen))

    # ---- Batch-B control: shuffled recipes among novel predicates with the same number of parts
    rng = np.random.default_rng(0)
    groups = {}
    for p in novel:
        groups.setdefault(len(phi[p]), []).append(p)
    perm = {}
    for k in sorted(groups):
        g = sorted(groups[k])
        if len(g) == 1:
            perm[g[0]] = g[0]
            continue
        while True:
            pi = rng.permutation(len(g))
            if all(pi[i] != i for i in range(len(g))):
                break
        for i, p in enumerate(g):
            perm[p] = g[pi[i]]                       # p receives the recipe of perm[p]
    fixed = sorted(p for p in perm if perm[p] == p)

    out = dict(
        frozen_check=[dict(path=r[0], sha256=r[2]) for r in frozen],
        novel_instances_dropped_at_parse=novel_dropped,
        videos=dict(fit=len(sp["fit"]), held=len(sp["held"]), early_stop=len(sp["es"]),
                    fit_train=len(sp["fit_train"]), excluded_no_ecc=sp["no_ecc"]),
        early_stop_videos=sp["es"],
        fit_support=[dict(output=f"{a}={v}", fit_instances=sup_all[(a, v)], after_leave_out=s_after[(a, v)])
                     for a, v in C.OUTPUTS],
        held_instances={p: held_inst[p] for p in base},
        fit_instances={p: fit_inst[p] for p in base},
        leave_out=dict(chosen=chosen,
                       rows=[dict(predicate=p, held_instances=held_inst[p], fit_instances=fit_inst[p],
                                  recipe=[list(x) for x in rec0[p]], learned_part=C.needs_learned_part(rec0[p]))
                             for p in chosen],
                       n_learned=sum(C.needs_learned_part(rec0[p]) for p in chosen),
                       selection_log=log),
        shuffled_recipes=dict(seed=0, groups={str(k): sorted(v) for k, v in sorted(groups.items())},
                              receives_recipe_of=perm, fixed_points=fixed),
        recipes_before_g1={p: [list(x) for x in rec0[p]] for p in sorted(phi)},
        collisions_before_g1=C.collision_groups(rec0))
    path = os.path.join(C.OUT, "step0_selection.json")
    json.dump(out, open(path, "w"), indent=1)
    C.write_log("step0_select", dict(N_LEAVE=N_LEAVE, MIN_LEARNED=MIN_LEARNED, SUPPORT_MIN=C.SUPPORT_MIN,
                                     ES_SEED=C.ES_SEED, ES_FRAC=C.ES_FRAC, permutation_seed=0),
                [C.PHI, C.SPLIT, C.ECC_TRAIN, C.PRED_SPLIT_INFO, C.PREREG], [path])

    print("novel instances dropped at parse:", novel_dropped)
    print("videos:", out["videos"] | {"excluded_no_ecc": len(sp["no_ecc"])})
    print("\nFIT support per learned output (instances):")
    for r in out["fit_support"]:
        print(f"  {r['output']:28s} {r['fit_instances']:6d}  after leave-out {r['after_leave_out']:6d}")
    print("\nleave-out set:")
    for r in out["leave_out"]["rows"]:
        print(f"  {r['predicate']:16s} HELD {r['held_instances']:4d} FIT {r['fit_instances']:5d} "
              f"learned={r['learned_part']} recipe={[(':'.join(x for x in f if x)) for f in r['recipe']]}")
    print("  n with a learned part:", out["leave_out"]["n_learned"])
    print("\nskipped on the way:")
    for p, why in log:
        if why != "chosen":
            print(f"  {p}: {why}")
    print("\nshuffle groups:", {k: len(v) for k, v in out["shuffled_recipes"]["groups"].items()}, "fixed points:", fixed)
    print("collisions before G1:", out["collisions_before_g1"])


if __name__ == "__main__":
    main()
