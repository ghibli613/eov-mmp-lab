#!/usr/bin/env python3
"""Stage 2 Batch A2, step 0 -- the second leave-out set (LO2) that amendment A7
freezes. No training, no scoring.

    python3 pilot_analysis/stage2/a2_step0_select.py

A6's eligibility rule unchanged (non-empty recipe, no shared recipe, every
consulted learned output keeps >= 50 FIT instances after removal, at least 4
with a learned part), plus: A6's 10 excluded; at most 3 per family; the 10 with
the most HELD instances. Family = the verb of a verb + spatial predicate
(stand_, walk_, ...); bare spatial terms and comparatives form one family
("no verb"); any other predicate (chase, ride, watch, ...) is its own family.
Counts from TRAINING annotations, base labels only. Writes a2_step0_selection.json.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import glob, json, os
from collections import Counter

import common as C

N_LEAVE, MIN_LEARNED, PER_FAMILY = 10, 4, 3
COMPARATIVES = {"larger", "taller", "faster"}
OUT = os.path.join(C.OUT, "a2_step0_selection.json")


def family(p):
    t = C.S0.tokenise(p)
    if len(t) == 2 and t[0] in C.S0.VERB:
        return t[0]
    if (len(t) == 1 and t[0] in C.S0.SPATIAL) or p in COMPARATIVES:
        return "no verb"
    return p


def main():
    if os.path.exists(OUT):
        sys.exit(f"{os.path.basename(OUT)} exists and is pinned by A7; not overwriting it")
    C.verify_frozen()
    phi, split = C.load_phi()
    base = sorted(p for p in phi if split[p] == "base")
    a6 = json.load(open(os.path.join(C.OUT, "step0_selection.json")))["leave_out"]["chosen"]
    sp = C.splits()
    fit, held = set(sp["fit"]), set(sp["held"])
    fit_inst, held_inst, nov = Counter(), Counter(), 0
    for f in sorted(glob.glob(os.path.join(C.ANNO_TRAIN, "*.json"))):
        v, _, _, rels, n_nov, _ = C.load_train_video(f, set(base))
        nov += n_nov
        for r in rels:
            (fit_inst if v in fit else held_inst if v in held else Counter())[r["predicate"]] += 1
    assert nov == 3301

    def support(removed):
        s = Counter()
        for p, n in fit_inst.items():
            if p not in removed:
                for out in C.instance_values(p, phi):
                    s[out] += n
        return s

    rec = {p: C.effective_recipe(p, phi) for p in phi}      # = A6's recipes after G1 (G1 dropped nothing)
    eligible = [p for p in base if rec[p] and held_inst[p] > 0 and p not in a6]
    order = sorted(eligible, key=lambda p: (-held_inst[p], p))
    chosen, log = [], []
    for p in order:
        if len(chosen) == N_LEAVE:
            break
        fam = family(p)
        if sum(family(q) == fam for q in chosen) >= PER_FAMILY:
            log.append((p, f"family '{fam}' already has {PER_FAMILY}"))
            continue
        if any(rec[p] == rec[q] for q in chosen):
            log.append((p, "shares a recipe with a chosen predicate"))
            continue
        trial = chosen + [p]
        s = support(set(trial))
        short = sorted({(q, f"{a}={v}", s[(a, v)]) for q in trial
                        for a, v in C.consulted_outputs(rec[q]) if s[(a, v)] < C.SUPPORT_MIN})
        if short:
            log.append((p, f"support would fall below {C.SUPPORT_MIN}: {short}"))
            continue
        if MIN_LEARNED - sum(C.needs_learned_part(rec[q]) for q in trial) > N_LEAVE - len(trial):
            log.append((p, "no learned part, and the remaining places are needed for learned ones"))
            continue
        chosen.append(p)
        log.append((p, "chosen"))
    if len(chosen) < 6:
        sys.exit(f"!! only {len(chosen)} eligible predicates (< 6): STOP")
    s_after = support(set(chosen))
    out = dict(rule=dict(a6_rule=True, exclude_a6=a6, per_family=PER_FAMILY, n=N_LEAVE, min_learned=MIN_LEARNED),
               families={p: family(p) for p in base},
               chosen=chosen, n_chosen=len(chosen),
               rows=[dict(predicate=p, family=family(p), held_instances=held_inst[p], fit_instances=fit_inst[p],
                          recipe=[list(x) for x in rec[p]], learned_part=C.needs_learned_part(rec[p])) for p in chosen],
               n_learned=sum(C.needs_learned_part(rec[p]) for p in chosen),
               consulted_support_after=sorted({(f"{a}={v}", s_after[(a, v)]) for q in chosen
                                               for a, v in C.consulted_outputs(rec[q])}),
               selection_log=log, novel_instances_dropped_at_parse=nov)
    json.dump(out, open(OUT, "w"), indent=1)
    C.write_log("a2_step0_select", dict(N_LEAVE=N_LEAVE, MIN_LEARNED=MIN_LEARNED, PER_FAMILY=PER_FAMILY,
                                        SUPPORT_MIN=C.SUPPORT_MIN), [C.PHI, C.SPLIT, C.PREREG], [OUT])
    for r in out["rows"]:
        print(f"  {r['predicate']:16s} family {r['family']:8s} HELD {r['held_instances']:4d} FIT {r['fit_instances']:5d} "
              f"learned={r['learned_part']} recipe={[':'.join(x for x in f if x) for f in r['recipe']]}")
    print("n learned:", out["n_learned"], "| consulted support after removal:", out["consulted_support_after"])
    print("skipped:")
    for p, why in log:
        if why != "chosen":
            print(f"  {p}: {why}")


if __name__ == "__main__":
    main()
