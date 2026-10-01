#!/usr/bin/env python
"""CHECK 1 -- does decomposing predicates into primitives buy supervision?

    python pilot_analysis/scripts/check1_supervision_multiplier.py

C1's premise is that primitives are commoner than the predicates that contain
them, so a classifier trained on primitives sees more instances than one trained
on predicates. This script measures that multiplier. CPU only, no model.

Supervision is counted over BASE predicates only: `Model.forward` trains on
`labels['pre_label'][:, :, base_pids]`, so a training instance whose predicate is
novel contributes zero gradient to anything.
"""
from __future__ import annotations

import glob, json, math, os, statistics, sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from utils import paths

MAP = os.path.join(ROOT, "pilot_analysis", "primitive_map.json")
# The AP ~ log(train count) slope fitted on the base split in SS B (h1_cause.py,
# ANALYSIS 2, n=71 with zeros kept).
BETA_LOGN = 6.14


def train_counts():
    c = Counter()
    for f in glob.glob(os.path.join(paths.ANNO_TRAIN_DIR, "*.json")):
        for r in json.load(open(f))["relation_instances"]:
            c[r["predicate"]] += 1
    return c


def geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    pm = json.load(open(MAP))
    tr = train_counts()
    base = [p for p, d in pm.items() if d["ov_split"] == "base"]
    novel = [p for p, d in pm.items() if d["ov_split"] == "novel"]

    # ---- primitive supervision: sum of base train instances containing it ---
    sup = Counter()
    members = defaultdict(list)
    for p in base:
        for slot in ("action", "spatial"):
            k = pm[p][slot]
            if k:
                sup[(slot, k)] += tr[p]
                members[(slot, k)].append(p)
    all_prims = {(s, pm[p][s]) for p in pm for s in ("action", "spatial") if pm[p][s]}

    print("=" * 74)
    print("CHECK 1 -- SUPERVISION MULTIPLIER")
    print("=" * 74)
    acts = sorted({k for s, k in all_prims if s == "action"})
    spas = sorted({k for s, k in all_prims if s == "spatial"})
    print(f"\n(b) VOCABULARY SIZE")
    print(f"  132 predicates -> {len(acts)} action primitives + {len(spas)} spatial "
          f"primitives = {len(acts)+len(spas)} total")
    print(f"  compression ratio: {132/(len(acts)+len(spas)):.2f}x fewer classes")

    print(f"\n  action primitives, base supervision (instances):")
    for k in sorted(acts, key=lambda k: -sup[("action", k)]):
        n = sup[("action", k)]
        print(f"    {k:10} {n:7d}  ({len(members[('action',k)])} base predicates)")
    print(f"\n  spatial primitives, base supervision (instances):")
    for k in sorted(spas, key=lambda k: -sup[("spatial", k)]):
        n = sup[("spatial", k)]
        print(f"    {k:10} {n:7d}  ({len(members[('spatial',k)])} base predicates)")

    # ---- (c) multiplier over base predicates -------------------------------
    # A predicate is scored by composing both of its primitives, so the binding
    # constraint is the rarer one.
    rows = []
    for p in base:
        own = tr[p]
        prims = [sup[(s, pm[p][s])] for s in ("action", "spatial") if pm[p][s]]
        eff = min(prims)
        rows.append((p, own, eff, eff / own if own else float("nan")))
    rows.sort(key=lambda r: r[1])

    mults = [r[3] for r in rows]
    per_pred = [r[1] for r in rows]
    per_prim = [r[2] for r in rows]
    print(f"\n(c) MULTIPLIER, {len(rows)} BASE predicates")
    print(f"  instances per PREDICATE  median {statistics.median(per_pred):8.1f}"
          f"   geomean {geomean(per_pred):8.1f}")
    print(f"  instances per PRIMITIVE  median {statistics.median(per_prim):8.1f}"
          f"   geomean {geomean(per_prim):8.1f}   (min of the two slots)")
    print(f"  multiplier               median {statistics.median(mults):8.2f}x"
          f"  geomean {geomean(mults):8.2f}x")

    q1 = statistics.quantiles(per_pred, n=4)[0]
    tail = [r for r in rows if r[1] <= q1]
    print(f"\n  bottom-quartile (rarest) base predicates: n={len(tail)}, "
          f"train count <= {q1:.0f}")
    print(f"    instances per PREDICATE  median {statistics.median([r[1] for r in tail]):8.1f}"
          f"   geomean {geomean([r[1] for r in tail]):8.1f}")
    print(f"    instances per PRIMITIVE  median {statistics.median([r[2] for r in tail]):8.1f}"
          f"   geomean {geomean([r[2] for r in tail]):8.1f}")
    print(f"    multiplier               median {statistics.median([r[3] for r in tail]):8.2f}x"
          f"  geomean {geomean([r[3] for r in tail]):8.2f}x")
    print(f"\n  rarest 15 base predicates:")
    print(f"    {'predicate':16} {'own':>6} {'prim(min)':>10} {'mult':>8}")
    for p, own, eff, m in rows[:15]:
        print(f"    {p:16} {own:6d} {eff:10d} {m:7.1f}x")
    print(f"\n  the 5 base predicates with the SMALLEST multiplier:")
    for p, own, eff, m in sorted(rows, key=lambda r: r[3])[:5]:
        print(f"    {p:16} {own:6d} {eff:10d} {m:7.2f}x")

    # ---- (d) novel coverage ------------------------------------------------
    print(f"\n(d) NOVEL COVERAGE, {len(novel)} novel predicates")
    unseen = []
    for p in sorted(novel, key=lambda p: -min(
            [sup[(s, pm[p][s])] for s in ("action", "spatial") if pm[p][s]])):
        a, s = pm[p]["action"], pm[p]["spatial"]
        sa = sup[("action", a)] if a else None
        ss = sup[("spatial", s)] if s else None
        vals = [v for v in (sa, ss) if v is not None]
        if min(vals) == 0:
            unseen.append(p)
    both = [p for p in novel if pm[p]["action"] and pm[p]["spatial"]]
    print(f"  novel predicates with BOTH slots filled : {len(both)}")
    print(f"  novel predicates with >=1 UNSEEN primitive (zero base support): "
          f"{len(unseen)}")
    dead_a = sorted({pm[p]['action'] for p in unseen
                     if pm[p]['action'] and sup[('action', pm[p]['action'])] == 0})
    dead_s = sorted({pm[p]['spatial'] for p in unseen
                     if pm[p]['spatial'] and sup[('spatial', pm[p]['spatial'])] == 0})
    print(f"    unseen action primitives : {dead_a}")
    print(f"    unseen spatial primitives: {dead_s}")
    print(f"    the {len(unseen)} predicates: {sorted(unseen)}")

    covered = [p for p in novel if p not in unseen]
    eff_nov = {p: min([sup[(s, pm[p][s])] for s in ("action", "spatial") if pm[p][s]])
               for p in covered}
    print(f"\n  the other {len(covered)} novel predicates -- base support of the "
          f"weaker primitive:")
    print(f"    median {statistics.median(eff_nov.values()):.0f}   "
          f"geomean {geomean(list(eff_nov.values())):.0f}   "
          f"min {min(eff_nov.values())}   max {max(eff_nov.values())}")
    print(f"    {'predicate':16} {'action':>10} {'spatial':>10} {'min':>8}")
    for p in sorted(covered, key=lambda p: eff_nov[p])[:12]:
        a, s = pm[p]["action"], pm[p]["spatial"]
        print(f"    {p:16} {sup[('action',a)] if a else '-':>10} "
              f"{sup[('spatial',s)] if s else '-':>10} {eff_nov[p]:8d}")

    # ---- (e) predicted gain ------------------------------------------------
    print(f"\n(e) PREDICTED GAIN  (beta = {BETA_LOGN:+.2f} mAP pts per ln-unit of "
          f"train count)")
    gains = [(p, BETA_LOGN * math.log(m)) for p, own, eff, m in rows]
    tot = sum(g for _, g in gains) / len(gains)
    tail_g = [BETA_LOGN * math.log(r[3]) for r in tail]
    print(f"  BASE predicates ({len(gains)}): mean predicted delta-AP "
          f"{tot:+.2f} pts, median {statistics.median([g for _,g in gains]):+.2f}")
    print(f"    bottom quartile ({len(tail_g)}): mean {sum(tail_g)/len(tail_g):+.2f} pts")
    print(f"    top 5 by predicted gain: "
          + ", ".join(f"{p} {g:+.1f}" for p, g in sorted(gains, key=lambda x: -x[1])[:5]))
    # novel: treat the weaker primitive's base support as the predicate's
    # effective supervision, against a baseline of zero trained parameters.
    # ln(1) = 0 is the floor, i.e. "one instance" is the notional starting point.
    nov_g = [BETA_LOGN * math.log(max(v, 1)) for v in eff_nov.values()]
    print(f"  NOVEL predicates with covered primitives ({len(nov_g)}): "
          f"mean predicted AP {sum(nov_g)/len(nov_g):+.2f} pts "
          f"(vs. a measured novel mAP of 15.79)")
    print(f"    the {len(unseen)} uncovered novel predicates gain nothing: +0.00")
    print(f"    novel-split average over all {len(novel)}: "
          f"{sum(nov_g)/len(novel):+.2f} pts")
    print("\n  CAVEAT: beta is a correlational fit (R2 0.237, +-3.17 at 95%) on "
          "base\n  predicates. Extrapolating it to primitive-pooled counts assumes "
          "pooled\n  instances are worth as much per instance as native ones. They "
          "are not:\n  a `walk_behind` instance is a noisier positive for `behind` "
          "than a\n  `behind` instance is. Read these as an upper bound.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
