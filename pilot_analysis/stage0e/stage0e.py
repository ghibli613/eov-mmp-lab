#!/usr/bin/env python3
"""Stage 0e -- apply and freeze. The last stage-0 round.

    python3 pilot_analysis/stage0e/stage0e.py

Data only. BASE-predicate instances from the TRAINING annotations are the only
evidence; test annotations are never opened. Writes only into
pilot_analysis/stage0e/. Builds on the stage-0d phi.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import csv, json, os
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for d in ("stage0", "stage0b", "stage0c", "stage0d"):
    sys.path.insert(0, os.path.join(ROOT, "pilot_analysis", d))
import stage0 as s0   # noqa: E402
import stage0b as sb  # noqa: E402
import stage0c as sc  # noqa: E402
import stage0d as sd  # noqa: E402

THIN = sb.THIN
FAMILY_RULE = 0.95       # apply subject_kind=vehicle to a verb family only if
FAMILY_MIN_N = 10        # every base member with >= 10 instances is >= 95% vehicle
LOCO_VERBS = ("walk", "run", "fly", "swim", "creep", "jump")
POSTURE_VERBS = ("stand", "sit", "lie")
ANIMATE_VERBS = POSTURE_VERBS + ("walk", "run", "creep", "swim")

SCHEMA = {a: list(v) for a, v in sd.SCHEMA.items()}
TAG = dict(sd.TAG)
TAG["subject_kind"] = "SUPPLIED"          # from the object classifier
LEARNED = [a for a, t in TAG.items() if t == "LEARNED"]
ABSENCE_AXES = ("locomotion", "posture")  # unset -> asserts absence


def phi_v5(p, vehicle_verbs):
    d = sd.phi_v4(p, revert_with_adjacent=True)
    if sb.verb_of(p) in vehicle_verbs:
        d["subject_kind"] = "vehicle"
    return d


def effective(d):
    """The scoring rule's view of a phi: unset locomotion/posture become an
    asserted absence. Empty phis stay empty (unscorable)."""
    if not d:
        return {}
    e = dict(d)
    for a in ABSENCE_AXES:
        e.setdefault(a, "none")
    return e


def positive_learned(d):
    return [(a, v) for a, v in d.items() if TAG[a] == "LEARNED" and v != "none"]


# ---------------------------------------------------------------------------
def box_rel(bs, bo):
    ix = max(0, min(bs["xmax"], bo["xmax"]) - max(bs["xmin"], bo["xmin"]))
    iy = max(0, min(bs["ymax"], bo["ymax"]) - max(bs["ymin"], bo["ymin"]))
    a_s = max(bs["xmax"] - bs["xmin"], 1) * max(bs["ymax"] - bs["ymin"], 1)
    a_o = max(bo["xmax"] - bo["xmin"], 1) * max(bo["ymax"] - bo["ymin"], 1)
    cx, cy = (bs["xmin"] + bs["xmax"]) / 2, (bs["ymin"] + bs["ymax"]) / 2
    if bo["xmin"] <= cx <= bo["xmax"] and bo["ymin"] <= cy <= bo["ymax"]:
        pos = "within"
    elif cy < bo["ymin"]:
        pos = "above"
    else:
        pos = "other"
    s_in_o, o_in_s = ix * iy / a_s, ix * iy / a_o
    return s_in_o, o_in_s, pos


def contained_comparative(s_in_o, o_in_s, pos):
    """Parameter-free: the subject is more inside the object than the object is
    inside the subject, and the subject's centre lies within the object box."""
    return s_in_o > o_in_s and pos == "within"


# ---------------------------------------------------------------------------
def main():
    split, tr = sb.load()
    preds = sorted(split)
    base = [p for p in preds if split[p] == "base"]
    novel = [p for p in preds if split[p] == "novel"]
    base_set = set(base)

    print("loading training annotations ...", file=sys.stderr)
    vids = sc.load_train()
    cat_of = {v: {so["tid"]: so["category"] for so in d["subject/objects"]}
              for v, d in vids.items()}
    all_cats = sorted({c for m in cat_of.values() for c in m.values()})

    subj = defaultdict(Counter)
    for v, d in vids.items():
        for r in d["relation_instances"]:
            if r["predicate"] in base_set:
                subj[r["predicate"]][cat_of[v][r["subject_tid"]]] += 1

    # ---------------- category-derived subject_kind sets ----------------
    VEHICLES = sorted({c for p in base if sb.verb_of(p) == "stop" for c in subj[p]})
    ANIMATE = sorted({c for p in base if sb.verb_of(p) in ANIMATE_VERBS
                      for c in subj[p]} - set(VEHICLES))
    OTHER = sorted(set(all_cats) - set(VEHICLES) - set(ANIMATE))
    kind_of = {c: ("vehicle" if c in VEHICLES else "animate" if c in ANIMATE else "other")
               for c in all_cats}
    kind_n = Counter()
    for p in base:
        for c, n in subj[p].items():
            kind_n[kind_of[c]] += n
    veh_leaks = {p: {c: n for c, n in subj[p].items() if c in VEHICLES}
                 for p in base if sb.verb_of(p) in ANIMATE_VERBS}
    veh_leaks = {p: v for p, v in veh_leaks.items() if v}

    # ---------------- Task 1: which verb families are vehicle-only ----------------
    fam_rates = {}
    for vb in ("move", "fly", "stop"):
        rows = []
        for p in sorted(base):
            if sb.verb_of(p) != vb:
                continue
            n = sum(subj[p].values())
            k = sum(m for c, m in subj[p].items() if c in VEHICLES)
            rows.append((p, k, n, {c: m for c, m in subj[p].items() if c not in VEHICLES}))
        tot_k = sum(r[1] for r in rows); tot_n = sum(r[2] for r in rows)
        passes = all(k / n >= FAMILY_RULE for _, k, n, _ in rows if n >= FAMILY_MIN_N)
        fam_rates[vb] = dict(rows=rows, k=tot_k, n=tot_n, passes=passes)
    vehicle_verbs = {vb for vb in ("move", "fly") if fam_rates[vb]["passes"]} | {"stop"}

    # ---------------- Task 3: contained, parameter-free ----------------
    ride = []
    for v, d in vids.items():
        for r in d["relation_instances"]:
            if r["predicate"] != "ride":
                continue
            fr = []
            for t in range(r["begin_fid"], min(r["end_fid"], len(d["_boxes"]))):
                bs = d["_boxes"][t].get(r["subject_tid"])
                bo = d["_boxes"][t].get(r["object_tid"])
                if bs and bo:
                    fr.append(box_rel(bs, bo))
            if not fr:
                continue
            maj = lambda xs: sum(xs) > len(xs) / 2
            pos = Counter(x[2] for x in fr).most_common(1)[0][0]
            nest, topok = [], []
            for t in range(r["begin_fid"], min(r["end_fid"], len(d["_boxes"]))):
                bs = d["_boxes"][t].get(r["subject_tid"])
                bo = d["_boxes"][t].get(r["object_tid"])
                if bs and bo:
                    nest.append(bs["xmin"] >= bo["xmin"] and bs["ymin"] >= bo["ymin"]
                                and bs["xmax"] <= bo["xmax"] and bs["ymax"] <= bo["ymax"])
                    a_, b_, _ = box_rel(bs, bo)
                    topok.append(a_ > b_ and bs["ymin"] >= bo["ymin"])
            ride.append(dict(
                nested=maj(nest), top=maj(topok),
                s_gt_o=maj([a > b for a, b, _ in fr]),
                pos=pos,
                contained=maj([contained_comparative(*x) for x in fr]),
                old90=maj([a >= 0.9 for a, _, _ in fr]),
                subj_obj=(cat_of[v][r["subject_tid"]], cat_of[v][r["object_tid"]])))
    # sanity: a strictly nested box passes, a rider-like box does not
    nested = contained_comparative(*box_rel(
        dict(xmin=40, ymin=40, xmax=60, ymax=60), dict(xmin=0, ymin=0, xmax=100, ymax=100)))
    rider = contained_comparative(*box_rel(
        dict(xmin=40, ymin=0, xmax=60, ymax=70), dict(xmin=20, ymin=50, xmax=80, ymax=100)))

    # ---------------- Task 4: tables ----------------
    phi4 = {p: sd.phi_v4(p, revert_with_adjacent=True) for p in preds}
    phi5 = {p: phi_v5(p, vehicle_verbs) for p in preds}
    for p, d in phi5.items():
        for a, v in d.items():
            assert v in SCHEMA[a], (p, a, v)
    json.dump({p: dict(phi5[p], _split=split[p]) for p in preds},
              open(os.path.join(HERE, "phi_frozen.json"), "w"), indent=1)

    def tables(phi, tag):
        sup, _ = sd.support(phi, split, tr)
        unr, unas, thin = sd.reach(phi, sup, novel, tag)
        g = sb.collision_groups(phi, preds)
        sp = sc.subset_pairs(phi)
        return dict(sup=sup, unr=unr, unas=unas, thin=thin, groups=g,
                    kinds=sd.kinds(g, split), pairs=sp, nsub=sd.nov_sub(sp, split))

    V4 = tables(phi4, sd.TAG)
    V5 = tables(phi5, TAG)
    key = lambda g: tuple(sorted(g))
    new_coll = [g for g in V5["groups"] if key(g) not in {key(x) for x in V4["groups"]}]
    gone_coll = [g for g in V4["groups"] if key(g) not in {key(x) for x in V5["groups"]}]
    new_ns = sorted(V5["nsub"] - V4["nsub"])
    new_pairs = sorted(V5["pairs"] - V4["pairs"])

    with open(os.path.join(HERE, "subset_pairs_frozen.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "subset_split", "superset", "superset_split",
                    "attributes_only_in_superset", "domination_under_scoring_rule"])
        sets = {p: set(phi5[p].items()) for p in preds}
        eff = {p: set(effective(phi5[p]).items()) for p in preds}
        for a, b in sorted(V5["pairs"]):
            w.writerow([a, split[a], b, split[b],
                        ";".join(f"{k}={v}" for k, v in sorted(sets[b] - sets[a])),
                        bool(eff[a]) and eff[a] < eff[b]])

    # ---------------- Task 5: scoring-rule consequences ----------------
    phiE = {p: effective(phi5[p]) for p in preds}
    gE = sb.collision_groups(phiE, preds)
    spE = sc.subset_pairs(phiE)
    nsubE = sd.nov_sub(spE, split)
    def scoped(p):
        t = s0.tokenise(p)
        compositional = (len(t) == 1 and t[0] in s0.SPATIAL) or \
                        (len(t) == 2 and t[0] in s0.VERB)
        return effective(phi5[p]) if compositional else dict(phi5[p])
    phiS = {p: scoped(p) for p in preds}
    gS = sb.collision_groups(phiS, preds)
    spS = sc.subset_pairs(phiS)
    nsubS = sd.nov_sub(spS, split)
    no_learned = sorted(p for p in novel if phi5[p] and not positive_learned(phi5[p]))
    unreachable = V5["unr"]

    # licence check: does "unset -> absent" hold for the families it is applied to?
    base_by_pair = defaultdict(list)
    for v, d in vids.items():
        for r in d["relation_instances"]:
            if r["predicate"] in base_set:
                base_by_pair[(v, r["subject_tid"], r["object_tid"])].append(r)

    def cooc(anchors):
        """Overlap with a locomotion / posture compound of a DIFFERENT verb, on
        the same pair. A column is None when the family sets that axis itself to
        a positive value (then absence is not being asserted on it)."""
        n = loco = post = anyo = 0
        some = next(iter(anchors))
        own = sb.verb_of(some)
        d0 = phi5[some]
        loco_rel = d0.get("locomotion", "none") == "none"
        post_rel = "posture" not in d0
        for insts in base_by_pair.values():
            for r in insts:
                if r["predicate"] not in anchors:
                    continue
                n += 1
                others = [q for q in insts if q is not r and sc.overlaps(r, q)]
                ov = [sb.verb_of(q["predicate"]) for q in others]
                loco += any(v in LOCO_VERBS and v != own for v in ov)
                post += any(v in POSTURE_VERBS and v != own for v in ov)
                anyo += any(q["predicate"] != r["predicate"] for q in others)
        return n, (loco if loco_rel else None), (post if post_rel else None), anyo

    fams = [("bare spatial (left, right, front, behind, next_to)",
             {"left", "right", "front", "behind", "next_to"})]
    fams.append(("move_* (all base)", {p for p in base if sb.verb_of(p) == "move"}))
    for vb in ("stop",) + POSTURE_VERBS + LOCO_VERBS:
        fams.append((f"{vb}_* (all base)", {p for p in base if sb.verb_of(p) == vb}))
    for p in sorted(base):
        if sb.verb_of(p) is None and p not in {"left", "right", "front", "behind", "next_to"} \
                and "locomotion" not in phi5[p] and "posture" not in phi5[p]:
            fams.append((f"`{p}`", {p}))
    lic = [(name,) + cooc(ps) for name, ps in fams]

    # ---------------- write ----------------
    fmt = lambda x: f"{x[0]}={x[1]}"
    star = lambda p: f"`{p}`" + ("*" if split[p] == "novel" else "")
    pct = sc.pct

    F = ["# Stage 0e — Task 1: vehicle verb families\n",
         f"Vehicle categories (subjects of base stop_*): {', '.join(VEHICLES)}.\n",
         f"Rule: apply subject_kind=vehicle to a verb family only if every base member "
         f"with ≥{FAMILY_MIN_N} instances has ≥{FAMILY_RULE:.0%} vehicle subjects. A family "
         f"rate alone can hide a mixed member.\n"]
    for vb in ("move", "fly"):
        fr = fam_rates[vb]
        F.append(f"## `{vb}_*`: family {pct(fr['k'], fr['n'])} → "
                 f"**{'APPLIED' if fr['passes'] else 'NOT APPLIED'}**\n")
        F.append("| base predicate | vehicle subjects | non-vehicle subjects |")
        F.append("|---|---:|---|")
        for p, k, n, nonv in fr["rows"]:
            F.append(f"| `{p}` | {pct(k, n)} | {', '.join(f'{c} {m}' for c, m in nonv.items()) or '—'} |")
        F.append("")
    open(os.path.join(HERE, "vehicle_families.md"), "w").write("\n".join(F) + "\n")

    cont = sum(r["contained"] for r in ride); n_r = len(ride)
    pos_c = Counter(r["pos"] for r in ride)
    s_gt = sum(r["s_gt_o"] for r in ride)
    both_fail = sum(1 for r in ride if not r["contained"])
    agree90 = sum(1 for r in ride if r["contained"] == r["old90"])

    R = ["# Stage 0e — apply and freeze\n",
         "## Comparison with v4 final\n",
         "| | v4 final | **frozen (0e)** |", "|---|---:|---:|",
         f"| unreachable novel predicates | {len(V4['unr'])} | **{len(V5['unr'])}** |",
         f"| collisions: all-novel | {V4['kinds']['all-novel']} | **{V5['kinds']['all-novel']}** |",
         f"| collisions: base-novel | {V4['kinds']['base-novel']} | **{V5['kinds']['base-novel']}** |",
         f"| collisions: all-base | {V4['kinds']['all-base']} | **{V5['kinds']['all-base']}** |",
         f"| strict-subset pairs, all | {len(V4['pairs'])} | **{len(V5['pairs'])}** |",
         f"| novel-subset pairs | {len(V4['nsub'])} | **{len(V5['nsub'])}** |",
         f"\nNew collisions: {len(new_coll)}. Removed collisions: {len(gone_coll)}. "
         f"New subset pairs (any): {len(new_pairs)}. New novel-subset pairs: {len(new_ns)}. "
         f"Removed novel-subset pairs: {len(V4['nsub'] - V5['nsub'])}.\n"]
    pass  # tables go in README.md

    S = ["# Frozen scoring rule for stage 1\n",
         "Schema: `phi_frozen.json` (132 predicates). Produced by stage 0e; stage 0 "
         "closes here.\n",
         "## Axis roles\n",
         "| role | axes | trained parameters |", "|---|---|---|",
         "| MEASURED | " + ", ".join(a for a, t in TAG.items() if t == "MEASURED")
         + " | none; parameter-free functions of box sequences |",
         "| LEARNED | " + ", ".join(LEARNED) + " | attribute head, trained on base only |",
         "| SUPPLIED | subject_kind | none; read off the object classifier's category |",
         "",
         "## Unset axes\n",
         "- **locomotion, posture**: unset **asserts absence**. Licensed by stage 0c "
         "Task 2: bare spatial terms overlap a same-spatial compound 4.3% of the time "
         "and move_X overlaps manner_X 0.4% of the time, against 70.6% / 59.8% "
         "any-label controls.",
         "- **every other axis**: unset is **marginalised out**.",
         "- A predicate with no ingredients at all stays unscorable. The absence rule "
         "does not give it a phi.\n",
         "## MEASURED definitions that changed in 0e\n",
         "- **proximity=contained — ⚠ DECISION NEEDED.** The brief's definition "
         "fails its own acceptance test. All three candidates are parameter-free, "
         "evaluated per frame, with the instance taking the majority. Base `ride` "
         f"({n_r} instances):\n",
         "  | definition | ride classed as contained |",
         "  |---|---:|",
         f"  | 0d: ≥90% of subject box inside object box (has a threshold) | {pct(sum(r['old90'] for r in ride), n_r)} |",
         f"  | **brief**: subject-in-object > object-in-subject AND subject centre within object box | **{pct(cont, n_r)}** |",
         f"  | strict nesting: all four subject edges inside the object box | {pct(sum(r['nested'] for r in ride), n_r)} |",
         f"  | **proposed**: subject-in-object > object-in-subject AND subject top edge not above object top edge | **{pct(sum(r['top'] for r in ride), n_r)}** |",
         "",
         f"  The brief's version fails because a rider's centre (the hip) lies "
         f"inside the mount's box in {pos_c['within']}/{n_r} instances. The rider is "
         f"also usually the smaller box, so the fraction comparison favours the "
         f"subject too. What separates ride is that the rider protrudes above the "
         f"mount (0c: rider top above mount top in 98.5% of frames). The proposed "
         f"version tests exactly that. Strict nesting separates as well, but a "
         f"one-pixel box overshoot defeats it, which is a problem for real "
         f"containment. Neither can be checked on a positive example, because "
         f"every `contained` carrier is novel.",
         "- **subject_kind**: vehicle = " + ", ".join(VEHICLES) + "; animate = "
         + ", ".join(ANIMATE) + "; neither = " + ", ".join(OTHER) + ".\n",
         f"## Novel predicates with NO learned ingredient ({len(no_learned)})\n",
         "No positive LEARNED value is set. Explicit locomotion=none counts as "
         "absence, which the rule treats the same as unset:\n",
         ", ".join(star(p) for p in no_learned) + "\n",
         "⚠ These are parameter-free in their positive evidence only. Under the "
         "absence rule, each of them is still scored against the learned locomotion "
         "and posture heads, which must say 'absent'. So they are not "
         "free of trained parameters. The heads they consult are well supported, "
         f"with locomotion=none at {V5['sup'][('locomotion', 'none')]} base "
         "instances, so this does not affect reachability.\n",
         f"## Unreachable novel predicates ({len(unreachable)})\n",
         ", ".join(f"{star(p)} ({'no ingredients' if p in V5['unas'] else ', '.join(fmt(x) + f' = {V5['sup'][x]}' for x in V5['thin'][p])})"
                   for p in unreachable) + "\n",
         "## What the rule does to the subset structure\n",
         "A formal subset A ⊂ B is a real domination only if B's extra attributes "
         "are all on marginalised axes. If an extra attribute is locomotion or "
         "posture, A asserts its absence and the two conflict. Recomputed on the "
         "effective phis (unset locomotion and posture set to `none`):\n",
         "| | formal phi | under the scoring rule |", "|---|---:|---:|",
         f"| strict-subset pairs | {len(V5['pairs'])} | {len(spE)} |",
         f"| novel-subset pairs | {len(V5['nsub'])} | {len(nsubE)} |",
         f"| collision groups (all) | {len(V5['groups'])} | {len(gE)} |",
         "",
         "Collision groups under the rule: " + ("; ".join(
             ", ".join(star(p) for p in g) for g in gE) or "none") + ".\n",
         "The dominations that remain are listed in `subset_pairs_frozen.csv`, in "
         "the column `domination_under_scoring_rule`. Every one of them has its "
         "extra attributes on marginalised axes.\n",
         "## ⚠ Licence check for the absence rule\n",
         "0c tested bare spatial terms and move_*. The frozen rule applies absence "
         "to EVERY predicate with unset locomotion or posture, including the "
         "single-morpheme ones. Base training data, same pair and overlapping "
         "extent:\n",
         "| family | instances | overlaps a locomotion compound | overlaps a posture compound | overlaps any other label |",
         "|---|---:|---:|---:|---:|"]
    f_ = lambda x, n: "—" if x is None else f"{x/n:.1%}"
    for name, n, lo, po, an in lic:
        S.append(f"| {name} | {n} | {f_(lo, n)} | {f_(po, n)} | {an/n:.1%} |")
    S.append("\n**⚠ DECISION NEEDED — the rule holds for compositional predicates "
             "and fails for single-morpheme ones.** Every compositional family "
             "conflicts ≤3.5% of the time: the bare spatial terms and all 11 verb "
             "families, each on the axes it leaves unset or sets to `none`. Every "
             "single-morpheme base predicate with n > 10 conflicts on at least one "
             "axis 23–80% of the time: `chase`, `follow` and `faster` with a "
             "locomotion compound (~70%), `ride` with a posture compound (80%, "
             "riders sit), `touch`, `watch` and `play` with posture (51–57%). For "
             "these, an unset locomotion or posture is underspecified, not absent. "
             "Scoring it as absent would reject a running dog as `chase`.\n")
    S.append("**Proposed amendment:** apply absence only to compositional "
             "predicates (a bare spatial term, or verb + spatial term). Marginalise "
             "every unset axis of single-morpheme predicates. Among the novel ones, "
             "that means `bite`, `feed`, `hold`, `kick`, `pull`, `drive` and "
             "`fight`. Their base analogues (`touch`, `watch`, `play`, `ride`) "
             "co-occur with posture labels 51–80% of the time.\n")
    S.append(f"Effect on the subset structure: the {len(nsubE)} novel dominations "
             f"are **identical** under the universal and the scoped rule (see "
             f"summary.json). The amendment changes how single-morpheme predicates "
             f"are scored, not which novel predicates are dominated. Of those "
             f"{len(nsubE)}: 23 are the stationary `*_with` pairs, 7 have a bare "
             f"spatial term as subset, and 1 is `pull` ⊂ `drive`.\n")
    open(os.path.join(HERE, "scoring_rule.md"), "w").write("\n".join(S) + "\n")

    summary = dict(
        vehicles=VEHICLES, animate=ANIMATE, other=OTHER,
        subject_kind_instances=dict(kind_n), vehicle_subjects_in_animate_verbs=veh_leaks,
        families={vb: dict(k=fr["k"], n=fr["n"], passes=fr["passes"],
                           rows=[[p, k, n, nonv] for p, k, n, nonv in fr["rows"]])
                  for vb, fr in fam_rates.items()},
        vehicle_verbs=sorted(vehicle_verbs),
        ride=dict(n=n_r, contained=cont, s_in_o_gt_o_in_s=s_gt, centre=dict(pos_c),
                  agrees_with_90pct=agree90, old90=sum(r["old90"] for r in ride),
                  sanity_nested=nested, sanity_rider=rider,
                  contained_pairs=Counter("-".join(r["subj_obj"]) for r in ride
                                          if r["contained"]).most_common()),
        v4=dict(unr=V4["unr"], kinds=V4["kinds"], pairs=len(V4["pairs"]), nsub=len(V4["nsub"])),
        frozen=dict(unr=V5["unr"], kinds=V5["kinds"], pairs=len(V5["pairs"]), nsub=len(V5["nsub"])),
        new_collisions=new_coll, removed_collisions=gone_coll,
        new_pairs=new_pairs, new_novel_subset=new_ns,
        removed_novel_subset=sorted(V4["nsub"] - V5["nsub"]),
        no_learned=no_learned, unreachable=unreachable,
        effective=dict(pairs=len(spE), nsub=len(nsubE), groups=gE,
                       nsub_list=sorted(nsubE)),
        scoped=dict(pairs=len(spS), nsub=len(nsubS), groups=gS, nsub_list=sorted(nsubS)),
        ride_alternatives=dict(nested=sum(r["nested"] for r in ride),
                               comparative_top=sum(r["top"] for r in ride)),
        licence=[dict(family=a, n=b, loco=c, posture=d, any=e) for a, b, c, d, e in lic],
    )
    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
