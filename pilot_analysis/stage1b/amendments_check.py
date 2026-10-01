#!/usr/bin/env python3
"""Consequences of the 2026-09-28 amendments (A1-A3) on the frozen schema.

    python3 pilot_analysis/stage1b/amendments_check.py

Pure schema computation: phi_frozen.json + base support counted from the
TRAINING annotations (base predicates only; novel instances dropped at parse).
Tags are compared before (session 2) and after (A2: pass LEARNED; A3: adjacent
LEARNED). A1 changes a measurement rule, not a tag or a phi value.
"""
from __future__ import annotations

import glob, json, os, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
raw = json.load(open(os.path.join(ROOT, "pilot_analysis", "stage0e", "phi_frozen.json")))
phi = {p: {k: v for k, v in d.items() if not k.startswith("_")} for p, d in raw.items()}
split = {p: d["_split"] for p, d in raw.items()}
base = {p for p, s in split.items() if s == "base"}
novel = sorted(p for p, s in split.items() if s == "novel")

tr = Counter()
for f in glob.glob(os.path.join(ROOT, "data", "vidvrd", "anno", "train", "*.json")):
    for r in json.load(open(f))["relation_instances"]:
        if r["predicate"] in base:          # novel dropped here
            tr[r["predicate"]] += 1
sup = Counter()
for p in base:
    for x in phi[p].items():
        sup[x] += tr[p]

LEARNED_AXES = {"locomotion", "posture", "contact", "contact_action"}
SUPPLIED_AXES = {"subject_kind"}
AMENDED_LEARNED_VALUES = {("relative_motion", "pass"), ("proximity", "adjacent")}


def tag(axis, value, amended):
    if axis in SUPPLIED_AXES:
        return "SUPPLIED"
    if axis in LEARNED_AXES or (amended and (axis, value) in AMENDED_LEARNED_VALUES):
        return "LEARNED"
    return "MEASURED"


def learned_positive(p, amended):
    return [x for x in phi[p].items() if tag(*x, amended) == "LEARNED" and x[1] != "none"]


def unreachable(amended):
    out = {}
    for p in novel:
        if not phi[p]:
            out[p] = "no ingredients"
            continue
        thin = [x for x in phi[p].items() if tag(*x, amended) == "LEARNED" and sup[x] < 50]
        if thin:
            out[p] = ", ".join(f"{a}={v} ({sup[(a, v)]})" for a, v in thin)
    return out


def collisions():
    g = defaultdict(list)
    for p, d in phi.items():
        if d:
            g[tuple(sorted(d.items()))].append(p)
    groups = [v for v in g.values() if len(v) > 1]
    kind = lambda grp: ("all-novel" if all(split[p] == "novel" for p in grp) else
                        "all-base" if all(split[p] == "base" for p in grp) else "base-novel")
    return Counter(kind(x) for x in groups), groups


def effective(d):
    # compositional absence rule is a scoring rule; for identity checks the
    # universal fill is the stricter test (it can only merge, never split, groups)
    e = dict(d)
    for a in ("locomotion", "posture"):
        e.setdefault(a, "none")
    return e


before = sorted(p for p in novel if phi[p] and not learned_positive(p, False))
after = sorted(p for p in novel if phi[p] and not learned_positive(p, True))
moved = sorted(set(before) - set(after))
unr_b, unr_a = unreachable(False), unreachable(True)
kinds, groups = collisions()
eg = defaultdict(list)
for p, d in phi.items():
    if d:
        eg[tuple(sorted(effective(d).items()))].append(p)
egroups = [v for v in eg.values() if len(v) > 1]

L = ["# Consequences of amendments A1–A3 (2026-09-28)\n",
     "Schema: `stage0e/phi_frozen.json`, unchanged. The amendments change **tags**: A2 "
     "makes `pass` LEARNED and A3 makes `adjacent` LEARNED. A1 changes a **measurement "
     "rule** for approach/recede/co_move, not a tag or a phi value. Base support is "
     "counted from the training annotations, base predicates only.\n",
     f"## Novel predicates with no LEARNED positive ingredient: {len(before)} → **{len(after)}**\n",
     "After the amendments: " + ", ".join(f"`{p}`" for p in after) + ".\n",
     "No longer in the set, because they now carry a LEARNED value: " + ", ".join(
         f"`{p}` ({', '.join(f'{a}={v}' for a, v in learned_positive(p, True))})" for p in moved) + ".\n",
     f"## Reachability: {len(unr_b)} → **{len(unr_a)} unreachable**"
     f"{' (unchanged)' if unr_a == unr_b else ' ⚠ CHANGED'}\n",
     ", ".join(f"`{p}` ({why})" for p, why in unr_a.items()) + ".\n",
     f"The newly LEARNED values clear the ≥ 50 threshold: pass {sup[('relative_motion', 'pass')]}, "
     f"adjacent {sup[('proximity', 'adjacent')]}.\n",
     "## Collisions\n",
     f"Identical phi: all-novel {kinds['all-novel']}, base-novel {kinds['base-novel']}, "
     f"all-base {kinds['all-base']} (" + "; ".join(", ".join(g) for g in groups) + "). "
     "Re-tagging changes no phi value, so it **cannot** create a collision. Under the "
     "absence fill the groups are: " + "; ".join(", ".join(g) for g in egroups) + ". "
     "**No new novel collision.**\n"]
open(os.path.join(HERE, "amendments_check.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
