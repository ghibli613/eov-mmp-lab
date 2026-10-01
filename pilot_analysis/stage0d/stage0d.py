#!/usr/bin/env python3
"""Stage 0d -- schema v4 freeze.

    python3 pilot_analysis/stage0d/stage0d.py

Data only. BASE-predicate instances from the TRAINING annotations are the only
evidence; test annotations are never opened. Writes only into
pilot_analysis/stage0d/. Builds on the stage-0c phi (imported with bytecode
writing disabled).
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import csv, json, os
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for d in ("stage0", "stage0b", "stage0c"):
    sys.path.insert(0, os.path.join(ROOT, "pilot_analysis", d))
import stage0 as s0   # noqa: E402
import stage0b as sb  # noqa: E402
import stage0c as sc  # noqa: E402

THIN = sb.THIN
RNG = np.random.default_rng(0)
N_BOOT = 1000
CONTAINED = 0.9      # fraction of the subject box inside the object box

# ---------------------------------------------------------------------------
# Schema v4
# ---------------------------------------------------------------------------
SCHEMA = {a: list(v) for a, v in sc.SCHEMA.items() if a != "state_change"}   # (a)
SCHEMA["proximity"] = SCHEMA["proximity"] + ["contained"]                    # (c)
SCHEMA["subject_kind"] = ["vehicle", "animate"]                               # (b)
TAG = {a: t for a, t in sc.TAG.items() if a != "state_change"}
TAG["subject_kind"] = "LEARNED"

CONTAINED_PREDS = {"drive", "lie_inside", "sit_inside", "stand_inside"}
WITH_STATIONARY = sc.WITH_STATIONARY


def phi_v4(p, revert_with_adjacent):
    d = sc.phi_v3(p)
    d.pop("state_change", None)                                     # (a)
    if sb.verb_of(p) == "stop":                                     # (b)
        d.update(subject_state="stationary", locomotion="none",
                 subject_kind="vehicle")
    if p in CONTAINED_PREDS:                                        # (c)
        d["proximity"] = "contained"
    if revert_with_adjacent and p in WITH_STATIONARY:               # Task 2 (B)
        d.pop("proximity", None)
    return d


def support(phi, split, tr):
    sup, nb = Counter(), Counter()
    for p, d in phi.items():
        if split[p] == "base":
            for x in d.items():
                sup[x] += tr[p]; nb[x] += 1
    return sup, nb


def reach(phi, sup, novel, tag):
    unas = sorted(p for p in novel if not phi[p])
    thin = {p: [x for x in phi[p].items() if tag[x[0]] == "LEARNED" and sup[x] < THIN]
            for p in novel}
    thin = {p: v for p, v in thin.items() if v}
    return sorted(set(unas) | set(thin)), unas, thin


def kinds(groups, split):
    c = Counter(sb.kind(g, split) for g in groups)
    return {k: c[k] for k in ("all-novel", "base-novel", "all-base")}


def nov_sub(pairs, split):
    return {(a, b) for a, b in pairs if split[a] == "novel"}


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

    # ---------------- (b) vehicle set, derived ----------------
    stop_base = sorted(p for p in base if sb.verb_of(p) == "stop")
    subj_cats = defaultdict(Counter)
    for v, d in vids.items():
        for r in d["relation_instances"]:
            if r["predicate"] in base_set:
                subj_cats[r["predicate"]][cat_of[v][r["subject_tid"]]] += 1
    veh = Counter()
    for p in stop_base:
        veh.update(subj_cats[p])
    VEHICLES = sorted(veh)
    veh_elsewhere = Counter()
    for p in base:
        if p in stop_base:
            continue
        for c, n in subj_cats[p].items():
            if c in VEHICLES:
                veh_elsewhere[p] += n
    veh_any = sum(n for p in base for c, n in subj_cats[p].items() if c in VEHICLES)
    verb_veh = Counter(); verb_n = Counter()
    for p in base:
        vb = sb.verb_of(p) or p
        for c, n in subj_cats[p].items():
            verb_n[vb] += n
            verb_veh[vb] += n * (c in VEHICLES)

    # ---------------- (c) ride containment ----------------
    ecc = json.load(open(os.path.join(ROOT, sc.ECC_FILE)))
    ride_rows, _ = sc.ride_geometry(vids, ecc, base_set)
    ride_major = sum(1 for r in ride_rows if r["cat"].startswith("contained"))
    ride_median = sum(1 for r in ride_rows if r["ios_median"] >= CONTAINED)

    # ---------------- Task 2 ----------------
    fam = {"with": [], "next_to": []}
    for p in base:
        for suf in fam:
            if p == suf or p.endswith("_" + suf):
                fam[suf].append(p)
    excluded = {suf: sorted(p for p in novel if p == suf or p.endswith("_" + suf))
                for suf in fam}
    inst = defaultdict(list)           # predicate -> [(video, same)]
    all_same = all_n = 0
    for v, d in vids.items():
        for r in d["relation_instances"]:
            p = r["predicate"]
            if p not in base_set:
                continue
            s = cat_of[v][r["subject_tid"]] == cat_of[v][r["object_tid"]]
            all_n += 1; all_same += s
            inst[p].append((v, s))
    base_rate = all_same / all_n

    def rate(ps):
        xs = [s for p in ps for _, s in inst[p]]
        return sum(xs), len(xs)

    pooled = {suf: rate(ps) for suf, ps in fam.items()}
    fp = Counter()
    for v, d in vids.items():
        for r in d["relation_instances"]:
            if r["predicate"] == "fly_with":
                fp[(cat_of[v][r["subject_tid"]], cat_of[v][r["object_tid"]])] += 1
    fly_pair = fp.most_common(1)[0]
    verbs_both = sorted({sb.verb_of(p) for p in fam["with"]} &
                        {sb.verb_of(p) for p in fam["next_to"]} - {None})

    def diff_ci(a, b):
        """with-rate minus next_to-rate, video-clustered bootstrap."""
        A, B = inst[a], inst[b]
        va, vb = sorted({v for v, _ in A}), sorted({v for v, _ in B})
        ia = defaultdict(list); ib = defaultdict(list)
        for v, s in A: ia[v].append(s)
        for v, s in B: ib[v].append(s)
        out = []
        for _ in range(N_BOOT):
            xa = [s for v in RNG.choice(va, len(va)) for s in ia[v]]
            xb = [s for v in RNG.choice(vb, len(vb)) for s in ib[v]]
            out.append(np.mean(xa) - np.mean(xb))
        d = np.mean([s for _, s in A]) - np.mean([s for _, s in B])
        return float(d), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

    matched = {vb: diff_ci(f"{vb}_with", f"{vb}_next_to") for vb in verbs_both}

    # The brief's criterion made explicit: (A) needs `with` predominantly
    # same-category (pooled >= 75%), `next_to` NOT (pooled no more than 5 pts
    # above the all-predicate base rate), and a same-verb gap whose CI excludes 0
    # for every verb where both families are base.
    w_rate = pooled["with"][0] / pooled["with"][1]
    n_rate = pooled["next_to"][0] / pooled["next_to"][1]
    outcome = "A" if (w_rate >= 0.75 and n_rate <= base_rate + 0.05 and
                      all(lo > 0 for _, lo, _ in matched.values())) else "B"
    if outcome == "A":
        raise SystemExit("Task 2 came out (A): pair_kind branch not implemented; "
                         "stop and report.")

    # ---------------- phi versions ----------------
    phi3 = {p: sc.phi_v3(p) for p in preds}
    phi4a = {p: phi_v4(p, revert_with_adjacent=False) for p in preds}
    phi4 = {p: phi_v4(p, revert_with_adjacent=True) for p in preds}
    for p, d in phi4.items():
        for a, v in d.items():
            assert v in SCHEMA[a], (p, a, v)
    json.dump({p: dict(phi4[p], _split=split[p]) for p in preds},
              open(os.path.join(HERE, "phi_v4.json"), "w"), indent=1)

    versions = {}
    for name, phi, tag in (("v3", phi3, sc.TAG), ("v4a", phi4a, TAG), ("v4", phi4, TAG)):
        sup, nb = support(phi, split, tr)
        unr, unas, thin = reach(phi, sup, novel, tag)
        g = sb.collision_groups(phi, preds)
        sp = sc.subset_pairs(phi)
        versions[name] = dict(phi=phi, sup=sup, nb=nb, unr=unr, unas=unas, thin=thin,
                              groups=g, kinds=kinds(g, split), pairs=sp,
                              nsub=nov_sub(sp, split))
    V3, V4a, V4 = versions["v3"], versions["v4a"], versions["v4"]
    key = lambda g: tuple(sorted(g))
    new_coll = [g for g in V4["groups"] if key(g) not in {key(x) for x in V3["groups"]}]
    gone_coll = [g for g in V3["groups"] if key(g) not in {key(x) for x in V4["groups"]}]
    new_ns = sorted(V4["nsub"] - V3["nsub"])
    gone_ns = sorted(V3["nsub"] - V4["nsub"])
    returned = sorted(V4["nsub"] - V4a["nsub"])      # pairs brought back by Task 2 (B)

    with open(os.path.join(HERE, "subset_pairs_v4.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "subset_split", "superset", "superset_split",
                    "attributes_only_in_superset", "new_vs_v3"])
        sets = {p: set(phi4[p].items()) for p in preds}
        for a, b in sorted(V4["pairs"]):
            w.writerow([a, split[a], b, split[b],
                        ";".join(f"{k}={v}" for k, v in sorted(sets[b] - sets[a])),
                        (a, b) not in V3["pairs"]])

    # ---------------- write ----------------
    fmt = lambda x: f"{x[0]}={x[1]}"
    star = lambda p: f"`{p}`" + ("*" if split[p] == "novel" else "")
    sup4 = V4["sup"]

    W = ["# Stage 0d — Task 2: is `with` a same-category convention?\n",
         "BASE training instances only. `same` = subject and object have the same "
         "object category. Novel `*_with` / `*_next_to` predicates are masked "
         "labels and are excluded (listed).\n",
         f"Base rate over all {all_n} base training instances: "
         f"**{base_rate:.1%}** same-category.\n",
         "| family | predicate | same / instances | fraction |", "|---|---|---:|---:|"]
    for suf, ps in fam.items():
        for p in sorted(ps, key=lambda p: -len(inst[p])):
            s = sum(x for _, x in inst[p]); n = len(inst[p])
            W.append(f"| `{suf}` | `{p}` | {s} / {n} | {s/n:.1%} |")
        s, n = pooled[suf]
        W.append(f"| `{suf}` | **pooled** | **{s} / {n}** | **{s/n:.1%}** |")
    W.append("\nExcluded (novel): `with`: " + ", ".join(excluded["with"])
             + "; `next_to`: " + ", ".join(excluded["next_to"]) + ". No bare `with` exists.\n")
    W.append("**Same verb, both families base** (with − next_to, video-clustered 95% CI):\n")
    W.append("| verb | with | next_to | difference | 95% CI |")
    W.append("|---|---:|---:|---:|---|")
    for vb, (dd, lo, hi) in matched.items():
        a, b = inst[f"{vb}_with"], inst[f"{vb}_next_to"]
        W.append(f"| {vb} | {sum(s for _, s in a)}/{len(a)} | {sum(s for _, s in b)}/{len(b)} | "
                 f"{dd:+.1%} | {lo:+.1%} … {hi:+.1%} |")
    W.append("\n## Conclusion: (B) no clear split\n")
    W.append(f"The pooled gap ({w_rate:.1%} vs {n_rate:.1%}) is a verb-mix artefact. "
             f"`fly_with` is {len(inst['fly_with'])} of the {pooled['with'][1]} `with` "
             f"instances, and {fly_pair[1]} of those are {fly_pair[0][0]}–{fly_pair[0][1]} "
             f"(formation flight), so it is 100% same-category. But `fly_next_to` is "
             f"100% same-category too, for the same reason. Holding the verb fixed, there "
             f"is no gap whose CI excludes zero. `next_to` is itself "
             f"predominantly same-category ({n_rate:.1%}, against a {base_rate:.1%} "
             f"base rate). So the second half of criterion (A) fails outright.\n")
    W.append("Also: no stationary `*_with` is base, and those three are exactly the "
             "novel predicates the axis was meant to separate. So even a real gap "
             "among moving verbs would have been an extrapolation to them.\n")
    W.append("**Applied:** stage-0c change (a) reverted. proximity=adjacent removed "
             "from " + ", ".join(f"`{p}`" for p in sorted(WITH_STATIONARY)) + ".\n")
    open(os.path.join(HERE, "with_same_category.md"), "w").write("\n".join(W) + "\n")

    # tables
    def row(label, f):
        return f"| {label} | " + " | ".join(str(f(V)) for V in (V3, V4a, V4)) + " |"
    head = ["| | 0c v3 strict | v4 (Task 1) | **v4 final** (Task 1 + 2B) |",
            "|---|---:|---:|---:|"]
    T1 = head + [row("unreachable novel predicates", lambda V: len(V["unr"]))]
    T2 = head + [row(k, lambda V, k=k: V["kinds"][k]) for k in ("all-novel", "base-novel", "all-base")] \
        + [row("novel collision groups (all-novel + base-novel)",
               lambda V: V["kinds"]["all-novel"] + V["kinds"]["base-novel"])]
    T3 = head + [row("strict-subset pairs, all", lambda V: len(V["pairs"])),
                 row("novel-subset pairs (subset is novel)", lambda V: len(V["nsub"]))]

    R = ["# Stage 0d — final tables for schema v4\n",
         "## Reachability (LEARNED threshold only, strict — no fallback)\n"] + T1 + [
        "\nUnreachable in v4 final: " + ", ".join(
            f"`{p}` ({'no ingredients' if p in V4['unas'] else ', '.join(fmt(x) + f' ({sup4[x]})' for x in V4['thin'][p])})"
            for p in V4["unr"]) + ".\n",
        "## Identical-phi collisions\n"] + T2 + [
        "\nv4 final groups:\n"] + [
        f"- {sb.kind(g, split)}: " + ", ".join(star(p) for p in g)
        + f" — {{{', '.join(fmt(x) for x in sorted(phi4[g[0]].items()))}}}"
        for g in V4["groups"]] + [
        f"\n**⚠ New vs v3: {len(new_coll)}**" + ("" if not new_coll else ":"),
    ] + [f"- {sb.kind(g, split)}: " + ", ".join(star(p) for p in g) for g in new_coll] + [
        f"\nRemoved vs v3: {len(gone_coll)}" + ("" if not gone_coll else ":")] + [
        f"- {sb.kind(g, split)}: " + ", ".join(star(p) for p in g) for g in gone_coll] + [
        "\n## Strict-subset pairs\n"] + T3 + [
        f"\n**⚠ New novel-subset pairs vs v3: {len(new_ns)}**" + ("" if not new_ns else ":")] + [
        f"- {star(a)} ⊂ {star(b)}" for a, b in new_ns] + [
        f"\nNovel-subset pairs removed vs v3: {len(gone_ns)}" + ("" if not gone_ns else ":")] + [
        f"- {star(a)} ⊂ {star(b)}" for a, b in gone_ns] + [
        f"\nOf the new pairs, {len(returned)} are the ones Task 2 (B) brings back "
        f"(they were removed in 0c by change (a)); the remaining "
        f"{len(set(new_ns) - set(returned))} come from Task 1.\n"]
    open(os.path.join(HERE, "tables_v4.md"), "w").write("\n".join(R) + "\n")

    summary = dict(
        vehicles=VEHICLES, vehicle_counts=dict(veh.most_common()),
        vehicle_subjects_in_other_base_predicates=dict(veh_elsewhere.most_common()),
        vehicle_subject_instances_any_base_predicate=veh_any,
        vehicle_share_by_verb={vb: [verb_veh[vb], verb_n[vb]] for vb in
                               sorted(verb_n, key=lambda v: -verb_veh[v] / verb_n[v])
                               if verb_veh[vb]},
        move_manner_pairs=[len([1 for a, b in versions["v4"]["pairs"]
                                if sb.verb_of(a) == "move" and sb.verb_of(b) in
                                ("walk", "run", "fly", "swim", "creep", "jump")]),
                           len([1 for a, b in versions["v4"]["nsub"]
                                if sb.verb_of(a) == "move" and sb.verb_of(b) in
                                ("walk", "run", "fly", "swim", "creep", "jump")])],
        support={fmt(x): sup4[x] for x in [("subject_kind", "vehicle"), ("subject_kind", "animate"),
                                         ("proximity", "contained"), ("proximity", "overlapping"),
                                         ("proximity", "adjacent"), ("contact", "contact")]},
        ride=dict(n=len(ride_rows), contained_majority_frames=ride_major,
                  contained_median=ride_median),
        task2=dict(outcome=outcome, base_rate=base_rate, pooled={k: list(v) for k, v in pooled.items()},
                   per={p: [sum(s for _, s in inst[p]), len(inst[p])] for ps in fam.values() for p in ps},
                   matched={k: list(v) for k, v in matched.items()}, excluded=excluded),
        tables={n: dict(unreachable=V["unr"], kinds=V["kinds"], pairs=len(V["pairs"]),
                        novel_subset=len(V["nsub"])) for n, V in versions.items()},
        new_collisions=new_coll, removed_collisions=gone_coll,
        new_novel_subset=new_ns, removed_novel_subset=gone_ns, returned_by_task2=returned,
    )
    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
