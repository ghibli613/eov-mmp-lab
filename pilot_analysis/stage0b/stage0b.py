#!/usr/bin/env python3
"""Stage 0b -- schema revision, reachability, collisions, reference frame.

    python3 pilot_analysis/stage0b/stage0b.py

Data only: reads the repo's annotations and split file, writes only into
pilot_analysis/stage0b/. Reuses the stage-0 phi rules (imported with bytecode
writing disabled, so nothing lands in stage0/). No checkpoint, no GPU.

Base support counts exclude novel-predicate instances: they are present in the
training annotations but masked from the loss.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import csv, glob, json, math, os
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "pilot_analysis", "stage0"))
import stage0 as s0  # noqa: E402

THIN = 50
RNG = np.random.default_rng(0)
N_BOOT = 1000

# ---------------------------------------------------------------------------
# Task 1 -- revised schema
# ---------------------------------------------------------------------------
SCHEMA = dict(s0.SCHEMA)
SCHEMA["posture"] = ["stand", "sit", "lie"]            # K; unset = absent
SCHEMA["state_change"] = ["halt", "start", "none"]     # L; unset = absent
AXES = list(SCHEMA)

# Task 2 -- MEASURED = computable from box sequences alone; LEARNED = pixels.
TAG = {a: "MEASURED" for a in AXES}
for a in ("locomotion", "contact_action", "posture"):
    TAG[a] = "LEARNED"

POSTURE_VERBS = {"stand", "sit", "lie"}


def verb_of(p):
    t = s0.tokenise(p)
    return t[0] if len(t) == 2 and t[0] in s0.VERB else None


def phi_b(p):
    d = s0.phi_of(p)
    v = verb_of(p)
    if v in POSTURE_VERBS:
        d["posture"] = v
    elif v == "stop":
        # `stop` is re-encoded as the event. Over an interval that contains a
        # halt the subject is both moving and stationary, so subject_state and
        # locomotion are left unset rather than asserting either half.
        d.pop("subject_state", None)
        d.pop("locomotion", None)
        d["state_change"] = "halt"
    return d


UNRESOLVED_B = [
    ("stop_*",
     "state_change=halt; subject_state and locomotion unset",
     "Stage 0 drafted stop as subject_state=stationary + locomotion=none. With a "
     "halt value available, the interval spans both states, so both axes are now "
     "unset. Unverified: whether VidVRD stop_* instances actually contain the "
     "moving->stationary transition inside their temporal extent, or simply "
     "label an already-stopped subject. If the latter, halt is MEASURED-false "
     "on most instances and stop should revert to stationary."),
    ("stand_*, sit_*, lie_*, and all moving verbs",
     "state_change left unset",
     "A sustained posture or gait arguably asserts state_change=none over the "
     "relation's extent. Left unset under the stage-0 convention (silent -> "
     "unset). Consequence: `none` and `start` are assigned to no predicate."),
    ("ride",
     "posture=sit",
     "Riding a bicycle, horse or motorbike is usually seated, but standing on a "
     "skateboard is also annotated as ride in similar datasets. Left unset."),
    ("*_inside",
     "posture as drafted (sit_inside -> sit, etc.)",
     "Posture is lexical here, so no doubt about the value; noted only because "
     "these three were an all-novel collision group in stage 0."),
    ("chase, follow, fall_off",
     "still identical: {subject_state=moving}",
     "Neither new axis touches them. All three are base, so this collision does "
     "not affect novel scoring, but the stage-0 proposals (relative_motion, "
     "depth, object_state) would separate them."),
]

# ---------------------------------------------------------------------------
def load():
    info = json.load(open(os.path.join(ROOT, s0.SPLIT_FILE)))
    split = {p: s for p, s in info["cls2split"].items() if p != "__background__"}
    tr = Counter()
    for f in glob.glob(os.path.join(ROOT, s0.TRAIN_DIR, "*.json")):
        for r in json.load(open(f))["relation_instances"]:
            tr[r["predicate"]] += 1
    return split, tr


def collision_groups(phi, preds):
    g = defaultdict(list)
    for p in preds:
        if phi[p]:
            g[tuple(sorted(phi[p].items()))].append(p)
    return [sorted(v) for v in g.values() if len(v) > 1]


def kind(group, split):
    s = {split[p] for p in group}
    return "all-novel" if s == {"novel"} else "all-base" if s == {"base"} else "base-novel"


# ---------------------------------------------------------------------------
# Task 4 -- reference frame
# ---------------------------------------------------------------------------
def box(b):
    return b["xmin"], b["ymin"], b["xmax"], b["ymax"]


def instance_features(d, r):
    """Per-instance geometry over the relation's own extent, or None."""
    traj = d["trajectories"]
    W, H = d["width"], d["height"]
    diag = math.hypot(W, H)
    s_t, o_t = r["subject_tid"], r["object_tid"]
    rows = []
    for fid in range(r["begin_fid"], min(r["end_fid"], len(traj))):
        bs = bo = None
        for e in traj[fid]:
            if e["tid"] == s_t:
                bs = box(e["bbox"])
            elif e["tid"] == o_t:
                bo = box(e["bbox"])
        if bs and bo:
            rows.append((bs, bo))
    if len(rows) < 2:
        return None

    def ctr(b):
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

    def area(b):
        return max(b[2] - b[0], 1) * max(b[3] - b[1], 1)

    log_area = float(np.mean([math.log(area(s) / area(o)) for s, o in rows]))
    dy_bottom = float(np.mean([(s[3] - o[3]) / H for s, o in rows]))
    dx = float(np.mean([(ctr(s)[0] - ctr(o)[0]) / W for s, o in rows]))

    (s0b, o0b), (s1b, o1b) = rows[0], rows[-1]
    cs0, cs1, co0, co1 = ctr(s0b), ctr(s1b), ctr(o0b), ctr(o1b)
    vs = np.array([cs1[0] - cs0[0], cs1[1] - cs0[1]]) / diag
    vo = np.array([co1[0] - co0[0], co1[1] - co0[1]]) / diag
    so = np.array([np.mean([ctr(o)[0] - ctr(s)[0] for s, o in rows]),
                   np.mean([ctr(o)[1] - ctr(s)[1] for s, o in rows])])
    n = np.linalg.norm(so)
    u = so / n if n > 0 else np.zeros(2)
    return dict(log_area=log_area, dy_bottom=dy_bottom, dx=dx,
                subj_proj=float(vs @ u),          # + : subject heads toward object
                obj_proj=float(vo @ -u),          # + : object heads toward subject
                subj_speed=float(np.linalg.norm(vs)),
                obj_speed=float(np.linalg.norm(vo)))


def auc(x, y):
    """P(score of a positive > score of a negative), ties 0.5. y in {0,1}."""
    x, y = np.asarray(x, float), np.asarray(y, int)
    pos, neg = x[y == 1], x[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(allv))
    r[order] = ranks
    rp = r[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auc_ci(x, y, vids):
    """Video-clustered bootstrap: instances from one video move together."""
    x, y, vids = np.asarray(x, float), np.asarray(y, int), np.asarray(vids)
    uv = np.unique(vids)
    idx = {v: np.where(vids == v)[0] for v in uv}
    bs = []
    for _ in range(N_BOOT):
        pick = np.concatenate([idx[v] for v in RNG.choice(uv, len(uv))])
        a = auc(x[pick], y[pick])
        if a == a:
            bs.append(a)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def reference_frame(split):
    FB = {"front": 1, "behind": 0}
    LR = {"left": 1, "right": 0}
    recs = []
    for f in sorted(glob.glob(os.path.join(ROOT, s0.TRAIN_DIR, "*.json"))):
        d = json.load(open(f))
        for r in d["relation_instances"]:
            p = r["predicate"]
            if split.get(p) != "base":
                continue
            sp = s0.tokenise(p)[-1]
            if sp not in FB and sp not in LR:
                continue
            ft = instance_features(d, r)
            if ft is None:
                continue
            v = verb_of(p)
            state = s0.VERB[v]["subject_state"] if v else "unspecified"
            recs.append(dict(ft, pred=p, spatial=sp, video=d["video_id"],
                             verb_state=state))
    return recs


def eval_features(recs, labels, feats, strata):
    """-> rows of (stratum, feature, hypothesis_sign, n, n_pos, auc, lo, hi)."""
    out = []
    for sname, keep in strata:
        R = [r for r in recs if r["spatial"] in labels and keep(r)]
        if not R:
            continue
        y = [labels[r["spatial"]] for r in R]
        vids = [r["video"] for r in R]
        for fname, key, sign in feats:
            x = [sign * r[key] for r in R]
            a = auc(x, y)
            lo, hi = auc_ci(x, y, vids)
            out.append((sname, fname, len(R), sum(y), a, lo, hi))
    return out


# ---------------------------------------------------------------------------
def main():
    split, tr = load()
    preds = sorted(split)
    base = [p for p in preds if split[p] == "base"]
    novel = [p for p in preds if split[p] == "novel"]
    assert len(preds) == 132 and len(novel) == 61

    phi = {p: phi_b(p) for p in preds}
    for p, d in phi.items():
        for a, v in d.items():
            assert v in SCHEMA[a], (p, a, v)
    out = {p: dict(phi[p], _split=split[p]) for p in preds}
    json.dump(out, open(os.path.join(HERE, "phi_b.json"), "w"), indent=1)

    # ---------------- support ----------------
    sup, nb, masked = Counter(), Counter(), Counter()
    for p in preds:
        for x in phi[p].items():
            if split[p] == "base":
                sup[x] += tr[p]; nb[x] += 1
            else:
                masked[x] += tr[p]
    used = {x for p in preds for x in phi[p].items()}

    # ---------------- Task 2: reachability ----------------
    unassigned = sorted(p for p in novel if not phi[p])
    learned_thin = {}
    for p in novel:
        bad = [x for x in phi[p].items() if TAG[x[0]] == "LEARNED" and sup[x] < THIN]
        if bad:
            learned_thin[p] = bad
    unreachable = sorted(set(unassigned) | set(learned_thin))
    reachable = [p for p in novel if p not in unreachable]
    measured_zero = {p: [x for x in phi[p].items()
                         if TAG[x[0]] == "MEASURED" and sup[x] == 0]
                     for p in reachable}
    measured_zero = {p: v for p, v in measured_zero.items() if v}

    # ---------------- Task 3: collisions ----------------
    phi0 = {p: s0.phi_of(p) for p in preds}
    g0 = collision_groups(phi0, preds)
    g1 = collision_groups(phi, preds)

    def fate(group):
        sub = defaultdict(list)
        for p in group:
            sub[tuple(sorted(phi[p].items()))].append(p)
        return "separated" if all(len(v) == 1 for v in sub.values()) else \
            "still collides: " + "; ".join(",".join(v) for v in sub.values() if len(v) > 1)

    # ---------------- Task 5: subset pairs ----------------
    sets = {p: set(phi[p].items()) for p in preds if phi[p]}
    pairs = [(a, b) for a in sets for b in sets if a != b and sets[a] < sets[b]]
    manner = ["walk", "run", "fly", "swim", "creep", "jump"]
    move_checks = []
    for p in preds:
        if verb_of(p) != "move":
            continue
        sp = s0.tokenise(p)[1]
        for m in manner:
            q = f"{m}_{sp}"
            if q in split:
                move_checks.append((p, q, sets[p] < sets[q]))
    with open(os.path.join(HERE, "subset_pairs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "subset_split", "superset", "superset_split",
                    "attributes_only_in_superset"])
        for a, b in sorted(pairs):
            extra = ";".join(f"{k}={v}" for k, v in sorted(sets[b] - sets[a]))
            w.writerow([a, split[a], b, split[b], extra])

    # ---------------- Task 4: reference frame ----------------
    recs = reference_frame(split)
    MOV = 0.02  # subject displacement over the extent, fraction of image diagonal
    strata = [
        ("all", lambda r: True),
        ("subject verb stationary (stand/sit/lie/stop)",
         lambda r: r["verb_state"] == "stationary"),
        ("subject verb moving", lambda r: r["verb_state"] == "moving"),
        (f"subject measurably moving (>{MOV:.0%} of diagonal)",
         lambda r: r["subj_speed"] > MOV),
        (f"object measurably moving (>{MOV:.0%} of diagonal)",
         lambda r: r["obj_speed"] > MOV),
    ]
    # hypothesis signs: the feature is multiplied by `sign` so that AUC > 0.5
    # means "agrees with the hypothesis that the label is FRONT (resp. LEFT)".
    fb_feats = [
        ("(i) log area ratio subj/obj  [camera: front=larger]", "log_area", +1),
        ("(ii) bottom-edge y diff  [camera: front=lower in image]", "dy_bottom", +1),
        ("(iii) subject velocity . subj->obj  [heading: front=moving away]", "subj_proj", -1),
        ("(iii-b) object velocity . obj->subj  [heading: front=obj approaches]", "obj_proj", +1),
    ]
    lr_feats = [
        ("signed horizontal offset  [camera: left=subject further left]", "dx", -1),
    ]
    fb = eval_features(recs, {"front": 1, "behind": 0}, fb_feats, strata)
    lr = eval_features(recs, {"left": 1, "right": 0}, lr_feats, strata[:3])
    n_fb = sum(1 for r in recs if r["spatial"] in ("front", "behind"))
    n_lr = sum(1 for r in recs if r["spatial"] in ("left", "right"))

    # ---------------- write reports ----------------
    fmt = lambda x: f"{x[0]}={x[1]}"
    L = ["# Stage 0b — coverage with MEASURED / LEARNED tags\n",
         "Base support = BASE-predicate instances in the 800 training videos. "
         "`masked` = novel-predicate instances in those videos, excluded from the "
         "loss, shown for reference only.\n",
         "| axis | tag | value | base train instances | base predicates | masked (not usable) |",
         "|---|---|---|---:|---:|---:|"]
    for a in AXES:
        for v in SCHEMA[a]:
            x = (a, v)
            note = " (never assigned)" if x not in used else (
                " ⚠" if TAG[a] == "LEARNED" and sup[x] < THIN else "")
            L.append(f"| {a} | {TAG[a]} | {v}{note} | {sup[x]} | {nb[x]} | {masked[x]} |")
    L.append(f"\n⚠ = LEARNED value with < {THIN} base instances. MEASURED values are "
             f"not held to the threshold: they are computed from boxes, not learned.\n")
    L.append("## Novel predicates: reachability\n")
    L.append(f"**Unreachable: {len(unreachable)} of 61.**\n")
    L.append("| novel predicate | reason |")
    L.append("|---|---|")
    for p in unreachable:
        why = ("no schema value fits (no ingredients)" if p in unassigned else
               "LEARNED ingredient below threshold: " +
               ", ".join(f"{fmt(x)} ({sup[x]})" for x in learned_thin[p]))
        L.append(f"| `{p}` | {why} |")
    L.append(f"\n**Reachable: {len(reachable)} of 61.** Of these, "
             f"{len(measured_zero)} depend on a MEASURED value that no base "
             f"predicate carries, so no base example exists to validate the "
             f"measurement:\n")
    for p, xs in sorted(measured_zero.items()):
        L.append(f"- `{p}` — " + ", ".join(fmt(x) for x in xs))
    L.append("")
    L.append("## Reachable novel predicates, weakest LEARNED ingredient\n")
    L.append("| novel predicate | weakest LEARNED ingredient | base support | all ingredients |")
    L.append("|---|---|---:|---|")
    rows = []
    for p in reachable:
        le = [x for x in phi[p].items() if TAG[x[0]] == "LEARNED"]
        m = min(le, key=lambda x: sup[x]) if le else None
        rows.append((sup[m] if m else float("inf"), p, m))
    for s, p, m in sorted(rows):
        L.append(f"| `{p}` | {fmt(m) if m else '— (all MEASURED)'} | "
                 f"{s if m else '—'} | " + ", ".join(fmt(x) for x in sorted(phi[p].items())) + " |")
    open(os.path.join(HERE, "coverage_b.md"), "w").write("\n".join(L) + "\n")

    C = ["# Stage 0b — identical-phi collisions\n",
         f"Stage 0: {len(g0)} groups, {sum(map(len, g0))} predicates. "
         f"Stage 0b: {len(g1)} groups, {sum(map(len, g1))} predicates. `*` = novel.\n"]
    for title, gs in (("Stage 0b", g1), ("Stage 0 (for reference)", g0)):
        C.append(f"## {title}\n")
        for k in ("all-novel", "base-novel", "all-base"):
            sel = [g for g in gs if kind(g, split) == k]
            C.append(f"**{k}: {len(sel)}**\n")
            for g in sel:
                line = "- " + ", ".join(f"`{p}`" + ("*" if split[p] == "novel" else "") for p in g)
                if title.startswith("Stage 0 ("):
                    line += f"  →  {fate(g)}"
                C.append(line)
            C.append("")
    open(os.path.join(HERE, "collisions_b.md"), "w").write("\n".join(C) + "\n")

    def table(rows):
        T = ["| stratum | feature | n | n front/left | AUC | 95% CI (video bootstrap) |",
             "|---|---|---:|---:|---:|---|"]
        for s, f_, n, npos, a, lo, hi in rows:
            T.append(f"| {s} | {f_} | {n} | {npos} | {a:.3f} | {lo:.3f}–{hi:.3f} |")
        return T

    Rf = ["# Stage 0b — reference-frame test\n",
          "BASE training instances only. Features are computed over each "
          "instance's own annotated extent from the GT boxes. Each feature is "
          "signed so that **AUC > 0.5 means it agrees with its hypothesis** and "
          "0.5 means no information. CIs resample whole videos.\n",
          f"Heading features use the displacement of the box centre between the "
          f"first and last frames of the extent, projected onto the mean "
          f"subject→object direction.\n",
          f"## front / behind ({n_fb} instances)\n"] + table(fb) + [
          f"\n## left / right ({n_lr} instances)\n"] + table(lr)
    open(os.path.join(HERE, "reference_frame.md"), "w").write("\n".join(Rf) + "\n")

    U = ["# Stage 0b — new judgement calls\n",
         "Stage-0 calls in `../stage0/unresolved.md` still apply unless superseded here.\n",
         "| predicate(s) | as drafted / best guess | reason for doubt |", "|---|---|---|"]
    for who, g, why in UNRESOLVED_B:
        U.append(f"| {who} | {g} | {why} |")
    open(os.path.join(HERE, "unresolved_b.md"), "w").write("\n".join(U) + "\n")

    summary = dict(
        support_posture={v: sup[("posture", v)] for v in SCHEMA["posture"]},
        support_state_change={v: sup[("state_change", v)] for v in SCHEMA["state_change"]},
        masked_posture={v: masked[("posture", v)] for v in SCHEMA["posture"]},
        masked_state_change={v: masked[("state_change", v)] for v in SCHEMA["state_change"]},
        unreachable=unreachable, unassigned=unassigned,
        learned_thin={p: [fmt(x) for x in v] for p, v in learned_thin.items()},
        reachable=len(reachable), measured_zero={p: [fmt(x) for x in v] for p, v in measured_zero.items()},
        collisions0={k: [g for g in g0 if kind(g, split) == k] for k in ("all-novel", "base-novel", "all-base")},
        collisions0_fate={",".join(g): fate(g) for g in g0},
        collisions_b={k: [g for g in g1 if kind(g, split) == k] for k in ("all-novel", "base-novel", "all-base")},
        subset_pairs=len(pairs),
        subset_pairs_novel_involved=sum(1 for a, b in pairs if "novel" in (split[a], split[b])),
        move_checks=[dict(move=a, manner=b, strict_subset=ok) for a, b, ok in move_checks],
        ref_fb=[dict(stratum=s, feature=f_, n=n, n_front=npos, auc=a, lo=lo, hi=hi)
                for s, f_, n, npos, a, lo, hi in fb],
        ref_lr=[dict(stratum=s, feature=f_, n=n, n_left=npos, auc=a, lo=lo, hi=hi)
                for s, f_, n, npos, a, lo, hi in lr],
    )
    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
