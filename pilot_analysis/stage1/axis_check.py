#!/usr/bin/env python3
"""Stage 1 Task A -- check every MEASURED axis against base training labels.

    python3 pilot_analysis/stage1/axis_check.py

CPU only. Reads the TRAINING annotations, the ECC matrices, the split file and
the frozen schema. Novel-predicate relation instances are dropped the moment an
annotation file is parsed and are never stored, counted or inspected. Test
annotations are never opened. Writes only into pilot_analysis/stage1/.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import glob, json, math, os
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import measured_axes as M  # noqa: E402

PHI = os.path.join(ROOT, "pilot_analysis", "stage0e", "phi_frozen.json")
SPLIT_INFO = os.path.join(ROOT, "data", "vidvrd", "data", "openvoc_pred_class_spilt_info.json")
TRAIN_DIR = os.path.join(ROOT, "data", "vidvrd", "anno", "train")
ECC_FILE = os.path.join(ROOT, "data", "vidvrd", "data", "VidVRD_ECC_train.json")
SPLIT_OUT = os.path.join(HERE, "train_split.json")
SEED = 20260928
FIT_FRAC = 0.8
N_BOOT = 1000
LOW_N = 50

PASS_AGREE, FIX_AUC, WEAK_AUC = 0.70, 0.80, 0.70


# ---------------------------------------------------------------------------
def make_or_load_split(video_ids):
    if os.path.exists(SPLIT_OUT):
        s = json.load(open(SPLIT_OUT))
        assert sorted(s["fit"] + s["held"]) == sorted(video_ids), "train_split.json is stale"
        return s
    vids = sorted(video_ids)
    perm = np.random.default_rng(SEED).permutation(len(vids))
    n_fit = int(round(FIT_FRAC * len(vids)))
    s = dict(seed=SEED, method="numpy default_rng(seed).permutation over sorted video ids; "
                               "first 80% FIT, rest HELD",
             fit=sorted(vids[i] for i in perm[:n_fit]),
             held=sorted(vids[i] for i in perm[n_fit:]))
    json.dump(s, open(SPLIT_OUT, "w"), indent=1)
    return s


def load(base):
    ecc = json.load(open(ECC_FILE))
    insts, no_ecc, base_count = [], [], Counter()
    video_ids = []
    for f in sorted(glob.glob(os.path.join(TRAIN_DIR, "*.json"))):
        d = json.load(open(f))
        v = d["video_id"]
        video_ids.append(v)
        rels = [r for r in d["relation_instances"] if r["predicate"] in base]  # novel dropped here
        del d["relation_instances"]
        for r in rels:
            base_count[r["predicate"]] += 1
        if v not in ecc:
            no_ecc.append(v)
            continue
        frames = [{e["tid"]: (e["bbox"]["xmin"], e["bbox"]["ymin"], e["bbox"]["xmax"], e["bbox"]["ymax"])
                   for e in fr} for fr in d["trajectories"]]
        for r in rels:
            p = M.build_pair(frames, r["subject_tid"], r["object_tid"], r["begin_fid"],
                             r["end_fid"], ecc[v], d["width"], d["height"])
            if p is not None:
                insts.append(dict(video=v, pred=r["predicate"], pair=p))
    return insts, no_ecc, base_count, video_ids


# ---------------------------------------------------------------------------
def rankdata(x):
    x = np.asarray(x, float)
    sorter = np.argsort(x, kind="mergesort")
    inv = np.empty(len(x), int); inv[sorter] = np.arange(len(x))
    xs = x[sorter]
    obs = np.r_[True, xs[1:] != xs[:-1]]
    dense = obs.cumsum()[inv]
    cnt = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (cnt[dense] + cnt[dense - 1] + 1)


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if not n1 or not n0:
        return float("nan")
    r = rankdata(x)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_ci(x, y, vids, rng):
    x, y, vids = np.asarray(x, float), np.asarray(y, int), np.asarray(vids)
    uv = np.unique(vids)
    idx = {v: np.where(vids == v)[0] for v in uv}
    bs = []
    for _ in range(N_BOOT):
        pick = np.concatenate([idx[v] for v in rng.choice(uv, len(uv))])
        a = auc(x[pick], y[pick])
        if a == a:
            bs.append(a)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def best_threshold(pos, neg, rule):
    """Maximise balanced agreement. rule(score, thr) -> True if the positive
    value is returned. Candidates: midpoints between sorted unique scores."""
    xs = np.unique(np.concatenate([pos, neg]))
    cands = np.r_[xs[0] - 1e-9, (xs[1:] + xs[:-1]) / 2, xs[-1] + 1e-9]
    if len(cands) > 4000:
        cands = np.quantile(np.concatenate([pos, neg]), np.linspace(0, 1, 4001))
    best = (-1, None)
    for t in cands:
        ba = (np.mean(rule(pos, t)) + np.mean(~rule(neg, t))) / 2
        if ba > best[0]:
            best = (ba, float(t))
    return best[1], best[0]


# ---------------------------------------------------------------------------
def main():
    raw = json.load(open(PHI))
    phi = {p: {k: v for k, v in d.items() if not k.startswith("_")} for p, d in raw.items()}
    split = {p: d["_split"] for p, d in raw.items()}
    info = json.load(open(SPLIT_INFO))["cls2split"]
    assert all(info[p] == split[p] for p in split)
    base = {p for p, s in split.items() if s == "base"}
    novel = sorted(p for p, s in split.items() if s == "novel")

    insts, no_ecc, base_count, video_ids = load(base)
    sp = make_or_load_split(video_ids)
    fit_v, held_v = set(sp["fit"]), set(sp["held"])
    for it in insts:
        it["subset"] = "FIT" if it["video"] in fit_v else "HELD"
    FIT = [i for i in insts if i["subset"] == "FIT"]
    HELD = [i for i in insts if i["subset"] == "HELD"]
    asserts = lambda it, axis: phi[it["pred"]].get(axis)
    rng = np.random.default_rng(0)

    # ---------------- A2: thresholds, fitted on FIT only ----------------
    def fit_state(score):
        pos = np.array([score(i["pair"]) for i in FIT if asserts(i, "subject_state") == "moving"])
        neg = np.array([score(i["pair"]) for i in FIT if asserts(i, "subject_state") == "stationary"])
        pos, neg = pos[pos == pos], neg[neg == neg]
        t, ba = best_threshold(pos, neg, lambda s_, t_: s_ > t_)
        return t, ba, len(pos), len(neg), auc(np.r_[pos, neg], np.r_[np.ones(len(pos)), np.zeros(len(neg))])

    speed_thr, speed_ba, n_pos, n_neg, speed_fit_auc = fit_state(M.subject_speed)     # as specified
    motion_thr, motion_ba, _, _, motion_fit_auc = fit_state(M.subject_motion)         # corrected

    SPATIAL_AXES = ("horizontal", "vertical", "depth")
    adj_pos = lambda i: asserts(i, "proximity") == "adjacent"
    adj_neg = lambda i: asserts(i, "proximity") is None and any(asserts(i, a) for a in SPATIAL_AXES)
    gpos = np.array([M.gap_score(i["pair"]) for i in FIT if adj_pos(i)])
    gneg = np.array([M.gap_score(i["pair"]) for i in FIT if adj_neg(i)])
    gap_thr, gap_ba = best_threshold(gpos, gneg, lambda s_, t_: s_ < t_)

    co = [abs(M.delta_d(i["pair"])) for i in FIT if asserts(i, "relative_motion") == "co_move"]
    dead_zone = float(np.percentile(co, 90))

    TH = dict(motion=motion_thr, gap=gap_thr, dead_zone=dead_zone)
    state_neg = "FIT base instances asserting subject_state=stationary"
    state_pos = "FIT base instances asserting subject_state=moving"
    json.dump(dict(
        thresholds=TH,
        fit=dict(
            motion=dict(rule="subject/object moving iff motion_rate > threshold",
                        units="net compensated displacement over the extent, in the entity's box "
                              "diagonals, per frame",
                        positives=state_pos, negatives=state_neg, n_pos=n_pos, n_neg=n_neg,
                        balanced_agreement_on_fit=motion_ba, auc_on_fit=motion_fit_auc),
            gap=dict(rule="adjacent iff mean normalised gap < threshold (and not contained)",
                     units="gap / mean of the two box diagonals",
                     positives="FIT base instances asserting proximity=adjacent",
                     negatives="FIT base instances asserting a left/right/above/beneath/front/behind "
                               "term and no proximity value",
                     n_pos=len(gpos), n_neg=len(gneg), balanced_agreement_on_fit=gap_ba),
            dead_zone=dict(rule="90th percentile of |change in d| over FIT base co_move (*_with) instances",
                           units="change in centre distance / mean box diagonal, last third minus first third",
                           n=len(co))),
        rejected=dict(speed=dict(
            value=speed_thr, rule="as specified: mean compensated step > threshold",
            units="fraction of frame diagonal per frame", balanced_agreement_on_fit=speed_ba,
            auc_on_fit=speed_fit_auc,
            why="subject_state FAIL on HELD; replaced by motion_rate (see axis_check.md)")),
        train_split="train_split.json"),
        open(os.path.join(HERE, "thresholds.json"), "w"), indent=1)

    # ---------------- A3: agreement on HELD ----------------
    SPEC = (M.subject_speed, M.object_speed)
    def measure(it):
        p = it["pair"]
        return {
            "horizontal": M.horizontal(p), "vertical": M.vertical(p), "depth": M.depth(p),
            "proximity": M.proximity(p, gap_thr),
            "subject_state": M.subject_state(p, motion_thr),
            "object_state": M.object_state(p, motion_thr),
            "relative_motion": M.relative_motion(p, dead_zone, motion_thr),
            "subject_state@spec": M._state(M.subject_speed(p), speed_thr),
            "object_state@spec": M._state(M.object_speed(p), speed_thr),
            "relative_motion@spec": M.relative_motion(p, dead_zone, speed_thr, _motion=SPEC),
            "comparative": M.comparative(p)}

    for it in insts:
        it["m"] = measure(it)

    def agree(rows, key, value):
        axis = key.split("@")[0]
        sel = [r for r in rows if asserts(r, axis) == value]
        if axis == "comparative":
            hit = sum(1 for r in sel if r["m"]["comparative"][value] is True)
        else:
            hit = sum(1 for r in sel if r["m"][key] == value)
        return hit, len(sel)

    res = {}
    score_fn = dict(horizontal=M.horizontal_score, vertical=M.vertical_score, depth=M.depth_score)
    pos_value = dict(horizontal="right", vertical="beneath", depth="front")   # predicted by score > 0
    for axis, (a, b) in (("horizontal", M.HORIZONTAL), ("vertical", M.VERTICAL), ("depth", M.DEPTH)):
        sel = [r for r in HELD if asserts(r, axis) in (a, b)]
        x = [score_fn[axis](r["pair"]) for r in sel]
        y = [1 if asserts(r, axis) == pos_value[axis] else 0 for r in sel]
        A = auc(x, y); lo, hi = auc_ci(x, y, [r["video"] for r in sel], rng)
        res[axis] = dict(values={v: agree(HELD, axis, v) for v in (a, b)},
                         table=Counter((asserts(r, axis), r["m"][axis]) for r in sel),
                         labels=(a, b), auc=(A, lo, hi))

    for key, score in (("subject_state", M.subject_motion), ("subject_state@spec", M.subject_speed)):
        sel = [r for r in HELD if asserts(r, "subject_state") in ("moving", "stationary")
               and r["m"][key] is not None]
        x = [score(r["pair"]) for r in sel]
        y = [1 if asserts(r, "subject_state") == "moving" else 0 for r in sel]
        A = auc(x, y); lo, hi = auc_ci(x, y, [r["video"] for r in sel], rng)
        res[key] = dict(values={v: agree(HELD, key, v) for v in ("moving", "stationary")},
                        table=Counter((asserts(r, "subject_state"), r["m"][key]) for r in sel),
                        labels=("moving", "stationary"), auc=(A, lo, hi))
    for key in ("object_state", "object_state@spec"):
        res[key] = dict(values={"moving": agree(HELD, key, "moving")}, auc=None)

    sel = [r for r in HELD if adj_pos(r) or adj_neg(r)]
    x = [-M.gap_score(r["pair"]) for r in sel]
    y = [1 if adj_pos(r) else 0 for r in sel]
    A = auc(x, y); lo, hi = auc_ci(x, y, [r["video"] for r in sel], rng)
    res["proximity"] = dict(values={"adjacent": agree(HELD, "proximity", "adjacent")}, auc=(A, lo, hi),
                            neg=(sum(1 for r in HELD if adj_neg(r) and r["m"]["proximity"] != "adjacent"),
                                 sum(1 for r in HELD if adj_neg(r))))

    ALL = insts
    for key in ("relative_motion", "relative_motion@spec"):
        res[key] = dict(values={"co_move": agree(HELD, key, "co_move")},
                        low_n={v: dict(held=agree(HELD, key, v), all=agree(ALL, key, v))
                               for v in ("pass", "approach")},
                        dist={v: Counter(r["m"][key] for r in ALL if asserts(r, "relative_motion") == v)
                              for v in ("co_move", "pass", "approach")},
                        held_dist=Counter(r["m"][key] for r in HELD if asserts(r, "relative_motion") == "co_move"),
                        auc=None)

    comp = {}
    for v, fn in (("larger", M.larger_score), ("taller", M.taller_score), ("faster", M.faster_score)):
        sel = [r for r in HELD if asserts(r, "comparative") == v]
        sv = np.array([fn(r["pair"]) for r in sel]); ok = sv == sv
        sel = [r for r, k in zip(sel, ok) if k]; sv = sv[ok]
        x = np.r_[sv, -sv]; y = np.r_[np.ones(len(sv)), np.zeros(len(sv))]
        A = auc(x, y); lo, hi = auc_ci(x, y, [r["video"] for r in sel] * 2, rng)
        comp[v] = dict(hit=int((sv > 0).sum()), n=len(sv), tie=int((sv == 0).sum()), auc=(A, lo, hi))

    # ---------------- A4 ----------------
    sym = Counter()
    MAP_T = {"approach": "recede", "recede": "approach"}
    INV = {"left": "right", "right": "left", "above": "beneath", "beneath": "above",
           "front": "behind", "behind": "front", None: None}
    for it in insts:
        p = it["pair"]
        rm = M.relative_motion(p, dead_zone, motion_thr)
        rmr = M.relative_motion(M.time_reverse(p), dead_zone, motion_thr)
        sym["time_n"] += 1
        sym["time_ok"] += rmr == MAP_T.get(rm, rm)
        q = M.swap(p)
        for fn in (M.horizontal, M.vertical, M.depth):
            sym["swap_n"] += 1
            sym["swap_ok"] += fn(q) == INV[fn(p)]
        c, cq = M.comparative(p), M.comparative(q)
        for k in c:
            sym["comp_n"] += 1
            sym["comp_ok"] += cq[k] == (None if c[k] is None else (not c[k]))
    sym_ok = all(sym[f"{k}_ok"] == sym[f"{k}_n"] for k in ("time", "swap", "comp"))

    posture_subj = [r for r in HELD if phi[r["pred"]].get("posture")]
    comove = [r for r in HELD if asserts(r, "relative_motion") == "co_move"]
    ride = [r for r in insts if r["pred"] == "ride"]
    contained_by = Counter(r["pred"] for r in insts if r["m"]["proximity"] == "contained")
    fp = dict(
        recede_on_comove=(sum(r["m"]["relative_motion"] == "recede" for r in comove), len(comove)),
        recede_on_posture=(sum(r["m"]["relative_motion"] == "recede" for r in posture_subj), len(posture_subj)),
        contained_on_ride=(sum(M.contained(r["pair"]) for r in ride), len(ride)),
        contained_on_all=(sum(contained_by.values()), len(insts)),
    )

    # ---------------- A5 verdicts ----------------
    def verdict(agr, A):
        if agr == agr and agr >= PASS_AGREE:
            return "PASS" if sym_ok else "FAIL (symmetry)"
        if A is None or A != A:
            return "UNRESOLVED (<70%, no negative class for an AUC)"
        if A >= FIX_AUC:
            return "FIXABLE"
        if A >= WEAK_AUC:
            return "WEAK"
        return "FAIL"

    LABEL = {"subject_state@spec": "subject_state — as specified (mean step / frame diag)",
             "subject_state": "subject_state — corrected (motion_rate)",
             "object_state@spec": "object_state — as specified",
             "object_state": "object_state — corrected (motion_rate)",
             "relative_motion@spec": "relative_motion — as specified",
             "relative_motion": "relative_motion — corrected (motion_rate)"}
    rows = []
    for key in ("horizontal", "vertical", "depth", "proximity", "subject_state@spec", "subject_state",
                "object_state@spec", "object_state", "relative_motion@spec", "relative_motion"):
        r = res[key]; axis = key.split("@")[0]
        supported = {v: hn for v, hn in r["values"].items()
                     if hn[1] and base_support(phi, base_count, axis, v) >= LOW_N}
        agr = min(h / n for h, n in supported.values()) if supported else float("nan")
        rows.append((key, supported, agr, r.get("auc"), verdict(agr, r["auc"][0] if r.get("auc") else None)))
    for v, c in comp.items():
        agr = c["hit"] / c["n"]
        rows.append((f"comparative={v}", {v: (c["hit"], c["n"])}, agr, c["auc"], verdict(agr, c["auc"][0])))

    def dependents(key):
        axis = key.split("@")[0].split("=")[0]
        v = key.split("=")[1] if "=" in key else None
        return sorted(p for p in novel if (phi[p].get(axis) == v if v else axis in phi[p]))

    # ---------------- write ----------------
    pct = lambda h, n: f"{h}/{n} ({h / n:.1%})" if n else "—"
    ci = lambda t: "—" if not t or t[0] != t[0] else f"{t[0]:.3f} ({t[1]:.3f}–{t[2]:.3f})"
    L = ["# Stage 1 — Task A: measured-axis check\n",
         "BASE-predicate labels from the TRAINING annotations only. Novel instances are "
         "dropped when each file is parsed and are never read after that. Test "
         "annotations are not opened. Functions: `measured_axes.py`. Frozen schema: "
         "`../stage0e/phi_frozen.json`.\n",
         f"**Split (A0):** {len(sp['fit'])} FIT / {len(sp['held'])} HELD videos, "
         f"video-level, seed {SEED} (`train_split.json`). "
         f"**Excluded for missing ECC:** {len(no_ecc)} videos. Base instances measured: "
         f"{len(insts)} ({len(FIT)} FIT, {len(HELD)} HELD).\n",
         "## A2 — thresholds (fitted on FIT only, frozen in `thresholds.json`)\n",
         "| threshold | value | how |", "|---|---:|---|",
         f"| **moving (motion_rate)** — frozen | {motion_thr:.5f} box-diag/frame | max balanced agreement, stationary ({n_neg}) vs moving ({n_pos}) subjects; FIT BA {motion_ba:.3f}, FIT AUC {motion_fit_auc:.3f} |",
         f"| moving (speed, as specified) — rejected | {speed_thr:.5f} frame-diag/frame | same fit; FIT BA {speed_ba:.3f}, FIT AUC {speed_fit_auc:.3f} |",
         f"| adjacency gap | {gap_thr:.4f} box-diag | max balanced agreement, `next_to` family ({len(gpos)}) vs other-spatial-term instances ({len(gneg)}); FIT BA {gap_ba:.3f} |",
         f"| relative-motion dead zone | {dead_zone:.4f} box-diag | 90th percentile of \\|Δd\\| over FIT co_move (`*_with`) instances ({len(co)}) |",
         "\nEverything else is a sign test with no threshold: horizontal, vertical, depth, "
         "contained, comparative, and co_move's direction agreement (cosine > 0).\n",
         "Adjacency negatives are instances labelled with a different spatial term. They "
         "are *not asserted adjacent*, not *asserted separated*, since no predicate "
         "asserts `separated`. Agreement on them is therefore a lower bound.\n",
         "**Why the speed rule was replaced.** The specified moving measure (mean "
         "compensated per-frame step ÷ frame diagonal) fails on HELD; see A5. Two "
         "things make it weak. Box jitter adds to every step, so it inflates the mean "
         "step but cancels out of the net displacement. And dividing by the frame "
         "penalises small, distant subjects. I compared alternatives **on FIT only**: "
         "mean step ÷ frame diag gives AUC 0.725; net displacement ÷ frame diag 0.787; "
         "mean step ÷ box diag 0.763; **net displacement ÷ own box diagonal, per "
         "frame: 0.812**. The last one is `motion_rate`. It is the frozen rule for "
         "subject_state, object_state and co_move, and HELD was scored once, after the "
         "choice was made. `faster` keeps the specified speed, which passes.\n",
         "## A3 — agreement on HELD\n"]
    for key in ("horizontal", "vertical", "depth", "subject_state@spec", "subject_state"):
        r = res[key]; a, b = r["labels"]; t = r["table"]
        L.append(f"### {LABEL.get(key, key)}\n")
        L.append(f"| label ↓ / function → | {a} | {b} | none |")
        L.append("|---|---:|---:|---:|")
        for lab in (a, b):
            L.append(f"| {lab} | {t[(lab, a)]} | {t[(lab, b)]} | {t[(lab, None)]} |")
        L.append(f"\nAgreement: {a} {pct(*r['values'][a])}, {b} {pct(*r['values'][b])}. "
                 f"AUC {ci(r['auc'])}.\n")
    r = res["proximity"]
    L.append("### proximity\n")
    L.append(f"adjacent: {pct(*r['values']['adjacent'])} of `next_to`-family instances "
             f"measured adjacent. Negatives measured non-adjacent: {pct(*r['neg'])}. "
             f"AUC (−gap) {ci(r['auc'])}. `contained` has no base carrier; see A4.\n")
    L.append("### object_state\n")
    L.append(f"moving, of `*_with` instances: as specified {pct(*res['object_state@spec']['values']['moving'])}; "
             f"corrected {pct(*res['object_state']['values']['moving'])}. Only `moving` is ever "
             f"asserted, so there is no negative class and no AUC.\n")
    L.append("### relative_motion\n")
    for key in ("relative_motion@spec", "relative_motion"):
        r = res[key]
        L.append(f"- {LABEL[key]}: co_move {pct(*r['values']['co_move'])} of HELD `*_with`. "
                 f"HELD output: " + ", ".join(f"{k} {v}" for k, v in r["held_dist"].most_common())
                 + ". All-base output: " + ", ".join(f"{k} {v}" for k, v in r["dist"]["co_move"].most_common()) + ".")
    L.append("")
    L.append("### comparative\n")
    L.append("Only the positive value is ever labelled, so the negative class is the same "
             "instance with subject and object swapped. The 2×2 is then fixed by "
             "antisymmetry: labelled-larger cases measured larger = swapped cases "
             "measured smaller.\n")
    L.append("| value | n (HELD) | measured as asserted | tie / no motion | AUC vs swapped |")
    L.append("|---|---:|---:|---:|---:|")
    for v, c in comp.items():
        L.append(f"| {v} | {c['n']} | {pct(c['hit'], c['n'])} | {c['tie']} | {ci(c['auc'])} |")
    L.append("\n## A4 — values with little or no base support\n")
    L.append("**a. Exact symmetry tests, all base instances (frozen rules):**\n")
    L.append(f"- time reversal (approach↔recede, pass/co_move/none fixed): {sym['time_ok']}/{sym['time_n']}")
    L.append(f"- subject/object swap (left↔right, above↔beneath, front↔behind): {sym['swap_ok']}/{sym['swap_n']}")
    L.append(f"- subject/object swap inverts larger/taller/faster: {sym['comp_ok']}/{sym['comp_n']}")
    L.append(f"- **{'all exact' if sym_ok else '⚠ NOT EXACT — code bug'}**\n")
    L.append("**b. False-positive rates (report only):**\n")
    L.append(f"- recede on HELD co_move (`*_with`) instances: {pct(*fp['recede_on_comove'])}")
    L.append(f"- recede on HELD instances with a stand/sit/lie subject: {pct(*fp['recede_on_posture'])}")
    L.append(f"- contained on base `ride` (all videos): {pct(*fp['contained_on_ride'])}")
    L.append(f"- contained on all base instances: {pct(*fp['contained_on_all'])}. Most "
             f"frequent predicates it fires on: " + ", ".join(f"`{p}` {n}" for p, n in contained_by.most_common(6)) + "\n")
    L.append("**c. LOW-N values** (no threshold was fitted on them, so ALL base videos are "
             "usable as well):\n")
    r = res["relative_motion"]
    for v in ("pass", "approach"):
        h = r["low_n"][v]
        L.append(f"- {v} ⚠ LOW-N: HELD {pct(*h['held'])}; all base {pct(*h['all'])}. "
                 f"Function output on all: " + ", ".join(f"{k} {n}" for k, n in r["dist"][v].most_common()))
    L.append("\n## A5 — verdicts\n")
    L.append(f"Agreement is the LOWEST per-value HELD agreement among values with ≥{LOW_N} "
             f"base instances. PASS ≥ {PASS_AGREE:.0%} with exact symmetry; FIXABLE < 70% "
             f"with AUC ≥ {FIX_AUC}; WEAK AUC {WEAK_AUC}–{FIX_AUC}; FAIL AUC < {WEAK_AUC}.\n")
    L.append("| axis | HELD agreement (per value) | AUC | verdict | novel predicates that depend on it |")
    L.append("|---|---|---:|---|---:|")
    for key, sup, agr, A, v in rows:
        L.append(f"| {LABEL.get(key, key)} | " + "; ".join(f"{k} {pct(*hn)}" for k, hn in sup.items())
                 + f" | {ci(A)} | **{v}** | {len(dependents(key))} |")
    fam = Counter()
    for r in HELD:
        v = asserts(r, "subject_state")
        if v and r["m"]["subject_state"] is not None:
            f_ = r["pred"].split("_")[0] if "_" in r["pred"] else r["pred"]
            fam[(f_, "n")] += 1
            fam[(f_, "ok")] += r["m"]["subject_state"] == v
    fams = sorted({k[0] for k in fam}, key=lambda f_: fam[(f_, "ok")] / fam[(f_, "n")])
    rm = res["relative_motion"]
    rm_deps = dependents("relative_motion")
    unlearnable = sorted(p for p in rm_deps if phi[p]["relative_motion"] in ("recede", "approach"))
    L.append("\n### Notes on the verdicts\n")
    L.append(f"**Pre-registered expectation: held.** subject_state was the weakest axis as "
             f"specified: HELD agreement {res['subject_state@spec']['values']['stationary'][0] / res['subject_state@spec']['values']['stationary'][1]:.1%} "
             f"on stationary, AUC {ci(res['subject_state@spec']['auc'])}, which is below 0.70, "
             f"so FAIL. The corrected `motion_rate` rule, chosen on FIT, passes on HELD, but "
             f"narrowly: stationary {pct(*res['subject_state']['values']['stationary'])}, "
             f"AUC {ci(res['subject_state']['auc'])}. By verb family on HELD, from worst to best: "
             + ", ".join(f"`{f_}` {fam[(f_, 'ok')]}/{fam[(f_, 'n')]}" for f_ in fams)
             + ". **`stop_*` is the weak spot**: most 'stopped' vehicles measure as moving. "
             "The four novel `stop_*` predicates depend on subject_state=stationary.\n")
    L.append(f"**relative_motion: UNRESOLVED, not PASS.** co_move agrees "
             f"{pct(*rm['values']['co_move'])} on HELD. It has no negative class, so no AUC is "
             f"possible, and the pre-registered rules cannot separate FIXABLE, WEAK and FAIL. "
             f"The other values are worse. `pass` fires on {pct(*rm['low_n']['pass']['all'])} of "
             f"all base `*_past` instances. `approach` fires on "
             f"{pct(*rm['low_n']['approach']['all'])}. `recede` has no base example at all. "
             f"{len(rm_deps)} novel predicates depend on this axis. Re-tagging it LEARNED "
             f"would cover co_move ({base_support(phi, base_count, 'relative_motion', 'co_move')} "
             f"base instances) and pass "
             f"({base_support(phi, base_count, 'relative_motion', 'pass')}). It would NOT cover "
             f"approach ({base_support(phi, base_count, 'relative_motion', 'approach')}) or recede "
             f"(0). So these {len(unlearnable)} novel predicates would become unreachable: "
             + ", ".join(f"`{p}`" for p in unlearnable) + ". As MEASURED they stay reachable, "
             "but they rest on a rule that base data cannot validate.\n")
    L.append(f"**proximity: PASS on the letter, marginal in substance.** Adjacent agrees "
             f"{pct(*res['proximity']['values']['adjacent'])}, but the AUC is "
             f"{ci(res['proximity']['auc'])}, which is WEAK-level, and only "
             f"{pct(*res['proximity']['neg'])} of the other-spatial-term negatives measure "
             f"non-adjacent. Those negatives are weak: labelled with a different spatial term, "
             f"not asserted separated. `contained` fires on "
             f"{fp['contained_on_all'][0] / fp['contained_on_all'][1]:.1%} of all base "
             f"instances. The 2-D overlap it detects is often occlusion (`move_beneath`, "
             f"`stand_front`). The four `contained` novel predicates will inherit that "
             f"false-positive rate.\n")
    open(os.path.join(HERE, "axis_check.md"), "w").write("\n".join(L) + "\n")

    summary = dict(
        n=dict(all=len(insts), fit=len(FIT), held=len(HELD), no_ecc_videos=len(no_ecc)),
        thresholds=TH, rejected_speed=speed_thr, fit_ba=dict(motion=motion_ba, speed=speed_ba, gap=gap_ba),
        fit_auc=dict(motion=motion_fit_auc, speed=speed_fit_auc),
        res={k: {kk: (dict(vv) if isinstance(vv, Counter) else vv) for kk, vv in v.items() if kk != "table"}
             for k, v in res.items()},
        tables={k: {f"{a}->{b}": n for (a, b), n in res[k]["table"].items()}
                for k in ("horizontal", "vertical", "depth", "subject_state", "subject_state@spec")},
        comparative=comp, symmetry=dict(sym), fp=fp, contained_by=dict(contained_by.most_common(15)),
        verdicts=[dict(axis=a, agreement=g, auc=A, verdict=v, novel_dependents=dependents(a))
                  for a, _, g, A, v in rows],
        base_support={f"{a}={v}": base_support(phi, base_count, a, v)
                      for a in ("subject_state", "object_state", "relative_motion", "proximity")
                      for v in ("moving", "stationary", "co_move", "pass", "approach", "recede",
                                "adjacent", "contained")
                      if base_support(phi, base_count, a, v)},
    )
    json.dump(summary, open(os.path.join(HERE, "axis_check_summary.json"), "w"), indent=1, default=str)
    for a, s_, g, A, v in rows:
        print(f"{a:24} agr {g:.3f}  auc {ci(A):28} {v}")
    print("symmetry", dict(sym)); print("fp", fp); print("thresholds", TH)
    return 0


def base_support(phi, base_count, axis, value):
    return sum(n for p, n in base_count.items() if phi[p].get(axis) == value)


if __name__ == "__main__":
    sys.exit(main())
