#!/usr/bin/env python
"""Does H1' reflect motion blindness or label granularity? CPU only.

    python pilot_analysis/scripts/h1_cause.py --preds <dir>

Expectations were pre-registered in report.md before this was run.

ANALYSIS 1 -- error destination. Granularity is a within-group handicap: a
dynamic label losing to its 77 dynamic neighbours. Motion blindness predicts
leakage ACROSS the group boundary, because without movement the evidence left is
the static configuration. So the discriminating statistic is the asymmetry
between dynamic->static and static->dynamic leakage.

ANALYSIS 2 -- granularity regression. Does a group indicator still explain
per-predicate AP once competition is controlled: log(train count) on base,
nearest-neighbour text similarity on novel (novel predicates are masked out of
supervision, so train counts are not a supervision proxy for them).
"""
from __future__ import annotations

import argparse, glob, json, math, os, random, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from inference.video_relation_detection_openvoc import evaluate, viou
from utils import paths

PART = "pilot_analysis/predicate_partition.json"
SEP = "pilot_analysis/clip_text_separability.json"
STATIC_VERBS = {"stand", "sit", "lie", "stop", "next"}


def verb(p):
    return p.split("_", 1)[0] if "_" in p else p


def spatial(p):
    return p.split("_", 1)[1] if "_" in p else None


def is_dyn(p, part):
    return part[p]["time"] == "dynamic"


# ---------------------------------------------------------------- confusions
def collect(segs, gt, part, thresh=0.5):
    """-> list of (gt_pred, top1_pred, matched) for every GT instance whose
    tracklet pair the model actually produced. `matched` means the top-1
    prediction equalled the truth."""
    rows = []
    top1_all = Counter()
    for vid, rels in gt.items():
        pairs = segs.get(vid, [])
        for r in rels:
            b, e = r["duration"]
            s, p, o = r["triplet"]
            best, best_ov = None, 0.0
            for pr in pairs:
                if pr["sbj_cls"] != s or pr["obj_cls"] != o:
                    continue
                ov = min(viou(pr["sbj_traj"], pr["duration"], r["sub_traj"], r["duration"]),
                         viou(pr["obj_traj"], pr["duration"], r["obj_traj"], r["duration"]))
                if ov > best_ov:
                    best, best_ov = pr, ov
            if best is None or best_ov < thresh:
                continue
            votes = Counter()
            for sg in best["segments"]:
                fs, fe = sg["frame_range"]
                if fe <= b or fs >= e or not sg["preds"]:
                    continue
                votes[sg["preds"][0][0]] += 1
            if not votes:
                continue
            top = votes.most_common(1)[0][0]
            rows.append((p, top, top == p))
    # the model's own prediction distribution, over every segment's top-1
    for vid, pairs in segs.items():
        for pr in pairs:
            for sg in pr["segments"]:
                if sg["preds"]:
                    top1_all[sg["preds"][0][0]] += 1
    return rows, top1_all


def analysis1(rows, top1_all, gt, part, label):
    errs = [(g, t) for g, t, ok in rows if not ok]
    dyn_err = [(g, t) for g, t in errs if is_dyn(g, part)]
    sta_err = [(g, t) for g, t in errs if not is_dyn(g, part)]

    def frac_static(pairs):
        return sum(1 for _, t in pairs if not is_dyn(t, part)) / len(pairs) if pairs else float("nan")

    def frac_dyn(pairs):
        return sum(1 for _, t in pairs if is_dyn(t, part)) / len(pairs) if pairs else float("nan")

    # nulls
    n_static_labels = sum(1 for p, d in part.items() if d["time"] == "static")
    null_uniform = n_static_labels / len(part)
    tot = sum(top1_all.values())
    null_model = sum(n for p, n in top1_all.items() if not is_dyn(p, part)) / tot if tot else float("nan")
    gt_c = Counter(r["triplet"][1] for rels in gt.values() for r in rels)
    null_gtfreq = sum(n for p, n in gt_c.items() if not is_dyn(p, part)) / sum(gt_c.values())

    print(f"\n--- {label} " + "-" * (64 - len(label)))
    print(f"  GT instances with a matched pair : {len(rows)}")
    print(f"    of which correct (top-1)       : {sum(1 for _,_,ok in rows if ok)}"
          f"  ({sum(1 for _,_,ok in rows if ok)/max(len(rows),1):.1%})")
    print(f"    errors                         : {len(errs)}"
          f"   (dynamic-GT {len(dyn_err)}, static-GT {len(sta_err)})")
    print()
    print(f"  dynamic-GT errors landing on a STATIC predicate : {frac_static(dyn_err):6.1%}"
          f"   ({sum(1 for _,t in dyn_err if not is_dyn(t,part))}/{len(dyn_err)})")
    print(f"  static-GT errors landing on a DYNAMIC predicate  : {frac_dyn(sta_err):6.1%}"
          f"   ({sum(1 for _,t in sta_err if is_dyn(t,part))}/{len(sta_err)})")
    print()
    print(f"  nulls for 'lands on static':")
    print(f"    uniform over the 132 labels          {null_uniform:6.1%}")
    print(f"    the MODEL's own top-1 distribution   {null_model:6.1%}   <- strictest")
    print(f"    GT-frequency weighted                {null_gtfreq:6.1%}")
    obs = frac_static(dyn_err)
    if obs == obs and null_model == null_model:
        print(f"\n  observed {obs:.1%} vs model null {null_model:.1%}"
              f"  -> ratio {obs/null_model:.2f}x")
    # asymmetry
    a, b = frac_static(dyn_err), frac_dyn(sta_err)
    if a == a and b == b:
        print(f"  ASYMMETRY: dyn->static {a:.1%} vs static->dyn {b:.1%}"
              f"   (difference {a-b:+.1%})")
    # static-counterpart sub-check
    cp = tot_cp = 0
    for g, t in dyn_err:
        if is_dyn(t, part):
            continue
        tot_cp += 1
        if spatial(g) and spatial(t) == spatial(g) and verb(t) in STATIC_VERBS:
            cp += 1
    if tot_cp:
        # chance: among static labels, how many share the GT's spatial term?
        exp = []
        for g, t in dyn_err:
            if is_dyn(t, part) or not spatial(g):
                continue
            share = sum(1 for p, d in part.items()
                        if d["time"] == "static" and spatial(p) == spatial(g))
            exp.append(share / n_static_labels)
        print(f"\n  of the {tot_cp} dyn->static errors, prediction is the STATIC COUNTERPART"
              f" (same spatial term, motion dropped): {cp} ({cp/tot_cp:.1%})")
        print(f"    chance among static labels: {sum(exp)/len(exp):.1%}")
    return dict(n_err=len(errs), dyn_to_static=a, static_to_dyn=b,
                null_model=null_model, counterpart=(cp/tot_cp if tot_cp else None))


# ------------------------------------------------------- per-predicate AP
def per_pred_ap(preds, gt, split, part):
    """AP for one predicate at a time, using the repo's evaluate()."""
    traj = json.load(open(paths.OBJ_SPLIT_INFO))["cls2split"]
    traj_ok = {c for c in traj if c != "__background__"}
    out = {}
    want = [p for p, d in part.items() if d["ov_split"] == split]
    for p in want:
        g = {}
        for v, rels in gt.items():
            k = [r for r in rels if r["triplet"][1] == p
                 and r["triplet"][0] in traj_ok and r["triplet"][2] in traj_ok]
            if k:
                g[v] = k
        if not g:
            continue
        pr = {v: [r for r in rels if r["triplet"][1] == p] for v, rels in preds.items()}
        pr = {v: r for v, r in pr.items() if r}
        # evaluate() iterates GT videos only, so what matters is whether any
        # video that HAS ground truth for this predicate also has a prediction
        # for it. If none does, every det_scores is empty, the concatenation is
        # empty and rec[-1] raises. AP is 0 by definition there: no true
        # positive is reachable.
        if not any(pr.get(v) for v in g):
            out[p] = 0.0
            continue
        try:
            m, _, _ = evaluate(g, pr, viou_threshold=0.5)
            out[p] = float(m)
        except IndexError:
            out[p] = 0.0
    return out


# ------------------------------------------------------------ regression
def ols(X, y):
    """Least squares with an intercept, via normal equations. Returns
    (betas, se, r2). X is a list of feature rows without the intercept."""
    n, k = len(y), len(X[0]) + 1
    A = [[1.0] + list(row) for row in X]
    # normal equations
    XtX = [[sum(A[i][a] * A[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    Xty = [sum(A[i][a] * y[i] for i in range(n)) for a in range(k)]
    # gaussian elimination with partial pivoting
    M = [row[:] + [Xty[i]] for i, row in enumerate(XtX)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            return None
        M[c], M[piv] = M[piv], M[c]
        for r in range(k):
            if r == c:
                continue
            f = M[r][c] / M[c][c]
            for cc in range(c, k + 1):
                M[r][cc] -= f * M[c][cc]
    beta = [M[i][k] / M[i][i] for i in range(k)]
    fit = [sum(beta[j] * A[i][j] for j in range(k)) for i in range(n)]
    ybar = sum(y) / n
    sst = sum((v - ybar) ** 2 for v in y)
    sse = sum((y[i] - fit[i]) ** 2 for i in range(n))
    r2 = 1 - sse / sst if sst else float("nan")
    dof = n - k
    s2 = sse / dof if dof > 0 else float("nan")
    # se from the inverse diagonal, recomputed by solving XtX e_j
    se = []
    for j in range(k):
        M2 = [row[:] for row in XtX]
        rhs = [1.0 if i == j else 0.0 for i in range(k)]
        Mx = [M2[i] + [rhs[i]] for i in range(k)]
        ok = True
        for c in range(k):
            piv = max(range(c, k), key=lambda r: abs(Mx[r][c]))
            if abs(Mx[piv][c]) < 1e-12:
                ok = False; break
            Mx[c], Mx[piv] = Mx[piv], Mx[c]
            for r in range(k):
                if r == c: continue
                f = Mx[r][c] / Mx[c][c]
                for cc in range(c, k + 1):
                    Mx[r][cc] -= f * Mx[c][cc]
        se.append(math.sqrt(s2 * (Mx[j][k] / Mx[j][j])) if ok and s2 == s2 else float("nan"))
    return beta, se, r2


def analysis2(ap, part, control, cname, label, drop_zeros):
    rows = [(p, a) for p, a in ap.items() if p in control]
    if drop_zeros:
        rows = [(p, a) for p, a in rows if a > 0]
    if len(rows) < 8:
        print(f"  {label}: only {len(rows)} predicates -- too few, skipped")
        return None
    y = [a * 100 for _, a in rows]
    ctrl = [control[p] for p, _ in rows]
    grp = [1.0 if part[p]["time"] == "dynamic" else 0.0 for p, _ in rows]
    r_ctrl = ols([[c] for c in ctrl], y)
    r_full = ols([[c, g] for c, g in zip(ctrl, grp)], y)
    if r_ctrl is None or r_full is None:
        print(f"  {label}: singular, skipped"); return None
    (b1, se1, r2_1), (b2, se2, r2_2) = r_ctrl, r_full
    partial = (r2_2 - r2_1) / (1 - r2_1) if r2_1 < 1 else float("nan")
    nd = int(sum(grp)); ns = len(grp) - nd
    print(f"\n  {label}  (n={len(rows)}: {ns} static, {nd} dynamic"
          f"{', zero-AP dropped' if drop_zeros else ', zeros kept'})")
    print(f"    AP ~ {cname:<26} R2 {r2_1:6.3f}")
    print(f"    AP ~ {cname} + group        R2 {r2_2:6.3f}   partial R2 of group {partial:6.3f}")
    print(f"    group coefficient (dynamic)  {b2[2]:+7.2f} mAP pts"
          f"  +-{1.96*se2[2]:5.2f} (95%)  -> "
          f"{'excludes 0' if abs(b2[2]) > 1.96*se2[2] else 'INCLUDES 0'}")
    print(f"    {cname} coefficient          {b2[1]:+7.2f}  +-{1.96*se2[1]:5.2f}")
    return dict(n=len(rows), r2_ctrl=r2_1, r2_full=r2_2, partial=partial,
                group=b2[2], group_ci=1.96*se2[2])


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--preds", default="pilot_analysis/preds_from_pod/full")
    a = ap_.parse_args()
    part = json.load(open(PART))
    gt = json.load(open(paths.TEST_RELATION_GT))

    print("=" * 74)
    print("ANALYSIS 1 -- ERROR DESTINATION")
    print("=" * 74)
    res1 = {}
    for split in ("all", "novel"):
        segs = json.load(open(f"{a.preds}/segments_raw_{split}.json"))
        # restrict GT to the predicates this pass could actually predict
        keep = {p for p, d in part.items() if split == "all" or d["ov_split"] == "novel"}
        g = {v: [r for r in rels if r["triplet"][1] in keep] for v, rels in gt.items()}
        g = {v: r for v, r in g.items() if r}
        rows, top1 = collect(segs, g, part)
        res1[split] = analysis1(rows, top1, g, part,
                                f"'{split}' pass  (GT restricted to {len(keep)} predicates)")

    print("\n" + "=" * 74)
    print("ANALYSIS 2 -- GRANULARITY REGRESSION")
    print("=" * 74)
    tr = Counter()
    for f in glob.glob(os.path.join(paths.ANNO_TRAIN_DIR, "*.json")):
        for r in json.load(open(f))["relation_instances"]:
            tr[r["predicate"]] += 1
    sep = json.load(open(SEP))["novel_within"]

    print("\nBASE split -- control: log(train instance count)")
    preds_all = json.load(open(f"{a.preds}/final_merged_all.json"))
    ap_base = per_pred_ap(preds_all, gt, "base", part)
    logn = {p: math.log(max(tr[p], 1)) for p in ap_base}
    nz = sum(1 for v in ap_base.values() if v == 0)
    print(f"  {len(ap_base)} base predicates scored, {nz} at exactly AP=0")
    for dz in (False, True):
        analysis2(ap_base, part, logn, "log(train count)", "base", dz)

    print("\nNOVEL split -- control: nearest-neighbour text similarity (SS B.11)")
    preds_nov = json.load(open(f"{a.preds}/final_merged_novel.json"))
    ap_nov = per_pred_ap(preds_nov, gt, "novel", part)
    nn = {p: sep[p]["sim"] for p in ap_nov if p in sep}
    nz = sum(1 for v in ap_nov.values() if v == 0)
    print(f"  {len(ap_nov)} novel predicates scored, {nz} at exactly AP=0")
    for dz in (False, True):
        analysis2(ap_nov, part, nn, "NN text similarity", "novel", dz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
