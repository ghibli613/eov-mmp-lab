#!/usr/bin/env python3
"""Missed novel-split GT: did the model predict the mirror on the reversed pair?

    python3 pilot_analysis/mirror_misses/mirror_misses.py

CPU only. Reads the EXISTING prediction dumps (pilot_analysis/preds_from_pod/full)
and the TEST relation GT -- unlike stage 0, this is an error analysis of the
evaluated model and needs test data by definition. Writes only into
pilot_analysis/mirror_misses/.

"Missed" replicates the official matcher (inference/video_relation_detection_openvoc.py
eval_detection_scores): greedy by score, exact triplet match, min(subject vIoU,
object vIoU) >= 0.5. Reported at two levels: missed by every dumped prediction
(<= 200 per video, the mAP level) and missed within each video's top 100 (R@100).
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import json, os
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from inference.video_relation_detection_openvoc import viou  # noqa: E402

PREDS = os.path.join(ROOT, "pilot_analysis", "preds_from_pod", "full")
GT = os.path.join(ROOT, "data", "vidvrd", "data", "test_relation_gt.json")
SPLIT = os.path.join(ROOT, "data", "vidvrd", "data", "openvoc_pred_class_spilt_info.json")
THR = 0.5
N_BOOT = 1000
RNG = np.random.default_rng(0)
SWAP = {"left": "right", "right": "left", "above": "beneath", "beneath": "above",
        "front": "behind", "behind": "front"}


def mirror(p, vocab):
    t = p.split("_")
    s = SWAP.get(t[-1])
    q = "_".join(t[:-1] + [s]) if s else None
    return q if q in vocab else None


def spatial_mirror_term(p):
    return SWAP.get(p.split("_")[-1])


def temporal_overlap(d1, d2):
    return max(d1[0], d2[0]) < min(d1[1], d2[1])


def match_flags(gts, preds):
    """Official greedy matcher. -> for each GT, the 0-based score rank of the
    prediction that detected it, or None."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i]["score"])
    hit_rank = [None] * len(gts)
    for rank, i in enumerate(order):
        pr = preds[i]
        best, k = -1.0, -1
        for g, gt in enumerate(gts):
            if hit_rank[g] is not None or tuple(pr["triplet"]) != tuple(gt["triplet"]):
                continue
            if not temporal_overlap(pr["duration"], gt["duration"]):
                continue
            ov = min(viou(pr["sub_traj"], pr["duration"], gt["sub_traj"], gt["duration"]),
                     viou(pr["obj_traj"], pr["duration"], gt["obj_traj"], gt["duration"]))
            if ov >= THR and ov > best:
                best, k = ov, g
        if k >= 0:
            hit_rank[k] = rank
    return hit_rank


def reversed_preds(gt, preds):
    """Predicates the model put on the reversed pair (GT object as subject, GT
    subject as object): trajectories match at vIoU >= 0.5 on both roles, which
    also guarantees an overlapping extent. Categories are not required to match."""
    out = set()
    for pr in preds:
        if not temporal_overlap(pr["duration"], gt["duration"]):
            continue
        s = viou(pr["sub_traj"], pr["duration"], gt["obj_traj"], gt["duration"])
        if s < THR:
            continue
        o = viou(pr["obj_traj"], pr["duration"], gt["sub_traj"], gt["duration"])
        if o >= THR:
            out.add(pr["triplet"][1])
    return out


def forward_preds(gt, preds):
    """Predicates on the forward pair (same roles as the GT), any category."""
    out = set()
    for pr in preds:
        if not temporal_overlap(pr["duration"], gt["duration"]):
            continue
        if viou(pr["sub_traj"], pr["duration"], gt["sub_traj"], gt["duration"]) < THR:
            continue
        if viou(pr["obj_traj"], pr["duration"], gt["obj_traj"], gt["duration"]) >= THR:
            out.add(pr["triplet"][1])
    return out


def boot(xs, vids):
    xs, vids = np.asarray(xs, float), np.asarray(vids)
    uv = np.unique(vids)
    idx = {v: np.where(vids == v)[0] for v in uv}
    b = [xs[np.concatenate([idx[v] for v in RNG.choice(uv, len(uv))])].mean()
         for _ in range(N_BOOT)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def analyse(pass_name, split):
    preds = json.load(open(os.path.join(PREDS, f"final_merged_{pass_name}.json")))
    gt_all = json.load(open(GT))
    vocab = set(split)
    novel = {p for p, s in split.items() if s == "novel"}
    cand = sorted(novel) if pass_name == "novel" else sorted(vocab)
    # the model's own marginal over predicates in this pass (for the weighted control)
    marg = Counter(r["triplet"][1] for rs in preds.values() for r in rs)

    rows = []
    for vid, rels in gt_all.items():
        gts = [r for r in rels if r["triplet"][1] in novel]     # novel-split GT
        if not gts:
            continue
        P = preds.get(vid, [])
        ranks = match_flags(gts, P)
        for gt, rk in zip(gts, ranks):
            p = gt["triplet"][1]
            q = mirror(p, vocab)
            if q is None:
                continue
            R = reversed_preds(gt, P)
            Fw = forward_preds(gt, P)
            others = [c for c in cand if c != q]
            mt = spatial_mirror_term(p)
            rows.append(dict(
                video=vid, pred=p, mirror=q, mirror_split=split[q],
                missed_all=rk is None, missed_100=rk is None or rk >= 100,
                mirror_hit=q in R,
                spatial_mirror_hit=any(c.split("_")[-1] == mt for c in R),
                any_rev=bool(R), any_fwd=bool(Fw), neither=not R and not Fw,
                fwd_has_p=p in Fw,
                ctrl_uniform=sum(c in R for c in others) / len(others),
                ctrl_marginal=(sum(marg[c] for c in others if c in R) /
                               max(sum(marg[c] for c in others), 1)),
                mirror_predictable=q in cand))
    return rows


def summarise(rows, label):
    out = {}
    n = len(rows)
    if not n:
        return dict(label=label, n=0)
    v = [r["video"] for r in rows]
    for k in ("mirror_hit", "spatial_mirror_hit", "any_rev", "ctrl_uniform", "ctrl_marginal",
              "any_fwd", "neither", "fwd_has_p"):
        xs = [float(r[k]) for r in rows]
        lo, hi = boot(xs, v)
        out[k] = (float(np.mean(xs)), lo, hi)
    cov = [r for r in rows if r["any_rev"]]
    out["cond_n"] = len(cov)
    out["cond_mirror"] = sum(r["mirror_hit"] for r in cov)
    out["cond_ctrl_uniform"] = float(np.mean([r["ctrl_uniform"] for r in cov])) if cov else float("nan")
    out["cond_ctrl_marginal"] = float(np.mean([r["ctrl_marginal"] for r in cov])) if cov else float("nan")
    out["n"] = n
    out["label"] = label
    return out


def main():
    info = json.load(open(SPLIT))["cls2split"]
    split = {p: s for p, s in info.items() if p != "__background__"}

    res = {}
    for pass_name in ("novel", "all"):
        rows = analyse(pass_name, split)
        pred_rows = [r for r in rows if r["mirror_predictable"]]
        res[pass_name] = dict(
            rows=rows,
            n_gt=len(rows),
            n_unpredictable=sum(1 for r in rows if not r["mirror_predictable"]),
            missed_all=summarise([r for r in pred_rows if r["missed_all"]], "missed (all dumped)"),
            missed_100=summarise([r for r in pred_rows if r["missed_100"]], "missed (R@100)"),
            hit=summarise([r for r in pred_rows if not r["missed_all"]], "detected (reference)"),
            per_pred=Counter((r["pred"], r["mirror"], r["missed_all"], r["mirror_hit"])
                             for r in pred_rows),
        )

    f = lambda t: f"{t[0]:.1%} ({t[1]:.1%}–{t[2]:.1%})"
    L = ["# Missed novel-split relations: was the mirror predicted on the reversed pair?\n",
         "**Data:** the existing prediction dumps (`preds_from_pod/full`) and the "
         "**test** relation GT. This is an error analysis of the evaluated model, so it "
         "needs test data, unlike stage 0. Anything designed from it has been informed "
         "by the test set.\n",
         "**Definitions.** Novel-split GT means test relations whose predicate is novel, "
         "which is the official filter. *Missed* uses the official matcher: greedy by "
         "score, exact triplet, min(subject, object) vIoU ≥ 0.5, first over all dumped "
         "predictions (≤200 per video) and then within each video's top 100. The "
         "*reversed pair* means predictions whose subject trajectory matches the GT "
         "object and whose object trajectory matches the GT subject, both at vIoU ≥ 0.5. "
         "That also guarantees an overlapping extent. Categories are not required to "
         "match.\n",
         "**Controls,** per missed GT, both exact expectations rather than samples: "
         "*uniform* = the chance that a predicate drawn uniformly from the pass's other "
         "candidates was predicted on the reversed pair; *marginal* = the same, with the "
         "draw weighted by how often the model emits each predicate. 95% CIs resample "
         "whole videos.\n"]
    for pass_name, title in (("novel", "Novel pass — the official novel split"),
                             ("all", "All pass — base predicates also scored (secondary)")):
        r = res[pass_name]
        L.append(f"## {title}\n")
        L.append(f"Novel-split GT instances whose predicate has a mirror in the vocabulary: "
                 f"**{r['n_gt']}**. Of these, {r['n_unpredictable']} have a mirror this pass "
                 f"cannot emit"
                 + (" (a BASE mirror, which the novel pass hard-zeroes) and are excluded below."
                    if pass_name == "novel" else "."))
        L.append("\n| GT set | n | exact mirror on reversed pair | mirror spatial term, any predicate | control: uniform random predicate | control: model-marginal random predicate | reversed pair has any prediction |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        for k in ("missed_all", "missed_100", "hit"):
            s = r[k]
            if not s["n"]:
                L.append(f"| {s['label']} | 0 | — | — | — | — | — |")
                continue
            L.append(f"| {s['label']} | {s['n']} | **{f(s['mirror_hit'])}** | "
                     f"{f(s['spatial_mirror_hit'])} | {f(s['ctrl_uniform'])} | "
                     f"{f(s['ctrl_marginal'])} | {f(s['any_rev'])} |")
        L.append("\n**Where the missed GTs went** (all-dumped level), with trajectories "
                 "matched at vIoU ≥ 0.5 and any category:\n")
        L.append("| missed GT | rate |")
        L.append("|---|---:|")
        s_ = r["missed_all"]
        if s_["n"]:
            L.append(f"| forward pair (A, B) has any prediction | {f(s_['any_fwd'])} |")
            L.append(f"| forward pair has the GT predicate, but the triplet still missed (category error) | {f(s_['fwd_has_p'])} |")
            L.append(f"| reversed pair (B, A) has any prediction | {f(s_['any_rev'])} |")
            L.append(f"| **neither direction has any prediction** | **{f(s_['neither'])}** |")
            c = s_["cond_n"]
            L.append(f"\nConditional on the reversed pair having a prediction (n = {c}): exact "
                     f"mirror {s_['cond_mirror']}/{c} ({s_['cond_mirror'] / c:.0%}). The controls "
                     f"on the same rows are {s_['cond_ctrl_uniform']:.0%} uniform and "
                     f"{s_['cond_ctrl_marginal']:.0%} model-marginal.\n" if c else "")
        L.append("")
        pp = defaultdict(lambda: [0, 0])
        for (p, q, missed, hit), n in r["per_pred"].items():
            if missed:
                pp[(p, q)][0] += n
                pp[(p, q)][1] += n * hit
        L.append("Missed GT by predicate (all-dumped level): mirror predicted / missed:\n")
        L.append(", ".join(f"`{p}`→`{q}` {h}/{n}" for (p, q), (n, h) in
                           sorted(pp.items(), key=lambda x: -x[1][0])) + "\n")
    a, b = res["novel"]["missed_all"], res["all"]["missed_all"]
    L.append("## Conclusion\n")
    L.append(f"**The mirror explains almost none of the misses.** On missed novel-split "
             f"GT, the model put the exact mirror predicate on the reversed pair "
             f"{a['mirror_hit'][0]:.1%} of the time in the novel pass (n = {a['n']}), and "
             f"{b['mirror_hit'][0]:.1%} in the all pass (n = {b['n']}). The random-pairing "
             f"control is {a['ctrl_uniform'][0]:.1%}–{a['ctrl_marginal'][0]:.1%} and "
             f"{b['ctrl_uniform'][0]:.1%}–{b['ctrl_marginal'][0]:.1%} respectively. That is "
             f"above chance in the all pass, but on the order of 2–3 percentage points "
             f"of misses. In the novel pass the CIs overlap.\n")
    L.append(f"**The reason is that the pair is usually absent altogether.** "
             f"{a['neither'][0]:.0%} (novel pass) and {b['neither'][0]:.0%} (all pass) of "
             f"missed GTs have NO dumped prediction on either direction of the pair, "
             f"meaning no prediction whose trajectories match the GT boxes at vIoU ≥ 0.5, "
             f"under any predicate or category. Where the reversed pair is covered, the "
             f"mirror rate rises to {a['cond_mirror']}/{a['cond_n']} and "
             f"{b['cond_mirror']}/{b['cond_n']}, against {a['cond_ctrl_marginal']:.0%} / "
             f"{b['cond_ctrl_marginal']:.0%} model-marginal controls. That is elevated, but "
             f"it rests on small counts.\n")
    h = res["novel"]["hit"]
    L.append(f"**When the model detects a relation, it is direction-consistent.** "
             f"{h['mirror_hit'][0]:.0%} of detected novel GT also carry the exact mirror on "
             f"the reversed pair, against a {h['ctrl_marginal'][0]:.0%} control. So the "
             f"model is not confusing directions. It is failing to produce the pair in the "
             f"first place.\n")
    L.append("**Build implication.** A mirror or argument-swap mechanism could recover "
             "at most the few percent of misses where the reversed pair is covered and "
             "the forward one isn't. The ceiling on these misses is pair coverage, which "
             "matches the pilot's earlier finding that the failure is upstream. One "
             "caveat on 'absent': the dump keeps ≤200 predictions per video and the top "
             "20 predicates per segment. So an absent pair is either a tracking or "
             "proposal failure, or a pair cut by truncation. The pre-merge "
             "`segments_raw_*.json` dumps can separate those two cases; that has not "
             "been done here.\n")
    open(os.path.join(HERE, "mirror_misses.md"), "w").write("\n".join(L) + "\n")

    json.dump({k: {kk: vv for kk, vv in v.items() if kk not in ("rows", "per_pred")}
               for k, v in res.items()},
              open(os.path.join(HERE, "summary.json"), "w"), indent=1)
    for k, v in res.items():
        print(k, v["n_gt"], v["n_unpredictable"])
        for kk in ("missed_all", "missed_100", "hit"):
            print("  ", kk, {a: b for a, b in v[kk].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
