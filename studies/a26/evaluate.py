#!/usr/bin/env python3
"""Laptop, after the copy-off: A26's numbers, the paired bootstrap and the reading fixed by A26.

    cd METOR/studies && python -m a26.evaluate --rec a26/data/rec

1. Every file matches the manifest; 200 videos x 4 configurations.
2. Official numbers: the unmodified evaluator, eval_relation_detection_openvoc(target_split_pred=split,
   rt_hit_infos=True), as cli/common.py:_predict_split calls it; mAP, R@50, R@100 per configuration
   and split.
3. R0 check: R0's mAP equals session 3's (S3_R0) within R0_TOL points; the videos whose predictions
   differ from session 3's (scores beyond 1e-6, or anything else) are listed. (The pilot's novel-branch
   predictions matched session 3's derived ones, scores within 1.1e-7: time_check_report.md:32.)
4. Per-video statistics from the evaluator's own functions (eval_detection_scores_v2, voc_ap): AP,
   TP within the first 50 and 100, number of GT. Their aggregates must equal the official numbers.
5. Paired video bootstrap: N_BOOT resamples per split, default_rng(BOOT_SEED), the same draws for
   every configuration; 95% percentile intervals of each configuration minus R0. On the first two
   resamples the aggregates are recomputed with the evaluator itself (duplicates as distinct keys,
   the GT resampled the same way) and must agree.
6. The reading (A26), applied to one quantity only: Δuw = R0uw - R0, SGDet novel mAP.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from a26.common import (BOOT_SEED, CI, CONFIG_NAMES, DATA, LARGE, METOR_ROOT, METRICS, MODERATE,  # noqa: E402
                        N_BOOT, R0_TOL, S3_DIR, S3_R0, S3_SUMS, S3_SUMS_SHA, SCORE_TOL, SPLITS)
from a26.run import sha256  # noqa: E402

GT_SHA = "b7a8219fa1b00e6d10c8bc5f37a1df7482b316dd401a1c870678817c4a921aeb"   # A25's pin


# ---------------------------------------------------------------- pure functions (tested on CPU)
def filter_split(gt: dict, preds: dict, split: str, obj_split: dict, pred_split: dict):
    """The official evaluator's category filter (video_relation_detection_openvoc.py:363-385):
    GT videos with no kept relation are dropped; predictions are filtered the same way."""
    traj = {c for c in obj_split["cls2split"] if c != "__background__"}
    pred = {c for c, s in pred_split["cls2split"].items() if c != "__background__" and (split == "all" or s == split)}
    keep = lambda r: r["triplet"][0] in traj and r["triplet"][1] in pred and r["triplet"][2] in traj
    g = defaultdict(list)
    for v, rels in gt.items():
        for r in rels:
            if keep(r):
                g[v].append(r)
    p = defaultdict(list)
    for v, rels in preds.items():
        for r in rels:
            if keep(r):
                p[v].append(r)
    return dict(g), dict(p)


def per_video(gt: dict, preds: dict) -> dict:
    """{video: (AP, TP@50, TP@100, #GT)} with the evaluator's own functions, in the evaluator's
    video order. `gt` and `preds` are already filtered."""
    from inference.video_relation_detection_openvoc import eval_detection_scores_v2, voc_ap
    out = {}
    for vid, g in gt.items():
        if not g:
            continue
        prec, rec, scores, _ = eval_detection_scores_v2(g, [dict(r) for r in preds.get(vid, [])], 0.5)
        tp = np.isfinite(scores)
        out[vid] = (float(voc_ap(rec, prec)), int(tp[:50].sum()), int(tp[:100].sum()), len(g))
    return out


def aggregate(stats: list) -> dict:
    """mAP and R@K (in %) from per-video statistics, as evaluate() aggregates them."""
    a = np.asarray(stats, dtype=np.float64)
    return {"mAP": 100 * float(np.mean(a[:, 0])), "R@50": 100 * float(a[:, 1].sum() / a[:, 3].sum()),
            "R@100": 100 * float(a[:, 2].sum() / a[:, 3].sum())}


def draws(n_videos: int, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """(n_boot, n_videos) indices, resampled with replacement; one matrix per split, shared by every
    configuration."""
    return np.random.default_rng(seed).integers(0, n_videos, size=(n_boot, n_videos))


def boot_metrics(stats: np.ndarray, idx: np.ndarray) -> dict:
    """stats: (n_videos, 4) in the order of `idx`'s columns' universe. -> {metric: (n_boot,)}."""
    s = stats[idx]                                   # (n_boot, n_videos, 4)
    return {"mAP": 100 * s[:, :, 0].mean(1), "R@50": 100 * s[:, :, 1].sum(1) / s[:, :, 3].sum(1),
            "R@100": 100 * s[:, :, 2].sum(1) / s[:, :, 3].sum(1)}


def reading(delta: float, lo: float, hi: float) -> dict:
    """A26's reading of Δuw (SGDet novel mAP points): the size word, and the direction in words."""
    excludes_zero = lo > 0 or hi < 0
    if excludes_zero and abs(delta) >= LARGE:
        size = "large"
    elif excludes_zero and abs(delta) >= MODERATE:
        size = "moderate"
    else:
        size = "small"
    direction = ("lower under the clean protocol" if delta < 0 else "higher under the clean protocol")
    return dict(size=size, direction=direction if size != "small" else "not read",
                delta=delta, ci=[lo, hi], ci_excludes_zero=excludes_zero)


# ---------------------------------------------------------------- the run
def load_split(rec: str, name: str, videos: list, split: str) -> dict:
    """{video: formatted predictions} for one configuration and split (one split at a time: 7 GB)."""
    out = {}
    for v in videos:
        with gzip.open(os.path.join(rec, name, f"{v}.json.gz"), "rt") as f:
            d = json.load(f)
        if d["config"] != name or d["video"] != v:
            raise SystemExit(f"!! STOP: {name}/{v} holds {d['config']}/{d['video']}")
        out[v] = d["predictions"][split]
    return out


def score_split(preds: dict, gt_all: dict, gt_path: str, split: str, obj_split: dict, pred_split: dict):
    """One configuration and split: the official numbers from the unmodified evaluator (as
    _predict_split calls it), and the per-video statistics, which must aggregate to them.
    -> (official, {video: stats}, filtered GT, filtered predictions)."""
    from inference.video_relation_detection_openvoc import eval_relation_detection_openvoc
    # the evaluator only adds 'hit_gt' to the dicts it is given; nothing it reads
    m, rec_at_n = eval_relation_detection_openvoc(target_split_pred=split, prediction_results=preds,
                                                  gt_json=gt_path, rt_hit_infos=True)
    off = {"mAP": 100 * float(m), "R@50": 100 * float(rec_at_n[50]), "R@100": 100 * float(rec_at_n[100])}
    g, p = filter_split(gt_all, preds, split, obj_split, pred_split)
    pv = per_video(g, p)
    agg = aggregate([pv[v] for v in pv])
    if abs(agg["mAP"] - off["mAP"]) > 1e-9 or any(abs(agg[k] - off[k]) > 1e-4 for k in ("R@50", "R@100")):
        raise SystemExit(f"!! STOP: {split}: per-video aggregate {agg} != official {off}")
    return off, pv, g, p


def resample_check(g: dict, p: dict, order: list, stats: np.ndarray, n: int = 2) -> None:
    """The first `n` resamples, scored by the evaluator itself (duplicates as distinct keys, the GT
    resampled the same way), must equal the per-video aggregates the bootstrap uses."""
    from inference.video_relation_detection_openvoc import evaluate_v2
    idx = draws(len(order))[:n]
    for b in range(n):
        keys = [order[j] for j in idx[b]]
        gb = {f"{v}#{k}": g[v] for k, v in enumerate(keys)}
        pb = {f"{v}#{k}": [dict(r) for r in p.get(v, [])] for k, v in enumerate(keys)}
        m, rec, _, _ = evaluate_v2(gb, pb, viou_threshold=0.5)
        bm = boot_metrics(stats, idx[b:b + 1])
        if (abs(100 * m - bm["mAP"][0]) > 1e-9 or abs(100 * rec[50] - bm["R@50"][0]) > 1e-4
                or abs(100 * rec[100] - bm["R@100"][0]) > 1e-4):
            raise SystemExit(f"!! STOP: resample {b}: evaluator {m} {rec} vs per-video {bm}")


def bootstrap_deltas(stats: dict, official: dict, order: dict) -> dict:
    """stats[split][config]: (n_videos, 4) in order[split]. -> {"<config>-R0": {split: {metric:
    {delta, ci}}}}; delta from the official numbers, the CI from the paired resamples."""
    out = {}
    for s in SPLITS:
        idx = draws(len(order[s]))
        bm = {n: boot_metrics(stats[s][n], idx) for n in CONFIG_NAMES}
        for n in CONFIG_NAMES[1:]:
            for k in METRICS:
                lo, hi = (float(x) for x in np.percentile(bm[n][k] - bm["R0"][k], CI))
                out.setdefault(f"{n}-R0", {}).setdefault(s, {})[k] = dict(
                    delta=official[n][s][k] - official["R0"][s][k], ci=[lo, hi])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default=os.path.join(DATA, "rec"))
    ap.add_argument("--out", default=os.path.join(DATA, "a26_results.json"))
    a = ap.parse_args()

    sys.path.insert(0, METOR_ROOT)
    os.chdir(METOR_ROOT)
    from utils import paths
    from a26.tracks import compare_predictions

    # 1. the copy-off
    man = [json.loads(l) for l in open(os.path.join(a.rec, "manifest.jsonl"))]
    videos = [r["video"] for r in man]
    if len(set(videos)) != 200 or len(videos) != 200:
        raise SystemExit(f"!! STOP: {len(set(videos))} videos in the manifest ({len(videos)} rows)")
    bad = [(r["video"], n) for r in man for n in CONFIG_NAMES
           if sha256(os.path.join(a.rec, n, f"{r['video']}.json.gz")) != r["sha256"][n]]
    if bad:
        raise SystemExit(f"!! STOP: {len(bad)} files do not match the manifest: {bad[:5]}")
    if sha256(paths.TEST_RELATION_GT) != GT_SHA:
        raise SystemExit("!! STOP: test_relation_gt.json differs from its pin")
    if sha256(S3_SUMS) != S3_SUMS_SHA:
        raise SystemExit("!! STOP: SHA256SUMS_session3.txt differs from its pin")
    s3_sums = dict(reversed(l.strip().split(None, 1)) for l in open(S3_SUMS))
    gt_all = json.load(open(paths.TEST_RELATION_GT))
    obj_split = json.load(open(paths.OBJ_SPLIT_INFO))
    pred_split = json.load(open(paths.PRED_SPLIT_INFO))

    res = dict(official={}, R0_check={}, bootstrap_videos={}, deltas={}, reading=None,
               settings=dict(n_boot=N_BOOT, seed=BOOT_SEED, ci=CI, large=LARGE, moderate=MODERATE))
    stats, order = {s: {} for s in SPLITS}, {}
    for name in CONFIG_NAMES:
        res["official"][name] = {}
        for s in SPLITS:
            preds = load_split(a.rec, name, videos, s)
            # 2 and 4. official numbers; per-video statistics that aggregate to them
            off, pv, g, p = score_split(preds, gt_all, paths.TEST_RELATION_GT, s, obj_split, pred_split)
            res["official"][name][s] = off
            order.setdefault(s, sorted(pv))
            if sorted(pv) != order[s]:
                raise SystemExit(f"!! STOP: {name} {s}: a different video set")
            stats[s][name] = np.asarray([pv[v] for v in order[s]], dtype=np.float64)
            if name == "R0":
                resample_check(g, p, order[s], stats[s][name])                       # 5
                path = os.path.join(S3_DIR, f"sgdet_{s}.json")                      # 3
                if sha256(path) != s3_sums[f"cache_s3_sgdet/sgdet_{s}.json"]:
                    raise SystemExit(f"!! STOP: sgdet_{s}.json differs from SHA256SUMS_session3.txt")
                s3 = json.load(open(path))
                differ = sorted(v for v in videos if compare_predictions(preds[v], s3.get(v, []), SCORE_TOL))
                del s3
                d_map = off["mAP"] - S3_R0[s]
                res["R0_check"][s] = dict(session3_mAP=S3_R0[s], diff=d_map, videos_differing=differ)
                if abs(d_map) > R0_TOL:
                    raise SystemExit(f"!! STOP: R0 {s} mAP differs from session 3's by {d_map}")
            del preds, g, p

    # 5-6. the paired bootstrap and the reading
    res["bootstrap_videos"] = {s: len(order[s]) for s in SPLITS}
    res["deltas"] = bootstrap_deltas(stats, res["official"], order)
    du = res["deltas"]["R0uw-R0"]["novel"]["mAP"]
    res["reading"] = dict(quantity="R0uw - R0, SGDet novel mAP", **reading(du["delta"], *du["ci"]))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"{a.out} sha256={sha256(a.out)}")
    for s in SPLITS:
        print(f"\nSGDet {s}")
        for n in CONFIG_NAMES:
            o = res["official"][n][s]
            line = f"  {n:5s} mAP {o['mAP']:6.2f}  R@50 {o['R@50']:6.2f}  R@100 {o['R@100']:6.2f}"
            if n != "R0":
                d = res["deltas"][f"{n}-R0"][s]["mAP"]
                line += f"   Δ mAP {d['delta']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]"
            print(line)
    print(f"\nreading: {res['reading']}")


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
