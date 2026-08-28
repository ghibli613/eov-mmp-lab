#!/usr/bin/env python
"""Phase 3.2(c) -- evaluate at a RELAXED vIoU threshold. CPU only.

    python pilot_analysis/scripts/eval_at_threshold.py

Why this file exists: eval_relation_detection_openvoc() hardcodes
`viou_threshold=0.5` at both of its call sites
(video_relation_detection_openvoc.py:387,389) and exposes no parameter for it.
The underlying evaluate()/evaluate_v2() DO accept the threshold, so the protocol's
Phase 3.2(c) ("evaluate with temporal-IoU threshold relaxed, to bound how much
signal exists before merging") needs the wrapper's category filtering
replicated with the threshold passed through. That is all this does -- the AP
computation is still the repo's own evaluate().

Run standalone it sweeps the threshold over the merge simulation, previewing the
shape Phase 3.2(c) will have on real predictions.
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from inference.video_relation_detection_openvoc import evaluate, load_json
from utils import paths


def eval_at(prediction_results, viou_threshold=0.5, target_split_traj="all",
            target_split_pred="all", enti_cls_split_info_path=None,
            pred_cls_split_info_path=None, gt_json=None):
    """eval_relation_detection_openvoc, with viou_threshold plumbed through.
    Filtering logic copied verbatim from that function so numbers stay comparable."""
    enti = enti_cls_split_info_path or paths.OBJ_SPLIT_INFO
    pred = pred_cls_split_info_path or paths.PRED_SPLIT_INFO
    gtp = gt_json or paths.TEST_RELATION_GT

    traj_info = load_json(enti)
    pred_info = load_json(pred)
    traj_cats = {c for c, s in traj_info["cls2split"].items()
                 if (s == target_split_traj or target_split_traj == "all")
                 and c != "__background__"}
    pred_cats = {c for c, s in pred_info["cls2split"].items()
                 if (s == target_split_pred or target_split_pred == "all")
                 and c != "__background__"}

    def keep(rels):
        out = []
        for r in rels:
            s, p, o = r["triplet"]
            if s in traj_cats and p in pred_cats and o in traj_cats:
                out.append(r)
        return out

    gt = {v: keep(r) for v, r in load_json(gtp).items()}
    gt = {v: r for v, r in gt.items() if r}
    pr = defaultdict(list)
    for v, r in prediction_results.items():
        pr[v] = keep(r)

    mean_ap, rec_at_n, _ = evaluate(gt, pr, viou_threshold=viou_threshold)
    return mean_ap, rec_at_n


def main():
    sys.path.insert(0, os.path.dirname(__file__))
    import random
    from merge_simulation import run, Args

    args = Args()
    gt = json.load(open(paths.TEST_RELATION_GT))

    print("Phase 3.2(c) preview -- vIoU threshold sweep over the merge simulation.")
    print("Relaxing the threshold bounds how much signal survives the merge: if the")
    print("greedy column recovers as the threshold drops, the predictions were")
    print("RIGHT but mislocalised in time; if it does not, the content is gone.\n")

    for dp in (0.05, 0.10):
        greedy, om, _, _, _ = run(gt, dp, 0, args)
        print(f"p(drop) = {dp:.2f}")
        print(f"  {'vIoU thr':>9} {'greedy':>9} {'oracle':>9} {'gap':>8}")
        for thr in (0.5, 0.4, 0.3, 0.2, 0.1):
            mg, _ = eval_at(greedy, viou_threshold=thr)
            mo, _ = eval_at(om, viou_threshold=thr)
            print(f"  {thr:9.1f} {mg:9.4f} {mo:9.4f} {mo-mg:8.4f}")
        print()


if __name__ == "__main__":
    main()
