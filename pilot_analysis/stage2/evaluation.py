"""Stage 2 Batch A -- HELD evaluation through the repo's own post-processing and
evaluator (A6). Nothing here computes a metric: process_pred -> association ->
format_ (inference/post_process.py), then eval_relation_detection_openvoc
(inference/video_relation_detection_openvoc.py) with gt_json = the HELD GT that
step 1 built from TRAINING annotations (base labels only)."""
from __future__ import annotations

import json, os
from types import SimpleNamespace

import numpy as np

import common as C


def names():
    from utils import paths
    return json.load(open(os.path.join(paths.META_DIR, "id2predicate.json")))


def predict(held_pairs, scores):
    """scores: (N, 132) over all slots (only HELD rows are read) -> {video: format_ output}."""
    import torch
    from utils import paths
    from inference.post_process import association, format_, process_pred
    id2pre = names()
    obj2id = json.load(open(os.path.join(paths.META_DIR, "object2id.json")))
    args = SimpleNamespace(**C.POST)
    preds = {}
    for v, pairs in held_pairs.items():
        rels = []
        for p in pairs:
            r0, r1 = p["rows"]
            pd = dict(sbj_cls=p["sbj_cls"], obj_cls=p["obj_cls"], sbj_scr=p["sbj_scr"], obj_scr=p["obj_scr"],
                      sbj_traj=[list(b) for b in p["sbj_traj"]], obj_traj=[list(b) for b in p["obj_traj"]],
                      duration=list(p["duration"]))
            rels.extend(association(process_pred(args, id2pre, obj2id, None,
                                                 torch.from_numpy(np.ascontiguousarray(scores[r0:r1], np.float32)), pd)))
        preds[v] = format_(args, rels)
    return preds


def split_file(path, competing, label="compete"):
    """A copy of the repo's predicate split file in which `competing` form their own split."""
    info = json.load(open(C.PRED_SPLIT_INFO))
    info["cls2split"] = {c: (label if c in competing else ("__background__" if c == "__background__" else "other"))
                         for c in info["cls2split"]}
    json.dump(info, open(path, "w"), indent=1)
    return path


def evaluate(preds, gt_path, split_name="base", split_info=C.PRED_SPLIT_INFO):
    from inference.video_relation_detection_openvoc import eval_relation_detection_openvoc
    m, rec = eval_relation_detection_openvoc(target_split_pred=split_name, pred_cls_split_info_path=split_info,
                                             prediction_results=preds, gt_json=gt_path)
    return dict(mAP=100 * float(m), R50=100 * float(rec[50]), R100=100 * float(rec[100]))


def eov(d):
    """The repo's per-slot score: sigmoid(logit) * sigmoid(interactiveness)."""
    s = 1 / (1 + np.exp(-d["logits"].astype(np.float64)))
    return s * (1 / (1 + np.exp(-d["int_logit"].astype(np.float64))))[:, None]


def only(scores, cols):
    out = np.zeros_like(scores)
    out[:, cols] = scores[:, cols]
    return out


def eov_task4(d, pairs):
    """EOV exactly as session 3 Task 4 scored it: stage1c/eval_from_cache.slot_scores (mode
    "all"), called per pair on that pair's (slots, 132) logits, as Task 4 did -- torch's CPU
    sigmoid rounds differently with tensor size, so it is not called on the whole matrix.
    Batch B uses this, so run 1 reproduces the Task-4 predictions bit for bit."""
    import torch
    import eval_from_cache as E4
    out = np.zeros(np.shape(d["logits"]), np.float64)
    for ps in pairs.values():
        for p in ps:
            r0, r1 = p["rows"]
            out[r0:r1] = E4.slot_scores(torch.from_numpy(np.asarray(d["logits"][r0:r1], np.float32)),
                                        torch.from_numpy(np.asarray(d["int_logit"][r0:r1], np.float32)),
                                        "all", None).numpy()
    return out
