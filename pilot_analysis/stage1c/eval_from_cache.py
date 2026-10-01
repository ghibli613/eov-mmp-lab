#!/usr/bin/env python3
"""Post-process cached logits with the repo's own pipeline and score them with
the official evaluator. Used by Task 4 (baseline PredCls), Task 5 (paraphrases)
and Task 6 (SGDet gate). No model calls; CPU only.

    python pilot_analysis/stage1c/eval_from_cache.py --cache CACHE --out CACHE/task4

Scores for a slot are sigmoid(logit) * sigmoid(interactiveness), exactly as
modelC returns them (models/relation_classifier.py:504-524). The `novel` split
zeroes the base columns, as modelC's novel branch does (rc:511-514). Then the
repo's process_pred (top-20 per slot) -> association -> format_ (top-200 per
video) -> eval_relation_detection_openvoc. Test relation labels are read only by
the official evaluator.

Task 4 gate: novel mAP must be within 3.0 points of the paper's PredCls 21.65,
otherwise exit code 3.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, glob, json, os
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

PAPER_PREDCLS = dict(novel=21.65, all=39.83)
GATE_TASK4 = 3.0
POST = SimpleNamespace(clip_len=30, clip_top_n=20, use_prior=False, max_per_video=200)


def meta():
    from utils import paths
    d = paths.META_DIR
    id2pre = json.load(open(os.path.join(d, "id2predicate.json")))
    obj2id = json.load(open(os.path.join(d, "object2id.json")))
    split = json.load(open(paths.PRED_SPLIT_INFO))
    base_ids = [int(i) for i, c in split["id2cls"].items() if c != "__background__" and split["cls2split"][c] == "base"]
    return id2pre, obj2id, base_ids


def load_cache(cache_dir, split="test"):
    """{video: [pair dicts]} from CACHE/<split>/*.pt."""
    import torch
    out = {}
    for f in sorted(glob.glob(os.path.join(cache_dir, split, "*.pt"))):
        d = torch.load(f, weights_only=False)
        out[d["video"]] = d["pairs"]
    return out


def slot_scores(logits, int_logit, mode, base_ids):
    import torch
    s = torch.sigmoid(logits.float()) * torch.sigmoid(int_logit.float()).unsqueeze(-1)
    if mode == "novel":
        s = s.clone()
        s[:, base_ids] = 0.0
    return s


def predict(cache, mode, logits_of=None):
    """-> {video: formatted relations}. `logits_of(video, k, pair)` may override
    the cached logits (Task 5)."""
    from inference.post_process import association, format_, process_pred
    id2pre, obj2id, base_ids = meta()
    preds = {}
    for v, pairs in cache.items():
        rels = []
        for k, p in enumerate(pairs):
            lg = logits_of(v, k, p) if logits_of else p["logits"]
            pd = dict(sbj_cls=p["sbj_cls"], obj_cls=p["obj_cls"], sbj_scr=p["sbj_scr"], obj_scr=p["obj_scr"],
                      sbj_traj=list(p["sbj_traj"]), obj_traj=list(p["obj_traj"]), duration=list(p["duration"]))
            rels.extend(association(process_pred(POST, id2pre, obj2id, None,
                                                 slot_scores(lg, p["int_logit"], mode, base_ids), pd)))
        preds[v] = format_(POST, rels)
    return preds


def _filtered(preds, split, exclude=(), only=None):
    """The official evaluator's own category filter (video_relation_detection_openvoc.py:357-385),
    with an optional predicate exclusion (A4) or a single-predicate restriction."""
    from utils import paths
    traj = json.load(open(paths.OBJ_SPLIT_INFO))["cls2split"]
    pred = json.load(open(paths.PRED_SPLIT_INFO))["cls2split"]
    tc = {c for c in traj if c != "__background__"}
    pc = {c for c, s in pred.items() if c != "__background__" and (split == "all" or s == split)}
    pc -= set(exclude)
    if only is not None:
        pc &= {only}
    keep = lambda r: r["triplet"][0] in tc and r["triplet"][1] in pc and r["triplet"][2] in tc
    gt = json.load(open(paths.TEST_RELATION_GT))
    g = {v: [r for r in rs if keep(r)] for v, rs in gt.items()}
    g = {v: r for v, r in g.items() if r}
    p = {v: [r for r in rs if keep(r)] for v, rs in preds.items()}
    return g, p


def score(preds, split, exclude=()):
    """mAP, R@50, R@100 (in %). With no exclusion this is exactly
    eval_relation_detection_openvoc; checked in main()."""
    from inference.video_relation_detection_openvoc import evaluate
    g, p = _filtered(preds, split, exclude)
    m, rec, _ = evaluate(g, p, viou_threshold=0.5)
    return dict(mAP=100 * float(m), R50=100 * float(rec[50]), R100=100 * float(rec[100]))


def per_predicate_ap(preds, split):
    """AP per predicate (the official evaluate() on one predicate at a time), with
    its GT instance count. A predicate with GT but no matching prediction scores 0."""
    from inference.video_relation_detection_openvoc import evaluate
    from utils import paths
    pred = json.load(open(paths.PRED_SPLIT_INFO))["cls2split"]
    out = {}
    for c, s in pred.items():
        if c == "__background__" or (split != "all" and s != split):
            continue
        g, p = _filtered(preds, split, only=c)
        n = sum(len(r) for r in g.values())
        if not n:
            continue
        if not any(p.get(v) for v in g):
            out[c] = (0.0, n)
            continue
        try:
            m, _, _ = evaluate(g, p, viou_threshold=0.5)
            out[c] = (100 * float(m), n)
        except IndexError:
            out[c] = (0.0, n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-gate", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from inference.video_relation_detection_openvoc import eval_relation_detection_openvoc
    cache = load_cache(a.cache)
    res = {}
    for split in ("novel", "all"):
        preds = predict(cache, split)
        r = score(preds, split)
        m_off, rec_off = eval_relation_detection_openvoc(target_split_pred=split, prediction_results=preds)
        assert abs(100 * m_off - r["mAP"]) < 1e-9, "own filter disagrees with the official evaluator"
        r["paper_PredCls_mAP"] = PAPER_PREDCLS[split]
        r["diff_vs_paper"] = r["mAP"] - PAPER_PREDCLS[split]
        res[split] = r
        json.dump(preds, open(os.path.join(a.out, f"predcls_{split}.json"), "w"))
    res["videos"] = len(cache)
    res["pairs"] = sum(len(v) for v in cache.values())
    json.dump(res, open(os.path.join(a.out, "task4_predcls.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    if not a.no_gate and abs(res["novel"]["diff_vs_paper"]) > GATE_TASK4:
        print(f"!! TASK 4 GATE: novel PredCls {res['novel']['mAP']:.2f} differs from "
              f"{PAPER_PREDCLS['novel']} by more than {GATE_TASK4} -- STOP")
        sys.exit(3)


if __name__ == "__main__":
    main()
