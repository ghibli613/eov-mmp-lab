#!/usr/bin/env python3
"""Object check O1 (A22), step 1 -- inventory (CPU, no test computation).

    python pilot_analysis/objects/o1_step1.py

1b  the detector inside the end-to-end checkpoint vs the standalone detector checkpoint (step 1 of
    the paper's training): tensor-by-tensor equality (memory-mapped, no forward pass), and the
    training configs both checkpoints carry; novel-category fragments in the trajectory file the
    repo's step-4 code turns into detector targets.
1c  GT object tracks in the 800 training videos, per category: annotated tids, option-B runs (A5),
    annotated boxes, videos.
1d  from D1's existing tc_missed.json only: the share of SGDet novel-pass U1 / U2 missed GT
    relations whose subject or object is a novel category.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import collections, json, os

import o1_common as O

E2E = os.path.join(O.ROOT, "output", "ckpt", "baseline_fbce_vidvrd_bs1_lr1e-05_dim512_none_rel_mot_clip_bbox_end2end_base-001.pth")
DET = os.path.join(O.ROOT, "output", "ckpt", "checkpoint_vidvrd0059_new_1e-5.pth")


def ckpt_compare():
    import torch
    e = torch.load(E2E, map_location="cpu", mmap=True, weights_only=False)
    d = torch.load(DET, map_location="cpu", mmap=True, weights_only=False)
    es = {k[len("modelA."):]: v for k, v in e["state_dict"].items() if k.startswith("modelA.")}
    ds = d["model"]
    common = sorted(set(es) & set(ds))
    n_eq, n_diff, worst, diff_names = 0, 0, 0.0, []
    for k in common:
        a, b = es[k], ds[k]
        if a.shape != b.shape:
            n_diff += 1; diff_names.append(k); continue
        if torch.equal(a, b):
            n_eq += 1
        else:
            n_diff += 1
            diff_names.append(k)
            worst = max(worst, float((a.float() - b.float()).abs().max()))
    ema = d.get("ema_model")
    ema_eq = None
    if isinstance(ema, dict):
        ek = [k for k in common if k in ema]
        ema_eq = sum(torch.equal(es[k], ema[k]) for k in ek), len(ek)
    cfg_e = vars(e["config"]) if hasattr(e.get("config"), "__dict__") else e.get("config")
    cfg_d = vars(d["args"]) if hasattr(d.get("args"), "__dict__") else d.get("args")
    pick = lambda c, keys: {k: c.get(k) for k in keys if c and k in c}
    keys_e = ("src_split", "train_split", "test_split", "tgt_split", "lr", "epochs", "frame_stride", "dataset", "all_train_token")
    keys_d = ("dataset_file", "all_train_token", "epochs", "lr", "label_map", "clip_feat_path", "distil_clip_embed", "prob",
              "train_split", "seen", "unseen", "zero_shot", "ov", "data_path", "coco_path", "output_dir", "resume")
    return dict(modelA_tensors_in_e2e=len(es), detector_ckpt_tensors=len(ds), compared=len(common), identical=n_eq,
                different=n_diff, max_abs_diff=worst, first_different=diff_names[:10],
                e2e_equals_det_ema=ema_eq, e2e_epoch=e.get("epoch"), det_epoch=d.get("epoch"),
                e2e_config=pick(cfg_e, keys_e), det_args=pick(cfg_d, keys_d),
                det_args_all_keys=sorted(cfg_d)[:400] if cfg_d else None)


def fragments():
    d = json.load(open(O.FRAG))
    c = collections.Counter()
    for v, ts in d.items():
        for t in ts:
            cat = t["category"][0] if isinstance(t["category"], list) else t["category"]
            c["novel" if cat in O.NOVEL else "base"] += 1
    return dict(videos=len(d), fragments=sum(c.values()), novel_fragments=c["novel"], base_fragments=c["base"])


def gt_counts():
    runs = json.load(open(O.GT_RUNS))
    tids, nruns, boxes, vids = collections.Counter(), collections.Counter(), collections.Counter(), collections.defaultdict(set)
    for v in sorted(runs):
        raw, cats = O.annotated_boxes(v)
        for tid, fb in raw.items():
            tids[cats[tid]] += 1
            boxes[cats[tid]] += len(fb)
            vids[cats[tid]].add(v)
        for r in runs[v]:
            nruns[r["category"]] += 1
    rows = {c: dict(novel=c in O.NOVEL, tids=tids[c], runs=nruns[c], annotated_boxes=boxes[c], videos=len(vids[c])) for c in O.CATS}
    tot = lambda nv, k: sum(r[k] for r in rows.values() if r["novel"] == nv)
    return dict(per_category=rows,
                novel=dict(tids=tot(True, "tids"), runs=tot(True, "runs"), annotated_boxes=tot(True, "annotated_boxes"),
                           videos=len(set().union(*[vids[c] for c in O.NOVEL]))),
                base=dict(tids=tot(False, "tids"), runs=tot(False, "runs"), annotated_boxes=tot(False, "annotated_boxes")),
                weak_check=tot(True, "tids") < 100)


def d1_crossref():
    rows = json.load(open(O.TC_MISSED))["novel"]["sgdet"]["rows"]
    out = {}
    for grp, bs in (("U1", ("U1",)), ("U2", ("U2",)), ("U1+U2", ("U1", "U2"))):
        sel = [r for r in rows if r["bucket"] in bs]
        nov = [r for r in sel if r["subject"] in O.NOVEL or r["object"] in O.NOVEL]
        out[grp] = dict(n=len(sel), with_novel_object=len(nov), share=len(nov) / len(sel) if sel else None,
                        subject_novel=sum(r["subject"] in O.NOVEL for r in sel), object_novel=sum(r["object"] in O.NOVEL for r in sel))
    out["missed_total"] = len(rows)
    return out


def main():
    O.C.verify_frozen()
    os.makedirs(O.OUT, exist_ok=True)
    res = dict(ckpt=ckpt_compare(), fragments=fragments(), gt=gt_counts(), d1=d1_crossref())
    path = os.path.join(O.OUT, "o1_step1.json")
    json.dump(res, open(path, "w"), indent=1, default=str)
    O.write_log("o1_step1", {}, [O.GT_RUNS, O.FRAG, O.TC_MISSED, O.OBJ_SPLIT], [path])
    c = res["ckpt"]
    print(f"1b: modelA in e2e vs detector ckpt: {c['identical']} identical / {c['different']} different of {c['compared']} "
          f"(max |diff| {c['max_abs_diff']:.3g}); fragments: {res['fragments']}")
    print(f"1c: novel tids {res['gt']['novel']['tids']}, runs {res['gt']['novel']['runs']}; base tids {res['gt']['base']['tids']}, "
          f"runs {res['gt']['base']['runs']}; weak check: {res['gt']['weak_check']}")
    print(f"1d: {json.dumps(res['d1'])}")


if __name__ == "__main__":
    main()
