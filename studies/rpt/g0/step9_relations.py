#!/usr/bin/env python3
"""RPT G0 step 9, EOV-MMP side, on the GPU (A28): each smoke video's SAM 3 file -> the adapter
(tracks, scores, tiers, duplicates) -> the confident tracks through gen_feats_test + modelC (session
3's run_pair, windows joined) with whole-video features on demand. Writes <out>/step9/<video>.pt;
post-processing, the evaluator and the report tables run on the laptop (step9_evaluate.py).

    OMP_NUM_THREADS=8 python studies/rpt/g0/step9_relations.py --sam3 OUT/sam3_out --out OUT
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RPT = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(RPT, "..", ".."))
for p in (RPT, REPO, os.path.join(REPO, "pilot_analysis", "stage1c")):
    if p not in sys.path:
        sys.path.insert(0, p)

from step8_reproduce import CKPT, CKPT_SHA256, VIDEOS, sha256  # noqa: E402

SMOKE = VIDEOS[:3]                                  # the first 3 of the 5 (A28)


def track_summary(t):
    from collections import Counter
    st = Counter(t["status"].values())
    det = [ps for ps in t["det_probs"].values() if ps]
    single = [ps[0] for ps in det if len(ps) == 1]          # recorded, not used (A28 rule 2)
    return dict(category=t["category"], obj_id=t["obj_id"], score=t["score"], tier=t["tier"],
                begin_fid=t["begin_fid"], end_fid=t["end_fid"], frames=len(t["trajectory"]),
                detected=st["detected"], propagated=st["propagated"], filled=st["filled"],
                frames_with_several_detections=sum(1 for ps in det if len(ps) > 1),
                mean_detection_probability_single_frames=(sum(single) / len(single)) if single else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam3", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--videos", nargs="*", default=SMOKE)
    a = ap.parse_args()
    import no_test_labels
    no_test_labels.install()
    import torch
    import adapter as A
    from cache_wrapper import build_modelC
    from relation import FrameFeatures, relation_head
    from utils import paths

    if sha256(CKPT) != CKPT_SHA256:
        raise SystemExit(f"!! STOP: {CKPT} is not the pinned end-to-end checkpoint")
    out_dir = os.path.join(a.out, "step9")
    os.makedirs(out_dir, exist_ok=True)
    m, _, load_info = build_modelC(CKPT)
    manifest = []
    for v in a.videos:
        t0 = time.time()
        src = os.path.join(a.sam3, f"{v}.json.gz")
        s3 = A.load_sam3(src)
        out = A.tracks_from_sam3(s3)
        torch.cuda.reset_peak_memory_stats()
        ff = FrameFeatures(v, paths.FRAME_DIR)
        pairs = relation_head(v, out["repo"], "train", ff, m) if out["repo"] else []
        gaps = A.gap_exposure(pairs, [t["status"] for t in out["confident"]])
        for p, g in zip(pairs, gaps):
            p["gap_flags"] = g
        rec = dict(video=v, sam3_file=os.path.basename(src), sam3_sha256=sha256(src), n_frames=len(ff.files),
                   tracks=[track_summary(t) for t in out["all"]], duplicates=out["duplicates"],
                   confident=[dict(track_summary(t), tid=i, status=t["status"], trajectory=t["trajectory"],
                                   box_sam3=t["box_sam3"]) for i, t in enumerate(out["confident"])],
                   pairs=[{k: p[k] for k in ("sub", "obj", "sub_tid", "obj_tid", "sbj_cls", "obj_cls", "sbj_scr", "obj_scr",
                                             "duration", "sbj_traj", "obj_traj", "logits", "int_logit", "windows",
                                             "mid_frames", "gap_flags", "fidelity_residual")} for p in pairs],
                   frames_encoded=sorted(ff.cache), seconds=round(time.time() - t0, 2),
                   peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2**30, modelC_load=load_info)
        path = os.path.join(out_dir, f"{v}.pt")
        torch.save(rec, path)
        row = dict(video=v, tracks=len(out["all"]), confident=len(out["confident"]), duplicates=len(out["duplicates"]),
                   pairs=len(pairs), clips=sum(len(p["logits"]) for p in pairs), frames_encoded=len(ff.cache),
                   clips_with_filled_middle_frame=sum(sum(g) for g in gaps), seconds=rec["seconds"],
                   file=os.path.relpath(path, a.out), sha256=sha256(path))
        manifest.append(row)
        print(json.dumps(row), flush=True)
    json.dump(dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0), videos=manifest),
              open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
