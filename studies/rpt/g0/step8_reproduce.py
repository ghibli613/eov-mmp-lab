#!/usr/bin/env python3
"""RPT G0 step 8, on the GPU (A28): session 3's GT tracks for the 5 pinned HELD videos, through the
adapter's track check and the whole-video feature wrapper, into gen_feats_test + modelC (session 3's
build_modelC and run_pair). Writes <out>/step8/<video>.pt and a manifest; the laptop compares them
with session 3's cache_s3 (step8_compare.py).

    OMP_NUM_THREADS=8 python studies/rpt/g0/step8_reproduce.py --out OUT [--videos V ...]
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RPT = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(RPT, "..", ".."))
for p in (RPT, REPO, os.path.join(REPO, "pilot_analysis", "stage1c")):
    if p not in sys.path:
        sys.path.insert(0, p)

VIDEOS = ["ILSVRC2015_train_00008005", "ILSVRC2015_train_00010024", "ILSVRC2015_train_00024005",
          "ILSVRC2015_train_00025029", "ILSVRC2015_train_00027003"]        # the first 5 HELD IDs (A28)
CKPT = os.path.join(REPO, "output", "ckpt",
                    "baseline_fbce_vidvrd_bs1_lr1e-05_dim512_none_rel_mot_clip_bbox_end2end_base-001.pth")
CKPT_SHA256 = "d306952a51e9ca94c201ecf580f75a99221ae658dc187ae0ff38ff8b7dfc0e70"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--videos", nargs="*", default=VIDEOS)
    ap.add_argument("--ckpt", default=CKPT)
    a = ap.parse_args()
    import no_test_labels
    no_test_labels.install()
    import torch
    from adapter import check_repo_tracks
    from cache_wrapper import build_modelC
    from gt_tracks import load_tracks
    from relation import FrameFeatures, relation_head
    from utils import paths

    if sha256(a.ckpt) != CKPT_SHA256:
        raise SystemExit(f"!! STOP: {a.ckpt} is not the pinned end-to-end checkpoint")
    out_dir = os.path.join(a.out, "step8")
    os.makedirs(out_dir, exist_ok=True)
    m, _, load_info = build_modelC(a.ckpt)
    env = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
               omp_num_threads=os.environ.get("OMP_NUM_THREADS"), modelC_load=load_info)
    print(json.dumps(env), flush=True)
    tracks = load_tracks("train")
    manifest = []
    for v in a.videos:
        t0 = time.time()
        ts = tracks[v]
        check_repo_tracks([dict(t, score=float(t["score"])) for t in ts])
        ff = FrameFeatures(v, paths.FRAME_DIR)
        pairs = relation_head(v, ts, "train", ff, m)
        rec = dict(video=v, env=env, n_frames=len(ff.files), seconds=round(time.time() - t0, 2),
                   frames={f: gp.half() for f, (gp, _) in ff.cache.items()},
                   pairs=[{k: p[k] for k in ("sub_tid", "obj_tid", "sbj_cls", "obj_cls", "duration", "logits",
                                             "int_logit", "windows", "slot_inputs", "mid_frames", "fidelity_residual")}
                          for p in pairs])
        path = os.path.join(out_dir, f"{v}.pt")
        torch.save(rec, path)
        row = dict(video=v, pairs=len(pairs), clips=sum(len(p["logits"]) for p in pairs), frames_encoded=len(ff.cache),
                   seconds=rec["seconds"], max_fidelity_residual=max((p["fidelity_residual"] for p in pairs), default=0.0),
                   file=os.path.relpath(path, a.out), sha256=sha256(path))
        manifest.append(row)
        print(json.dumps(row), flush=True)
    json.dump(dict(env=env, videos=manifest), open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
