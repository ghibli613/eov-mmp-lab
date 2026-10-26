#!/usr/bin/env python3
"""RPT G0, code check 3: which frames does gen_feats_test read, and how big is a per-video
feature cache that holds only those frames?

CPU only; HELD training videos only (pilot_analysis/stage1/train_split.json). The repo's own
gen_feats_test (models/gen_labels.py:642) runs on session 3's GT tracks (option B,
pilot_analysis/stage1c/gt_tracks.py), with proxy feature tensors that record every frame index
requested and return zeros. No model runs and no feature is computed.

    cd ov-vidvrd-lab && ~/miniconda3/envs/repro-next/bin/python studies/rpt/g0/check3_middle_frames.py
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import json
import os
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "pilot_analysis", "stage1c"))

from models.gen_labels import gen_feats_test  # noqa: E402
from gt_tracks import load_tracks  # noqa: E402
from utils import paths  # noqa: E402

PP_BYTES = 576 * 768 * 2 + 768 * 2          # patch_proj + global_proj per frame, fp16 (CLIP on CUDA is fp16)


class Recorder:
    """Stands in for patch_proj / global_proj (1, T, ...): x[0][a:a+1, ...] logs frame a."""

    def __init__(self, shape_tail):
        self.shape_tail, self.frames = shape_tail, []

    def __getitem__(self, key):
        if key == 0:
            return self
        sl = key[0] if isinstance(key, tuple) else key
        assert isinstance(sl, slice) and sl.stop == sl.start + 1, key
        self.frames.append(sl.start)
        return torch.zeros(1, *self.shape_tail)


def video_stats(video, tracks):
    anno = json.load(open(os.path.join(paths.ANNO_DIR, "train", f"{video}.json")))
    w, h = anno["width"], anno["height"]
    pp, gp = Recorder((576, 768)), Recorder((768,))
    items = gen_feats_test([video], tracks, "train", pp, gp, w, h)
    clips = sum(int(it["clip_feat"].shape[1]) for it in items.values())
    assert len(pp.frames) == 3 * clips and len(gp.frames) == clips      # 3 boxes + 1 global per clip
    mids = sorted(set(gp.frames))
    n_frames = len(os.listdir(os.path.join(paths.FRAME_DIR, video)))
    return dict(video=video, frames=n_frames, tracks=len(tracks), pairs=len(items), clips=clips,
                mid_frames=len(mids), max_pair_clips=max((int(it["clip_feat"].shape[1]) for it in items.values()), default=0))


def main():
    torch.Tensor.cuda = lambda self, *a, **k: self                          # CPU run; nothing is computed on GPU
    split = json.load(open(os.path.join(ROOT, "pilot_analysis", "stage1", "train_split.json")))
    held = sorted(split["held"])
    tracks = load_tracks("train")
    rows = [video_stats(v, tracks[v]) for v in held]
    fr = np.array([r["frames"] for r in rows])
    mf = np.array([r["mid_frames"] for r in rows])
    out = dict(
        videos=len(rows), frames_total=int(fr.sum()), frames_median=float(np.median(fr)), frames_max=int(fr.max()),
        mid_frames_total=int(mf.sum()), mid_frames_median=float(np.median(mf)), mid_frames_max=int(mf.max()),
        share_of_frames_read=float(mf.sum() / fr.sum()),
        reduction_factor=float(fr.sum() / max(mf.sum(), 1)),
        cache_mb_per_video_all_frames=dict(median=float(np.median(fr) * PP_BYTES / 2**20), max=float(fr.max() * PP_BYTES / 2**20)),
        cache_mb_per_video_mid_frames=dict(median=float(np.median(mf) * PP_BYTES / 2**20), max=float(mf.max() * PP_BYTES / 2**20)),
        pairs_total=int(sum(r["pairs"] for r in rows)), clips_total=int(sum(r["clips"] for r in rows)),
        pairs_over_40_clips=int(sum(1 for r in rows if r["max_pair_clips"] > 40)),
        videos_without_pairs=int(sum(1 for r in rows if r["pairs"] == 0)),
        clips_per_mid_frame=float(sum(r["clips"] for r in rows) / max(mf.sum(), 1)),
        bytes_per_frame_fp16=PP_BYTES,
        per_video=rows,
    )
    json.dump(out, open(os.path.join(HERE, "logs", "check3_middle_frames.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_video"}, indent=1))


if __name__ == "__main__":
    main()
