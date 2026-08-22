#!/usr/bin/env python
"""Build the per-category CLIP image-embedding bank the detector conditions on.

`models/methods/detectors/ov_prompt/model.py` loads this at construction:

    self.clip_feat = torch.load(clip_feat_path)          # line 274

and uses it in `forward_train` (OV-DETR's conditioning scheme):

    index = torch.randperm(len(self.clip_feat[cat_id]))[0:1]
    img_query.append(self.clip_feat[cat_id][index])
    ...
    clip_query = text_query * mask + img_query * (1 - mask)   # 75% text / 25% image

So it is a mapping

    category_id (0..34, from openvoc_obj_class_spilt_info.json) -> Tensor (K, 768)

of CLIP ViT-L/14@336px **image** embeddings of ground-truth object crops. Several
exemplars per category, one sampled at random each step.

────────────────────────────────────────────────────────────────────────────────
THIS IS A RECONSTRUCTION, NOT THE AUTHORS' FILE.

The original `clip_L14_feat_vidvrd.pkl` was not published with the code. Every
input here is real -- real GT boxes, real frames, the real CLIP encoder -- but the
authors' exact recipe is unknown:

  * how many exemplars per category they kept        (here: --per-class, default 50)
  * which frames they sampled them from              (here: evenly spaced over each
                                                      trajectory's frame range)
  * whether crops were padded / squared before CLIP  (here: --pad, default 0.0)
  * whether they used the train split only           (here: yes, train only)

Different choices shift the image queries and therefore the trained detector.
Treat any number produced with this bank as *not* directly comparable to the
paper's until the original file is obtained. See docs/PORT_STATUS.md.
────────────────────────────────────────────────────────────────────────────────

    python tools/build_clip_object_bank.py                  # default 50/class
    python tools/build_clip_object_bank.py --per-class 20 --limit 50   # quick run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import torch
from PIL import Image
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from utils import paths
from vlm.backbones import clip

OUT_DEFAULT = os.path.join(paths.META_DIR, "clip_L14_feat_vidvrd.pkl")


def crop_boxes_for_category(traj_json: dict, cls2id: dict, per_class: int):
    """category_id -> list of (video, frame_id, box) sampled across the split."""
    per_cat = defaultdict(list)
    for video, trajs in traj_json.items():
        for t in trajs:
            cid = cls2id.get(t["category"])
            if cid is None:
                continue
            frames = sorted(int(k) for k in t["trajectory"])
            if not frames:
                continue
            # evenly spaced samples along this trajectory, at most 3 per tracklet
            # so no single long tracklet dominates a category
            step = max(1, len(frames) // 3)
            for f in frames[::step][:3]:
                per_cat[cid].append((video, f, t["trajectory"][str(f)]))

    # deterministic subsample to per_class
    g = torch.Generator().manual_seed(0)
    for cid, items in per_cat.items():
        if len(items) > per_class:
            idx = torch.randperm(len(items), generator=g)[:per_class].tolist()
            per_cat[cid] = [items[i] for i in sorted(idx)]
    return per_cat


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=50)
    ap.add_argument("--pad", type=float, default=0.0, help="fraction of box size to pad")
    ap.add_argument("--backbone", default="ViT-L/14@336px")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--limit", type=int, default=None, help="only N categories (debug)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(paths.OBJ_SPLIT_INFO) as f:
        cls2id = json.load(f)["cls2id"]
    with open(paths.TRAIN_TRAJ) as f:
        traj = json.load(f)
    print(f"  categories: {len(cls2id)}   train videos: {len(traj)}")

    per_cat = crop_boxes_for_category(traj, cls2id, args.per_class)
    cats = sorted(per_cat)
    if args.limit:
        cats = cats[:args.limit]
    empty = [c for c in cls2id.values() if c not in per_cat]
    print(f"  categories with exemplars: {len(per_cat)}/{len(cls2id)}"
          + (f"   MISSING: {sorted(empty)}" if empty else ""))

    model, preprocess = clip.load(args.backbone, device=args.device)
    model.eval()

    bank = {}
    for cid in tqdm(cats, desc="encoding"):
        crops = []
        for video, fid, box in per_cat[cid]:
            # frames are 1-indexed on disk; trajectory frame ids are 0-indexed
            p = os.path.join(paths.FRAME_DIR, video, f"{fid + 1:06d}.jpg")
            if not os.path.exists(p):
                continue
            im = Image.open(p).convert("RGB")
            x1, y1, x2, y2 = box
            if args.pad:
                dw, dh = (x2 - x1) * args.pad, (y2 - y1) * args.pad
                x1, y1, x2, y2 = x1 - dw, y1 - dh, x2 + dw, y2 + dh
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(im.width, int(x2)), min(im.height, int(y2))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            crops.append(preprocess(im.crop((x1, y1, x2, y2))))
        if not crops:
            continue
        with torch.no_grad():
            out = model.encode_image(torch.stack(crops).to(args.device))
            # this vendored CLIP's visual tower returns (patch_tokens, projected);
            # data_loading/dataset.py does `_, gp = self.clip.encode_image(...)`
            feats = out[1] if isinstance(out, (tuple, list)) else out
            feats = feats / feats.norm(dim=-1, keepdim=True)
        bank[cid] = feats.float().cpu()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # torch.save, not pickle.dump: the consumer is torch.load in
    # models/methods/detectors/ov_prompt/model.py, which cannot read a bare pickle.
    torch.save(bank, args.out)

    sizes = {c: tuple(v.shape) for c, v in bank.items()}
    print(f"\n  wrote {args.out}")
    print(f"  categories: {len(bank)}   embedding dim: {next(iter(bank.values())).shape[-1]}")
    print(f"  exemplars/category: min {min(v[0] for v in sizes.values())}, "
          f"max {max(v[0] for v in sizes.values())}")
    if len(bank) < len(cls2id):
        print(f"  WARNING: {len(cls2id) - len(bank)} categories have no exemplars; "
              f"forward_train will KeyError on them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
