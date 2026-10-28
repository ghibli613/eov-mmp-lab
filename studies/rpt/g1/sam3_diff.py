"""RPT G1 check 3 (A29): when an option's output is not bitwise identical to G0's, the size of the
difference: max |delta| of out_probs and of box coordinates, mask IoU over shared output frames,
and masklets added or lost. Per prompt common to both files; masklets paired by SAM 3 ID."""
from __future__ import annotations

import gzip
import json


def load(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def size(a: dict, b: dict) -> dict:
    from pycocotools import mask as mask_util
    pa, pb = {p["class"]: p for p in a["prompts"]}, {p["class"]: p for p in b["prompts"]}
    d_prob = d_box = 0.0
    ious, added, lost, frames_added, frames_lost = [], 0, 0, 0, 0
    H, W = a["height"], a["width"]
    for c in sorted(set(pa) & set(pb)):
        mx = {m["obj_id"]: {r["f"]: r for r in m["frames"]} for m in pa[c]["masklets"]}
        my = {m["obj_id"]: {r["f"]: r for r in m["frames"]} for m in pb[c]["masklets"]}
        added += len(set(my) - set(mx))
        lost += len(set(mx) - set(my))
        for o in set(mx) & set(my):
            frames_added += len(set(my[o]) - set(mx[o]))
            frames_lost += len(set(mx[o]) - set(my[o]))
            for f in set(mx[o]) & set(my[o]):
                x, y = mx[o][f], my[o][f]
                d_prob = max(d_prob, abs(x["prob"] - y["prob"]))
                d_box = max(d_box, max(abs(u - v) for u, v in zip(x["box"] + x["box_sam3"], y["box"] + y["box_sam3"])))
                rx = {"size": [H, W], "counts": x["rle"].encode("ascii")}
                ry = {"size": [H, W], "counts": y["rle"].encode("ascii")}
                ious.append(float(mask_util.iou([rx], [ry], [0])[0][0]))
    ious.sort()
    return dict(max_abs_delta_out_probs=d_prob, max_abs_delta_box=d_box,
                mask_iou_min=ious[0] if ious else None, mask_iou_mean=sum(ious) / len(ious) if ious else None,
                mask_iou_below_1=sum(i < 1.0 for i in ious), shared_masks=len(ious),
                masklets_added=added, masklets_lost=lost, frames_added=frames_added, frames_lost=frames_lost)
