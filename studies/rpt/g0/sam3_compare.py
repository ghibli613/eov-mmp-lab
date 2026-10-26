#!/usr/bin/env python3
"""RPT G0 runtime measurement (A28): is SAM 3's output bitwise identical between two runs of the
runner on the same video, and how long did each take?

    python studies/rpt/g0/sam3_compare.py A.json.gz B.json.gz [--out OUT.json]

Compared per prompt common to both: the masklets (IDs, and every per-frame field: mask RLE, tight
box, SAM 3's box, out_probs, status, recorded detection and tracker scores) and the raw detections
in [0.1, 0.5). Timing and memory fields are not compared; they give the speedup.
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter

TIMING = {"seconds", "seconds_per_frame", "peak_gpu_allocated_gib", "peak_gpu_reserved_gib"}


def load(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def compare(a, b):
    pa, pb = {p["class"]: p for p in a["prompts"]}, {p["class"]: p for p in b["prompts"]}
    common = [c for c in pa if c in pb]
    diffs, prompts_identical = Counter(), 0
    for c in common:
        x, y = pa[c], pb[c]
        same = True
        for k in sorted((set(x) | set(y)) - TIMING - {"masklets"}):
            if x.get(k) != y.get(k):
                diffs[f"prompt.{k}"] += 1
                same = False
        mx, my = {m["obj_id"]: m["frames"] for m in x["masklets"]}, {m["obj_id"]: m["frames"] for m in y["masklets"]}
        if set(mx) != set(my):
            diffs["masklet ids"] += 1
            same = False
        for o in set(mx) & set(my):
            fx, fy = {r["f"]: r for r in mx[o]}, {r["f"]: r for r in my[o]}
            if set(fx) != set(fy):
                diffs["masklet frames"] += 1
                same = False
            for f in set(fx) & set(fy):
                for k in set(fx[f]) | set(fy[f]):
                    if fx[f].get(k) != fy[f].get(k):
                        diffs[f"frame.{k}"] += 1
                        same = False
        prompts_identical += same
    sa, sb = sum(pa[c]["seconds"] for c in common), sum(pb[c]["seconds"] for c in common)
    return dict(video=a["video"], prompts_compared=len(common), prompts_identical=prompts_identical,
                bitwise_identical=prompts_identical == len(common) and not diffs, differences=dict(diffs),
                seconds_a=round(sa, 2), seconds_b=round(sb, 2), speedup_a_over_b=round(sa / sb, 3) if sb else None,
                reuse_backbone=(a.get("reuse_backbone"), b.get("reuse_backbone")),
                backbone_cache=b.get("backbone_cache") or a.get("backbone_cache"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--out")
    args = ap.parse_args()
    res = compare(load(args.a), load(args.b))
    res.update(a=args.a, b=args.b)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
