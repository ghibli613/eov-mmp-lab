#!/usr/bin/env python
"""Is the 30-frame segment grid itself a ceiling on vIoU? No model, no GPU.

    python pilot_analysis/scripts/temporal_grid_ceiling.py

post_process.process_pred cuts the tracklet pair into NON-OVERLAPPING 30-frame
blocks starting at the pair's own begin_fid, so any predicted temporal extent is
a union of consecutive [phase + 30k, phase + 30(k+1)) blocks. GT extents, by
contrast, sit on a 15-frame grid. This asks: assuming perfect boxes, what is the
best vIoU a grid-aligned prediction can reach against each GT instance?

Assumption (stated because it is load-bearing): constant per-frame box area, so
vIoU reduces to I / (P + G - I) on frame counts. The evaluator's real vIoU
(inference/video_relation_detection_openvoc.py) weights by box area, so this is
an idealisation, not a bound on the real metric.
"""
import glob
import json
from collections import Counter

SEG = 30
THRESH = 0.5  # evaluator uses `ov >= viou_threshold`
SPLIT = json.load(open("configs/VidVRD_pred_class_spilt_info_v2.json"))["cls2split"]


def best_viou(b, e, phase=0):
    """Max I/(P+G-I) over predicted extents that are unions of grid blocks."""
    g = e - b
    best = 0.0
    k0 = (b - phase) // SEG - 1
    k1 = (e - phase) // SEG + 1
    for i in range(k0, k1 + 1):
        for j in range(i + 1, k1 + 2):
            ps, pe = phase + i * SEG, phase + j * SEG
            inter = max(0, min(e, pe) - max(b, ps))
            if inter <= 0:
                continue
            best = max(best, inter / ((pe - ps) + g - inter))
    return best


def main():
    rows = []
    for f in sorted(glob.glob("data/vidvrd/anno/test/*.json")):
        d = json.load(open(f))
        for r in d["relation_instances"]:
            rows.append((r["begin_fid"], r["end_fid"],
                         SPLIT.get(r["predicate"], "?")))
    n = len(rows)

    for phase in (0, 15):
        vs = [best_viou(b, e, phase) for b, e, _ in rows]
        nov = [v for v, (_, _, s) in zip(vs, rows) if s == "novel"]
        bad = sum(v < THRESH for v in vs)
        badn = sum(v < THRESH for v in nov)
        print(f"grid phase {phase:2d}: mean best vIoU {sum(vs)/n:.3f} | "
              f"unmatchable (<{THRESH}): {bad}/{n} = {bad/n:.1%} "
              f"(novel {badn}/{len(nov)} = {badn/len(nov):.1%})")

    vs = [best_viou(b, e, 0) for b, e, _ in rows]
    print("\nphase 0, margin above the threshold:")
    for thr in (0.50, 0.55, 0.60, 0.70):
        c = sum(v <= thr + 1e-9 for v in vs)
        print(f"  best vIoU <= {thr:.2f}: {c:5d} / {n}  ({c/n:5.1%})")
    zero = sum(abs(v - THRESH) < 1e-9 for v in vs)
    print(f"  exactly {THRESH} (zero margin): {zero} ({zero/n:.1%})")

    print("\nphase 0, unmatchable by GT duration:")
    buck, bad = Counter(), Counter()
    for (b, e, _), v in zip(rows, vs):
        k = min((e - b) // SEG, 5)
        buck[k] += 1
        if v < THRESH:
            bad[k] += 1
    for k in sorted(buck):
        lab = f"{k*SEG}-{k*SEG+SEG-1}" if k < 5 else f"{5*SEG}+"
        print(f"  dur {lab:>8}: {buck[k]:5d} instances, {bad[k]:5d} unmatchable "
              f"({bad[k]/buck[k]:5.1%})")


if __name__ == "__main__":
    main()
