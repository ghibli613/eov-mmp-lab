#!/usr/bin/env python
"""Phase 3.1 -- GT relation-instance duration statistics. No model, no GPU.

    python pilot_analysis/scripts/gt_duration_stats.py

Reads data/vidvrd/anno/test/ and configs/VidVRD_pred_class_spilt_info_v2.json.
Segment length is args.clip_len == 30 (see inference/post_process.py).
"""
import glob
import json
import statistics as st
from collections import Counter

SEG = 30
SPLIT = json.load(open("configs/VidVRD_pred_class_spilt_info_v2.json"))["cls2split"]


def load():
    rows = []
    nvid = 0
    for f in sorted(glob.glob("data/vidvrd/anno/test/*.json")):
        d = json.load(open(f))
        nvid += 1
        for r in d["relation_instances"]:
            rows.append((r["begin_fid"], r["end_fid"],
                         r["predicate"], SPLIT.get(r["predicate"], "?")))
    return nvid, rows


def main():
    nvid, rows = load()
    durs = [e - b for b, e, _, _ in rows]
    print(f"test videos: {nvid}   GT relation instances: {len(rows)}")
    print(f"duration frames: min {min(durs)}  median {st.median(durs)}  "
          f"mean {st.mean(durs):.1f}  max {max(durs)}\n")

    print(f"{'group':10} {'n':>6} {'>1x30':>8} {'>2x60':>8} {'>4x120':>8} {'med':>6} {'mean':>7}")
    for name, keep in (("all", lambda s: True),
                       ("base", lambda s: s == "base"),
                       ("novel", lambda s: s == "novel")):
        v = [e - b for b, e, _, s in rows if keep(s)]
        n = len(v)
        print(f"{name:10} {n:6d} "
              f"{sum(x > SEG for x in v)/n:7.1%} "
              f"{sum(x > 2*SEG for x in v)/n:7.1%} "
              f"{sum(x > 4*SEG for x in v)/n:7.1%} "
              f"{st.median(v):6.0f} {st.mean(v):7.1f}")

    print("\nsegments spanned (ceil dur/30), 9+ collapsed:")
    spans = Counter(min(-(-d // SEG), 9) for d in durs)
    for k in sorted(spans):
        print(f"  {k}{'+' if k == 9 else ' '}: {spans[k]:5d}  ({spans[k]/len(durs):5.1%})")

    # Annotation grid: are extents 30-aligned, or 15-aligned?
    off = Counter(d % SEG for d in durs)
    beg = Counter(b % SEG for b, _, _, _ in rows)
    nm = sum(v for k, v in off.items() if k)
    print(f"\nduration NOT a multiple of {SEG}: {nm} ({nm/len(durs):.2%}); "
          f"remainders {dict((k, v) for k, v in off.items() if k)}")
    print(f"begin_fid mod {SEG}: {dict(beg)}")


if __name__ == "__main__":
    main()
