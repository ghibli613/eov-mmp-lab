#!/usr/bin/env python
"""Phase 3.1, remaining half -- per-predicate durations joined with the
partition, plus the CSV and histogram the protocol asks for. No GPU.

    python pilot_analysis/scripts/duration_by_predicate.py

gt_duration_stats.py covers the aggregate distribution. This adds what the
protocol also requires: "per-predicate mean duration (join with the partition --
expectation: dynamic predicates last longer). Output a histogram figure + CSV."

Writes:
  pilot_analysis/duration_by_predicate.csv    one row per predicate
  pilot_analysis/duration_histogram.png       the figure
"""
import csv
import glob
import json
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEG = 30
OUT = "pilot_analysis"
GROUPS = ("geometric_static", "geometric_dynamic", "appearance_static", "appearance_dynamic")


def main():
    part = json.load(open(f"{OUT}/predicate_partition.json"))
    durs = defaultdict(list)
    for f in sorted(glob.glob("data/vidvrd/anno/test/*.json")):
        for r in json.load(open(f))["relation_instances"]:
            durs[r["predicate"]].append(r["end_fid"] - r["begin_fid"])

    # ---------------------------------------------------------------- CSV
    rows = []
    for p, meta in sorted(part.items()):
        v = durs.get(p, [])
        rows.append({
            "predicate": p,
            "group": meta["group"],
            "evidence": meta["evidence"],
            "time": meta["time"],
            "ov_split": meta["ov_split"],
            "instances": len(v),
            "mean_frames": round(st.mean(v), 1) if v else "",
            "median_frames": st.median(v) if v else "",
            "max_frames": max(v) if v else "",
            "mean_segments": round(st.mean(v) / SEG, 2) if v else "",
            "frac_over_1seg": round(sum(x > SEG for x in v) / len(v), 3) if v else "",
            "review": meta["review"],
        })
    with open(f"{OUT}/duration_by_predicate.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------- the protocol's expectation
    print("PROTOCOL'S EXPECTATION: 'dynamic predicates last longer'\n")
    print(f"{'group':22} {'preds':>6} {'inst':>6} {'mean dur':>9} {'median':>8} {'mean segs':>10}")
    per_group = {}
    for g in GROUPS:
        v = [d for p, m in part.items() if m["group"] == g for d in durs.get(p, [])]
        per_group[g] = v
        print(f"{g:22} {sum(1 for m in part.values() if m['group']==g):6d} {len(v):6d} "
              f"{st.mean(v):9.1f} {st.median(v):8.0f} {st.mean(v)/SEG:10.2f}")

    stat = [d for g in ("geometric_static", "appearance_static") for d in per_group[g]]
    dyn = [d for g in ("geometric_dynamic", "appearance_dynamic") for d in per_group[g]]
    print(f"\n{'all static':22} {'':6} {len(stat):6d} {st.mean(stat):9.1f} {st.median(stat):8.0f}")
    print(f"{'all dynamic':22} {'':6} {len(dyn):6d} {st.mean(dyn):9.1f} {st.median(dyn):8.0f}")
    verdict = ("CONFIRMED" if st.mean(dyn) > st.mean(stat) * 1.05 else
               "NOT CONFIRMED" if st.mean(dyn) < st.mean(stat) * 0.95 else
               "NO DIFFERENCE")
    print(f"\n  -> dynamic/static mean-duration ratio {st.mean(dyn)/st.mean(stat):.3f}  **{verdict}**")

    print("\n10 longest-lasting predicates:")
    top = sorted((r for r in rows if r["instances"] >= 5),
                 key=lambda r: -r["mean_frames"])[:10]
    for r in top:
        print(f"  {r['predicate']:16} {r['mean_frames']:7.1f} frames "
              f"({r['mean_segments']:4.1f} segs, n={r['instances']:4d})  [{r['group']}]")

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    allv = [d for v in durs.values() for d in v]
    bins = list(range(0, 601, 30))
    axes[0].hist([min(x, 600) for x in allv], bins=bins, color="#4C72B0", edgecolor="white")
    for k, ls in ((1, "-"), (2, "--"), (4, ":")):
        axes[0].axvline(k * SEG, color="#C44E52", ls=ls, lw=1.4,
                        label=f"{k}x segment ({k*SEG}f)")
    axes[0].set_xlabel("GT relation-instance duration (frames, 600+ clipped)")
    axes[0].set_ylabel("instances")
    axes[0].set_title(f"All {len(allv)} test instances (segment length = {SEG})")
    axes[0].legend(frameon=False, fontsize=9)

    data = [per_group[g] for g in GROUPS]
    bp = axes[1].boxplot([[min(x, 600) for x in d] for d in data], vert=True,
                         patch_artist=True, showfliers=False,
                         tick_labels=[g.replace("_", "\n") for g in GROUPS])
    for patch, c in zip(bp["boxes"], ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]):
        patch.set_facecolor(c)
    axes[1].axhline(SEG, color="#C44E52", ls="--", lw=1.2, label=f"1 segment ({SEG}f)")
    axes[1].set_ylabel("duration (frames)")
    axes[1].set_title("Duration by partition group")
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/duration_histogram.png", dpi=150)

    print(f"\nwrote {OUT}/duration_by_predicate.csv ({len(rows)} rows)")
    print(f"wrote {OUT}/duration_histogram.png")


if __name__ == "__main__":
    main()
