#!/usr/bin/env python
"""CHECK 2 -- is there a base/novel score bias?  Gates C3.  CPU only.

    python pilot_analysis/scripts/check2_base_novel_bias.py

(a) is answered by code inspection and then verified against the dump.
(b)/(c) are computed on the ALL-split dump, because that is the only pass in
which base and novel predicates compete for the same score -- see the report.

Instance-level scores are the mean over every dumped segment that overlaps the
GT duration (the dump keeps the top 20 predicates per segment, so a predicate
absent from a segment's list contributes 0 there).
"""
from __future__ import annotations

import json, os, statistics, sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from inference.video_relation_detection_openvoc import viou
from utils import paths

PREDS = os.path.join(ROOT, "pilot_analysis", "preds_from_pod", "full")
SPLIT_INFO = os.path.join(ROOT, "configs", "VidVRD_pred_class_spilt_info_v2.json")


def instance_scores(segs, gt, thresh=0.5):
    """-> list of (gt_predicate, ranked [(pred, score), ...]) for every GT
    instance whose tracklet pair the model actually produced."""
    out = []
    for vid, rels in gt.items():
        pairs = segs.get(vid, [])
        for r in rels:
            b, e = r["duration"]
            s, p, o = r["triplet"]
            best, best_ov = None, 0.0
            for pr in pairs:
                if pr["sbj_cls"] != s or pr["obj_cls"] != o:
                    continue
                ov = min(viou(pr["sbj_traj"], pr["duration"], r["sub_traj"], r["duration"]),
                         viou(pr["obj_traj"], pr["duration"], r["obj_traj"], r["duration"]))
                if ov > best_ov:
                    best, best_ov = pr, ov
            if best is None or best_ov < thresh:
                continue
            acc, n = Counter(), 0
            for sg in best["segments"]:
                fs, fe = sg["frame_range"]
                if fe <= b or fs >= e or not sg["preds"]:
                    continue
                n += 1
                for name, sc in sg["preds"]:
                    acc[name] += sc
            if not n:
                continue
            ranked = sorted(((k, v / n) for k, v in acc.items()),
                            key=lambda x: -x[1])
            out.append((p, ranked))
    return out


def main():
    info = json.load(open(SPLIT_INFO))
    split = {p: s for p, s in info["cls2split"].items() if p != "__background__"}
    is_base = {p: s == "base" for p, s in split.items()}
    n_base = sum(is_base.values())
    gt = json.load(open(paths.TEST_RELATION_GT))

    print("=" * 74)
    print("CHECK 2 -- BASE/NOVEL SCORE BIAS")
    print("=" * 74)

    # ---- (a) what does the novel pass actually score? ----------------------
    print("\n(a) DOES THE NOVEL PASS SCORE ALL 132 PREDICATES, OR ONLY THE 61?")
    print("  By inspection -- models/relation_classifier.py L509-L513:")
    print("    text embeddings come from split_text_embeddings(split='novel'),")
    print("    so pre_scores has 61 columns; they are then scattered into a")
    print("    zeros([.., .., 132]) at self.pre_text_encoder.novel_pids.")
    print("  => base predicates carry a hard zero. They cannot be predicted.")
    print("  Verifying against the dump:")
    for name in ("novel", "all"):
        segs = json.load(open(f"{PREDS}/segments_raw_{name}.json"))
        seen = Counter()
        nseg = 0
        for pairs in segs.values():
            for pr in pairs:
                for sg in pr["segments"]:
                    nseg += 1
                    for k, _ in sg["preds"]:
                        seen[k] += 1
        nb = sum(1 for k in seen if is_base[k])
        print(f"    {name:6} pass: {len(seen):3d} distinct predicates ever in a "
              f"top-20 list  ({nb} base, {len(seen)-nb} novel), {nseg} segments")

    # ---- (b)/(c) on the all-split pass -------------------------------------
    segs = json.load(open(f"{PREDS}/segments_raw_all.json"))
    rows = instance_scores(segs, gt)
    print(f"\n  ALL-split pass: {len(rows)} GT instances matched to a predicted "
          f"pair (vIoU>=0.5)")
    nb_gt = sum(1 for p, _ in rows if is_base[p])
    print(f"    of which base-GT {nb_gt}, novel-GT {len(rows)-nb_gt}")

    print(f"\n(b) WHERE DO ERRORS LAND?   candidate set is "
          f"{n_base}/132 = {n_base/132:.1%} base")
    print(f"    {'GT group':10} {'n':>5} {'err':>5} {'top1 base':>11} {'top5 base':>11}")
    stats = {}
    for grp, want_base in (("base-GT", True), ("novel-GT", False)):
        sel = [(p, r) for p, r in rows if is_base[p] == want_base]
        err = [(p, r) for p, r in sel if r and r[0][0] != p]
        t1 = sum(1 for _, r in err if is_base[r[0][0]])
        t5n = t5d = 0
        for _, r in err:
            for k, _s in r[:5]:
                t5d += 1
                t5n += is_base[k]
        stats[grp] = dict(n=len(sel), err=len(err),
                          top1_base=t1 / len(err) if err else float("nan"),
                          top5_base=t5n / t5d if t5d else float("nan"))
        print(f"    {grp:10} {len(sel):5d} {len(err):5d} "
              f"{stats[grp]['top1_base']:10.1%} {stats[grp]['top5_base']:10.1%}")
    print(f"    {'chance':10} {'':5} {'':5} {n_base/132:10.1%} {n_base/132:10.1%}")

    # ---- (c) raw score distributions ---------------------------------------
    print(f"\n(c) RAW SCORE DISTRIBUTIONS (all-split pass)")
    top1_b, top1_n = [], []
    best_b, best_n = [], []
    for _, r in rows:
        (k, s) = r[0]
        (top1_b if is_base[k] else top1_n).append(s)
        bb = next((s for k, s in r if is_base[k]), 0.0)
        nn = next((s for k, s in r if not is_base[k]), 0.0)
        best_b.append(bb); best_n.append(nn)
    print(f"  top-1 winner is a BASE predicate  in {len(top1_b)}/{len(rows)} "
          f"({len(top1_b)/len(rows):.1%}) of instances   (chance {n_base/132:.1%})")
    print(f"    score when the winner is base : mean {statistics.mean(top1_b):.4f}"
          f"  median {statistics.median(top1_b):.4f}")
    if top1_n:
        print(f"    score when the winner is novel: mean {statistics.mean(top1_n):.4f}"
              f"  median {statistics.median(top1_n):.4f}")
    print(f"  paired, within the same instance -- best base vs best novel score:")
    print(f"    best BASE  mean {statistics.mean(best_b):.4f}  "
          f"median {statistics.median(best_b):.4f}")
    print(f"    best NOVEL mean {statistics.mean(best_n):.4f}  "
          f"median {statistics.median(best_n):.4f}")
    d = [a - b for a, b in zip(best_b, best_n)]
    win = sum(1 for x in d if x > 0)
    print(f"    base > novel in {win}/{len(d)} ({win/len(d):.1%}) of instances; "
          f"mean gap {statistics.mean(d):+.4f}")

    # per-predicate mean score, base vs novel, over every segment
    per = defaultdict(list)
    for pairs in segs.values():
        for pr in pairs:
            for sg in pr["segments"]:
                for k, s in sg["preds"]:
                    per[k].append(s)
    mb = [statistics.mean(v) for k, v in per.items() if is_base[k]]
    mn = [statistics.mean(v) for k, v in per.items() if not is_base[k]]
    print(f"  per-predicate mean score when it appears in a top-20 list:")
    print(f"    base  ({len(mb)} predicates) mean-of-means {statistics.mean(mb):.4f}"
          f"  median {statistics.median(mb):.4f}")
    print(f"    novel ({len(mn)} predicates) mean-of-means {statistics.mean(mn):.4f}"
          f"  median {statistics.median(mn):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
