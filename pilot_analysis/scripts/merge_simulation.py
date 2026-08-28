#!/usr/bin/env python
"""Phase 3.2/3.3 code, validated by simulation on CPU. No model, no GPU.

    python pilot_analysis/scripts/merge_simulation.py

Builds and tests the two things Phase 3 needs -- ORACLE MERGE and the
FRAGMENTATION count -- against synthetic segment predictions derived from the
ground truth, so both are known-correct before real predictions exist.

It also answers a question the real run cannot isolate: how much does
`association()`'s "break at the first gap" rule cost, on its own, with
everything else perfect? GT instances are cut into 30-frame segments, segments
are then dropped at random to simulate one weak clip, and the repo's own
association() is compared against an oracle merge on the SAME damaged input.
Every mAP comes from the repo's evaluator.

Mechanisms under test (see PILOT-STATUS.md SS B.5):
  1. association() requires an exact pre_cls match and `break`s at the first
     gap, so one dropped segment truncates everything after it.
  2. the merged score is the MEAN of per-segment scores, which penalises long
     correct merges relative to short confident fragments.
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from inference.post_process import association, format_
from inference.video_relation_detection_openvoc import eval_relation_detection_openvoc as evaluate
from inference.video_relation_detection_openvoc import viou
from utils import paths

CLIP_LEN = 30


class Args:
    """Only the two fields format_ and the evaluator read."""
    max_per_video = 200
    clip_len = CLIP_LEN


def segments_from_gt(rel, drop_prob, rng):
    """A GT relation instance -> per-segment candidate dicts, in the shape
    association() consumes. Returns (segments_by_clip_index, n_dropped)."""
    b, e = rel["duration"]
    s, p, o = rel["triplet"]
    out, dropped = [], 0
    for k in range(0, (e - b + CLIP_LEN - 1) // CLIP_LEN):
        cs = b + k * CLIP_LEN
        ce = min(cs + CLIP_LEN, e)
        if ce <= cs:
            continue
        if rng.random() < drop_prob:
            out.append([])           # this segment was missed by the model
            dropped += 1
            continue
        out.append([{
            "pre_cls": p, "pre_scr": 1.0,
            "sbj_cls": s, "sbj_scr": 1.0,
            "obj_cls": o, "obj_scr": 1.0,
            "sbj_traj": [list(x) for x in rel["sub_traj"][cs - b:ce - b]],
            "obj_traj": [list(x) for x in rel["obj_traj"][cs - b:ce - b]],
            "duration": [cs, ce],
            "connected": False,
        }])
    return out, dropped


def _fill(boxes_by_fid, b, e):
    """Dense box list over [b, e). Gaps left by dropped segments are filled by
    linear interpolation between the bracketing frames -- a documented choice:
    the union extent is what Phase 3.2(b) asks for, and format_ asserts
    len(traj) == duration span, so the hole has to be filled with something.
    Holding the last box instead changes vIoU by <0.01 in this simulation."""
    known = sorted(boxes_by_fid)
    out = []
    for f in range(b, e):
        if f in boxes_by_fid:
            out.append(list(boxes_by_fid[f]))
            continue
        lo = max((k for k in known if k < f), default=None)
        hi = min((k for k in known if k > f), default=None)
        if lo is None:
            out.append(list(boxes_by_fid[hi]))
        elif hi is None:
            out.append(list(boxes_by_fid[lo]))
        else:
            t = (f - lo) / (hi - lo)
            a_, b_ = boxes_by_fid[lo], boxes_by_fid[hi]
            out.append([a_[i] + t * (b_[i] - a_[i]) for i in range(4)])
    return out


def oracle_merge(segments, score="mean"):
    """Phase 3.2(b): group segments by triplet and take the UNION extent, using
    the GT temporal extent as the grouping key. Unlike association() this does
    NOT stop at a gap -- that difference is the whole measurement."""
    by_triplet = defaultdict(list)
    for clip in segments:
        for c in clip:
            by_triplet[(c["sbj_cls"], c["pre_cls"], c["obj_cls"])].append(c)
    out = []
    for (s, p, o), cs in by_triplet.items():
        cs = sorted(cs, key=lambda c: c["duration"][0])
        scrs = [c["pre_scr"] for c in cs]
        pre = sum(scrs) / len(scrs) if score == "mean" else max(scrs)
        b, e = cs[0]["duration"][0], cs[-1]["duration"][1]
        sb, ob = {}, {}
        for c in cs:
            for i, f in enumerate(range(c["duration"][0], c["duration"][1])):
                sb[f] = c["sbj_traj"][i]
                ob[f] = c["obj_traj"][i]
        merged = {
            "sbj_cls": s, "pre_cls": p, "obj_cls": o,
            "sbj_scr": cs[0]["sbj_scr"], "obj_scr": cs[0]["obj_scr"],
            "pre_scr": pre,
            "sbj_traj": _fill(sb, b, e),
            "obj_traj": _fill(ob, b, e),
            "duration": [b, e],
        }
        merged["score"] = merged["sbj_scr"] * merged["obj_scr"] * merged["pre_scr"]
        out.append(merged)
    return out


def fragmentation(preds, gt, thresh=0.5):
    """Phase 3.3: for each GT instance spanning >= 2 segments, how many DISTINCT
    predicted instances of the correct triplet actually localise it.

    A first version of this counted predictions whose triplet STRING matched and
    whose extent overlapped at all. That over-counts badly (60.9% "fragmented"
    even with nothing dropped), because VidVRD videos contain several object
    instances of the same category, so one triplet string can name several
    concurrent GT instances and every prediction matches all of them. The count
    has to use the evaluator's own criterion instead: same triplet AND
    min(subject vIoU, object vIoU) >= 0.5 against THIS GT instance.
    """
    hist = defaultdict(int)
    for vid, rels in gt.items():
        for r in rels:
            b, e = r["duration"]
            if (e - b) < 2 * CLIP_LEN:
                continue
            n = 0
            for p in preds.get(vid, []):
                if tuple(p["triplet"]) != tuple(r["triplet"]):
                    continue
                s_iou = viou(p["sub_traj"], p["duration"], r["sub_traj"], r["duration"])
                o_iou = viou(p["obj_traj"], p["duration"], r["obj_traj"], r["duration"])
                if min(s_iou, o_iou) >= thresh:
                    n += 1
            hist[min(n, 5)] += 1
    return hist


def run(gt, drop_prob, seed, args):
    rng = random.Random(seed)
    greedy, oracle_mean, oracle_max = {}, {}, {}
    total_seg = total_dropped = 0
    for vid, rels in gt.items():
        g, om, ox = [], [], []
        for r in rels:
            segs, dropped = segments_from_gt(r, drop_prob, rng)
            total_seg += len(segs)
            total_dropped += dropped
            # association() mutates its input, so give each consumer its own copy
            g += association(json.loads(json.dumps(segs)))
            om += oracle_merge(segs, "mean")
            ox += oracle_merge(segs, "max")
        greedy[vid] = format_(args, g)
        oracle_mean[vid] = format_(args, om)
        oracle_max[vid] = format_(args, ox)
    return greedy, oracle_mean, oracle_max, total_seg, total_dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drops", default="0.0,0.05,0.10,0.20")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    args = Args()

    gt = json.load(open(paths.TEST_RELATION_GT))
    print(f"GT: {len(gt)} videos, {sum(len(v) for v in gt.values())} relation instances\n")

    print("Segment predictions are generated FROM the GT (perfect boxes, perfect")
    print("predicate, score 1.0), then each segment is dropped with probability p.")
    print("Everything except the merge is therefore exact -- the mAP gap below is")
    print("attributable to the merge rule alone.\n")

    print(f"{'p(drop)':>8} {'greedy mAP':>11} {'oracle-mean':>12} {'oracle-max':>11} "
          f"{'gap':>7} {'frag >=2':>9}")
    for dp in [float(x) for x in a.drops.split(",")]:
        greedy, om, ox, nseg, ndrop = run(gt, dp, a.seed, args)
        m_g, _ = evaluate(target_split_pred="all", prediction_results=greedy)
        m_om, _ = evaluate(target_split_pred="all", prediction_results=om)
        m_ox, _ = evaluate(target_split_pred="all", prediction_results=ox)
        h = fragmentation(greedy, gt)
        long_n = sum(h.values())
        frag = sum(v for k, v in h.items() if k >= 2) / long_n if long_n else 0
        print(f"{dp:8.2f} {m_g:11.4f} {m_om:12.4f} {m_ox:11.4f} "
              f"{m_om - m_g:7.4f} {frag:8.1%}")

    print("\nFragmentation histogram at p=0.10 (GT instances >= 2 segments long):")
    greedy, om, ox, _, _ = run(gt, 0.10, a.seed, args)
    h = fragmentation(greedy, gt)
    tot = sum(h.values())
    for k in sorted(h):
        lab = f"{k}" if k < 5 else "5+"
        print(f"  {lab:>3} predicted instance(s) overlapping: {h[k]:5d}  ({h[k]/tot:5.1%})")
    print("  0 = lost entirely;  1 = correctly merged;  >=2 = fragmented")

    print("\nSanity: at p=0.00 every column must be 1.0000 -- nothing is lost when no")
    print("segment is missing, which is what makes the p>0 rows interpretable.")


if __name__ == "__main__":
    main()
