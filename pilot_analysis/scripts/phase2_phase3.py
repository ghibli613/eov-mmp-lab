#!/usr/bin/env python
"""Phases 2 and 3 from saved predictions. CPU only -- no GPU, no model.

    python pilot_analysis/scripts/phase2_phase3.py --preds pilot_analysis/preds/full

Consumes what dump_predictions.py wrote and produces the numbers the study
actually turns on:

  H1'  geometric_dynamic vs geometric_static mAP, both splits, against the
       <= 0.5x criterion -- plus a per-verb-family breakdown, because the two
       groups differ in label granularity (78 predicates vs 44) and that is a
       complete alternative explanation for any gap (see PILOT-STATUS.md SS B.19).
  H1'  mechanism check: are novel-predicate confusions concentrated WITHIN a
       verb family (fly_front -> fly_right) rather than across verbs? Predicted
       in SS B.11 before any prediction existed.
  H2   oracle-merge gain and the loss rate, rebuilt from segments_raw_*.json.
       Fragmentation is reported too, but SS C.5 shows the loss rate is the
       criterion that means anything here.

Every mAP comes from the repo's own evaluator (ground rule 2).

VALIDATION STATUS. This script has been exercised end to end on synthetic
predictions derived from GT: every table renders, no crashes, and the Phase 2 /
2.3 numbers behave sensibly. The Phase 3 numbers from that fixture are NOT
meaningful, and the reason is worth knowing: the fixture's merged predictions
were GT-clean (one correct instance each) while its segment dump carried five
candidates per segment, so the oracle rebuild wore several times the
false-positive load of the baseline and reported a negative gain. Real dumps do
not have that asymmetry -- association() also emits candidates from every one of
the clip_top_n predicates. For a validated H2 direction on consistent inputs see
merge_simulation.py, which builds both sides from the same segments and measures
an 11-26 mAP oracle gain. Treat a negative gain here as a signal to check the
inputs, not as evidence against H2.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))

from eval_at_threshold import eval_at
from inference.video_relation_detection_openvoc import viou
from utils import paths

GROUPS = ("geometric_static", "geometric_dynamic", "appearance_static", "appearance_dynamic")
PART = "pilot_analysis/predicate_partition.json"
GROUP_SPLIT = "pilot_analysis/splits/pred_split_group_x_ov.json"
CLIP_LEN = 30


def verb_of(pred):
    return pred.split("_", 1)[0] if "_" in pred else pred


def write_verb_split(part, out):
    """A cls2split file keyed by verb family x base/novel, so the evaluator can
    score one family at a time with no new AP code."""
    m = {p: f"{verb_of(p)}__{d['ov_split']}" for p, d in part.items()}
    m["__background__"] = "__background__"
    json.dump({"cls2split": m}, open(out, "w"), indent=1)
    return out


# ----------------------------------------------------------------- Phase 2
def phase2(preds, part):
    print("=" * 74)
    print("PHASE 2 / H1'  -- per-group mAP (repo evaluator, group split file)")
    print("=" * 74)
    res = {}
    print(f"\n{'group':22} {'split':6} {'mAP':>8} {'R@50':>8} {'R@100':>8}")
    for g in GROUPS:
        for sp in ("base", "novel"):
            try:
                m, r = eval_at(preds, target_split_pred=f"{g}__{sp}",
                               pred_cls_split_info_path=GROUP_SPLIT)
            except Exception as e:
                print(f"{g:22} {sp:6}  failed: {type(e).__name__}")
                continue
            res[(g, sp)] = m
            note = "" if g.startswith("geometric") else "  <- descriptive, small n"
            print(f"{g:22} {sp:6} {m*100:8.2f} {r[50]*100:8.2f} {r[100]*100:8.2f}{note}")

    print("\n--- H1' VERDICT " + "-" * 58)
    for sp in ("base", "novel"):
        st = res.get(("geometric_static", sp))
        dy = res.get(("geometric_dynamic", sp))
        if st is None or dy is None or st == 0:
            print(f"  {sp:6}: insufficient data")
            continue
        ratio = dy / st
        verdict = "SUPPORTED" if ratio <= 0.5 else "NOT SUPPORTED"
        print(f"  {sp:6}: dynamic {dy*100:.2f} / static {st*100:.2f} = ratio {ratio:.3f}"
              f"   -> {verdict} (criterion <= 0.500)")
    print("  NOTE: SS B.19 argues the 0.5x bar is a large leap from the mechanism")
    print("        evidence (an 0.0086 gap in text separability). Report the ratio")
    print("        with an interval; treat 0.5 as a landmark, not a pass/fail line.")
    return res


def phase2_verb_control(preds, part, tmpdir):
    print("\n" + "=" * 74)
    print("PHASE 2 CONTROL -- per verb family (holds label granularity constant)")
    print("=" * 74)
    vs = write_verb_split(part, os.path.join(tmpdir, "pred_split_verb_x_ov.json"))
    fams = defaultdict(lambda: {"n": 0, "group": None})
    for p, d in part.items():
        if "_" not in p:
            continue
        f = fams[verb_of(p)]
        f["n"] += 1
        f["group"] = d["group"]
    rows = []
    for v in sorted(fams, key=lambda v: (fams[v]["group"], v)):
        if fams[v]["n"] < 4:            # `fall` and `next` are 1 variant each
            continue
        for sp in ("base", "novel"):
            try:
                m, _ = eval_at(preds, target_split_pred=f"{v}__{sp}",
                               pred_cls_split_info_path=vs)
            except Exception:
                continue
            rows.append((v, fams[v]["group"], sp, fams[v]["n"], m))
    print(f"\n{'verb':8} {'group':20} {'split':6} {'variants':>9} {'mAP':>8}")
    for v, g, sp, n, m in rows:
        print(f"{v:8} {g:20} {sp:6} {n:9d} {m*100:8.2f}")
    print("\n  matched pairs (similar variant counts, one static one dynamic):")
    for a, b in (("stand", "walk"), ("sit", "run"), ("lie", "fly"), ("stop", "creep")):
        for sp in ("base", "novel"):
            ma = next((m for v, _, s, _, m in rows if v == a and s == sp), None)
            mb = next((m for v, _, s, _, m in rows if v == b and s == sp), None)
            if ma is None or mb is None:
                continue
            r = (mb / ma) if ma else float("nan")
            print(f"    {sp:6} {a:6}(static) {ma*100:6.2f}  vs  {b:6}(dynamic) "
                  f"{mb*100:6.2f}   ratio {r:.3f}")
    return rows


# ---------------------------------------------------- Phase 2.3 confusion
def phase2_confusion(segs, gt, part, split="novel"):
    print("\n" + "=" * 74)
    print(f"PHASE 2.3 -- what {split} predicates get confused with (mechanism check)")
    print("=" * 74)
    within = across = 0
    table = Counter()
    for vid, rels in gt.items():
        for r in rels:
            p = r["triplet"][1]
            if part.get(p, {}).get("ov_split") != split:
                continue
            b, e = r["duration"]
            best, best_ov = None, 0.0
            for pair in segs.get(vid, []):
                if pair["sbj_cls"] != r["triplet"][0] or pair["obj_cls"] != r["triplet"][2]:
                    continue
                s_iou = viou(pair["sbj_traj"], pair["duration"], r["sub_traj"], r["duration"])
                o_iou = viou(pair["obj_traj"], pair["duration"], r["obj_traj"], r["duration"])
                ov = min(s_iou, o_iou)
                if ov > best_ov:
                    best, best_ov = pair, ov
            if best is None or best_ov < 0.5:
                continue
            # top-1 predicate over the segments overlapping this GT instance
            votes = Counter()
            for s in best["segments"]:
                fs, fe = s["frame_range"]
                if fe <= b or fs >= e:
                    continue
                if s["preds"]:
                    votes[s["preds"][0][0]] += 1
            if not votes:
                continue
            top = votes.most_common(1)[0][0]
            if top == p:
                continue
            table[(p, top)] += 1
            if verb_of(top) == verb_of(p):
                within += 1
            else:
                across += 1
    tot = within + across
    print(f"\nmisclassified {split} instances with a matched tracklet pair: {tot}")
    if tot:
        print(f"  confusion WITHIN the same verb family : {within:5d}  ({within/tot:5.1%})")
        print(f"  confusion ACROSS verb families        : {across:5d}  ({across/tot:5.1%})")
        pred = "CONFIRMED" if within / tot > 0.5 else "NOT CONFIRMED"
        print(f"\n  SS B.11 predicted errors cluster WITHIN a verb family -> {pred}")
        print("\n  top 15 confusions (GT -> predicted):")
        for (a, b_), n in table.most_common(15):
            mark = "same-verb" if verb_of(a) == verb_of(b_) else "cross-verb"
            print(f"    {a:16} -> {b_:16} {n:4d}   {mark}")
    return table


# ----------------------------------------------------------------- Phase 3
def rebuild_oracle(segs, gt, score="mean", max_per_video=200):
    """Phase 3.2(b): group each pair's segments by predicate over the UNION extent
    instead of stopping at the first gap, and slice the pair's trajectory to match.

    The result is then sorted by score and truncated to max_per_video, exactly as
    inference/post_process.format_ does to the greedy output. Without that the two
    are not comparable: the oracle emits up to clip_top_n instances per pair, and
    the extra low-scoring ones destroy precision, which shows up as a NEGATIVE
    oracle gain and would read as "H2 not supported" for purely procedural
    reasons. format_'s truncation is part of the pipeline, so the oracle has to
    wear it too.
    """
    out = defaultdict(list)
    for vid, pairs in segs.items():
        for pair in pairs:
            pb = pair["duration"][0]
            by_pred = defaultdict(list)
            for s in pair["segments"]:
                for pre, scr in s["preds"]:
                    by_pred[pre].append((s["frame_range"], scr))
            for pre, items in by_pred.items():
                items.sort(key=lambda x: x[0][0])
                b = items[0][0][0]
                e = items[-1][0][1]
                scrs = [x[1] for x in items]
                pre_scr = sum(scrs) / len(scrs) if score == "mean" else max(scrs)
                out[vid].append({
                    "triplet": [pair["sbj_cls"], pre, pair["obj_cls"]],
                    "score": pair["sbj_scr"] * pair["obj_scr"] * pre_scr,
                    "sub_traj": pair["sbj_traj"][b - pb:e - pb],
                    "obj_traj": pair["obj_traj"][b - pb:e - pb],
                    "duration": [b, e],
                })
    for vid in out:
        out[vid].sort(key=lambda r: r["score"], reverse=True)
        del out[vid][max_per_video:]
    return dict(out)


def loss_and_fragmentation(preds, gt, thresh=0.5):
    """Phase 3.3, using the evaluator's own matching criterion."""
    hist = Counter()
    for vid, rels in gt.items():
        for r in rels:
            b, e = r["duration"]
            if (e - b) < 2 * CLIP_LEN:
                continue
            n = 0
            for p in preds.get(vid, []):
                if tuple(p["triplet"]) != tuple(r["triplet"]):
                    continue
                if min(viou(p["sub_traj"], p["duration"], r["sub_traj"], r["duration"]),
                       viou(p["obj_traj"], p["duration"], r["obj_traj"], r["duration"])) >= thresh:
                    n += 1
            hist[min(n, 5)] += 1
    return hist


def phase3(merged, segs, gt):
    print("\n" + "=" * 74)
    print("PHASE 3 / H2 -- the price of the greedy merge")
    print("=" * 74)
    res = {}
    for name, pr in (("(a) greedy association()", merged),
                     ("(b) oracle merge, mean", rebuild_oracle(segs, gt, "mean")),
                     ("(b) oracle merge, max ", rebuild_oracle(segs, gt, "max"))):
        for sp in ("all", "novel"):
            m, _ = eval_at(pr, target_split_pred=sp)
            res[(name, sp)] = m
    print(f"\n{'variant':26} {'all mAP':>9} {'novel mAP':>11}")
    for name in ("(a) greedy association()", "(b) oracle merge, mean", "(b) oracle merge, max "):
        print(f"{name:26} {res[(name,'all')]*100:9.2f} {res[(name,'novel')]*100:11.2f}")
    for sp in ("all", "novel"):
        gain = max(res[("(b) oracle merge, mean", sp)],
                   res[("(b) oracle merge, max ", sp)]) - res[("(a) greedy association()", sp)]
        v = "SUPPORTED" if gain * 100 >= 2.0 else "NOT SUPPORTED"
        print(f"\n  {sp:6}: oracle-merge gain {gain*100:+.2f} mAP -> H2 {v} (criterion >= 2.0)")

    print("\n--- Phase 3.2(c) relaxed vIoU (how much is mislocalisation) ---")
    print(f"  {'thr':>5} {'greedy':>9} {'oracle':>9} {'gap':>8}")
    om = rebuild_oracle(segs, gt, "mean")
    for thr in (0.5, 0.3, 0.1):
        a, _ = eval_at(merged, viou_threshold=thr, target_split_pred="all")
        b, _ = eval_at(om, viou_threshold=thr, target_split_pred="all")
        print(f"  {thr:5.1f} {a*100:9.2f} {b*100:9.2f} {(b-a)*100:8.2f}")

    print("\n--- Phase 3.3 -- loss rate, not fragmentation (SS C.5) ---")
    h = loss_and_fragmentation(merged, gt)
    tot = sum(h.values()) or 1
    print(f"  GT instances >= 2 segments long: {tot}")
    for k in sorted(h):
        lab = "lost entirely" if k == 0 else ("correctly merged" if k == 1 else f"{k} pieces")
        print(f"    {k}: {h[k]:5d} ({h[k]/tot:5.1%})  {lab}")
    lost = h[0] / tot
    frag = sum(v for k, v in h.items() if k >= 2) / tot
    print(f"\n  LOSS rate {lost:.1%}   FRAGMENTATION rate {frag:.1%}")
    print("  SS C.5: use the loss rate. Fragmentation reads ~1% even when the merge")
    print("          is costing double digits, because it truncates rather than splits.")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="pilot_analysis/preds/full")
    ap.add_argument("--split", default="all", choices=["all", "novel"],
                    help="which merged/segment dump to load for Phase 3")
    a = ap.parse_args()

    part = json.load(open(PART))
    gt = json.load(open(paths.TEST_RELATION_GT))
    mp = os.path.join(a.preds, f"final_merged_{a.split}.json")
    sp = os.path.join(a.preds, f"segments_raw_{a.split}.json")
    if not os.path.exists(mp):
        raise SystemExit(f"no {mp} -- run dump_predictions.py first")
    merged = json.load(open(mp))
    print(f"loaded {len(merged)} videos of merged predictions from {mp}")

    phase2(merged, part)
    phase2_verb_control(merged, part, a.preds)

    if os.path.exists(sp):
        segs = json.load(open(sp))
        print(f"\nloaded {sum(len(v) for v in segs.values())} tracklet pairs from {sp}")
        phase2_confusion(segs, gt, part, "novel")
        phase3(merged, segs, gt)
    else:
        print(f"\n!! no {sp} -- Phase 3 and the confusion table need the pre-merge dump")
    return 0


if __name__ == "__main__":
    sys.exit(main())
