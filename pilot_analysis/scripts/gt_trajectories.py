#!/usr/bin/env python
"""Phase 4 feasibility -- build GT trajectories in the schema modelC consumes.

    python pilot_analysis/scripts/gt_trajectories.py            # validate + cost
    python pilot_analysis/scripts/gt_trajectories.py --dump     # write JSON

No GPU, no model. This answers the protocol's Phase 4 question ("the repo's
cascaded ancestry means the classifier consumes tracklet pairs -- find that
interface") and proves GT trajectories can be substituted for detected ones.

The interface, traced in models/end2end_model.py (test branch):

    modelA -> deep_sort -> AFLink -> format_trajectories_test(temp)
                                          |
                                    `trajectories`  <-- SUBSTITUTE HERE
                                          |
                        gen_feats_test(video_name, trajectories, 'test',
                                       patch_proj, global_proj, w, h)
                                          |
                                    modelC(feats, seq_lens)

Replacing `trajectories` at that one point bypasses the detector, the tracker
and AFLink entirely, while leaving the relation classifier untouched. The
required schema, read off format_trajectories_test (models/gen_labels.py:586)
and the fields gen_feats_test actually reads (gen_labels.py:642):

    {'tid': int, 'category': str, 'score': float,
     'trajectory': {frame_id(int): [x1, y1, x2, y2], ...},
     'begin_fid': int, 'end_fid': int}   # end_fid == max(fid) + 1

Invariants gen_feats_test relies on, which this script checks:
  - `trajectory` is DENSE over [begin_fid, end_fid): it indexes
    s_traj['trajectory'][fid] for every fid in that range and will KeyError on
    a gap (gen_labels.py:670-673). RAW GT VIOLATES THIS -- 43 trajectories have
    gaps where the object leaves and re-enters frame. The detected path never
    hits it because format_trajectories_test runs add_initial_frames() and
    interpolate_and_adjust_frames() first, so this script runs GT through the
    SAME two functions (--mode repo, the default) rather than inventing a fix.
  - a trajectory shorter than 12 frames is dropped by format_trajectories_test
    (`if len(track) >= 12`), so GT ones below that would never have existed.
  - a PAIR is used only if end_fid - begin_fid >= 10 (gen_labels.py:659).
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# The repo's own gap handling, so GT and detected trajectories get identical
# downstream treatment and the only difference between conditions is the source.
from models.gen_labels import add_initial_frames, interpolate_and_adjust_frames

CLIP_LEN = 30      # models/gen_labels.py:13
MIN_TRACK = 12     # format_trajectories_test: `if len(track) >= 12`
MIN_PAIR = 10      # gen_feats_test: `if (end_fid - begin_fid) < 10: continue`
ANNO = "data/vidvrd/anno/test"
OBJ_SPLIT = "configs/VidVRD_class_spilt_info.json"


def build(anno, mode="repo"):
    """GT annotation -> list of trajectories in format_trajectories_test's schema.

    mode='repo'   : run the repo's add_initial_frames + interpolate_and_adjust_frames,
                    exactly as format_trajectories_test does for detected tracks.
    mode='strict' : interpolate gaps only -- no phantom pre-roll frames. Cleaner,
                    but no longer identical to the detected path.
    """
    tid2cat = {so["tid"]: so["category"] for so in anno["subject/objects"]}
    boxes = {}   # tid -> {fid: [x1,y1,x2,y2]}
    for fid, frame in enumerate(anno["trajectories"]):
        for box in frame:
            b = box["bbox"]
            boxes.setdefault(box["tid"], {})[fid] = [
                b["xmin"], b["ymin"], b["xmax"], b["ymax"]]
    out, rejected = [], 0
    for tid in sorted(boxes):
        traj = boxes[tid]
        if len(traj) < MIN_TRACK:
            continue
        if mode == "repo":
            traj = add_initial_frames(dict(traj))
        traj = interpolate_and_adjust_frames(traj)
        if not traj:                 # their 65%-coverage rule rejected it
            rejected += 1
            continue
        traj = {int(f): b for f, b in traj.items()}
        out.append({
            "tid": len(out),
            "category": tid2cat[tid],
            "score": 1.0,          # GT is certain; detected trajs carry modelB's score
            "trajectory": traj,
            "begin_fid": min(traj),
            "end_fid": max(traj) + 1,
            "gt_tid": tid,         # extra, ignored downstream; keeps GT identity for Phase 2.3
        })
    return out, rejected


def check(trajs, categories):
    """Return list of invariant violations."""
    bad = []
    for t in trajs:
        span = range(t["begin_fid"], t["end_fid"])
        missing = [f for f in span if f not in t["trajectory"]]
        if missing:
            bad.append(f"tid {t['tid']}: {len(missing)} gaps in [{t['begin_fid']},{t['end_fid']}) "
                       f"-- gen_feats_test would KeyError")
        if t["category"] not in categories:
            bad.append(f"tid {t['tid']}: category {t['category']!r} not in object vocabulary")
        for f, b in t["trajectory"].items():
            if not (b[2] > b[0] and b[3] > b[1]):
                bad.append(f"tid {t['tid']} frame {f}: degenerate box {b}")
                break
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true",
                    help="write pilot_analysis/gt_trajectories.json")
    ap.add_argument("--mode", choices=("repo", "strict"), default="repo",
                    help="repo: use the repo's own gap handling (default). "
                         "strict: interpolate only, no phantom pre-roll frames.")
    a = ap.parse_args()

    categories = set(json.load(open(OBJ_SPLIT))["cls2split"]) - {"__background__"}

    all_trajs, violations = {}, []
    n_traj = n_pair = n_seg = 0
    dropped_short_traj = dropped_short_pair = dropped_coverage = 0
    neg_fid = 0
    pairs_per_video = Counter()

    for fn in sorted(os.listdir(ANNO)):
        anno = json.load(open(os.path.join(ANNO, fn)))
        vid = anno["video_id"]
        raw = len(anno["subject/objects"])
        trajs, rejected = build(anno, a.mode)
        dropped_coverage += rejected
        dropped_short_traj += raw - len(trajs) - rejected
        violations += [f"{vid}: {v}" for v in check(trajs, categories)]
        all_trajs[vid] = trajs
        n_traj += len(trajs)
        neg_fid += sum(1 for t in trajs if t["begin_fid"] < 0)
        # replicate gen_feats_test's ordered-pair enumeration and segment count
        for i, s in enumerate(trajs):
            for j, o in enumerate(trajs):
                if i == j:
                    continue
                b = max(s["begin_fid"], o["begin_fid"])
                e = min(s["end_fid"], o["end_fid"])
                if (e - b) < MIN_PAIR:
                    dropped_short_pair += 1
                    continue
                n_pair += 1
                pairs_per_video[vid] += 1
                clips = (e - b) // CLIP_LEN
                tail = (e - b) % CLIP_LEN
                if tail >= 10:
                    clips += 1
                n_seg += clips

    print(f"videos                        {len(all_trajs)}")
    print(f"mode                          {a.mode}")
    print(f"GT trajectories built         {n_traj}   (dropped {dropped_short_traj} "
          f"with < {MIN_TRACK} frames, {dropped_coverage} by the repo's "
          f"65%-coverage rule)")
    if neg_fid:
        print(f"  !! {neg_fid} trajectories have begin_fid < 0 -- add_initial_frames "
              f"prepends two frames before the first. Harmless for the dict lookup, "
              f"but gen_feats_test's feature extractor will be asked for frames that "
              f"do not exist. Check this on the first GPU run.")
    print(f"ordered trajectory pairs      {n_pair}   (dropped {dropped_short_pair} "
          f"with overlap < {MIN_PAIR} frames, matching gen_feats_test)")
    print(f"segments to classify          {n_seg}   <-- modelC forward passes for Phase 4")
    print(f"pairs per video               min {min(pairs_per_video.values())}, "
          f"median {sorted(pairs_per_video.values())[len(pairs_per_video)//2]}, "
          f"max {max(pairs_per_video.values())}")
    print()
    if violations:
        print(f"SCHEMA VIOLATIONS: {len(violations)}")
        for v in violations[:20]:
            print("  " + v)
        if len(violations) > 20:
            print(f"  ... and {len(violations)-20} more")
        return 1
    print("SCHEMA OK -- every trajectory is dense over its span, every category is in")
    print("the object vocabulary, no degenerate boxes. gen_feats_test can consume these")
    print("as a drop-in replacement for format_trajectories_test's output.")

    if a.dump:
        os.makedirs("pilot_analysis", exist_ok=True)
        with open("pilot_analysis/gt_trajectories.json", "w") as f:
            json.dump(all_trajs, f)
        mb = os.path.getsize("pilot_analysis/gt_trajectories.json") / 1e6
        print(f"\nwrote pilot_analysis/gt_trajectories.json ({mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
