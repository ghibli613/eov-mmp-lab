#!/usr/bin/env python
"""Unit-tests the Phase 1 dump logic without a GPU.

    python pilot_analysis/scripts/test_dump_logic.py

The one-video probe (pilot_analysis/logs/dump-probe-1video.txt) never reached
the dump code -- it spent 31 minutes inside video 1 and was killed. So the dump
block would otherwise run for the first time on paid hardware. This drives it
with synthetic pre_preds/pair_data through the REAL process_pred, the real
extract_segments, the real association() and the real format_, so the whole
chain is exercised on CPU.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))

from dump_predictions import extract_segments
from inference.post_process import association, format_, process_pred

CLIP_LEN = 30
N_PRED = 132


class Args:
    clip_len = CLIP_LEN
    clip_top_n = 20        # the real default (parser_func.py:77)
    use_prior = False
    max_per_video = 200


def make_pair(n_clips, begin=0, tail=0):
    """A pair_data dict shaped like gen_feats_test's output."""
    n = n_clips * CLIP_LEN + tail
    box = [10.0, 20.0, 110.0, 220.0]
    return {
        "sbj_cls": "dog", "obj_cls": "person",
        "sbj_scr": 0.9, "obj_scr": 0.8,
        "sbj_traj": [list(box) for _ in range(n)],
        "obj_traj": [[v + 5 for v in box] for _ in range(n)],
        "duration": [begin, begin + n],
    }


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return cond


def main():
    args = Args()
    id2pre = {i: f"pred_{i}" for i in range(N_PRED)}
    obj2id = {"dog": 0, "person": 1}
    prior = torch.zeros(2, 2, N_PRED)

    ok = True
    print("1. exact multiple of clip_len (4 clips, 120 frames)")
    pair = make_pair(4)
    scores = torch.rand(4, N_PRED)
    clip_rels = process_pred(args, id2pre, obj2id, prior, scores, pair)
    segs = extract_segments(clip_rels)
    ok &= check("one entry per (clip, top-n) pair",
                len(segs) == 4 * args.clip_top_n, f"{len(segs)} records")
    ok &= check("segment_index covers every clip",
                sorted({s['segment_index'] for s in segs}) == [0, 1, 2, 3])
    ok &= check("frame_range tiles [0,120) with no overlap",
                sorted({tuple(s['frame_range']) for s in segs})
                == [(0, 30), (30, 60), (60, 90), (90, 120)])
    ok &= check("triplet is [sbj, pre, obj] order",
                all(s['triplet'][0] == 'dog' and s['triplet'][2] == 'person' for s in segs))
    ok &= check("confidence is a float, not a tensor",
                all(isinstance(s['confidence'], float) for s in segs))

    print("\n2. ragged tail (3 clips + 17 frames, 107 total)")
    pair = make_pair(3, tail=17)
    clip_rels = process_pred(args, id2pre, obj2id, prior, torch.rand(4, N_PRED), pair)
    segs = extract_segments(clip_rels)
    rng = sorted({tuple(s['frame_range']) for s in segs})
    ok &= check("last segment ends at the true end, not a multiple of 30",
                rng[-1] == (90, 107), f"{rng[-1]}")

    print("\n3. non-zero begin_fid (offset 45) -- the 15-frame grid case (SS B.3)")
    pair = make_pair(2, begin=45)
    clip_rels = process_pred(args, id2pre, obj2id, prior, torch.rand(2, N_PRED), pair)
    segs = extract_segments(clip_rels)
    ok &= check("frame_range is absolute, offset by begin_fid",
                sorted({tuple(s['frame_range']) for s in segs}) == [(45, 75), (75, 105)])

    print("\n4. dump happens BEFORE association() mutates clip_rels")
    pair = make_pair(3)
    clip_rels = process_pred(args, id2pre, obj2id, prior, torch.rand(3, N_PRED), pair)
    before = extract_segments(clip_rels)
    lens_before = [len(c['sbj_traj']) for clip in clip_rels for c in clip]
    merged = association(clip_rels)
    after = extract_segments(clip_rels)
    lens_after = [len(c['sbj_traj']) for clip in clip_rels for c in clip]
    ok &= check("association() DOES mutate the trajectories in place",
                lens_before != lens_after,
                "confirms the dump must run first, as it does")
    # Re-extracting AFTER association gives corrupted records: association()
    # rewrites duration[1] to the end of the merged chain and overwrites pre_scr
    # with the chain mean. So `after` != `before` is the POINT -- it is what
    # proves the ordering in predict_and_dump matters. (An earlier version of
    # this test asserted before == after, which had the logic backwards.)
    ok &= check("re-extracting after association is CORRUPTED (so order matters)",
                before != after,
                f"{sum(1 for x, y in zip(before, after) if x != y)} of "
                f"{len(before)} records differ")
    ok &= check("the pre-merge snapshot is immune to the mutation",
                sorted({tuple(x['frame_range']) for x in before})
                == [(0, 30), (30, 60), (60, 90)],
                "frame_range still tiles the original segments")
    stretched = [tuple(x['frame_range']) for x in after
                 if x['frame_range'][1] - x['frame_range'][0] > CLIP_LEN]
    ok &= check("and the corruption is visible: post-merge spans exceed clip_len",
                len(stretched) > 0, f"{len(stretched)} stretched spans, e.g. {stretched[:2]}")

    print("\n5. full chain through format_, the evaluator's input shape")
    fmt = format_(args, merged)
    ok &= check("format_ accepts association()'s output", len(fmt) > 0, f"{len(fmt)} instances")
    keys = set(fmt[0])
    ok &= check("keys match the evaluator's expectation",
                {"triplet", "score", "sub_traj", "obj_traj", "duration"} <= keys,
                str(sorted(keys)))
    ok &= check("len(sub_traj) == duration span (format_'s own assert)",
                all(len(f['sub_traj']) == f['duration'][1] - f['duration'][0] for f in fmt))

    print("\n6. JSON-serialisable (json.dump would not raise)")
    import json
    try:
        json.dumps({"v": before})
        json.dumps({"v": fmt})
        ok &= check("both dumps serialise", True)
    except TypeError as e:
        ok &= check("both dumps serialise", False, str(e))

    print("\n" + ("ALL CHECKS PASSED" if ok else "*** FAILURES ABOVE ***"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
