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

from dump_predictions import _BothSplits, extract_segments
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
    recs = extract_segments(clip_rels, pair)
    ok &= check("one record per tracklet pair", len(recs) == 1, f"{len(recs)}")
    r = recs[0]
    ok &= check("trajectories stored once per pair, not per candidate",
                len(r["sbj_traj"]) == 120 and len(r["obj_traj"]) == 120,
                f"{len(r['sbj_traj'])} boxes")
    ok &= check("one segment entry per clip", len(r["segments"]) == 4)
    ok &= check("top-n predicates per segment",
                all(len(s["preds"]) == args.clip_top_n for s in r["segments"]))
    ok &= check("frame_range tiles [0,120) with no overlap",
                [tuple(s["frame_range"]) for s in r["segments"]]
                == [(0, 30), (30, 60), (60, 90), (90, 120)])
    ok &= check("scores are floats, not tensors",
                all(isinstance(p[1], float) for s in r["segments"] for p in s["preds"]))

    print("\n2. ragged tail (3 clips + 17 frames, 107 total)")
    pair = make_pair(3, tail=17)
    recs = extract_segments(
        process_pred(args, id2pre, obj2id, prior, torch.rand(4, N_PRED), pair), pair)
    rng = [tuple(s["frame_range"]) for s in recs[0]["segments"]]
    ok &= check("last segment ends at the true end, not a multiple of 30",
                rng[-1] == (90, 107), f"{rng[-1]}")
    ok &= check("trajectory length matches the pair duration",
                len(recs[0]["sbj_traj"]) == 107)

    print("\n3. non-zero begin_fid (offset 45) -- the 15-frame grid case (SS B.3)")
    pair = make_pair(2, begin=45)
    recs = extract_segments(
        process_pred(args, id2pre, obj2id, prior, torch.rand(2, N_PRED), pair), pair)
    ok &= check("frame_range is absolute, offset by begin_fid",
                [tuple(s["frame_range"]) for s in recs[0]["segments"]]
                == [(45, 75), (75, 105)])
    ok &= check("duration is absolute too", recs[0]["duration"] == [45, 105])

    print("\n4. dump happens BEFORE association() mutates clip_rels")
    pair = make_pair(3)
    clip_rels = process_pred(args, id2pre, obj2id, prior, torch.rand(3, N_PRED), pair)
    before = extract_segments(clip_rels, pair)
    lens_before = [len(c['sbj_traj']) for clip in clip_rels for c in clip]
    merged = association(clip_rels)
    lens_after = [len(c['sbj_traj']) for clip in clip_rels for c in clip]
    ok &= check("association() DOES mutate the trajectories in place",
                lens_before != lens_after,
                "confirms the dump must run first, as it does")
    ok &= check("the pre-merge snapshot still tiles the original segments",
                [tuple(s["frame_range"]) for s in before[0]["segments"]]
                == [(0, 30), (30, 60), (60, 90)])

    print("\n5. Phase 3.2(b) is reconstructible from the dump")
    r = before[0]
    ok &= check("segment boxes recoverable by slicing the pair trajectory",
                all(len(r["sbj_traj"][s["frame_range"][0] - r["duration"][0]:
                                      s["frame_range"][1] - r["duration"][0]])
                    == s["frame_range"][1] - s["frame_range"][0]
                    for s in r["segments"]),
                "oracle merge can rebuild instances without another GPU pass")

    print("\n6. full chain through format_, the evaluator's input shape")
    fmt = format_(args, merged)
    ok &= check("format_ accepts association()'s output", len(fmt) > 0, f"{len(fmt)} instances")
    ok &= check("keys match the evaluator's expectation",
                {"triplet", "score", "sub_traj", "obj_traj", "duration"} <= set(fmt[0]))
    ok &= check("len(sub_traj) == duration span (format_'s own assert)",
                all(len(f['sub_traj']) == f['duration'][1] - f['duration'][0] for f in fmt))

    print("\n7. JSON-serialisable (json.dump would not raise)")
    import json
    try:
        json.dumps({"v": before}); json.dumps({"v": fmt})
        ok &= check("both dumps serialise", True)
    except TypeError as e:
        ok &= check("both dumps serialise", False, str(e))

    print("\n8. --fuse-splits: one pipeline pass, both predicate splits")

    class _FakeC(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.tgt_split = "all"; self.calls = []
        def forward(self, feats, seq_lens, labels=None):
            self.calls.append(self.tgt_split)
            v = 1.0 if self.tgt_split == "all" else 2.0
            return (torch.full((1, 3, N_PRED), v), torch.full((1, 35), v),
                    torch.full((1, 35), v))

    class _FakeE2E(torch.nn.Module):
        def __init__(self, mc):
            super().__init__(); self.modelC = mc

    inner = _FakeC(); e2e = _FakeE2E(inner)
    e2e.modelC = _BothSplits(e2e.modelC)   # TypeError here if not an nn.Module
    ok &= check("wrapper can replace a child Module",
                isinstance(e2e.modelC, torch.nn.Module))
    e2e.modelC.tgt_split = "novel"
    ok &= check("tgt_split delegates to the inner module", inner.tgt_split == "novel")
    e2e.modelC.other.clear(); inner.calls.clear()
    a, _, _ = e2e.modelC(None, torch.tensor([3]))
    ok &= check("one call runs the inner module for BOTH splits",
                inner.calls == ["all", "novel"], str(inner.calls))
    ok &= check("returns the 'all' output so end2end_model's unpacking holds",
                a[0, 0, 0].item() == 1.0)
    ok &= check("stashes the 'novel' output for the caller",
                e2e.modelC.other[0][0][0, 0, 0].item() == 2.0)
    e2e.modelC.other.clear()
    for _ in range(4):
        e2e.modelC(None, torch.tensor([3]))
    ok &= check("call order preserved, one stash per pair",
                len(e2e.modelC.other) == 4,
                "final_results[k] must line up with other[k]")
    e2e.modelC.other.clear(); inner.calls.clear()
    e2e.modelC(None, torch.tensor([3]), labels={"x": 1})
    ok &= check("training path passes straight through",
                len(e2e.modelC.other) == 0 and len(inner.calls) == 1)
    e2e.eval()
    ok &= check("model.eval() still reaches the inner module", not inner.training)

    print("\n" + ("ALL CHECKS PASSED" if ok else "*** FAILURES ABOVE ***"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
