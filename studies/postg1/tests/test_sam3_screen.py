"""Post-G1 Task 2: the screen's rules on hand-built tracks, and its subset scorer against A29's own
code (metrics.py) on every subset of random synthetic videos. CPU only, no data files.

Run from studies/:  python -m pytest postg1/tests/test_sam3_screen.py
"""
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import sam3_screen as S  # noqa: E402

M = S.M


def track(tid, cat, b, n, box, score=0.8):
    return dict(tid=tid, category=cat, begin_fid=b, end_fid=b + n, boxes=[list(box)] * n, score=score)


def moving(tid, cat, b, n, x0, y0, w, h, dx, score=0.8):
    return dict(tid=tid, category=cat, begin_fid=b, end_fid=b + n, score=score,
                boxes=[[x0 + dx * i, y0, x0 + dx * i + w - 1, y0 + h - 1] for i in range(n)])


def test_tiny_uses_inclusive_width():
    tr = {0: track(0, "dog", 0, 5, (10, 10, 16, 40)), 1: track(1, "dog", 0, 5, (10, 10, 17, 40))}  # widths 7, 8
    assert S.rule_tiny(tr, 8) == {0}


def test_edge_sliver_left_and_right():
    tr = {0: track(0, "dog", 0, 10, (0, 10, 9, 50)),          # left edge, width 10
          1: track(1, "dog", 0, 10, (617, 10, 637, 50)),      # right edge of a 640-wide frame (637 >= 640-1-2), width 21
          2: track(2, "dog", 0, 10, (3, 10, 9, 50)),          # 3 px from the left edge: not touching
          3: track(3, "dog", 0, 10, (300, 10, 309, 50))}
    assert S.rule_edge(tr, 640, 0.5, 16) == {0}
    assert S.rule_edge(tr, 640, 0.5, 32) == {0, 1}
    half = {0: dict(track(0, "dog", 0, 10, (0, 10, 9, 50)))}
    half[0]["boxes"] = [[0, 10, 9, 50]] * 5 + [[100, 10, 109, 50]] * 5
    assert S.rule_edge(half, 640, 0.5, 16) == {0} and S.rule_edge(half, 640, 0.8, 16) == set()


def test_fragment_needs_a_larger_track_and_enough_frames():
    big = track(0, "person", 0, 10, (0, 0, 99, 99))
    small = track(1, "bicycle", 0, 10, (10, 10, 19, 19))       # fully inside on all 10 shared frames
    half_in = dict(track(2, "dog", 0, 10, (10, 10, 19, 19)))
    half_in["boxes"] = [[10, 10, 19, 19]] * 6 + [[200, 200, 209, 209]] * 4   # inside on 6 of 10
    tr = {0: big, 1: small, 2: half_in}
    assert S.rule_fragment(tr, 0.8, 0.5) == {1, 2}
    assert S.rule_fragment(tr, 0.8, 0.8) == {1}
    assert 0 not in S.rule_fragment(tr, 0.8, 0.5)              # the largest track is never a fragment


def test_partner_by_centre_distance_and_overlap():
    a = track(0, "person", 0, 20, (0, 0, 9, 9))               # diagonal ~14.1
    b = track(1, "dog", 0, 20, (20, 0, 29, 9))                # centres 20 apart
    c = track(2, "car", 0, 9, (0, 0, 9, 9))                   # overlaps a and b in time by only 9 frames
    tr = {0: a, 1: b, 2: c}
    assert S.rule_partner(tr, 1) == {0, 1, 2}                 # 20 > 1 x 14.1
    assert S.rule_partner(tr, 2) == {2}                       # 20 <= 2 x 14.1; c never has 10 shared frames


def test_score_and_gate():
    tr = {0: track(0, "dog", 0, 5, (0, 0, 9, 9), 0.65), 1: track(1, "cat", 0, 5, (0, 0, 9, 9), 0.95)}
    assert S.rule_score(tr, 0.7) == {0} and S.rule_score(tr, 0.6) == set()
    assert S.rule_gate(tr, {"dog"}) == {1}


def synthetic(seed):
    rng = np.random.default_rng(seed)
    cats = ["dog", "person", "car"]
    gts = []
    for g in range(4):
        b, n = int(rng.integers(0, 30)), int(rng.integers(20, 60))
        gts.append(moving(g, cats[g % 3], b, n, int(rng.integers(0, 200)), int(rng.integers(0, 200)),
                          int(rng.integers(20, 80)), int(rng.integers(20, 80)), int(rng.integers(-2, 3))))
    runs = [dict(rid=i, gt_tid=g["tid"] // 2, category=g["category"], begin_fid=g["begin_fid"], end_fid=g["end_fid"],
                 boxes=g["boxes"], annotated=list(range(g["begin_fid"], g["end_fid"]))) for i, g in enumerate(gts)]
    tracks = {}
    for t in range(6):
        g = gts[int(rng.integers(0, 4))]
        jit = lambda: int(rng.integers(-6, 7))
        tr = dict(tid=t, category=g["category"] if rng.random() < 0.7 else cats[int(rng.integers(0, 3))],
                  begin_fid=g["begin_fid"] + abs(jit()), score=0.7, boxes=None)
        n = max(5, g["end_fid"] - tr["begin_fid"] - abs(jit()))
        off = g["begin_fid"]
        tr["boxes"] = [[x + jit() for x in g["boxes"][min(len(g["boxes"]) - 1, tr["begin_fid"] - off + i)]]
                       for i in range(n)]
        tr["boxes"] = [[min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])] for b in tr["boxes"]]
        tr["end_fid"] = tr["begin_fid"] + n
        tracks[t] = tr
    rels = []
    for s, o in ((0, 1), (1, 2), (2, 3)):
        gs, go = gts[s], gts[o]
        lo, hi = max(gs["begin_fid"], go["begin_fid"]), min(gs["end_fid"], go["end_fid"])
        if hi - lo < 5:
            continue
        rels.append(dict(duration=[lo, hi], triplet=[gs["category"], "next_to", go["category"]],
                         sub_traj=gs["boxes"][lo - gs["begin_fid"]:hi - gs["begin_fid"]],
                         obj_traj=go["boxes"][lo - go["begin_fid"]:hi - go["begin_fid"]]))
    return tracks, runs, rels


def test_subset_scorer_equals_metrics_on_every_subset():
    for seed in range(8):
        tracks, runs, rels = synthetic(seed)
        V = S.Video(tracks, runs, rels)
        for k in range(len(tracks) + 1):
            for keep in itertools.combinations(sorted(tracks), k):
                kept = {t: tracks[t] for t in keep}
                ref = M.video_record(kept, runs, rels)
                got = V.score(frozenset(keep))
                assert got["matched_runs"] == len(ref["matched"])
                assert got["unmatched"] == ref["unmatched"]
                assert got["recalled"] == sum(p["bucket"] == "recalled" for p in ref["pairs"]), (seed, keep)


def test_grid_is_1000_and_selection_rule():
    cfgs = list(S.configs())
    assert len(cfgs) == 1000 and cfgs[0] == dict(tiny=None, edge=None, fragment=None, partner=None, gate=False)
    rows = [dict(index=0, pass_=True, config=dict(cfgs[1]), recall_gain=9.0, pair_gain=8.5, burden=1.3),
            dict(index=1, config=dict(cfgs[3]), recall_gain=8.5, pair_gain=9.5, burden=1.2),
            dict(index=2, config=dict(cfgs[0]), recall_gain=12.0, pair_gain=12.0, burden=1.0)]
    for r in rows:
        r["pass"] = True
    assert S.select(rows)["index"] == 1                        # tie on min gain 8.5: lower burden; no-gate row ignored
