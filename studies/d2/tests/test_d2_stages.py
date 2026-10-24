"""The analysis core on synthetic cases: geometry, oracle, the filter rules on the repo's own
functions, fragmentation, the stage classifier, and D1's bootstrap reused unchanged."""
import ast
import os

import numpy as np
import pytest

from d2 import stages as S
from d2.common import EMPTY, THESIS


def rows_for(tid, frames, box, score_cls=5):
    """`temp` rows as end2end_model.py:295-313 builds them: frame, id, x1y1x2y2, flag, 3 unused, 35 scores."""
    out = []
    for f in frames:
        r = np.zeros(45)
        r[0], r[1], r[2:6], r[6] = f, tid, box, 1
        r[10 + score_cls] = 0.8
        out.append(r)
    return np.array(out)


# --------------------------------------------------------------------------- geometry
def test_iou_many_equals_the_repo_iou(repo_fns):
    _, iou, _ = repo_fns
    rng = np.random.default_rng(0)
    b = rng.uniform(-20, 300, (500, 4))
    b[:250, 2:] = b[:250, :2] + rng.uniform(0, 80, (250, 2))   # proper boxes; the rest arbitrary
    for g in ([10, 10, 100, 120], [0, 0, 0, 0], [50.5, 3.2, 51.0, 400.0]):
        got = S.iou_many(b, g)
        want = np.array([iou(list(x), g) for x in b])
        assert np.array_equal(got, want)


def test_empty_frames_add_nothing_to_the_repo_viou(repo_fns):
    viou, _, _ = repo_fns
    g = [10, 10, 49, 49]                                     # area 40 x 40 = 1600 (+1 convention)
    assert viou([EMPTY, EMPTY], [0, 2], [g, g], [0, 2]) == 0.0
    assert viou([g, EMPTY], [0, 2], [g, g], [0, 2]) == pytest.approx(1600 / (1600 + 3200 - 1600))
    assert viou([g, g], [0, 2], [g, EMPTY], [0, 2]) == pytest.approx(1600 / (3200 + 1600 - 1600))
    assert viou([[-5, -5, 30, 30], EMPTY], [0, 2], [EMPTY, EMPTY], [0, 2]) == 0.0


def test_clip_boxes_as_end2end():
    b = S.clip_boxes(np.array([[-3, -1, 700, 500], [1, 2, 3, 4]]), 640, 480)
    assert b.tolist() == [[0, 0, 640, 480], [1, 2, 3, 4]]


def test_oracle_picks_highest_iou_first_on_ties_and_no_box_at_zero(repo_fns):
    viou, _, _ = repo_fns
    g = [10, 10, 49, 49]
    sets = {0: np.array([[0, 0, 20, 20], [10, 10, 49, 49]]),          # the exact box wins
            1: np.array([[10, 10, 49, 49], [10, 10, 49, 49.0]]),       # tie: the first
            2: np.array([[200, 200, 220, 220]]),                       # IoU 0: no box
            3: None}
    traj, chosen = S.oracle_track(sets.get, [g] * 4, 0)
    assert chosen == [[10, 10, 49, 49]] * 2 + [None, None]
    assert traj[2] == EMPTY and traj[3] == EMPTY
    v, _ = S.oracle_viou(sets.get, [g] * 4, 0, viou)
    assert v == pytest.approx(0.5)
    # only the frames of the GT extent count, and the GT's own empty frames give no box
    traj, chosen = S.oracle_track(sets.get, [g, EMPTY], 0)
    assert chosen[1] is None


def test_cut_viou_is_d1s_convention(repo_fns):
    viou, _, _ = repo_fns
    g = [10, 10, 49, 49]
    # a track over frames 0..19, GT over 5..14: only the overlap counts, the extra frames don't hurt
    assert S.cut_viou(0, [g] * 20, [g] * 10, 5, viou) == pytest.approx(1.0)
    assert S.cut_viou(0, [g] * 5, [g] * 10, 5, viou) == 0.0                  # no overlap
    assert S.cut_viou(10, [g] * 10, [g] * 10, 5, viou) == pytest.approx(0.5)  # half the GT


# --------------------------------------------------------------------------- the repo's filter
def test_length_rule_is_ten_frames_on_the_repo_function(repo_fns):
    """gen_labels.py:626 keeps len(track) >= 12, and the dict holds 'category' and 'score':
    8 tracker rows + the 2 added frames = 10 frames survive; 7 + 2 = 9 do not."""
    _, _, G = repo_fns
    box = [10, 10, 60, 60]
    kept = G.format_trajectories_test(list(rows_for(1, range(20, 28), box)))
    assert len(kept) == 1 and len(kept[0]["trajectory"]) == 10 and kept[0]["begin_fid"] == 18
    assert G.format_trajectories_test(list(rows_for(1, range(20, 27), box))) == []


def test_replay_track_rules(repo_fns):
    _, _, G = repo_fns
    box = [10, 10, 60, 60]
    assert S.replay_track(G, rows_for(1, range(0, 100, 2), box), 100)["rule"] == "F-cov"      # 52 of 101
    r = S.replay_track(G, rows_for(1, range(20, 27), box), 100)
    assert r["rule"] == "F-len" and r["n_frames"] == 9
    r = S.replay_track(G, rows_for(1, range(20, 40), box), 200)                                # 22 <= 30
    assert r["rule"] == "F-range" and r["n_frames"] == 22
    r = S.replay_track(G, rows_for(1, range(20, 40), box), 100)                                # 22 > 15
    assert r["rule"] is None and r["begin_fid"] == 18 and r["end_fid"] == 40 and not r["trimmed"]
    assert r["boxes"][0] == [10, 10, 60, 60]
    # a gap > 15% of the span: trimmed to the larger side, interpolated from all rows, kept
    r = S.replay_track(G, rows_for(1, list(range(0, 48)) + list(range(70, 100)), box), 100)
    assert r["rule"] is None and r["trimmed"] and (r["begin_fid"], r["end_fid"]) == (-2, 48)


def test_replay_video_is_format_plus_the_range_rule(repo_fns):
    _, _, G = repo_fns
    temp = np.concatenate([rows_for(4, range(0, 60), [10, 10, 60, 60]),
                           rows_for(9, range(0, 10), [100, 100, 150, 150]),        # 12 frames, not > 15% of 100
                           rows_for(2, range(30, 90), [200, 10, 260, 60], 7)])
    out = S.replay_video(G, temp, 100)
    assert [(t["tid"], t["begin_fid"], t["end_fid"]) for t in out] == [(0, -2, 60), (1, 28, 90)]
    assert out[0]["category"] != out[1]["category"]


def test_pre_filter_tracks_interpolate_inside_gaps():
    temp = np.concatenate([rows_for(3, [5, 6, 9], [0, 0, 10, 10])])
    temp[2, 2:6] = [3, 3, 13, 13]
    p = S.pre_filter_tracks(temp)[3]
    assert (p["begin_fid"], p["end_fid"]) == (5, 10)
    assert p["boxes"][3] == [2, 2, 12, 12] and len(p["boxes"]) == 5
    assert p["frames"].tolist() == [5, 6, 9]


def test_fragmentation():
    g = [10, 10, 49, 49]
    gt = [g] * 100
    chosen = [list(g)] * 100
    two = {1: dict(frames=np.arange(0, 45), raw_boxes=np.array([g] * 45)),
           2: dict(frames=np.arange(55, 100), raw_boxes=np.array([g] * 45))}
    f = S.fragmentation(chosen, 0, 100, two, gt)
    assert f["frag"] and f["good_frames"] == 100 and f["assigned"] == 90 and not f["untracked"]
    one = {1: dict(frames=np.arange(0, 45), raw_boxes=np.array([g] * 45)),
           2: dict(frames=np.arange(95, 100), raw_boxes=np.array([g] * 5))}      # 5 < 10% of 100
    f = S.fragmentation(chosen, 0, 100, one, gt)
    assert not f["frag"] and not f["untracked"] and f["assigned"] == 50          # 50 of 100 is not < 50%
    few = {1: dict(frames=np.arange(0, 20), raw_boxes=np.array([g] * 20))}
    f = S.fragmentation(chosen, 0, 100, few, gt)
    assert not f["frag"] and f["untracked"]                                      # 20 of 100 < 50%
    f = S.fragmentation(chosen, 0, 100, {}, gt)
    assert not f["frag"] and f["untracked"] and f["assigned"] == 0


# --------------------------------------------------------------------------- classifier
def test_classify_side_every_branch():
    nf = dict(frag=False, untracked=True)
    assert S.classify_side(0.4, 0.4, 0.4, 0.0, nf) == ("O1a", None)
    assert S.classify_side(0.6, 0.4, 0.4, 0.9, nf) == ("O1b", None)             # order: O1b before O3
    assert S.classify_side(0.6, 0.6, 0.4, 0.2, nf) == ("O2", "dedupe")
    assert S.classify_side(0.6, 0.6, 0.6, 0.2, dict(frag=True, untracked=False)) == ("O2", "frag")
    assert S.classify_side(0.6, 0.6, 0.6, 0.2, nf) == ("O2", "other-untracked")
    assert S.classify_side(0.6, 0.6, 0.6, 0.2, dict(frag=False, untracked=False)) == ("O2", "other-tracked")
    for rule in ("F-cov", "F-len", "F-range"):
        assert S.classify_side(0.6, 0.6, 0.6, 0.7, nf, dict(rule=rule)) == ("O3", rule)
    assert S.classify_side(0.6, 0.6, 0.6, 0.7, nf, dict(rule=None, final_viou=0.4, trimmed=True)) == ("O3", "F-reshape-trim")
    assert S.classify_side(0.6, 0.6, 0.6, 0.7, nf, dict(rule=None, final_viou=0.4, trimmed=False)) == ("O3", "F-reshape-other")
    assert S.classify_side(0.6, 0.6, 0.6, 0.7, nf, dict(rule=None, final_viou=0.6, trimmed=False)) == ("O4", None)
    assert S.classify_side(0.5, 0.5, 0.5, 0.5, nf, dict(rule="F-cov")) == ("O3", "F-cov")   # 0.5 passes


def test_relation_label_is_the_most_upstream():
    assert S.relation_label([("O3", "F-len"), ("O1b", None)]) == ("O1b", None)
    assert S.relation_label([("O2", "other-tracked"), ("O2", "dedupe")]) == ("O2", "dedupe")
    assert S.relation_label([("O3", "F-reshape-other"), ("O3", "F-cov")]) == ("O3", "F-cov")
    assert S.relation_label([]) == ("S0", None)


# --------------------------------------------------------------------------- bootstrap
def _d1_boot_share():
    """D1's boot_share, taken from tc_run.py's source (pinned by A25), not re-typed."""
    src = open(os.path.join(THESIS, "ov-vidvrd-lab/pilot_analysis/time_check/tc_run.py")).read()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "boot_share")
    ns = dict(np=np, N_BOOT=1000)
    exec(compile(ast.Module([fn], []), "tc_run.py", "exec"), ns)
    return ns["boot_share"]


def test_boot_share_is_d1s():
    d1 = _d1_boot_share()
    rng = np.random.default_rng(5)
    vids = rng.choice([f"v{i}" for i in range(30)], 200)
    mask = rng.random(200) < 0.3
    assert S.boot_share(mask, vids) == d1(mask, vids)
    assert S.boot_share([], []) == d1([], [])


def test_boot_diff_is_paired():
    vids = np.repeat([f"v{i}" for i in range(20)], 5)
    a = np.arange(100) % 2 == 0
    lo, hi = S.boot_diff(a, a, vids)
    assert lo == hi == 0.0
    lo, hi = S.boot_diff(np.ones(100), np.zeros(100), vids)
    assert lo == hi == 1.0
