"""A33's naming check: the pooling against an independent implementation, the class reorder, the stop
rule, and the renaming through A29's matching on hand-built tracks. No HELD data, no GPU."""
import importlib
import os
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
D7 = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(D7))
sys.modules["held_b_guard"] = types.SimpleNamespace(install=lambda: None)   # no audit hook in the test process
sys.path.insert(0, D7)
N = importlib.import_module("naming_check")
del sys.modules["held_b_guard"]
sys.path.insert(0, os.path.join(REPO, "studies", "rpt", "g1"))
import metrics as M  # noqa: E402


def coverage_mask(box, w, h):
    """Independent: each 24 x 24 cell's overlap area with the box scaled to the grid."""
    x1, y1, x2, y2 = (max(box[0], 0) / w * 24, max(box[1], 0) / h * 24, min(box[2], w) / w * 24, min(box[3], h) / h * 24)
    m = np.zeros((24, 24))
    for i in range(24):
        for j in range(24):
            m[i, j] = max(0.0, min(x2, j + 1) - max(x1, j)) * max(0.0, min(y2, i + 1) - max(y1, i))
    return m.flatten()


def test_pooling_matches_an_independent_coverage_mean():
    g = torch.Generator().manual_seed(0)
    pp = torch.randn(1, 576, 768, generator=g)
    for box, w, h in [([10, 20, 300, 200], 640, 360), ([0, 0, 1279, 719], 1280, 720), ([5.5, 3, 40, 41], 320, 240),
                      ([-4, -2, 700, 400], 640, 360)]:
        got = N.pooled_feature(pp, box, w, h)
        assert tuple(got.shape) == (1, 768)          # what modelB is given in end2end_model.py:253-256
        m = torch.from_numpy(coverage_mask(box, w, h)).float()
        ref = (pp[0] * m[:, None]).sum(0) / m.sum().clamp(min=1e-9)
        ref = ref / ref.norm()
        assert torch.allclose(got[0], ref, atol=1e-5), box
        assert abs(float(got.norm()) - 1) < 1e-5


def test_reorder_is_end2end_new_order():
    probs = list(range(35))
    assert N.reorder(probs) == N.NEW_ORDER
    assert sorted(N.NEW_ORDER) == list(range(35))


def test_stop_rule():
    assert N.stop_rule(2.0, 5.0).startswith("STOP: the prompt-name gain")
    assert N.stop_rule(10.0, 4.99).startswith("STOP: modelB")
    assert N.stop_rule(10.0, 5.0).startswith("CONTINUE")


def box_track(tid, cat, b, e, box):
    return dict(tid=tid, category=cat, begin_fid=b, end_fid=e, boxes=[box] * (e - b), score=0.9)


def test_renaming_changes_only_class_aware_matches():
    runs = [dict(rid=0, gt_tid=0, category="dog", begin_fid=0, end_fid=20, boxes=[[0, 0, 10, 10]] * 20, annotated=list(range(20))),
            dict(rid=1, gt_tid=1, category="cat", begin_fid=0, end_fid=20, boxes=[[50, 50, 60, 60]] * 20, annotated=list(range(20)))]
    repo = [box_track(0, "dog", 0, 20, [0, 0, 10, 10]), box_track(1, "cat", 0, 20, [50, 50, 60, 60])]
    a = M.video_record(M.tracks_from_list(repo), runs, [])
    b = M.video_record(M.tracks_from_list(N.renamed(repo, ["dog", "sheep"])), runs, [])
    assert set(a["matched"]) == {0, 1} and set(b["matched"]) == {0}
    assert set(a["matched_agnostic"]) == set(b["matched_agnostic"]) == {0, 1}
    assert b["unmatched"] == [1]
