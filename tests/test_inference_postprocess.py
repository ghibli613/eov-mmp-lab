"""Characterisation tests for the pure functions in the inference path.

These pin *current* behaviour, including behaviour that is arguably wrong. The
association step in particular is crude -- it chains clips only on exact
predicate equality and stops at the first gap -- and docs/02_Code-walkthrough.md
proposes changing it. These tests exist so that such a change shows up as a
deliberate, reviewed diff rather than a silent shift in the numbers.
"""
import numpy as np
import pytest
import torch

from inference.post_process import association, format_, process_pred
from utils.eov_utils import gen_union_bbox
from third_party.vidvrd_eval_api.common import voc_ap


def clip_entry(pre_cls, pre_scr, n_frames=2, **kw):
    d = dict(pre_cls=pre_cls, pre_scr=pre_scr, sbj_cls="dog", sbj_scr=1.0,
             obj_cls="person", obj_scr=1.0,
             sbj_traj=[[0, 0, 10, 10]] * n_frames,
             obj_traj=[[5, 5, 15, 15]] * n_frames,
             duration=[0, n_frames], connected=False)
    d.update(kw)
    return d


# ------------------------------------------------------------------ geometry
def test_union_box_is_the_enclosing_box():
    assert gen_union_bbox([0, 0, 10, 10], [5, 5, 20, 20]) == [0, 0, 20, 20]
    assert gen_union_bbox([0, 0, 10, 10], [2, 2, 4, 4]) == [0, 0, 10, 10]


# --------------------------------------------------------------- association
def test_consecutive_clips_with_the_same_predicate_merge():
    clips = [[clip_entry("run_left", 0.8, duration=[0, 2])],
             [clip_entry("run_left", 0.6, duration=[2, 4])]]
    out = association(clips)
    assert len(out) == 1
    assert out[0]["duration"] == [0, 4]
    assert len(out[0]["sbj_traj"]) == 4
    assert out[0]["pre_scr"] == pytest.approx(0.7)      # mean over the chain
    assert out[0]["score"] == pytest.approx(0.7)        # sbj_scr * obj_scr * pre_scr


def test_score_is_the_product_of_the_three_confidences():
    clips = [[clip_entry("run_left", 0.5, sbj_scr=0.4, obj_scr=0.5)]]
    out = association(clips)
    assert out[0]["score"] == pytest.approx(0.5 * 0.4 * 0.5)


def test_a_one_clip_gap_splits_the_instance():
    """Known-crude behaviour: the chain stops at the first clip that does not
    carry the same predicate, so a single dropped clip yields two instances."""
    clips = [[clip_entry("run_left", 0.9, duration=[0, 2])],
             [clip_entry("walk_left", 0.9, duration=[2, 4])],
             [clip_entry("run_left", 0.9, duration=[4, 6])]]
    out = association(clips)
    assert sorted(r["pre_cls"] for r in out) == ["run_left", "run_left", "walk_left"]


def test_different_predicates_never_merge():
    clips = [[clip_entry("run_left", 0.9)], [clip_entry("sit_above", 0.9)]]
    assert len(association(clips)) == 2


# -------------------------------------------------------------------- format
def test_format_truncates_to_max_per_video_and_sorts_by_score():
    class A:
        max_per_video = 2
    rels = [dict(clip_entry("p%d" % i, 0.1 * i), score=0.1 * i) for i in range(5)]
    out = format_(A(), rels)
    assert len(out) == 2
    assert [r["score"] for r in out] == sorted([r["score"] for r in out], reverse=True)
    assert out[0]["triplet"] == ["dog", "p4", "person"]


# ----------------------------------------------------------------- top-N cut
def test_process_pred_keeps_the_top_n_predicates_per_clip():
    class A:
        clip_len, clip_top_n, use_prior = 30, 3, False
    scores = torch.zeros(1, 5)
    scores[0] = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3])
    pair = dict(duration=[0, 30], sbj_cls="dog", sbj_scr=1.0, obj_cls="person",
                obj_scr=1.0, sbj_traj=[[0, 0, 1, 1]] * 30,
                obj_traj=[[0, 0, 1, 1]] * 30)
    id2pre = {i: f"p{i}" for i in range(5)}
    out = process_pred(A(), id2pre, {}, None, scores, pair)
    assert [c["pre_cls"] for c in out[0]] == ["p1", "p3", "p4"]   # descending


# ----------------------------------------------------------------- eval API
def test_voc_ap_matches_hand_computation():
    rec = np.array([0.0, 0.5, 0.5, 1.0])
    prec = np.array([1.0, 1.0, 0.5, 0.5])
    assert 0.0 <= voc_ap(rec, prec) <= 1.0
    assert voc_ap(np.array([0.0, 1.0]), np.array([1.0, 1.0])) == pytest.approx(1.0)
