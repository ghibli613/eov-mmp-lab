"""A31 §6's counts on a hand-built video (no FIT data)."""
import importlib
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.modules["held_b_guard"] = types.SimpleNamespace(install=lambda: None)   # no audit hook in the test process
F = importlib.import_module("fit_counts")
del sys.modules["held_b_guard"]


def frag(tid, a, b):
    return dict(tid=tid, category="dog", trajectory={str(f): [0, 0, 1, 1] for f in range(a, b)})


def test_hand_built_video():
    # 100 frames -> sampled 0, 30, 60, 90. tid 0: frames 0-39 and 70-99 (gap 40-69). tid 1: frames 20-35.
    videos = {"v": [frag(0, 0, 40), frag(0, 70, 100), frag(1, 20, 36)]}
    c = F.counts(videos, {"v": 100})
    assert c["sampled_frames"] == 4 and c["tracks"] == 2
    assert c["no_box_share_all_sampled"] == 1 / 4                 # frame 60 has no box
    assert c["sampled_frames_inside_some_span"] == 4 and c["no_box_share_inside_spans"] == 1 / 4
    # pairs in span: tid 0 at 0, 30, 60, 90 (60 in its gap); tid 1 at 30 -> 5 pairs, 1 in a gap
    assert c["frame_track_pairs_in_span"] == 5 and c["in_gap_share"] == 1 / 5
    assert abs(c["coverage_mean"] - (70 / 100 + 1.0) / 2) < 1e-12
    assert c["coverage_full_share"] == 0.5 and c["coverage_under_65_share"] == 0.0


def test_frames_outside_spans():
    videos = {"v": [frag(0, 40, 50)]}
    c = F.counts(videos, {"v": 100})
    assert c["sampled_frames"] == 4 and c["no_box_share_all_sampled"] == 1.0
    assert c["sampled_frames_inside_some_span"] == 0 and c["no_box_share_inside_spans"] is None
