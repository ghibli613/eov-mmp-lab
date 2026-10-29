"""RPT G1, A29d's futility bounds on hand-built video records: SAM 3's best case on the videos without
SAM 3 files, EOV-MMP's real records on all, through A29's own bars (metrics.bars).

Run from studies/:  python -m pytest rpt/tests/test_g1_futility.py
"""
import os
import sys

import pytest

RPT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RPT, "g1"))

import futility as F  # noqa: E402


def rec(matched, n_unmatched, buckets):
    return dict(matched={rid: (0, 0.9) for rid in matched}, matched_agnostic={rid: (0, 0.9) for rid in matched},
                unmatched=list(range(n_unmatched)), unmatched_agnostic=list(range(n_unmatched)),
                pairs=[dict(bucket=b) for b in buckets], pairs_scoreable=[dict(bucket=b) for b in buckets],
                n_tracks=len(matched) + n_unmatched)


RUNS = {v: [dict(rid=i, category="dog", gt_tid=i) for i in range(n)] for v, n in (("a", 2), ("b", 1), ("c", 3))}
RELS = {"a": [{}, {}], "b": [{}], "c": [{}, {}]}
HELD = ["a", "b", "c"]
SAM3 = {"a": rec([0], 2, ["recalled", "U1"]), "b": rec([], 1, ["U2"])}                # "c" has no SAM 3 file


def eov(recalled_runs_c, unmatched_c):
    return {"a": rec([], 1, ["U1", "U1"]), "b": rec([], 1, ["U1"]),
            "c": rec(list(range(recalled_runs_c)), unmatched_c, ["U1", "U1"])}


def test_best_case_adds_exactly_the_missing_videos():
    bars, added = F.best_case(SAM3, eov(0, 1), RUNS, RELS, [0.5, 0.6], ["a", "b"], HELD, set())
    assert added == dict(videos=["c"], gt_runs_matched=3, base_relations_recalled=2, unmatched_tracks=0,
                         box_style_pairs_at_1=3)
    assert bars["overall_recall"]["sam3"] == pytest.approx(4 / 6)                  # (1 + 3) / (3 + 3)
    assert bars["pair_recall"]["sam3"] == pytest.approx(3 / 5)                     # (1 + 2) / (3 + 2)
    assert bars["false_track_burden"]["sam3"] == pytest.approx(3 / 3)              # (2 + 1 + 0) / 3 videos
    assert bars["false_track_burden"]["eov"] == pytest.approx(3 / 3)               # EOV-MMP's real, on all 3
    assert bars["box_style"]["median"] == 1.0 and bars["box_style"]["n_pairs"] == 5
    assert all(b["pass_"] for b in bars.values())


def test_a_bar_failing_at_the_best_case():
    # EOV-MMP recalls all 3 runs of "c": 3/6 = 50%; SAM 3's best 4/6 = 66.7%, +16.7 points: still passes.
    # Its 4 unmatched tracks on "c" make its mean 6/3 = 2.0; SAM 3's best 3/3 = 1.0: ratio 0.5, passes.
    bars, _ = F.best_case(SAM3, eov(3, 4), RUNS, RELS, [0.5, 0.6], ["a", "b"], HELD, set())
    assert bars["overall_recall"]["diff_points"] == pytest.approx(100 * (4 / 6 - 3 / 6))
    assert bars["overall_recall"]["pass_"] and bars["false_track_burden"]["ratio"] == pytest.approx(0.5)
    # Box style: with many low pairs, the 3 best-case pairs at 1.0 do not lift the median to 0.70.
    bars, _ = F.best_case(SAM3, eov(0, 1), RUNS, RELS, [0.1] * 10, ["a", "b"], HELD, set())
    assert bars["box_style"]["median"] == pytest.approx(0.1) and not bars["box_style"]["pass_"]
