"""RPT G1 metrics unit tests (A29), on synthetic inputs, CPU only.

Run from studies/:  python -m pytest rpt/tests/test_g1_metrics.py
"""
import os
import sys

import pytest

RPT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RPT, "g1"))

import metrics as M  # noqa: E402

BOX = [10.0, 10.0, 49.0, 49.0]
FAR = [200.0, 200.0, 239.0, 239.0]


def track(tid, b, e, cat="dog", box=BOX, boxes=None):
    return dict(tid=tid, category=cat, begin_fid=b, end_fid=e, boxes=boxes or [list(box)] * (e - b), score=0.9)


def run(rid, gt_tid, b, e, cat="dog", box=BOX, annotated=None):
    return dict(rid=rid, gt_tid=gt_tid, category=cat, begin_fid=b, end_fid=e, boxes=[list(box)] * (e - b),
                annotated=list(range(b, e)) if annotated is None else annotated)


def tr(*ts):
    return {t["tid"]: t for t in ts}


# ---------------------------------------------------------------- cut-vIoU and assignment
def test_bridging_track_matches_both_runs_of_one_tid():
    """An unbroken track over a > 30-frame gap matches both runs of the tid (A29 item 2's reason)."""
    runs = [run(0, 7, 0, 40), run(1, 7, 80, 120)]
    m = M.match(tr(track(0, 0, 120)), runs)
    assert m == {0: (0, 1.0), 1: (0, 1.0)}
    # without the cut, the whole track against one run would fail
    assert M.viou([BOX] * 120, [0, 120], [BOX] * 40, [0, 40]) < 0.5


def test_track_over_runs_of_two_tids_matches_only_one_tid():
    runs = [run(0, 1, 0, 50), run(1, 2, 50, 100)]
    m = M.match(tr(track(0, 0, 100)), runs)
    assert len(m) == 1 and m[0][0] == 0           # equal cut-vIoU: the lower run index wins
    # a second track can take the other tid's run
    m2 = M.match(tr(track(0, 0, 100), track(1, 50, 100)), runs)
    assert m2 == {0: (0, 1.0), 1: (1, 1.0)}


def test_track_matched_to_one_tid_skips_the_other_tid_even_if_better_later():
    runs = [run(0, 1, 0, 50), run(1, 2, 50, 100), run(2, 1, 100, 150)]
    m = M.match(tr(track(0, 0, 150)), runs)
    assert set(m) == {0, 2} and all(t == 0 for t, _ in m.values())


def test_each_run_matched_at_most_once():
    runs = [run(0, 1, 0, 50)]
    m = M.match(tr(track(0, 0, 50), track(1, 0, 50), track(2, 0, 50)), runs)
    assert m == {0: (0, 1.0)}
    assert M.unmatched_tracks(tr(track(0, 0, 50), track(1, 0, 50), track(2, 0, 50)), m) == [1, 2]


def test_tie_breaks_track_id_then_run_index():
    runs = [run(3, 1, 0, 50), run(1, 2, 0, 50)]
    # two tracks tie on both runs: track 0 takes run 1 (lower run index), track 5 takes run 3
    m = M.match(tr(track(5, 0, 50), track(0, 0, 50)), runs)
    assert m == {1: (0, 1.0), 3: (5, 1.0)}


def test_descending_cut_viou_beats_track_id():
    shifted = [12.0, 12.0, 51.0, 51.0]
    runs = [run(0, 1, 0, 50)]
    m = M.match(tr(track(0, 0, 50, box=shifted), track(1, 0, 50)), runs)
    assert m[0][0] == 1


def test_cut_viou_partial_cover():
    """A track over half the run: cut to [max(gb, begin), min(ge, end)), compared with the whole run."""
    r = run(0, 1, 0, 40)
    t = track(0, 20, 100)
    assert M.cut_viou(t, r) == pytest.approx(0.5)
    assert M.cut_viou(track(0, 25, 100), r) < 0.5
    assert M.cut_viou(track(0, 40, 60), r) == 0.0              # no overlap


def test_class_aware_and_agnostic():
    runs = [run(0, 1, 0, 50, cat="dog")]
    t = tr(track(0, 0, 50, cat="cat"))
    assert M.match(t, runs, True) == {} and M.match(t, runs, False) == {0: (0, 1.0)}


def test_threshold_is_inclusive():
    r = run(0, 1, 0, 40)
    assert M.match(tr(track(0, 20, 60)), [r]) == {0: (0, 0.5)}


# ---------------------------------------------------------------- box style
def test_box_style_uses_annotated_frames_only():
    r = run(0, 1, 0, 10, annotated=[0, 9])            # frames 1-8 interpolated
    boxes = [list(BOX) if f in (0, 9) else list(FAR) for f in range(10)]
    t = track(0, 0, 10, boxes=boxes)
    assert M.box_style_pair(t, r) == 1.0
    r_all = run(0, 1, 0, 10)
    assert M.box_style_pair(t, r_all) == pytest.approx(0.2)
    assert M.box_style_pair(track(0, 1, 9), r) is None    # no annotated frame inside the track


# ---------------------------------------------------------------- pair recall
def rel(b, e, sub="dog", obj="cat", sub_box=BOX, obj_box=FAR):
    return dict(duration=[b, e], triplet=[sub, "next_to", obj], sub_traj=[list(sub_box)] * (e - b),
                obj_traj=[list(obj_box)] * (e - b))


def test_pair_buckets():
    r = rel(0, 30)
    good = tr(track(0, 0, 30, "dog", BOX), track(1, 0, 30, "cat", FAR))
    assert M.bucket(good, r)["bucket"] == "recalled"
    named = tr(track(0, 0, 30, "dog", BOX), track(1, 0, 30, "sheep", FAR))
    assert M.bucket(named, r)["bucket"] == "U2"
    assert M.bucket(tr(track(0, 0, 30, "dog", BOX)), r)["bucket"] == "U1"


def test_best_pair_equals_d1_quantity():
    from d2.analyse import best_pair_min
    r = rel(0, 30)
    ts = tr(track(0, 0, 30, "dog", BOX), track(1, 0, 30, "cat", FAR), track(2, 5, 30, "cat", FAR))
    final = {k: dict(begin_fid=t["begin_fid"], boxes=t["boxes"]) for k, t in ts.items()}
    assert M.best_pair(ts, r)[0] == best_pair_min(final, r, M.viou)


def test_scoreable_pairs_exclude_overlap_under_10():
    r = rel(0, 30)
    ts = tr(track(0, 0, 15, "dog", BOX), track(1, 6, 30, "cat", FAR))      # overlap 9 frames
    assert M.overlap(ts[0], ts[1]) == 9
    assert M.bucket(ts, r, scoreable=True)["bucket"] == "U1"
    assert M.best_pair(ts, r)[1:] == (0, 1)                                 # the binding bar keeps it
    ts10 = tr(track(0, 0, 16, "dog", BOX), track(1, 6, 30, "cat", FAR))     # overlap 10 frames
    assert M.best_pair(ts10, r, scoreable=True)[1:] == (0, 1)


# ---------------------------------------------------------------- item 8, bins, inputs
def test_raw_fraction_same_class_annotated_frames():
    r = run(0, 1, 0, 4, annotated=[0, 1, 3])
    raw = [[0] + BOX + [0.3], [2] + BOX + [0.3], [3] + FAR + [0.4]]
    assert M.raw_fraction(r, raw) == pytest.approx(1 / 3)


def test_n_bins():
    assert [M.n_bin(n) for n in (0, 1, 2, 3, 4, 5, 8, 9, 40)] == ["0", "1", "2", "3-4", "3-4", "5-8", "5-8", ">=9", ">=9"]


def test_runs_of_reads_annotated_frames_and_checks_boxes():
    anno = dict(trajectories=[[dict(tid=4, bbox=dict(xmin=1, ymin=2, xmax=3, ymax=4))], [],
                              [dict(tid=4, bbox=dict(xmin=1, ymin=2, xmax=3, ymax=4))]])
    ob = [dict(tid=0, gt_tid=4, run=0, category="dog", begin_fid=0, end_fid=3, boxes=[[1, 2, 3, 4]] * 3)]
    assert M.runs_of(ob, anno)[0]["annotated"] == [0, 2]
    ob[0]["boxes"][2] = [1, 2, 3, 5]
    with pytest.raises(ValueError):
        M.runs_of(ob, anno)


def test_sparse_tracks_are_refused():
    with pytest.raises(ValueError):
        M.check_tracks({0: dict(tid=0, category="dog", begin_fid=0, end_fid=5, boxes=[BOX] * 4)})


def test_sensitivity_inputs_cut_to_span():
    r = rel(10, 40)
    s3 = tr(track(0, 0, 30, "dog"), track(1, 20, 60, "cat", FAR))
    inp = M.sensitivity_inputs(r, dict(sam3=(0, 1), eov=(0, 1)), s3, s3)
    assert [(t["begin_fid"], t["end_fid"]) for t in inp["sam3"]] == [(10, 30), (20, 40)]
    assert [(t["begin_fid"], t["end_fid"]) for t in inp["gt"]] == [(10, 40), (10, 40)]


def test_top5_base_hit():
    logits = [0.0] * 10
    for i, v in ((1, 5.0), (2, 4.0), (3, 3.0), (4, 2.0), (5, 1.0), (6, 0.5), (7, 9.0)):
        logits[i] = v
    base = [1, 2, 3, 4, 5, 6]                  # 7 is novel: ignored
    assert M.top5_base_hit(logits, 0.0, 5, base) and not M.top5_base_hit(logits, 0.0, 6, base)
