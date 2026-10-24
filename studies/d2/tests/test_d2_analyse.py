"""The Step-3 analysis end to end on one synthetic video whose every relation is built to land in
a known stage. No test label is read: the GT here is made up, and only the class/predicate split
metadata is loaded."""
import json
import os

import numpy as np
import pytest

from d2 import analyse as A
from d2 import stages as S
from d2.common import META
from d2.tests.test_d2_stages import rows_for

VIDEO, NF, W, H = "synthetic_d2_analysis", 200, 640, 480
PRED = "walk_past"

# name: (GT tid, class, box, GT frames, det frames, det score, selected, in tracker input, temp tracks)
OBJ = {
    "S1": (0, "dog", [10, 10, 110, 110], range(0, 200), range(0, 200), 0.9, True, True, [(1, range(0, 200))]),
    "O1": (1, "antelope", [300, 10, 400, 110], range(0, 100), range(0, 100), 0.2, False, False, []),
    "O2": (2, "domestic_cat", [10, 200, 110, 300], range(0, 100), range(0, 100), 0.03, False, False, []),
    "O3": (3, "car", [300, 200, 400, 300], range(0, 100), range(0, 100), 0.8, True, False, []),
    "O4": (4, "horse", [450, 10, 550, 110], range(0, 100), range(0, 100), 0.8, True, True, [(4, range(0, 100, 2))]),
    "O5": (5, "bus", [450, 200, 550, 300], range(0, 100), range(0, 100), 0.8, True, True,
           [(5, range(0, 45)), (6, range(55, 100))]),
    "O6": (6, "fox", [10, 350, 110, 450], range(120, 135), range(120, 135), 0.8, True, True, [(7, range(120, 135))]),
    "O7": (7, "bird", [300, 350, 400, 450], range(150, 158), range(150, 158), 0.8, True, True, [(8, range(150, 157))]),
    "O8": (8, "dog", [10, 10, 110, 110], range(0, 100), [], 0, False, False, []),          # same box as S1
    "O10": (10, "zebra", [200, 120, 280, 180], range(0, 100), range(0, 100), 0.8, True, True,
            [(10, list(range(0, 48)) + list(range(70, 100)))]),
}
# relation -> (object, duration, expected label at floors 0.01 / 0.05 / 0.1)
REL = {
    "R1": ("O1", (0, 100), [("O1b", None)] * 3),
    "R2": ("O2", (0, 100), [("O1b", None), ("O1a", None), ("O1a", None)]),
    "R3": ("O3", (0, 100), [("O2", "dedupe")] * 3),
    "R4": ("O4", (0, 100), [("O3", "F-cov")] * 3),
    "R5": ("O5", (0, 100), [("O2", "frag")] * 3),
    "R6": ("O6", (120, 135), [("O3", "F-range")] * 3),
    "R7": ("O7", (150, 158), [("O3", "F-len")] * 3),
    "R8": ("O8", (0, 100), [("S0", None)] * 3),
    "R10": ("O10", (0, 100), [("O3", "F-reshape-trim")] * 3),
}


@pytest.fixture(scope="module")
def splits():
    obj = json.load(open(os.path.join(META, "openvoc_obj_class_spilt_info.json")))["cls2split"]
    pred = json.load(open(os.path.join(META, "openvoc_pred_class_spilt_info.json")))["cls2split"]
    assert all(o[1] in obj for o in OBJ.values()) and pred[PRED] == "novel"
    return obj, pred


@pytest.fixture(scope="module")
def video(repo_fns):
    _, _, G = repo_fns
    det, ti, temp = [], [], []
    for q, (name, (tid, cls, box, gtf, detf, sc, sel, inti, tracks)) in enumerate(OBJ.items()):
        for f in detf:
            det.append((f, q, box, sc, sel))
            if inti:
                ti.append((f, box))
        for t, frames in tracks:
            temp.append(rows_for(t, frames, box))
    det.sort(key=lambda d: (d[0], d[1]))
    arr = dict(det_frame=np.array([d[0] for d in det], np.int32), det_q=np.array([d[1] for d in det], np.int16),
               det_box=np.array([d[2] for d in det], np.float32), det_score=np.array([d[3] for d in det], np.float32),
               det_sel=np.array([d[4] for d in det], bool), ti_frame=np.array([t[0] for t in ti], np.int32),
               ti_box=np.array([t[1] for t in ti], np.float32), temp=np.concatenate(temp))
    base = S.replay_video(G, arr["temp"], NF)
    meta = dict(video=VIDEO, begin_fid=0, end_fid=NF, w=W, h=H, n_post=NF, post_box_mismatch=0,
                post_cut_mismatch=0, final=base)
    anno = {"subject/objects": [dict(tid=o[0], category=o[1]) for o in OBJ.values()],
            "trajectories": [[dict(tid=o[0], bbox=dict(xmin=o[2][0], ymin=o[2][1], xmax=o[2][2], ymax=o[2][3]))
                              for o in OBJ.values() if f in o[3]] for f in range(NF)]}
    gts = {}
    for r, (obj, (gb, ge), _) in REL.items():
        s, o = OBJ["S1"], OBJ[obj]
        gts[r] = dict(triplet=[s[1], PRED, o[1]], subject_tid=s[0], object_tid=o[0], duration=[gb, ge],
                      sub_traj=[s[2]] * (ge - gb), obj_traj=[o[2]] * (ge - gb))
    return arr, meta, base, anno, gts


def test_check_video_passes_on_a_consistent_recording(video, repo_fns):
    _, _, G = repo_fns
    arr, meta, base, _, _ = video
    assert A.check_video(G, arr, meta, base) == []
    bad = dict(meta, final=[dict(t, boxes=t["boxes"][:-1] + [[0, 0, 1, 1]]) if i == 0 else t for i, t in enumerate(base)])
    assert any("ground rule 3" in e for e in A.check_video(G, arr, bad, base))
    assert any("replay" in e for e in A.check_video(G, arr, bad, bad["final"]))


def test_every_relation_lands_in_its_stage(video, repo_fns, splits):
    viou, _, G = repo_fns
    arr, meta, base, _, gts = video
    finals = {t["tid"]: t for t in base}
    joined = [dict(row=dict(best_pair_min_viou=A.best_pair_min(finals, g, viou)), gt=g) for g in gts.values()]
    rels = A.analyse_relations(VIDEO, joined, arr, meta, base, splits[0], G, viou)
    for (name, (_, _, want)), r in zip(REL.items(), rels):
        got = [tuple(r["labels"][str(f)]) for f in (0.01, 0.05, 0.1)]
        assert got == want, (name, got, r["sides"]["object"])
        assert tuple(r["label"]) == want[1]
        assert r["best_pair_min_viou"] < 0.5, name                      # every one is a U1 relation
    by = {n: r for n, r in zip(REL, rels)}
    assert not by["R1"]["sides"]["subject"]["missed"] and by["R1"]["sides"]["object"]["missed"]
    assert not by["R8"]["sides"]["object"]["missed"]                     # S0: the same track is best for both
    assert by["R4"]["sides"]["object"]["best_pre"] > 0.9                 # interpolated pre-filter track
    assert by["R10"]["sides"]["object"]["replay"]["final_viou"] == pytest.approx(0.48)
    assert by["R10"]["sides"]["object"]["replay"]["in_final"]
    assert by["R1"]["rel_split"] == "involves novel" and by["R3"]["rel_split"] == "base-only"

    T = A.tables(rels, [])
    assert T["t1"]["relation"]["n"] == 9
    assert T["t1"]["relation"]["top"]["O3"]["count"] == 4 and T["t1"]["relation"]["top"]["S0"]["count"] == 1
    assert T["t1"]["side"]["n"] == 8                                     # R8 has no missed side
    assert T["t3"]["rules"]["F-cov"]["count"] == 1 and T["t3"]["n_o3"] == 4
    assert T["t4"]["0.01"]["relation"]["O1a"]["count"] == 0 and T["t4"]["0.05"]["relation"]["O1a"]["count"] == 1
    assert T["checks"] == dict(o4_missed_sides=0, s0_relations=1, upstream_label_but_tracked=0,
                               replay_kept_not_in_final=0)
    R = A.reading(rels)
    assert R["largest"] == ["link"] and R["groups"]["link"]["count"] == 6 and R["second"] == "rescore"
    assert R["separated"] and "relation-aware track linking and keeping" in R["line"]
    assert "Reading (A25)" in A.render_md(T, R)

    # A25c, Table 6: R1's object is the only O1b side; its oracle boxes all score 0.2 (float32)
    sc = by["R1"]["sides"]["object"]["o1_scores"]
    assert len(sc) == 100 and all(s == pytest.approx(0.2) and sel is False for s, sel in sc)
    t6 = T["t6"]["all"]
    assert t6["n_sides"] == 1 and t6["n_frames"] == 100 and t6["pooled"]["median"] == pytest.approx(0.2)
    assert (t6["pooled"]["ge_0.1"], t6["pooled"]["ge_0.2"], t6["pooled"]["ge_0.3"], t6["pooled"]["selected"]) == (1, 1, 0, 0)
    assert T["t6"]["novel"]["n_sides"] == 1 and T["t6"]["base"]["n_sides"] == 0     # antelope is novel
    assert "Table 6" in A.render_md(T, R) and "Table 7" not in A.render_md(T, R)


def test_o1b_scores_follow_the_oracles_own_choice():
    """Two boxes in one frame: the score reported is the one of the box the oracle picked (the
    higher IoU), not the higher score; a frame with no box counts as below every level."""
    g = [10, 10, 49, 49]
    arr = dict(det_frame=np.array([0, 0, 1], np.int32), det_q=np.array([0, 1, 0], np.int16),
               det_box=np.array([[10, 10, 49, 49], [0, 0, 30, 30], [200, 200, 220, 220]], np.float32),
               det_score=np.array([0.15, 0.6, 0.25], np.float32), det_sel=np.array([False, True, False]),
               ti_frame=np.zeros(0, np.int32), ti_box=np.zeros((0, 4), np.float32))
    ds = S.DetSets(arr, 640, 480)
    out = S.oracle_scores(ds, [g, g], 0, 0.05)
    assert out[0][0] == pytest.approx(0.15) and out[0][1] is False and out[1] == (None, None)
    t6 = A.o1b_score_table([dict(video="v", label=("O1b", None), split="base", o1_scores=out)])["all"]
    assert t6["pooled"]["ge_0.1"] == 0.5 and t6["pooled"]["ge_0.2"] == 0 and t6["pooled"]["median"] == pytest.approx(0.15)


def test_frame_range_table():
    metas = [dict(video="a", video_len=100, begin_fid=0, end_fid=100),
             dict(video="b", video_len=200, begin_fid=50, end_fid=150),
             dict(video="c", video_len=40, begin_fid=10, end_fid=40)]
    t = A.frame_range_table(metas)
    assert (t["frames_total"], t["frames_outside"], t["videos_with_frames_outside"]) == (340, 110, 2)
    assert t["share_max"] == 0.5 and t["share_max_video"] == "b" and t["share_median"] == 0.25
    assert t["share_total"] == pytest.approx(110 / 340) and t["ranges_beyond_video"] == 0


def test_reading_names_ties_and_unseparated_groups():
    rel = lambda v, st: dict(video=v, label=(st, None))
    rels = [rel(f"v{i}", "O1a") for i in range(5)] + [rel(f"v{i}", "O1b") for i in range(5, 10)]
    R = A.reading(rels)
    assert R["largest"] == ["rescore", "detector"] and "tied largest" in R["line"]
    rels = [rel(f"v{i % 10}", "O1a" if i % 2 else ("O2" if i % 3 else "O1b")) for i in range(60)]
    R = A.reading(rels)
    assert R["largest"] == ["detector"]
    assert (not R["separated"]) == ("not separated" in R["line"])


def test_funnel(video, repo_fns, splits):
    viou, _, G = repo_fns
    arr, meta, base, anno, _ = video
    fun = {x["tid"]: x["cumulative"] for x in A.analyse_funnel(VIDEO, anno, arr, meta, base, splits[0], G, viou)}
    last = lambda c: max([0] + [i + 1 for i, k in enumerate(("F1", "F2", "F3", "F4", "F5")) if c[k]])
    assert {OBJ[n][0]: last(fun[OBJ[n][0]]) for n in OBJ} == dict(
        {0: 5, 1: 1, 2: 0, 3: 2, 4: 4, 5: 3, 6: 4, 7: 4, 8: 5, 10: 4})


def test_main_verifies_everything_before_opening_labels(tmp_path, monkeypatch):
    """A25, Order 3: pins, code, baseline and recordings first; with no recordings it stops there,
    and the test labels are never opened. On a code root other than the pinned EOV front end
    (D2_CODE_ROOT=METOR) it stops earlier, at the code pins."""
    import sys
    from d2.common import CODE_PINS, CODE_ROOT, THESIS
    if not os.path.exists(CODE_PINS):
        pytest.skip("A25a's code pins are not written yet")

    def opened():
        raise AssertionError("test labels opened before verification finished")
    monkeypatch.setattr(A, "open_labels", opened)
    monkeypatch.setattr(sys, "argv", ["analyse", "--rec_dir", str(tmp_path)])
    eov = os.path.abspath(CODE_ROOT) == os.path.join(THESIS, "ov-vidvrd-lab")
    with pytest.raises(SystemExit, match="no recordings" if eov else "differs from A25a's pins"):
        A.main()


def test_join_matches_by_value_and_stops_when_counts_differ(video, repo_fns, splits):
    viou, _, _ = repo_fns
    _, _, base, _, gts = video
    finals = {VIDEO: {t["tid"]: t for t in base}}
    keep = A.keep_filter(*splits)
    g1 = gts["R1"]
    g1b = dict(g1, object_tid=99, obj_traj=[[0, 400, 50, 450]] * 100)    # same key and value as R1
    g3 = gts["R3"]
    row = lambda g: dict(video=VIDEO, subject=g["triplet"][0], predicate=PRED, object=g["triplet"][2],
                         duration=list(g["duration"]), best_pair_min_viou=A.best_pair_min(finals[VIDEO], g, viou))
    out = A.join_u1([row(g1), row(g1b), row(g3)], {VIDEO: [g1, g1b, g3]}, finals, keep, viou)
    assert sorted(id(j["gt"]) for j in out) == sorted(map(id, (g1, g1b, g3)))
    with pytest.raises(SystemExit):                                       # two GT entries, one row
        A.join_u1([row(g1)], {VIDEO: [g1, g1b]}, finals, keep, viou)
    with pytest.raises(SystemExit):                                       # D1's value not reproduced
        A.join_u1([dict(row(g3), best_pair_min_viou=0.3)], {VIDEO: [g3]}, finals, keep, viou)
