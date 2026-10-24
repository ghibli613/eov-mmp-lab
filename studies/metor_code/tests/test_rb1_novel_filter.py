"""RB1 (pre-registration A23): --exclude_novel_train removes every novel item
from step-4 training targets, and with the flag off nothing changes.

Two kinds of test. Pure ones build tiny annotations by hand and pin each filter
(F1 detector fragments, F2/F3 relations and tracks) and each assertion. The
real-data ones read two training videos' annotations (no model, no CUDA; skipped
when the data is not present) and check the same properties on real files.
"""
from __future__ import annotations

import copy
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models import novel_filter as nf
from models.gen_labels import gen_box_label, gen_feats, gen_frame_pre_label
from utils import paths

NOVEL_OBJ = frozenset({"horse"})
NOVEL_PRED = frozenset({"jump_past"})
VIDEO = "ILSVRC2015_train_00227012"          # 45 frames, 1 novel object, mixed relations
have_data = os.path.exists(os.path.join(paths.ANNO_TRAIN_DIR, VIDEO + ".json"))
needs_data = pytest.mark.skipif(not have_data, reason="training annotations not present")


def _anno():
    traj = [[{"tid": 0, "bbox": {"xmin": 1, "ymin": 1, "xmax": 9, "ymax": 9}},
             {"tid": 1, "bbox": {"xmin": 2, "ymin": 2, "xmax": 8, "ymax": 8}},
             {"tid": 2, "bbox": {"xmin": 3, "ymin": 3, "xmax": 7, "ymax": 7}}]]
    return {
        "video_id": "v", "width": 10, "height": 10,
        "trajectories": traj,
        "subject/objects": [{"tid": 0, "category": "dog"},
                            {"tid": 1, "category": "horse"},
                            {"tid": 2, "category": "car"}],
        "relation_instances": [
            {"subject_tid": 0, "object_tid": 2, "predicate": "walk_past", "begin_fid": 0, "end_fid": 5},
            {"subject_tid": 0, "object_tid": 2, "predicate": "jump_past", "begin_fid": 0, "end_fid": 5},
            {"subject_tid": 0, "object_tid": 1, "predicate": "walk_past", "begin_fid": 0, "end_fid": 5},
            {"subject_tid": 1, "object_tid": 2, "predicate": "walk_past", "begin_fid": 0, "end_fid": 5},
        ],
    }


# ---------------------------------------------------------------- F1 / F2 / F3
def test_f1_removes_only_novel_fragments_and_counts_them():
    st = nf.NovelStats()
    frags = [{"category": "dog", "tid": 0, "trajectory": {"0": [1, 1, 2, 2]}},
             {"category": ["horse"], "tid": 1, "trajectory": {"0": [1, 1, 2, 2]}}]
    kept = nf.filter_gt_traj(frags, NOVEL_OBJ, st)
    assert [f["tid"] for f in kept] == [0]
    assert st.snapshot() == {"frag_removed_novel": 1, "frag_total": 2}


def test_f2_drops_novel_predicate_novel_subject_and_novel_object():
    a = _anno()
    st = nf.NovelStats()
    kept = nf.filter_relations(a["relation_instances"], nf.tid_categories(a),
                               NOVEL_OBJ, NOVEL_PRED, st)
    assert [(r["subject_tid"], r["object_tid"], r["predicate"]) for r in kept] == [(0, 2, "walk_past")]
    s = st.snapshot()
    assert (s["rel_total"], s["rel_removed"]) == (4, 3)
    assert s["rel_novel_predicate"] == 1 and s["rel_novel_object"] == 2


def test_f2_accepts_the_predicate_as_a_one_element_list():
    a = _anno()
    rels = copy.deepcopy(a["relation_instances"])
    for r in rels:
        r["predicate"] = [r["predicate"]]
    kept = nf.filter_relations(rels, nf.tid_categories(a), NOVEL_OBJ, NOVEL_PRED, nf.NovelStats())
    assert len(kept) == 1


def test_f3_removes_novel_tracks_and_does_not_mutate_the_input():
    a = _anno()
    before = copy.deepcopy(a)
    out = nf.filter_vid_anno(a, NOVEL_OBJ, NOVEL_PRED, nf.NovelStats())
    assert a == before
    assert [b["tid"] for b in out["trajectories"][0]] == [0, 2]
    assert [o["tid"] for o in out["subject/objects"]] == [0, 2]
    assert len(out["relation_instances"]) == 1


def test_all_novel_frames_are_counted_from_the_unfiltered_fragments():
    frags = [{"category": "horse", "tid": 1, "trajectory": {str(i): [0, 0, 1, 1] for i in range(0, 10)}},
             {"category": "dog", "tid": 0, "trajectory": {str(i): [0, 0, 1, 1] for i in range(5, 10)}}]
    assert nf.count_all_novel_frames(frags, 0, 10, NOVEL_OBJ) == 5     # frames 0-4
    assert nf.count_all_novel_frames(frags, 2, 4, NOVEL_OBJ) == 2      # range-limited


def test_assertions_pass_on_clean_input_and_abort_on_a_novel_item():
    nf.assert_base_object_ids([0, 2, 31], novel_obj_ids=frozenset({1}))
    with pytest.raises(AssertionError):
        nf.assert_base_object_ids([0, 1], novel_obj_ids=frozenset({1}))
    clean = np.zeros((2, 5)); nf.assert_clean_pre_label(clean, novel_pred_ids=frozenset({3}))
    dirty = np.zeros((2, 5)); dirty[1, 3] = 1
    with pytest.raises(AssertionError):
        nf.assert_clean_pre_label(dirty, novel_pred_ids=frozenset({3}))


# ------------------------------------------------- the repo's own split files
@needs_data
def test_split_ids_match_the_id_spaces_the_targets_use():
    """The filter's novel ids are only meaningful if they index the same spaces
    `gen_feats` writes: predicate2id for pre_label columns, object2id for
    classes. If these ever diverge the zero-count assertions would pass for the
    wrong reason."""
    no, np_, nobj_ids, npred_ids = nf.split_info()
    assert len(no) == 10 and len(np_) == 61
    p2id = json.load(open(os.path.join(paths.META_DIR, "predicate2id.json")))
    o2id = json.load(open(os.path.join(paths.META_DIR, "object2id.json")))
    assert npred_ids == frozenset(p2id[p] for p in np_)
    assert nobj_ids == frozenset(o2id[o] for o in no)
    assert {"antelope", "bear", "cattle", "fox", "giant_panda", "horse", "lizard",
            "motorcycle", "red_panda", "watercraft"} == set(no)


# ------------------------------------------------------ target builders, real data
def _real_anno(video=VIDEO):
    return json.load(open(os.path.join(paths.ANNO_TRAIN_DIR, video + ".json")))


@needs_data
def test_released_target_builders_run_on_this_repos_annotation_format():
    """RB1 found that `gen_frame_pre_label` and `gen_box_label` raised on the
    files in this repo (predicate and category are strings, boxes are flat). They
    now accept both forms; this pins that they run and give sane outputs."""
    anno = _real_anno()
    traj = json.load(open(paths.TRAIN_TRAJ))[VIDEO]
    w, h = anno["width"], anno["height"]
    seen_label = False
    for fid in range(anno["frame_count"]):
        lab = gen_frame_pre_label(fid, anno["relation_instances"])
        assert len(lab) == 71 and set(lab) <= {0, 1}
        seen_label |= any(lab)
        xyxy, cxcywh, labels = gen_box_label(fid, traj, w, h)
        assert len(xyxy) == len(cxcywh) == len(labels)
        assert all(0 <= l < 35 for l in labels)
    assert seen_label


@needs_data
def test_exclude_novel_leaves_no_novel_target_on_a_real_video():
    anno = _real_anno()
    traj = json.load(open(paths.TRAIN_TRAJ))[VIDEO]
    novel_obj, novel_pred, nobj_ids, _ = nf.split_info()
    assert any(f["category"] in novel_obj for f in traj), "pick a video with a novel object"
    ftraj = nf.filter_gt_traj(traj)
    rel = nf.filter_relations(anno["relation_instances"], nf.tid_categories(anno))
    t2c = nf.tid_categories(anno)
    assert all(f["category"] not in novel_obj for f in ftraj)
    assert rel and len(rel) < len(anno["relation_instances"])
    for r in rel:
        assert r["predicate"] not in novel_pred
        assert t2c[r["subject_tid"]] not in novel_obj and t2c[r["object_tid"]] not in novel_obj
    for fid in range(anno["frame_count"]):
        _, _, labels = gen_box_label(fid, ftraj, anno["width"], anno["height"])
        assert not set(labels) & nobj_ids


# ------------------------------------------------------------------ gen_feats
def _tracks_from_annotation(anno):
    """Pipeline-shaped tracks built from the GT, so gen_feats has pairs to label.
    Every tid becomes a track named with its GT category (novel ones included, as
    a detector might produce them)."""
    t2c = nf.tid_categories(anno)
    boxes = {}
    for fid, frame in enumerate(anno["trajectories"]):
        for b in frame:
            bb = b["bbox"]
            boxes.setdefault(b["tid"], {})[fid] = [bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"]]
    tracks = []
    for tid, fb in boxes.items():
        lo, hi = min(fb), max(fb)
        full, last = {}, None
        for f in range(lo, hi + 1):
            last = fb.get(f, last)
            full[f] = last
        tracks.append({"category": t2c[tid], "trajectory": full, "begin_fid": lo,
                       "end_fid": hi + 1, "tid": len(tracks), "score": 1.0})
    return tracks


@pytest.fixture
def cpu_cuda(monkeypatch):
    # gen_feats's clip-feature step calls .cuda(); the tensors are tiny, so run
    # them on the CPU and keep this test GPU-free.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)


@needs_data
def test_gen_feats_with_the_flag_off_keeps_novel_labels_and_with_it_on_has_none(cpu_cuda):
    anno = _real_anno()
    tracks = _tracks_from_annotation(anno)
    T = anno["frame_count"]
    patch = torch.zeros(1, 1, 576, 768).expand(1, T, 576, 768)
    glob = torch.zeros(1, 1, 768).expand(1, T, 768)
    args = ([VIDEO], tracks, "train", patch, glob, anno["width"], anno["height"])
    _, novel_pred, nobj_ids, npred_ids = nf.split_info()
    cols = sorted(npred_ids)

    off = gen_feats(*args)
    nf.STATS.reset()
    on = gen_feats(*args, exclude_novel=True)

    def novel_entries(items):
        return sum(int(it["pre_label"][..., cols].sum()) for it in items.values())

    def interactive(items):
        return sum(int((it["pre_label"].sum(-1) > 0).sum()) for it in items.values())

    assert off and on and len(off) == len(on)      # same pairs; only the labels differ
    assert novel_entries(off) > 0                  # released: novel predicates are in pre_label
    assert novel_entries(on) == 0                  # RB1: none, so none reach int_labels
    # the interactiveness target (any predicate set) carries strictly less
    # positive signal once novel predicates / novel objects are gone
    assert interactive(on) < interactive(off)
    s = nf.STATS.snapshot()
    assert s["novel_predicate_entries_in_pre_label"] == 0
    assert s["novel_gt_tids_in_label_matching"] == 0
    assert s["rel_removed"] > 0


@needs_data
def test_flag_off_gen_feats_is_the_released_computation(cpu_cuda):
    """Calling gen_feats without the new argument and with exclude_novel=False
    are the same call and give identical targets."""
    anno = _real_anno()
    tracks = _tracks_from_annotation(anno)
    T = anno["frame_count"]
    patch = torch.zeros(1, 1, 576, 768).expand(1, T, 576, 768)
    glob = torch.zeros(1, 1, 768).expand(1, T, 768)
    args = ([VIDEO], tracks, "train", patch, glob, anno["width"], anno["height"])
    a, b = gen_feats(*args), gen_feats(*args, exclude_novel=False)
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k]["pre_label"], b[k]["pre_label"])
        assert torch.equal(a[k]["sbj_label"], b[k]["sbj_label"])
