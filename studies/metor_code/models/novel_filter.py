"""RB1: remove novel categories from the EOV-MMP step-4 training targets.

Everything here is behind `--exclude_novel_train` (default off). Off, none of
these functions is called and step 4 is exactly the released code. On, three
filters are applied (pre-registration A23, "What clean means"):

    F1  filter_gt_traj     detector box targets: novel-category fragments removed
    F2  filter_relations   relation instances removed if the predicate is novel
                           or the subject/object has a novel category
    F3  filter_vid_anno    the same two removals on the annotation `gen_feats`
                           reads, so a track over a novel object has no GT hit

Novel objects are not relabelled: they are dropped, so they become background.
Frames are kept (the processed frame range is computed before filtering).

`STATS` counts what was removed and what was used, per epoch. The `assert_*`
helpers abort a run if a novel item reaches a training target.
"""
from __future__ import annotations

import copy
import json
from collections import Counter
from functools import lru_cache

from utils import paths


@lru_cache(maxsize=None)
def split_info():
    """(novel object names, novel predicate names, novel object ids,
    novel predicate ids) from the repo's own split files."""
    obj = json.load(open(paths.OBJ_SPLIT_INFO, "r"))
    pred = json.load(open(paths.PRED_SPLIT_INFO, "r"))
    novel_obj = frozenset(c for c, s in obj["cls2split"].items() if s == "novel")
    novel_pred = frozenset(c for c, s in pred["cls2split"].items() if s == "novel")
    novel_obj_ids = frozenset(obj["cls2id"][c] for c in novel_obj)
    novel_pred_ids = frozenset(pred["cls2id"][c] for c in novel_pred)
    return novel_obj, novel_pred, novel_obj_ids, novel_pred_ids


def name_of(x):
    """A category or predicate may be a string or a one-element list.

    The released target builders index `[0]`, which assumes the list form; the
    annotation files in this repo hold strings. Both are accepted here.
    """
    return x[0] if isinstance(x, (list, tuple)) else x


class NovelStats:
    """Per-epoch counters. `reset()` at the start of an epoch."""

    def __init__(self):
        self.c = Counter()

    def reset(self):
        self.c = Counter()

    def add(self, key, n=1):
        self.c[key] += n

    def snapshot(self):
        return dict(sorted(self.c.items()))


STATS = NovelStats()


# ------------------------------------------------------------------ filters
def filter_gt_traj(gt_traj, novel_objects=None, stats=STATS):
    """F1. `gt_traj` is a list of trajectory fragments with a 'category'."""
    if novel_objects is None:
        novel_objects = split_info()[0]
    kept = []
    for frag in gt_traj:
        stats.add("frag_total")
        if name_of(frag["category"]) in novel_objects:
            stats.add("frag_removed_novel")
        else:
            kept.append(frag)
    return kept


def tid_categories(vid_anno):
    """{tid: category name} from the annotation's 'subject/objects'."""
    return {o["tid"]: name_of(o["category"]) for o in vid_anno["subject/objects"]}


def filter_relations(relations, tid2cat, novel_objects=None, novel_predicates=None,
                     stats=STATS):
    """F2/F3. Drop a relation whose predicate is novel, or whose subject or
    object is a novel-category object."""
    if novel_objects is None or novel_predicates is None:
        no, np_, _, _ = split_info()
        novel_objects = no if novel_objects is None else novel_objects
        novel_predicates = np_ if novel_predicates is None else novel_predicates
    kept = []
    for rel in relations:
        stats.add("rel_total")
        bad_pred = name_of(rel["predicate"]) in novel_predicates
        bad_obj = (tid2cat.get(rel["subject_tid"]) in novel_objects
                   or tid2cat.get(rel["object_tid"]) in novel_objects)
        if bad_pred:
            stats.add("rel_novel_predicate")
        if bad_obj:
            stats.add("rel_novel_object")
        if bad_pred or bad_obj:
            stats.add("rel_removed")
        else:
            kept.append(rel)
    return kept


def filter_vid_anno(vid_anno, novel_objects=None, novel_predicates=None, stats=STATS):
    """F3. A copy of a video annotation with novel objects and every relation
    that touches a novel object or predicate removed. `vid_anno` is not mutated."""
    if novel_objects is None:
        novel_objects = split_info()[0]
    tid2cat = tid_categories(vid_anno)
    novel_tids = {t for t, c in tid2cat.items() if c in novel_objects}
    out = copy.copy(vid_anno)
    out["trajectories"] = [[b for b in frame if b["tid"] not in novel_tids]
                           for frame in vid_anno["trajectories"]]
    out["subject/objects"] = [o for o in vid_anno["subject/objects"]
                              if o["tid"] not in novel_tids]
    out["relation_instances"] = filter_relations(
        vid_anno["relation_instances"], tid2cat, novel_objects, novel_predicates, stats)
    stats.add("tids_total", len(tid2cat))
    stats.add("tids_removed_novel", len(novel_tids))
    return out


def count_all_novel_frames(gt_traj, begin_fid, end_fid, novel_objects=None):
    """Frames in [begin_fid, end_fid) that hold novel objects and no base
    object. After F1 these have no detector target and the existing
    `len(labels) > 0` gate skips them (released behaviour for an object-free
    frame). `gt_traj` is the UNFILTERED fragment list."""
    if novel_objects is None:
        novel_objects = split_info()[0]
    base_frames, novel_frames = set(), set()
    for frag in gt_traj:
        target = novel_frames if name_of(frag["category"]) in novel_objects else base_frames
        target.update(int(k) for k in frag["trajectory"])
    return sum(1 for f in novel_frames - base_frames if begin_fid <= f < end_fid)


# ---------------------------------------------------------------- assertions
def assert_base_object_ids(labels, novel_obj_ids=None, where="detector targets"):
    """No novel object id may be a detector target / text query."""
    if novel_obj_ids is None:
        novel_obj_ids = split_info()[2]
    bad = [int(x) for x in labels if int(x) in novel_obj_ids]
    STATS.add("novel_object_ids_in_targets", len(bad))
    if bad:
        raise AssertionError(f"novel object ids {sorted(set(bad))} in {where}")


def assert_clean_pre_label(pre_label, novel_pred_ids=None):
    """No novel predicate column may be set in a relation `pre_label`
    (shape (..., n_pred)); this is what feeds the interactiveness target."""
    if novel_pred_ids is None:
        novel_pred_ids = split_info()[3]
    cols = sorted(novel_pred_ids)
    n = float(pre_label[..., cols].sum())
    STATS.add("novel_predicate_entries_in_pre_label", int(n))
    if n != 0:
        raise AssertionError(f"{int(n)} novel-predicate entries set in pre_label")
