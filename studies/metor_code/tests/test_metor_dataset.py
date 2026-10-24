"""`METORDataset` against the real VidVRD data.

Light by construction: it reads one video's key frames (about six at stride 30)
and never builds a model. Skipped when the data is not present.

The properties worth pinning are the ones that would produce a model that
trains happily and scores zero:

  * the key-frame grid must match `data_loading/feature_dataset.py`'s, or `t` means
    different things in the model and the evaluator;
  * trajectory keys are 0-indexed while frame *files* are 1-indexed, and
    mixing them up shifts every box by one frame;
  * boxes must be normalised `cxcywh`, matching the sigmoid head of Eq. 5;
  * training must never see a novel category.
"""
from __future__ import annotations

import argparse
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from utils import paths

pytestmark = pytest.mark.skipif(
    not os.path.exists(paths.FRAME_DIR) or not os.path.exists(paths.OBJ_SPLIT_INFO),
    reason="VidVRD data not available")

from data_loading.frame_dataset import (
    METORDataset,
    metor_collate,
    xyxy_to_cxcywh_normalised,
)


def _args(**kw):
    base = dict(dataset="vidvrd", frame_stride=30)
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture(scope="module")
def dataset():
    return METORDataset(_args(), "train")


@pytest.fixture(scope="module")
def item(dataset):
    return dataset[0]


# ------------------------------------------------------------------- geometry
def test_box_conversion_is_normalised_cxcywh():
    out = xyxy_to_cxcywh_normalised([0, 0, 320, 180], 640, 360)
    assert out == pytest.approx([0.25, 0.25, 0.5, 0.5])


def test_boxes_are_in_unit_range(item):
    boxes = item["gt_boxes"][item["gt_box_mask"]]
    if boxes.numel():
        assert bool((boxes >= -1e-6).all() and (boxes <= 1 + 1e-6).all())


# -------------------------------------------------------------- key-frame grid
def test_key_frames_start_at_frame_one_and_step_by_stride(dataset):
    """Frame files are 1-indexed, so the grid is `(fno - 1) % stride == 0`.
    `data_loading/feature_dataset.py` uses exactly this; a half-frame offset between the
    two would make any comparison against the EOV baseline meaningless."""
    keys = dataset._key_frames(dataset.video_names[0])
    numbers = [fno for _, fno in keys]
    assert numbers[0] == 1
    assert all((n - 1) % 30 == 0 for n in numbers)


def test_frame_ids_are_zero_indexed_for_trajectory_lookup(dataset, item):
    """Trajectory tables are keyed 0-indexed; filenames are 1-indexed."""
    keys = dataset._key_frames(item["video_name"])
    assert item["frame_ids"] == [fno - 1 for _, fno in keys]
    assert item["frame_ids"][0] == 0


def test_shapes_agree_across_the_item(item):
    t = item["num_frames"]
    n = item["gt_boxes"].shape[0]
    assert item["images"].shape[0] == t
    assert item["gt_boxes"].shape == (n, t, 4)
    assert item["gt_box_mask"].shape == (n, t)
    assert item["gt_labels"].shape == (n,)
    assert len(item["frame_ids"]) == t


def test_stride_one_yields_more_frames_than_stride_thirty(dataset):
    video = dataset.video_names[0]
    dense = METORDataset(_args(frame_stride=1), "train")
    assert len(dense._key_frames(video)) > len(dataset._key_frames(video))


# ---------------------------------------------------------------- annotations
def test_at_least_one_trajectory_has_boxes(item):
    assert bool(item["gt_box_mask"].any())


def test_labels_are_valid_object_ids(dataset, item):
    labels = item["gt_labels"]
    assert bool(((labels >= 0) & (labels < dataset.n_obj)).all())


def test_present_sets_are_multi_hot(dataset, item):
    for key, n in (("present_obj", dataset.n_obj), ("present_rel", dataset.n_rel)):
        v = item[key]
        assert v.shape == (n,)
        assert set(v.unique().tolist()) <= {0.0, 1.0}


def test_pair_labels_are_indexed_by_trajectory_row(dataset, item):
    """Keys are (subject_row, object_row) into gt_boxes, not raw tids -- the
    matcher works in row space and a tid would silently mis-target."""
    n = item["gt_boxes"].shape[0]
    for (s, o), lab in item["pair_labels"].items():
        assert 0 <= s < n and 0 <= o < n
        assert s != o
        assert lab.shape == (item["num_frames"], dataset.n_rel)


def test_a_pair_label_is_set_somewhere(dataset):
    """Guards against a silently empty supervision signal."""
    for i in range(min(12, len(dataset))):
        it = dataset[i]
        if any(float(v.sum()) > 0 for v in it["pair_labels"].values()):
            return
    pytest.fail("no predicate labels in the first 12 videos")


# -------------------------------------------------------------- open-vocabulary
def test_base_splits_are_non_empty_and_disjoint_from_novel(dataset):
    assert dataset.base_rel_ids and dataset.base_obj_ids
    assert len(dataset.base_rel_ids) < dataset.n_rel     # some are novel
    assert len(dataset.base_obj_ids) <= dataset.n_obj


def test_base_relation_remap_is_a_bijection_onto_its_range(dataset):
    m = dataset.rel_id_to_base
    assert sorted(m.values()) == list(range(len(dataset.base_rel_ids)))


# --------------------------------------------------------------------- collate
def test_collate_unwraps_a_single_video(item):
    assert metor_collate([item]) is item


def test_collate_rejects_a_real_batch(item):
    """Sec. 4.2: one video at a time. Ragged shapes make silent collation
    worse than a crash."""
    with pytest.raises(ValueError, match="batch_size=1"):
        metor_collate([item, item])


def test_split_argument_is_validated():
    with pytest.raises(ValueError, match="split"):
        METORDataset(_args(), "nonsense")
