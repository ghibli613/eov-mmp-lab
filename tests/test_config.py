"""The split files and the argument defaults are load-bearing constants.

Every number reported by this repository is relative to these. A silent edit --
a re-generated split, a changed default -- would move results without moving any
code, so they are pinned here.
"""
import json

from utils import paths


def test_predicate_split_is_the_published_one():
    info = json.load(open(paths.PRED_SPLIT_INFO))
    split = {k: v for k, v in info["cls2split"].items() if k != "__background__"}
    base = [k for k, v in split.items() if v == "base"]
    novel = [k for k, v in split.items() if v == "novel"]
    assert len(split) == 132, "VidVRD has 132 predicates"
    assert (len(base), len(novel)) == (71, 61)
    assert not set(base) & set(novel)


def test_object_split_is_the_published_one():
    info = json.load(open(paths.OBJ_SPLIT_INFO))
    split = info["cls2split"]
    base = [k for k, v in split.items() if v == "base"]
    novel = [k for k, v in split.items() if v == "novel"]
    assert len(split) == 35, "VidVRD has 35 object categories"
    assert (len(base), len(novel)) == (25, 10)


def test_ids_are_dense_and_consistent(  ):
    for path in (paths.PRED_SPLIT_INFO, paths.OBJ_SPLIT_INFO):
        info = json.load(open(path))
        id2cls, cls2id = info["id2cls"], info["cls2id"]
        assert sorted(int(i) for i in id2cls) == list(range(len(id2cls)))
        for i, name in id2cls.items():
            assert cls2id[name] == int(i)


def test_defaults_are_the_papers_config(args):
    assert args.dataset == "vidvrd"
    assert args.clip_len == 30           # VidVRD-II segment length
    assert args.batch_size == 1
    assert args.lr == 1e-5               # paper step 4
    assert args.train_split == "base"    # train on base, evaluate on all
    assert args.test_split == "all"
    assert args.temperature == 0.01      # paper tau


def test_cleanup_flags_default_to_upstream_behaviour(args):
    """The flags added in the 2026-08-22 cleanup must not change any result
    unless explicitly set."""
    assert args.frame_stride == 1, "1 = encode every frame, as upstream ships"
    assert args.normalize_visual_feats is False
    for stream in ("rel_feat", "mot_feat", "clip_feat", "bbox_feat"):
        assert getattr(args, stream) is True


def test_feat_types_reports_only_consumed_streams(args):
    from utils.eov_utils import get_feat_types
    assert get_feat_types(args) == ["rel_feat", "mot_feat", "clip_feat", "bbox_feat"]
    args.clip_feat = False
    try:
        assert get_feat_types(args) == ["rel_feat", "mot_feat", "bbox_feat"]
    finally:
        args.clip_feat = True
