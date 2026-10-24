"""RB1 (A23): what `Dataset_new` hands the training loop with --exclude_novel_train
off (the released item) and on (F1/F2). Frames are real; the CLIP encoders are
stubs so no GPU or weights are needed."""
from __future__ import annotations

import copy
import json
import os

import pytest

torch = pytest.importorskip("torch")

from data_loading import feature_dataset as fd
from models import novel_filter as nf
from utils import paths

VIDEO = "ILSVRC2015_train_00227012"
pytestmark = pytest.mark.skipif(
    not (os.path.isdir(os.path.join(paths.FRAME_DIR, VIDEO))
         and os.path.exists(paths.TRAIN_TRAJ)),
    reason="training frames / trajectories not present")


class _StubClip:
    def encode_image(self, x):
        return None, torch.zeros(1, 4)

    def encode_image_tagclip(self, x, h, w, attn_mask=1):
        return torch.zeros(1, 4), torch.zeros(1, 4, 4)


@pytest.fixture
def make_ds(monkeypatch, args):
    monkeypatch.setattr(fd, "_shared_encoder", lambda *a, **k: _StubClip())
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)

    def make(exclude):
        a = copy.copy(args)
        a.exclude_novel_train = exclude
        ds = fd.Dataset_new(a, "train")
        ds.path_list = [VIDEO]
        return ds
    return make


def test_flag_off_item_is_the_unfiltered_annotation(make_ds):
    item = make_ds(False)[0]
    assert item["gt_traj"] == json.load(open(paths.TRAIN_TRAJ))[VIDEO]
    anno = json.load(open(os.path.join(paths.ANNO_TRAIN_DIR, VIDEO + ".json")))
    assert item["gt_rel"] == anno["relation_instances"]


def test_flag_on_filters_both_and_keeps_the_released_frame_range(make_ds):
    off, nf_stats = make_ds(False)[0], None
    nf.STATS.reset()
    on = make_ds(True)[0]
    assert (on["begin_fid"], on["end_fid"]) == (off["begin_fid"], off["end_fid"])
    novel_obj, novel_pred, _, _ = nf.split_info()
    assert len(on["gt_traj"]) < len(off["gt_traj"])
    assert all(f["category"] not in novel_obj for f in on["gt_traj"])
    assert 0 < len(on["gt_rel"]) < len(off["gt_rel"])
    assert all(r["predicate"] not in novel_pred for r in on["gt_rel"])
    s = nf.STATS.snapshot()
    assert s["frag_removed_novel"] > 0 and s["rel_removed"] > 0
    assert "frames_all_novel" in s


def test_the_flag_only_applies_to_the_training_split(monkeypatch, args):
    monkeypatch.setattr(fd, "_shared_encoder", lambda *a, **k: _StubClip())
    a = copy.copy(args)
    a.exclude_novel_train = True
    assert fd.Dataset_new(a, "val").exclude_novel is False
