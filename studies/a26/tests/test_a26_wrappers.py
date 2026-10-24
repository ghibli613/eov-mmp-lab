"""The whole-video wrapper, the fusion switch and BothSplits, on their own."""
import argparse
import json
import os

import numpy as np
import pytest
import torch
from torch import nn

from a26.wrappers import BothSplits, config_batch, set_fusion, whole_video


def _batch(b=10, e=50, n=60):
    return dict(video_name=["v"], begin_fid=torch.tensor([b]), end_fid=torch.tensor([e]), video_len=torch.tensor([n]),
                patch_=torch.zeros(1, n, 3, 2), image_size=[torch.tensor([64]), torch.tensor([48])])


def test_whole_video_sets_only_the_range():
    batch = _batch()
    w = whole_video(batch)
    assert int(w["begin_fid"][0]) == 0 and int(w["end_fid"][0]) == 60
    assert w["begin_fid"].dtype == batch["begin_fid"].dtype and w["end_fid"].dtype == batch["end_fid"].dtype
    assert w["begin_fid"].shape == w["end_fid"].shape == (1,)
    for k in batch:
        if k not in ("begin_fid", "end_fid"):
            assert w[k] is batch[k]
    assert int(batch["begin_fid"][0]) == 10 and int(batch["end_fid"][0]) == 50     # the input is untouched
    assert config_batch(batch, "gt") is batch
    assert int(config_batch(batch, "whole")["end_fid"][0]) == 60
    with pytest.raises(ValueError):
        config_batch(batch, "other")


@pytest.mark.parametrize("b,e,n", [(10, 10, 60), (20, 10, 60), (0, 61, 60), (-1, 30, 60)])
def test_whole_video_refuses_a_range_outside_the_video(b, e, n):
    with pytest.raises(ValueError):
        whole_video(_batch(b, e, n))


class _StubEncoder:
    """CLIP / TagCLIP stand-ins for Dataset_new.__getitem__: features carry the frame's mean pixel."""

    def encode_image(self, x):
        return None, x.mean().reshape(1, 1).expand(1, 8).clone()

    def encode_image_tagclip(self, x, h, w, attn_mask=1):
        m = x.mean()
        return m.reshape(1, 1, 1).expand(1, 577, 4).clone(), m.reshape(1, 1, 1).expand(1, 576, 6).clone()


def test_real_dataset_item_then_whole_video(tmp_path, cpu_cuda):
    """Dataset_new.__getitem__ itself (stub encoders, synthetic frames): the GT-derived range, then the
    whole video through the DataLoader's collation, as run.py receives it."""
    from PIL import Image
    from data_loading.feature_dataset import Dataset_new, _transform_resize
    vid, n = "synthetic_a26", 12
    os.makedirs(tmp_path / vid)
    for i in range(n):
        Image.fromarray(np.full((24, 32, 3), 10 * i, np.uint8)).save(tmp_path / vid / f"{i + 1:06d}.jpg")
    ds = Dataset_new.__new__(Dataset_new)
    ds.split, ds.FEAT_ROOT, ds.frame_stride = "val", str(tmp_path), 1
    ds.gt_traj = {vid: [dict(begin_fid=3, end_fid=8, category="dog"), dict(begin_fid=5, end_fid=10, category="cat")]}
    ds.path_list = [vid]
    ds.preprocess = _transform_resize(336, 336)
    ds.clip = ds.tagclip = _StubEncoder()
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)))
    assert (int(batch["begin_fid"][0]), int(batch["end_fid"][0]), int(batch["video_len"][0])) == (3, 10, n)
    assert batch["patch_"].shape[:2] == (1, n)
    w = whole_video(batch)
    assert (int(w["begin_fid"][0]), int(w["end_fid"][0])) == (0, n)
    assert w["patch_"] is batch["patch_"] and w["patch_proj"] is batch["patch_proj"]


def test_set_fusion():
    m = argparse.Namespace(args=argparse.Namespace())
    set_fusion(m, True)
    assert m.args.uniform_fusion is True
    set_fusion(m, 0)
    assert m.args.uniform_fusion is False


class _StubRelation(nn.Module):
    """modelC-shaped: the novel split zeroes the base predicate columns, as the real one does."""

    def __init__(self, novel_pids):
        super().__init__()
        self.tgt_split = "all"
        self.novel_pids = novel_pids
        self.featEmbedding = nn.Identity()
        self.calls = []

    def forward(self, feats, seq_lens, labels=None):
        self.calls.append(self.tgt_split if labels is None else "train")
        if labels is not None:
            return torch.tensor(1.0)
        x = feats["clip_feat"].sum(-1, keepdim=True)
        pre = torch.sigmoid(x + torch.arange(132.0) / 50)
        if self.tgt_split == "novel":
            z = torch.zeros_like(pre)
            z[..., self.novel_pids] = pre[..., self.novel_pids]
            pre = z
        return pre, torch.zeros(1, 35), torch.zeros(1, 35)


def _novel_pids():
    from utils import paths
    info = json.load(open(paths.PRED_SPLIT_INFO))
    return [info["cls2id"][c] for c in info["cls2id"] if info["cls2split"][c] == "novel"]


def test_both_splits():
    inner = _StubRelation(_novel_pids())
    both = BothSplits(inner)
    feats = {"clip_feat": torch.randn(1, 3, 4, generator=torch.Generator().manual_seed(0))}
    out = both(feats, torch.tensor([3]))
    inner.tgt_split = "all"
    ref_all = inner(feats, torch.tensor([3]))
    inner.tgt_split = "novel"
    ref_novel = inner(feats, torch.tensor([3]))
    inner.tgt_split = "all"
    assert torch.equal(out[0], ref_all[0])
    assert len(both.other) == 1 and torch.equal(both.other[0][0], ref_novel[0])
    assert inner.tgt_split == "all" and both.tgt_split == "all"
    assert both.featEmbedding is inner.featEmbedding                     # run.py's startup check reaches through
    base = [i for i in range(132) if i not in set(inner.novel_pids)]
    assert float(both.other[0][0][..., base].abs().sum()) == 0.0
    inner.calls.clear()
    assert float(both(feats, torch.tensor([3]), labels={"x": 1})) == 1.0      # training: passed through
    assert inner.calls == ["train"] and len(both.other) == 1
