"""A26's two wrappers. Neither edits the pipeline; both act from outside it.

whole_video(batch)   the loader's batch with the frame range set to the whole video
config_batch(...)    the batch for one configuration's frame range
set_fusion(...)      the fusion switch for one configuration (A23's --uniform_fusion)
BothSplits           modelC wrapped so one pass gives both predicate splits (pilot B.21)
"""
from __future__ import annotations

import torch
from torch import nn


def whole_video(batch: dict) -> dict:
    """A shallow copy of a DataLoader batch (batch size 1) from `Dataset_new(args, 'val')`, with
    `begin_fid = 0` and `end_fid = video_len`; every other entry is the same object.

    End2End_Model's test path reads its frame range only from these two entries: the detection
    loop runs over range(begin_fid, end_fid) (end2end_model.py:205-210) and the 15% length rule
    uses min_l = end_fid - begin_fid (:322-327). Dataset_new sets them from the GT object
    trajectories (feature_dataset.py:123-138). Here both come from the frames alone.
    """
    n = int(batch["video_len"][0])
    b, e = int(batch["begin_fid"][0]), int(batch["end_fid"][0])
    if not 0 <= b < e <= n:
        raise ValueError(f"GT range [{b}, {e}) is not inside the video's {n} frames")
    out = dict(batch)
    out["begin_fid"] = torch.zeros_like(batch["begin_fid"])
    out["end_fid"] = batch["video_len"].clone().to(batch["end_fid"].dtype)
    return out


def config_batch(batch: dict, frame_range: str) -> dict:
    if frame_range == "gt":
        return batch
    if frame_range == "whole":
        return whole_video(batch)
    raise ValueError(frame_range)


def set_fusion(model, uniform: bool) -> None:
    """End2End_Model reads `self.args.uniform_fusion` at every detection (end2end_model.py:271-272)."""
    model.args.uniform_fusion = bool(uniform)


class BothSplits(nn.Module):
    """modelC, called twice per tracklet pair: 'all' (returned, so End2End_Model is unchanged) and
    'novel' (stashed in `other`, in call order). The pilot's `_BothSplits`
    (pilot_analysis/scripts/dump_predictions.py), validated exact on GPU (PILOT-STATUS B.21).

    End2End_Model calls `self.modelC(feats, seq_lens)` once per pair and appends one final_result
    per call, so `other[k]` belongs to final_results[k]. A module, because PyTorch refuses a plain
    object where a child module lives. The training path is passed through untouched.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.other = []

    @property
    def tgt_split(self):
        return self.inner.tgt_split

    @tgt_split.setter
    def tgt_split(self, v):
        self.inner.tgt_split = v

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["inner"], name)

    def forward(self, feats, seq_lens, labels=None):
        if labels is not None:
            return self.inner(feats, seq_lens, labels)
        self.inner.tgt_split = "all"
        first = self.inner(feats, seq_lens)
        self.inner.tgt_split = "novel"
        second = self.inner(feats, seq_lens)
        self.inner.tgt_split = "all"
        self.other.append(tuple(t.detach() if torch.is_tensor(t) else t for t in second))
        return first
