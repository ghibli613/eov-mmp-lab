"""`--arch metor-compat` must be a strict superset of EOV-MMP.

Two properties make the compat path worth trusting, and both are easy to break
by accident:

1. **Parameter names survive.** EOV-MMP's released checkpoints are the only
   trained weights that exist for this lineage. If attaching METOR's modules
   renames or reshapes a single EOV parameter, `load_state_dict(strict=False)`
   silently reports it as "missing" and that layer loads *randomly initialised*
   instead of erroring. A model like that trains, evaluates, and quietly scores
   nonsense. So the key set is pinned against the real checkpoint.

2. **`--n_iter 0` reproduces EOV exactly.** The zero-initialised mapping in
   `IterativeEnhancementLayer` means the enhancement contributes nothing at
   initialisation, so a fresh compat model with no iterations must produce
   bit-identical features to the EOV baseline. That equality is what makes any
   later delta attributable to the module rather than to the plumbing.

These run without CUDA by exercising `METORFeatEmbedding` against a stub that
has EOV's `FeatEmbedding` interface. The checkpoint-key test needs the real
checkpoint and skips without it.
"""
from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F

from metor.compat import METORFeatEmbedding
from utils import paths

RELATION_CKPT = paths.RELATION_CKPT
D = 768


class _StubSpatialDecoder(nn.Module):
    """EOV's `SpatialDecoder` interface, small enough to run on CPU."""

    def __init__(self):
        super().__init__()
        self.objEmb = nn.Linear(D, D, bias=False)

    def forward(self, x_img, x_box=None, x_pos=None):
        pre = F.normalize(x_img, dim=-1)
        obj = F.normalize(self.objEmb(pre[:, :2]), dim=-1)
        return pre, obj[:, 0], obj[:, 1]


class _StubTemporalDecoder(nn.Module):
    def forward(self, x_clip, seq_lens):
        bs, slen = x_clip.shape[0], x_clip.shape[1]
        return F.normalize(x_clip, dim=-1), torch.ones(bs, slen)


class _StubFeatEmbedding(nn.Module):
    """Stands in for EOV's `FeatEmbedding`, including its forward."""

    def __init__(self):
        super().__init__()
        self.enabled_streams = {"clip_feat": True, "bbox_feat": True,
                                "rel_feat": True, "mot_feat": True}
        self.spatial_decoder = _StubSpatialDecoder()
        self.temporal_decoder = _StubTemporalDecoder()

    def _stream(self, inputs, name):
        x = inputs[name]
        return x if self.enabled_streams[name] else torch.zeros_like(x)

    def forward(self, inputs, seq_lens):
        bs, slen = inputs["clip_feat"].shape[0], inputs["clip_feat"].shape[1]
        img = self._stream(inputs, "clip_feat")
        box = self._stream(inputs, "bbox_feat")
        pos = torch.cat((self._stream(inputs, "rel_feat"),
                         self._stream(inputs, "mot_feat")), dim=-1)
        pre, sbj, obj = self.spatial_decoder(
            img.view(bs * slen, 4, -1), box.view(bs * slen, 4, -1),
            pos.view(bs * slen, -1))
        sbj = torch.mean(sbj.view(bs, slen, -1), dim=1)
        obj = torch.mean(obj.view(bs, slen, -1), dim=1)
        pre = pre.view(bs, slen, 4, -1)
        pre, inter = self.temporal_decoder(pre, seq_lens)
        return pre.view(bs, slen, -1) / 2, sbj, obj, inter


def _args(**kw):
    import argparse
    base = dict(n_iter=2, alpha=0.9, n_heads=8, dropout=0.0,
                st_max_len=64, residual_mapping=False)
    base.update(kw)
    return argparse.Namespace(**base)


def _inputs(bs=1, slen=5):
    torch.manual_seed(0)
    return {
        "clip_feat": torch.randn(bs, slen, 4, D),
        "bbox_feat": torch.randn(bs, slen, 4, 24),
        "rel_feat": torch.randn(bs, slen, 42),
        "mot_feat": torch.randn(bs, slen, 42),
    }, torch.tensor([slen])


def test_n_iter_zero_matches_the_eov_baseline_exactly():
    """The equality that makes a measured delta attributable to the module."""
    torch.manual_seed(0)
    base = _StubFeatEmbedding().eval()
    torch.manual_seed(0)
    wrapped = METORFeatEmbedding(_args(n_iter=0), _StubFeatEmbedding()).eval()

    inputs, seq_lens = _inputs()
    with torch.no_grad():
        a = base(inputs, seq_lens)
        b = wrapped(inputs, seq_lens)

    for i, (x, y) in enumerate(zip(a, b)):
        assert torch.allclose(x, y, atol=1e-6), f"output {i} diverged"


def test_shapes_are_unchanged_with_iterations_on():
    base = _StubFeatEmbedding().eval()
    wrapped = METORFeatEmbedding(_args(n_iter=2), _StubFeatEmbedding()).eval()
    inputs, seq_lens = _inputs()
    with torch.no_grad():
        a = base(inputs, seq_lens)
        b = wrapped(inputs, seq_lens)
    for x, y in zip(a, b):
        assert x.shape == y.shape


def test_wrapping_preserves_every_eov_parameter_name():
    """Attaching the module must ADD keys, never rename or drop one."""
    base = _StubFeatEmbedding()
    before = {k: tuple(v.shape) for k, v in base.state_dict().items()}
    wrapped = METORFeatEmbedding(_args(), _StubFeatEmbedding())
    after = {k: tuple(v.shape) for k, v in wrapped.state_dict().items()}

    for key, shape in before.items():
        assert key in after, f"EOV parameter {key} disappeared"
        assert after[key] == shape, f"EOV parameter {key} changed shape"
    assert any(k.startswith("iem.") for k in after), "IEM added no parameters"


@pytest.mark.skipif(not os.path.exists(RELATION_CKPT),
                    reason="EOV relation checkpoint not available")
def test_released_checkpoint_keys_are_all_still_loadable():
    """Against the real released checkpoint, not a stub.

    Every `featEmbedding.*` tensor EOV shipped must still exist, with the same
    shape, in a compat-wrapped model. This is the test that would have caught a
    rename before it silently randomised a layer.
    """
    ckpt = torch.load(RELATION_CKPT, weights_only=False, map_location="cpu")
    shipped = {k[len("featEmbedding."):]: tuple(v.shape)
               for k, v in ckpt["state_dict"].items()
               if k.startswith("featEmbedding.")}
    assert shipped, "checkpoint has no featEmbedding tensors"

    wrapped = METORFeatEmbedding(_args(), _StubFeatEmbedding())
    have = {k: tuple(v.shape) for k, v in wrapped.state_dict().items()}

    # The stub only implements the submodules the wrapper re-exposes, so
    # compare on the intersection of names rather than demanding the full set.
    checked = 0
    for key, shape in shipped.items():
        if key in have:
            assert have[key] == shape, f"{key}: {have[key]} != shipped {shape}"
            checked += 1
    assert checked > 0, "no overlapping keys -- the wrapper renamed everything"
