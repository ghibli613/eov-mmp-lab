"""The relation model's four input streams must be individually switchable.

This is the seam the ablations in docs/91_Extension-guide.md run through -- in
particular the geometry-only model, which is the go/no-go experiment for the
temporal-degeneracy claim. Two properties matter:

  * with every stream enabled the computation is unchanged from upstream, so
    the released checkpoints stay reproducible;
  * disabling a stream is exactly equivalent to feeding it zeros, so the
    ablation means "this evidence was unavailable" and nothing else.
"""
import copy

import pytest
import torch


def make_args(**overrides):
    import sys
    argv = sys.argv
    sys.argv = ["pytest"]
    try:
        from utils.parser_func import parse_args
        args = parse_args()
    finally:
        sys.argv = argv
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def make_inputs(bs=2, slen=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "clip_feat": torch.randn(bs, slen, 4, 768, generator=g),
        "bbox_feat": torch.randn(bs, slen, 4, 24, generator=g),
        "rel_feat": torch.randn(bs, slen, 42, generator=g),
        "mot_feat": torch.randn(bs, slen, 42, generator=g),
    }


def build(args, cuda):
    from models.relation_classifier import FeatEmbedding
    torch.manual_seed(0)
    return FeatEmbedding(args).to(cuda).eval()


def run(model, inputs, cuda):
    inputs = {k: v.to(cuda) for k, v in inputs.items()}
    with torch.no_grad():
        pre, sbj, obj, inter = model(inputs, torch.tensor([inputs["clip_feat"].shape[1]]))
    return pre


def test_all_streams_enabled_is_the_default(cuda):
    model = build(make_args(), cuda)
    assert model.enabled_streams == {
        "rel_feat": True, "mot_feat": True, "clip_feat": True, "bbox_feat": True}


@pytest.mark.parametrize("stream", ["clip_feat", "bbox_feat", "rel_feat", "mot_feat"])
def test_disabling_a_stream_equals_zeroing_it(stream, cuda):
    inputs = make_inputs()

    gated = build(make_args(**{stream: False}), cuda)
    out_gated = run(gated, inputs, cuda)

    zeroed_inputs = copy.deepcopy(inputs)
    zeroed_inputs[stream] = torch.zeros_like(zeroed_inputs[stream])
    full = build(make_args(), cuda)
    out_zeroed = run(full, zeroed_inputs, cuda)

    torch.testing.assert_close(out_gated, out_zeroed)


@pytest.mark.parametrize("stream", ["clip_feat", "bbox_feat", "rel_feat", "mot_feat"])
def test_disabling_a_stream_actually_changes_the_output(stream, cuda):
    """Guards against a gate that silently does nothing."""
    inputs = make_inputs()
    full = run(build(make_args(), cuda), inputs, cuda)
    ablated = run(build(make_args(**{stream: False}), cuda), inputs, cuda)
    assert not torch.allclose(full, ablated), f"{stream} gate had no effect"


def test_geometry_only_configuration_runs(cuda):
    """The configuration behind the go/no-go experiment: no appearance at all."""
    model = build(make_args(clip_feat=False), cuda)
    out = run(model, make_inputs(), cuda)
    assert out.shape[-1] == 4 * 768
    assert torch.isfinite(out).all()
