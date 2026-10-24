"""Whole-framework tests for `--arch metor`, on a stub CLIP.

Covers Eq. 1-5 end to end: that every tensor has the shape Sec. 3 says it has,
that the ablation switches actually change what runs, and that gradients reach
the parameters the paper says are learned -- including the context tokens that
sit inside the frozen ViT, which is the one place a plausible-looking
implementation can silently learn nothing.

These use `tests/stub_clip.py` rather than ViT-L/14@336px. See that module for
why (short version: the real backbone needs several GB and this machine has 7).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from metor.model import METOR
from tests.stub_clip import stub_clip

OBJ = ["person", "dog", "zebra", "car", "frisbee"]
REL = ["run_past", "stand_left", "ride", "fly_with"]
RES = 32          # stub CLIP's input resolution
T = 4             # timesteps (key frames)


def build(**kw):
    clip = stub_clip(resolution=RES)
    defaults = dict(n_queries=12, n_iter=2, alpha=0.9, d_model=32,
                    n_heads=2, n_dec_layers=2, cre_mode="inject",
                    n_ctx_obj=3, n_ctx_rel=3, n_prompt_vec=4,
                    max_pair_objects=4, st_max_len=32, dropout=0.0)
    defaults.update(kw)
    return METOR(clip, OBJ, REL, **defaults)


def frames(t=T):
    torch.manual_seed(0)
    return torch.randn(t, 3, RES, RES)


# ------------------------------------------------------------------ Eq. 1 - 5
def test_forward_produces_every_output_with_the_documented_shapes():
    model = build().eval()
    with torch.no_grad():
        out = model(images=frames())

    n_pairs = 4 * 3                       # max_pair_objects=4 -> ordered pairs
    assert out["obj_feats"].shape == (12, T, 32)          # Eq. 2, N_q x T x d
    assert out["rel_feats"].shape == (n_pairs, T, 32)     # Eq. 3
    assert out["rel_scores"].shape == (n_pairs, T, len(REL))   # Eq. 4
    assert out["sbj_logits"].shape == (n_pairs, len(OBJ))
    assert out["obj_logits"].shape == (n_pairs, len(OBJ))
    assert out["pred_sbj_boxes"].shape == (n_pairs, T, 4)      # Eq. 5
    assert out["pred_obj_boxes"].shape == (n_pairs, T, 4)
    assert out["text_obj"].shape[0] == len(OBJ)                # Eq. 9
    assert out["text_rel"].shape[0] == len(REL)                # Eq. 10
    assert out["ctx_obj"].shape[0] == 3                        # n_ctx_obj
    assert out["ctx_rel"].shape[0] == 3


def test_scores_are_in_range():
    model = build().eval()
    with torch.no_grad():
        out = model(images=frames())
    for key in ("rel_scores", "sbj_scores", "obj_scores",
                "pred_sbj_boxes", "pred_obj_boxes"):
        v = out[key]
        assert bool((v >= 0).all() and (v <= 1).all()), key


def test_text_features_are_l2_normalised():
    """Eq. 4 is a cosine, so the text bank must be unit-norm or the temperature
    means something different per class."""
    model = build().eval()
    with torch.no_grad():
        out = model(images=frames())
    for key in ("text_obj", "text_rel"):
        norms = out[key].norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), key


@pytest.mark.parametrize("t", [1, 2, 7])
def test_variable_video_length(t):
    model = build().eval()
    with torch.no_grad():
        out = model(images=frames(t))
    assert out["rel_scores"].shape[1] == t


def test_n_queries_is_honoured():
    """N_q = 100 in the paper; EOV uses 300. Mixing them up is exactly why the
    released detector cannot be loaded into --arch metor."""
    model = build(n_queries=7).eval()
    with torch.no_grad():
        out = model(images=frames())
    assert out["obj_feats"].shape[0] == 7


# ------------------------------------------------------------------ ablations
def test_no_crq_skips_query_refinement():
    """Their Table 3 'w/o CRQ'. The refiner must be absent, not merely bypassed
    -- a present-but-unused module would still take optimiser state."""
    model = build(refine_queries=False).eval()
    assert not hasattr(model.cre, "query_refiner")
    with torch.no_grad():
        assert model(images=frames())["rel_scores"] is not None


def test_no_crt_requires_fixed_text_banks():
    """'w/o CRT' falls back to unconditioned CLIP prompts. Those have to be
    supplied; failing loudly beats silently scoring against random vectors."""
    model = build(refine_text=False).eval()
    assert not hasattr(model.cre, "obj_text")
    with pytest.raises(RuntimeError, match="fixed text banks"):
        model(images=frames())


def test_query_refiner_starts_as_the_identity():
    """Zero-initialised output projection: Eq. 8 must reduce to Q_hat = Q at
    step 0, or the COCO-pretrained queries are corrupted before training."""
    model = build().eval()
    q = model.query_embed.weight
    ctx = torch.randn(T, 3, model.cre.ctx_dim)
    with torch.no_grad():
        refined = model.cre.query_refiner(q, ctx)
    assert torch.allclose(refined, q.unsqueeze(0).expand(T, -1, -1), atol=1e-6)


def test_n_iter_zero_still_produces_relationship_features():
    model = build(n_iter=0).eval()
    with torch.no_grad():
        out = model(images=frames())
    assert out["rel_feats"].shape[1] == T


def test_pool_mode_runs_from_cached_features():
    """--cre_mode pool consumes pre-encoded patches instead of pixels."""
    model = build(cre_mode="pool").eval()
    width = model.patch_dim
    h_patch = torch.randn(T, 16, width)
    h_cls = torch.randn(T, width)
    with torch.no_grad():
        out = model(h_patch=h_patch, h_cls=h_cls)
    assert out["rel_scores"].shape[1] == T


def test_pool_mode_rejects_missing_inputs():
    model = build(cre_mode="pool").eval()
    with pytest.raises(ValueError, match="h_patch"):
        model(images=frames())


def test_inject_mode_rejects_missing_pixels():
    model = build(cre_mode="inject").eval()
    with pytest.raises(ValueError, match="images"):
        model(h_patch=torch.randn(T, 16, 64), h_cls=torch.randn(T, 64))


def test_invalid_cre_mode_is_rejected():
    with pytest.raises(ValueError, match="cre_mode"):
        build(cre_mode="magic")


# ------------------------------------------------------------------ gradients
def test_clip_backbone_stays_frozen():
    """"the ViT-L/14 variant of CLIP with fixed parameters" (Sec. 4.2)."""
    model = build()
    injector = model.cre.context
    assert all(not p.requires_grad for p in injector.visual.parameters())


def test_context_tokens_receive_gradient_through_the_frozen_vit():
    """The load-bearing test for Eq. 7.

    c_o and c_r are the only new inputs to a frozen encoder. If they are
    detached, appended after the transformer, or otherwise disconnected, the
    model still runs and still trains -- the contextual refinement simply never
    learns anything, and the paper's largest ablation gap becomes unreachable.
    Nothing else in the suite would notice.
    """
    model = build()
    out = model(images=frames())
    (out["rel_scores"].sum() + out["sbj_logits"].sum()).backward()

    injector = model.cre.context
    for name, p in (("ctx_obj", injector.ctx_obj), ("ctx_rel", injector.ctx_rel)):
        assert p.grad is not None, f"{name} got no gradient"
        assert float(p.grad.abs().sum()) > 0, f"{name} gradient is all zero"


def test_prompt_vectors_and_mapping_layers_train():
    """v_o / v_r and M_o / M_r of Eq. 9-10."""
    model = build()
    out = model(images=frames())
    out["rel_scores"].sum().backward()

    checked = 0
    for name, p in model.named_parameters():
        if "prompt_vectors" in name or ".mapping.net" in name:
            assert p.requires_grad, name
            if p.grad is not None and float(p.grad.abs().sum()) > 0:
                checked += 1
    assert checked > 0, "no prompt/mapping parameter received gradient"


def test_trainable_parameter_count_excludes_the_backbone():
    model = build()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert 0 < trainable < total


# ------------------------------------------------- [REL] position (lineage)
def test_three_quarter_token_position_produces_valid_text_features():
    """Both predecessors put the relationship token at 75% of the prompt.

    MMP: "The [CLS] token is positioned at 75% of the token length."
    EOV-MMP: "positioning the relationship token [REL] at 75% of the token
    length" (while [OBJ] stays at the end).

    METOR's Eq. 10 prints [REL] last, so 'end' is the default -- but the
    alternative has to actually work, because it is the reading the authors'
    own two prior systems support.
    """
    model = build(rel_token_position="three_quarter").eval()
    with torch.no_grad():
        out = model(images=frames())
    assert out["text_rel"].shape[0] == len(REL)
    norms = out["text_rel"].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_object_text_stays_at_end_even_in_three_quarter_mode():
    """[OBJ] is at the end in both predecessors; only [REL] moves."""
    model = build(rel_token_position="three_quarter").eval()
    assert model.cre.obj_text.token_position == "end"
    assert model.cre.rel_text.token_position == "three_quarter"


def test_the_two_token_positions_give_different_text_features():
    """Guards against the flag being silently inert."""
    torch.manual_seed(0)
    a = build(rel_token_position="end").eval()
    torch.manual_seed(0)
    b = build(rel_token_position="three_quarter").eval()
    with torch.no_grad():
        ta = a(images=frames())["text_rel"]
        tb = b(images=frames())["text_rel"]
    assert not torch.allclose(ta, tb, atol=1e-4)


def test_invalid_token_position_is_rejected():
    with pytest.raises(ValueError, match="token_position"):
        build(rel_token_position="middle")


def test_lineage_defaults_are_eight_and_eight():
    """MMP/EOV-MMP: "eight tokens each for learnable continuous prompts and
    learnable conditional prompts". v is the continuous half, M(C) the
    conditional half."""
    from metor.model import METOR
    from tests.stub_clip import stub_clip
    m = METOR(stub_clip(resolution=RES), OBJ, REL, d_model=32, n_heads=2,
              n_dec_layers=1, st_max_len=32)
    assert m.cre.context.n_ctx_obj == 8
    assert m.cre.context.n_ctx_rel == 8
    assert m.cre.obj_text.n_prompt_vec == 8
