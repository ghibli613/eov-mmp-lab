"""Shape and behaviour tests for the modules METOR contributes.

These do not need CUDA, CLIP weights or the dataset, so they run anywhere and
are the first thing to run after changing anything in `metor/modules/`.

What is actually being pinned here is the algebra of Sec. 3.4 and Sec. 3.5 --
the parts of the paper that are stated precisely enough to be tested. Nothing
here validates that the model is *correct*, only that it is what the equations
say. See docs/33_Verification-status.md for the difference.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from metor.modules.criterion import METORCriterion
from metor.modules.heads import CosineScoreHead, TrajectoryRegressionHead
from metor.modules.iterative_enhancement import (
    IterativeEnhancementLayer,
    IterativeEnhancementModule,
    SpatioTemporalTransformer,
)

P, T, D = 5, 9, 64


def _inputs(seed=0):
    torch.manual_seed(seed)
    return (torch.randn(P, T, D), torch.randn(P, T, D), torch.randn(T, D))


# --------------------------------------------------------------- Eq. 11 shapes
def test_st_transformer_maps_role_triple_to_relationship():
    st = SpatioTemporalTransformer(D).eval()
    tokens = torch.randn(P, T, 3, D)
    assert st(tokens).shape == (P, T, D)


def test_iem_preserves_object_shapes():
    s, o, cls = _inputs()
    iem = IterativeEnhancementModule(D, n_iter=2).eval()
    s_hat, o_hat, rel = iem(s, o, cls)
    assert s_hat.shape == s.shape
    assert o_hat.shape == o.shape
    assert rel.shape == (P, T, D)


def test_iem_accepts_per_pair_cls_features():
    s, o, _ = _inputs()
    cls = torch.randn(P, T, D)
    iem = IterativeEnhancementModule(D, n_iter=1).eval()
    assert iem(s, o, cls)[0].shape == s.shape


# ------------------------------------------------------- Eq. 11, the algebra
def test_paper_literal_update_scales_subject_object_difference_by_alpha():
    """Eq. 11 gives both entities the same additive term, so their difference
    can only be scaled by alpha, never rotated. With a zero-initialised M_f the
    scaling is exactly alpha**n_iter, which makes this checkable in closed form.

    This is the property discussed in the module docstring, and it is the reason
    to be careful about attributing gains to "mutual enhancement": the module
    provably cannot change how a subject differs from its object.
    """
    s, o, cls = _inputs()
    for n_iter in (1, 2, 3):
        iem = IterativeEnhancementModule(D, n_iter=n_iter).eval()
        s_hat, o_hat, _ = iem(s, o, cls)
        expected = (s - o) * (0.9 ** n_iter)
        assert torch.allclose(s_hat - o_hat, expected, atol=1e-5), n_iter


def test_residual_variant_preserves_the_difference_exactly():
    """The documented DEVIATION: with residual_mapping the difference survives
    unscaled. Pinned so the two readings cannot silently converge."""
    s, o, cls = _inputs()
    iem = IterativeEnhancementModule(D, n_iter=2, residual_mapping=True).eval()
    s_hat, o_hat, _ = iem(s, o, cls)
    assert torch.allclose(s_hat - o_hat, s - o, atol=1e-5)


def test_zero_initialised_mapping_starts_as_pure_alpha_rescale():
    s, o, cls = _inputs()
    layer = IterativeEnhancementLayer(D, alpha=0.9).eval()
    s_hat, o_hat, _ = layer(s, o, cls)
    assert torch.allclose(s_hat, 0.9 * s, atol=1e-6)
    assert torch.allclose(o_hat, 0.9 * o, atol=1e-6)


def test_n_iter_zero_is_the_ablation_baseline():
    """Table 4's "Iteration Number 0" row: relationship features are produced,
    but nothing feeds back into the objects."""
    s, o, cls = _inputs()
    iem = IterativeEnhancementModule(D, n_iter=0).eval()
    s_hat, o_hat, rel = iem(s, o, cls)
    assert s_hat is s and o_hat is o
    assert rel.shape == (P, T, D)


def test_negative_n_iter_is_rejected():
    with pytest.raises(ValueError):
        IterativeEnhancementModule(D, n_iter=-1)


def test_sequence_longer_than_max_len_raises_clearly():
    st = SpatioTemporalTransformer(D, max_len=4).eval()
    with pytest.raises(ValueError, match="st_max_len"):
        st(torch.randn(2, 8, 3, D))


# --------------------------------------------------------------------- Eq. 5
def test_trajectory_head_emits_normalised_boxes():
    head = TrajectoryRegressionHead(D)
    out = head(torch.randn(P, T, D))
    assert out.shape == (P, T, 4)
    assert bool((out >= 0).all() and (out <= 1).all())


# --------------------------------------------------------------------- Eq. 4
def test_cosine_score_head_modes():
    head = CosineScoreHead(temperature=0.01)
    v, t = torch.randn(P, D), torch.randn(7, D)
    assert bool((head(v, t, "sigmoid") >= 0).all())
    assert torch.allclose(head(v, t, "softmax").sum(-1), torch.ones(P), atol=1e-5)
    with pytest.raises(ValueError):
        head(v, t, "nonsense")


# ------------------------------------------------------------- Eq. 12 - 16
def _criterion_batch(n_rel=8, n_obj=6):
    torch.manual_seed(1)
    outputs = {
        "rel_scores": torch.rand(P, T, n_rel),
        "sbj_logits": torch.randn(P, n_obj),
        "obj_logits": torch.randn(P, n_obj),
        "pred_sbj_boxes": torch.rand(P, T, 4),
        "pred_obj_boxes": torch.rand(P, T, 4),
        "ctx_obj": torch.randn(4, 16),
        "ctx_rel": torch.randn(4, 16),
        "text_obj": torch.randn(n_obj, 16),
        "text_rel": torch.randn(n_rel, 16),
    }
    targets = {
        "rel_labels": (torch.rand(P, T, n_rel) > 0.7).float(),
        "sbj_labels": torch.randint(0, n_obj, (P,)),
        "obj_labels": torch.randint(0, n_obj, (P,)),
        "gt_sbj_boxes": torch.rand(P, T, 4),
        "gt_obj_boxes": torch.rand(P, T, 4),
        "sbj_box_mask": torch.ones(P, T, dtype=torch.bool),
        "obj_box_mask": torch.ones(P, T, dtype=torch.bool),
        "present_obj": (torch.rand(n_obj) > 0.5).float(),
        "present_rel": (torch.rand(n_rel) > 0.5).float(),
    }
    return outputs, targets


def test_criterion_reports_every_term_of_eq_12():
    crit = METORCriterion(n_base_rel=8)
    total, parts = crit(*_criterion_batch())
    for key in ("rel_ctr", "obj_ctr", "traj", "rel_ctx", "obj_ctx", "total"):
        assert key in parts, key
    assert torch.isfinite(total)


def test_criterion_weights_match_the_paper():
    """theta_traj=1.0, theta_ctx=0.2 (Sec. 4.2). Recompute Eq. 12 by hand and
    check the total is that sum -- a silently mis-weighted term would still
    train, just not to the paper's objective."""
    crit = METORCriterion(n_base_rel=8, theta_traj=1.0, theta_ctx=0.2)
    total, p = crit(*_criterion_batch())
    expected = (p["rel_ctr"] + p["obj_ctr"] + 1.0 * p["traj"]
                + 0.2 * (p["rel_ctx"] + p["obj_ctx"]))
    assert torch.allclose(total, expected, atol=1e-6)


def test_trajectory_consistency_is_zero_for_a_static_trajectory():
    """L_cst penalises frame-to-frame movement of the *prediction*, so a box
    that never moves has none. Pins the sign convention of Eq. 15."""
    crit = METORCriterion(n_base_rel=4)
    static = torch.rand(P, 1, 4).expand(P, T, 4).contiguous()
    _, _, l_cst = crit.trajectory(static, torch.rand(P, T, 4))
    assert float(l_cst) == pytest.approx(0.0, abs=1e-7)


def test_ignored_object_labels_do_not_contribute():
    """A pair whose subject and object are both unmatched must produce exactly
    zero object loss, not NaN -- unmatched queries are common."""
    crit = METORCriterion(n_base_rel=4)
    logits = torch.randn(3, 5)
    labels = torch.full((3,), -1, dtype=torch.long)
    assert float(crit.object_contrastive(logits, logits, labels, labels)) == 0.0


def test_missing_pieces_are_skipped_not_faked():
    """Ablations legitimately drop terms; the criterion must degrade to the
    terms it was given rather than inventing zeros for the rest."""
    crit = METORCriterion(n_base_rel=8)
    outputs, targets = _criterion_batch()
    total, parts = crit({"rel_scores": outputs["rel_scores"]},
                        {"rel_labels": targets["rel_labels"]})
    assert set(parts) == {"rel_ctr", "total"}
    assert torch.allclose(total, parts["rel_ctr"])


def test_gradients_reach_the_enhancement_module():
    """The end-to-end claim only holds if the loss actually reaches Psi."""
    s, o, cls = _inputs()
    iem = IterativeEnhancementModule(D, n_iter=2)
    s_hat, o_hat, rel = iem(s, o, cls)
    rel.sum().backward()
    grads = [p.grad for p in iem.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)


# ------------------------------------------- Eq. 13, normalisation (lineage)
def test_relationship_loss_is_a_mean_over_categories_not_a_double_division():
    """`1/|C_b_r|` in Eq. 13 is the average over predicate categories.

    EOV-MMP writes the same loss with the summation explicit (its Eq. 25):
    `(1/|C_R_b|) * SUM_{r in C_R_b} BCE(.)`. So a plain `reduction="mean"`
    already is the normalisation, and dividing by |C_b_r| a second time would
    shrink this term by ~100x on VidVRD -- effectively deleting the predicate
    signal from Eq. 12. This pins the correct scale.
    """
    n_rel = 8
    crit = METORCriterion(n_base_rel=n_rel)
    scores = torch.full((3, 4, n_rel), 0.5)
    labels = torch.zeros(3, 4, n_rel)
    expected = -torch.log(torch.tensor(0.5))       # BCE(0.5, 0) = -log(0.5)
    got = crit.relationship_contrastive(scores, labels)
    assert torch.allclose(got, expected, atol=1e-5), (float(got), float(expected))


def test_relationship_loss_rejects_the_wrong_vocabulary_size():
    """Passing full-vocabulary scores where base-only was expected would
    normalise against the wrong |C_b_r| and silently mis-weight training."""
    crit = METORCriterion(n_base_rel=8)
    with pytest.raises(ValueError, match="n_base_rel"):
        crit.relationship_contrastive(torch.rand(2, 3, 132), torch.zeros(2, 3, 132))
