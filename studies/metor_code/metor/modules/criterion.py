"""The training objective -- METOR Sec. 3.5, Eq. 12-16.

    L = L_rel-ctr + L_obj-ctr + th_traj*L_traj + th_ctx*(L_rel-ctx + L_obj-ctx)

with th_traj = 1.0, th_ctx = 0.2, th_cst = 0.1 (Sec. 4.2).

Five terms, and it is worth being clear about what each one is for, because
three of them are doing jobs that EOV-MMP did not have:

  L_rel-ctr   BCE on predicate scores. Multi-label -- a dog can be running
              behind *and* larger than a person -- hence BCE, not CE.
  L_obj-ctr   CE on subject and object category scores. Single-label.
  L_traj      Smooth-L1 box regression + a temporal-consistency term. METOR
              regresses trajectories directly (Eq. 5), so it needs this; the
              cascaded predecessors got trajectories from a separate tracker.
  L_*-ctx     The contextual losses. These are what force `C_o` and `C_r` to
              actually *be* about the objects and relations present, rather
              than drifting into arbitrary features that happen to help. They
              supervise the context tokens against the per-frame category sets.

Normalisation is written as the paper writes it: `L_rel-ctr` carries an explicit
`1/|C_b_r|` factor while `L_obj-ctr` does not.
Narrative walkthrough: docs/15_Training-objective.md
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class METORCriterion(nn.Module):
    """Eq. 12. Call with whatever the forward pass produced; missing pieces are
    skipped rather than faked, and `loss_dict` reports every term separately so
    an ablation can see which one moved.
    """

    def __init__(self, n_base_rel: int, theta_traj: float = 1.0,
                 theta_ctx: float = 0.2, theta_cst: float = 0.1,
                 ignore_index: int = -1):
        super().__init__()
        if n_base_rel <= 0:
            raise ValueError(f"n_base_rel must be positive, got {n_base_rel}")
        self.n_base_rel = n_base_rel
        self.theta_traj = theta_traj
        self.theta_ctx = theta_ctx
        self.theta_cst = theta_cst
        self.ignore_index = ignore_index

    # -------------------------------------------------------------- Eq. 13
    def relationship_contrastive(self, rel_scores, rel_labels):
        """`L_rel-ctr = (1/|C_b_r|) BCE(S_r, r~)`.

        The `1/|C_b_r|` is an average **over predicate categories**, not an
        extra scale factor on top of one. EOV-MMP writes the same loss out in
        full (its Eq. 25):

            L_rel = (1/|C_R_b|) * SUM_{r in C_R_b} BCE(y_hat_r, y_r)

        -- a sum over categories divided by their count, i.e. a mean. METOR's
        Eq. 13 is the same expression with the sum folded into `BCE(.)`.

        So `reduction="mean"` already *is* the `1/|C_b_r|`: it averages over the
        category axis (and over pairs and timesteps). Dividing by
        `n_base_rel` again on top of it would shrink this term by another
        factor of ~100 on VidVRD and effectively remove the predicate signal
        from Eq. 12. An earlier version of this file did exactly that; it was
        caught by reading EOV-MMP's Eq. 25, which spells out the summation
        METOR leaves implicit.

        Scores arrive post-sigmoid from Eq. 4, so this is `binary_cross_entropy`
        rather than the `_with_logits` form. Clamped because a saturated
        sigmoid gives log(0).
        """
        if rel_scores.shape[-1] != self.n_base_rel:
            raise ValueError(
                f"rel_scores has {rel_scores.shape[-1]} categories but "
                f"n_base_rel={self.n_base_rel}. Training scores only base "
                f"categories -- slice the novel columns off before the loss, "
                f"or the 1/|C_b_r| normalisation of Eq. 13 is against the "
                f"wrong vocabulary.")
        rel_scores = rel_scores.clamp(min=1e-7, max=1 - 1e-7)
        return F.binary_cross_entropy(
            rel_scores, rel_labels.to(rel_scores.dtype), reduction="mean")

    # -------------------------------------------------------------- Eq. 14
    def object_contrastive(self, sbj_logits, obj_logits, sbj_labels, obj_labels):
        """`L_obj-ctr = CE(S_s, s~) + CE(S_o, o~)`.

        Takes *logits*, not the softmax scores of Eq. 4 -- `cross_entropy`
        applies its own log-softmax, and feeding it softmaxed input would apply
        it twice. Pairs whose subject or object has no ground-truth label
        (`ignore_index`) drop out; a video where neither is labelled
        contributes zero rather than NaN.
        """
        total = sbj_logits.new_zeros(())
        for logits, labels in ((sbj_logits, sbj_labels), (obj_logits, obj_labels)):
            valid = labels != self.ignore_index
            if valid.any():
                total = total + F.cross_entropy(logits[valid], labels[valid])
        return total

    # -------------------------------------------------------------- Eq. 15
    def trajectory(self, pred_boxes, gt_boxes, valid_mask=None):
        """`L_traj = L_box + th_cst * L_cst`.

        `pred_boxes`/`gt_boxes`: (..., T, 4) normalised cxcywh.
        `L_box` is Smooth-L1 per frame per entity; `L_cst` penalises
        frame-to-frame box movement in the *prediction*, which is what makes a
        regressed trajectory smooth instead of jittering between frames.

        Note `L_cst` is a prior, not a fit: it is computed on predictions only
        and would be minimised by a motionless box. `th_cst = 0.1` is what
        keeps it from winning.
        """
        if valid_mask is None:
            valid_mask = torch.ones(pred_boxes.shape[:-1], dtype=torch.bool,
                                    device=pred_boxes.device)
        if not valid_mask.any():
            zero = pred_boxes.sum() * 0.0
            return zero, zero, zero

        l_box = F.smooth_l1_loss(pred_boxes[valid_mask], gt_boxes[valid_mask],
                                 reduction="mean")

        if pred_boxes.shape[-2] > 1:
            delta = (pred_boxes[..., 1:, :] - pred_boxes[..., :-1, :]).abs()
            pair_valid = valid_mask[..., 1:] & valid_mask[..., :-1]
            l_cst = (delta.sum(-1)[pair_valid].mean() if pair_valid.any()
                     else delta.sum() * 0.0)
        else:
            l_cst = pred_boxes.sum() * 0.0

        return l_box + self.theta_cst * l_cst, l_box, l_cst

    # -------------------------------------------------------------- Eq. 16
    def contextual(self, ctx_embeddings, text_embeddings, present_labels):
        """`L_ctx = BCE(cos(C, T), Y~)` for one of the two streams.

        `present_labels` is the multi-hot set of categories actually present in
        the frame, so this asks the context tokens to be linearly readable as
        "which objects / which relations are here". Cosine is mapped from
        [-1, 1] into [0, 1] before the BCE, since a raw cosine is not a
        probability. PAPER-GAP: Eq. 16 writes `BCE(cos(...), Y~)` with no
        stated squashing, and BCE on a negative input is undefined.
        """
        ctx = F.normalize(ctx_embeddings, dim=-1)
        txt = F.normalize(text_embeddings, dim=-1)
        sim = ctx @ txt.t()                      # (n_ctx_tokens, n_classes)
        # Any one context token may carry the evidence, so pool over tokens
        # before comparing to the frame's label set.
        sim = sim.max(dim=0).values if sim.dim() == 2 else sim
        prob = ((sim + 1.0) / 2.0).clamp(min=1e-7, max=1 - 1e-7)
        return F.binary_cross_entropy(prob, present_labels.to(prob.dtype),
                                      reduction="mean")

    # -------------------------------------------------------------- Eq. 12
    def forward(self, outputs: dict, targets: dict):
        """Returns `(total_loss, loss_dict)`.

        `outputs` may contain: rel_scores, sbj_logits, obj_logits, pred_sbj_boxes,
        pred_obj_boxes, ctx_obj, ctx_rel, text_obj, text_rel.
        `targets`  may contain: rel_labels, sbj_labels, obj_labels, gt_sbj_boxes,
        gt_obj_boxes, sbj_box_mask, obj_box_mask, present_obj, present_rel.
        """
        losses = {}
        device = next((v.device for v in outputs.values()
                       if isinstance(v, torch.Tensor)), torch.device("cpu"))
        total = torch.zeros((), device=device)

        if "rel_scores" in outputs and "rel_labels" in targets:
            l = self.relationship_contrastive(outputs["rel_scores"], targets["rel_labels"])
            losses["rel_ctr"] = l
            total = total + l

        if "sbj_logits" in outputs and "sbj_labels" in targets:
            l = self.object_contrastive(
                outputs["sbj_logits"], outputs["obj_logits"],
                targets["sbj_labels"], targets["obj_labels"])
            losses["obj_ctr"] = l
            total = total + l

        if "pred_sbj_boxes" in outputs and "gt_sbj_boxes" in targets:
            l_s, box_s, cst_s = self.trajectory(
                outputs["pred_sbj_boxes"], targets["gt_sbj_boxes"],
                targets.get("sbj_box_mask"))
            l_o, box_o, cst_o = self.trajectory(
                outputs["pred_obj_boxes"], targets["gt_obj_boxes"],
                targets.get("obj_box_mask"))
            # Eq. 15 sums over entities e in {s, o} inside L_box.
            l_traj = l_s + l_o
            losses["traj"] = l_traj
            losses["traj_box"] = box_s + box_o
            losses["traj_cst"] = cst_s + cst_o
            total = total + self.theta_traj * l_traj

        ctx_total = torch.zeros((), device=device)
        has_ctx = False
        if "ctx_rel" in outputs and "present_rel" in targets:
            l = self.contextual(outputs["ctx_rel"], outputs["text_rel"], targets["present_rel"])
            losses["rel_ctx"] = l
            ctx_total = ctx_total + l
            has_ctx = True
        if "ctx_obj" in outputs and "present_obj" in targets:
            l = self.contextual(outputs["ctx_obj"], outputs["text_obj"], targets["present_obj"])
            losses["obj_ctx"] = l
            ctx_total = ctx_total + l
            has_ctx = True
        if has_ctx:
            total = total + self.theta_ctx * ctx_total

        losses["total"] = total
        return total, losses
