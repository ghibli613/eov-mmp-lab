"""Matching predicted object queries to ground-truth trajectories.

Narrative walkthrough: docs/16_Matching-and-pair-formation.md

METOR is query-based, so nothing tells you which of the `N_q` queries is
supposed to be the dog and which the frisbee. DETR solves this with a Hungarian
assignment between predictions and ground truth, and the paper inherits that
machinery implicitly -- it initialises its decoder from a COCO-pretrained
detector, which is trained that way.

PAPER-GAP: METOR never states its matching cost. This uses the DETR recipe
adapted from boxes to *trajectories*: the L1 cost is averaged over the frames
where a ground-truth box exists, so a query that tracks an object well for its
whole duration beats one that happens to overlap on a single frame. That
temporal averaging is the only real departure from DETR, and it is forced --
DETR matches one box, METOR matches a sequence.

Unmatched queries are simply not supervised. There is no "no-object" class here
because Eq. 4 scores objects against CLIP text embeddings, which have no such
row; the corresponding DETR term has nowhere to go.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

try:
    from scipy.optimize import linear_sum_assignment
    _HAVE_SCIPY = True
except ImportError:                                     # pragma: no cover
    _HAVE_SCIPY = False


def _greedy_assignment(cost: torch.Tensor):
    """Fallback when SciPy is absent: repeatedly take the cheapest free cell.

    Not optimal, but deterministic and never worse than random. SciPy is in
    requirements.txt; this exists so an import error in a fresh environment
    degrades instead of crashing a training run at hour three.
    """
    cost = cost.clone()
    rows, cols = [], []
    n = min(cost.shape)
    for _ in range(n):
        idx = torch.argmin(cost)
        r = int(idx // cost.shape[1])
        c = int(idx % cost.shape[1])
        if not torch.isfinite(cost[r, c]):
            break
        rows.append(r)
        cols.append(c)
        cost[r, :] = float("inf")
        cost[:, c] = float("inf")
    return rows, cols


class TrajectoryHungarianMatcher:
    """Assign queries to ground-truth trajectories by trajectory cost.

    cost = w_box * mean_t |pred_box - gt_box|  -  w_cls * P(gt class)
    """

    def __init__(self, cost_box: float = 5.0, cost_class: float = 1.0):
        self.cost_box = cost_box
        self.cost_class = cost_class

    @torch.no_grad()
    def __call__(self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor,
                 gt_mask: torch.Tensor,
                 pred_class_probs: Optional[torch.Tensor] = None,
                 gt_labels: Optional[torch.Tensor] = None
                 ) -> Tuple[List[int], List[int]]:
        """
        pred_boxes        (Q, T, 4) normalised cxcywh
        gt_boxes          (G, T, 4)
        gt_mask           (G, T) bool -- where the ground truth actually exists
        pred_class_probs  (Q, C) optional
        gt_labels         (G,) optional

        Returns `(query_indices, gt_indices)`, parallel lists.
        """
        q, t, _ = pred_boxes.shape
        g = gt_boxes.shape[0]
        if q == 0 or g == 0:
            return [], []

        # (Q, G, T) L1 distance, averaged over frames the GT covers.
        diff = (pred_boxes.unsqueeze(1) - gt_boxes.unsqueeze(0)).abs().sum(-1)
        mask = gt_mask.unsqueeze(0).to(diff.dtype)               # (1, G, T)
        denom = mask.sum(-1).clamp(min=1.0)                      # (1, G)
        box_cost = (diff * mask).sum(-1) / denom                 # (Q, G)

        # A trajectory with no boxes on the key-frame grid cannot be matched.
        box_cost = box_cost.masked_fill(gt_mask.any(-1).unsqueeze(0) == 0,
                                        float("inf"))

        cost = self.cost_box * box_cost
        if pred_class_probs is not None and gt_labels is not None:
            valid = gt_labels >= 0
            cls_cost = torch.zeros_like(cost)
            if valid.any():
                cls_cost[:, valid] = -pred_class_probs[:, gt_labels[valid]]
            cost = cost + self.cost_class * cls_cost

        cost_cpu = cost.detach().cpu()
        if _HAVE_SCIPY:
            finite = torch.isfinite(cost_cpu)
            if not finite.any():
                return [], []
            # linear_sum_assignment rejects inf; substitute a value that is
            # worse than every real cost so those cells are chosen last.
            big = cost_cpu[finite].max().item() + 1e3
            rows, cols = linear_sum_assignment(
                torch.where(finite, cost_cpu, torch.full_like(cost_cpu, big)).numpy())
            keep = [(int(r), int(c)) for r, c in zip(rows, cols)
                    if torch.isfinite(cost_cpu[r, c])]
            if not keep:
                return [], []
            rows, cols = zip(*keep)
            return list(rows), list(cols)

        return _greedy_assignment(cost_cpu)


def build_pair_targets(dataset, item, query_rows: List[int], gt_rows: List[int],
                       pair_index: torch.Tensor, device) -> dict:
    """Turn a query<->GT assignment into the tensors `METORCriterion` wants.

    `pair_index` holds `(i, j)` indices into the *kept* query list; `query_rows`
    maps kept-query position -> matched GT row, or -1 if that query matched
    nothing. Pairs where either side is unmatched are still returned, with an
    all-zero relation label and a box mask of False: they are legitimate
    negatives for the predicate loss and are excluded from the trajectory loss.
    """
    q_to_gt = {q: g for q, g in zip(query_rows, gt_rows)}
    p = pair_index.shape[0]
    t = item["gt_boxes"].shape[1]
    n_rel = dataset.n_rel

    gt_boxes = item["gt_boxes"].to(device)
    gt_mask = item["gt_box_mask"].to(device)
    gt_labels = item["gt_labels"].to(device)

    sbj_boxes = torch.zeros(p, t, 4, device=device)
    obj_boxes = torch.zeros(p, t, 4, device=device)
    sbj_mask = torch.zeros(p, t, dtype=torch.bool, device=device)
    obj_mask = torch.zeros(p, t, dtype=torch.bool, device=device)
    sbj_labels = torch.full((p,), -1, dtype=torch.long, device=device)
    obj_labels = torch.full((p,), -1, dtype=torch.long, device=device)
    rel_labels = torch.zeros(p, t, n_rel, device=device)

    for pi in range(p):
        si = int(pair_index[pi, 0])
        oi = int(pair_index[pi, 1])
        s_gt = q_to_gt.get(si, -1)
        o_gt = q_to_gt.get(oi, -1)
        if s_gt >= 0:
            sbj_boxes[pi] = gt_boxes[s_gt]
            sbj_mask[pi] = gt_mask[s_gt]
            sbj_labels[pi] = gt_labels[s_gt]
        if o_gt >= 0:
            obj_boxes[pi] = gt_boxes[o_gt]
            obj_mask[pi] = gt_mask[o_gt]
            obj_labels[pi] = gt_labels[o_gt]
        if s_gt >= 0 and o_gt >= 0:
            lab = item["pair_labels"].get((s_gt, o_gt))
            if lab is not None:
                rel_labels[pi] = lab.to(device)

    # The predicate head is trained on base categories only.
    base = torch.tensor(dataset.base_rel_ids, device=device)
    return {
        "rel_labels": rel_labels.index_select(-1, base),
        "sbj_labels": sbj_labels,
        "obj_labels": obj_labels,
        "gt_sbj_boxes": sbj_boxes,
        "gt_obj_boxes": obj_boxes,
        "sbj_box_mask": sbj_mask,
        "obj_box_mask": obj_mask,
        "present_obj": item["present_obj"].to(device),
        "present_rel": item["present_rel"].to(device),
    }
