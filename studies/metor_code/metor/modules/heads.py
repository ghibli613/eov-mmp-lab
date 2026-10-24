"""Prediction heads -- METOR Eq. 4 and Eq. 5.

Narrative walkthrough: docs/14_Scoring-and-trajectory-heads.md
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryRegressionHead(nn.Module):
    """`M_b` of Eq. 5 -- "a multi-layer perceptron".

        tau_s = M_b(O_s_hat)        tau_o = M_b(O_o_hat)

    Takes the enhanced per-timestep object features and emits a box per
    timestep, so a "trajectory" is just the stack of them. Boxes come out in
    normalised `cxcywh` through a sigmoid, matching Deformable-DETR's
    convention and therefore the detector this framework initialises from.

    The head is shared between subject and object: Eq. 5 writes `M_b` once for
    both, and a subject and an object are the same kind of thing.
    """

    def __init__(self, dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        dims = [dim] + [hidden_dim] * (n_layers - 1) + [4]
        self.layers = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        # DETR's init: start every box at the centre of the frame rather than
        # at a random corner, so early training is not dominated by nonsense.
        nn.init.constant_(self.layers[-1].weight, 0.0)
        nn.init.constant_(self.layers[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., dim) -> (..., 4) normalised cxcywh in [0, 1]."""
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x.sigmoid()


class CosineScoreHead(nn.Module):
    """Eq. 4 -- `S = sigma(cos(visual, text))`, with the paper's temperature.

    Objects and predicates are scored the same way but read differently
    downstream: a box is one category (softmax over classes) while a pair can
    hold several relations at once (independent sigmoids). The paper's Eq. 4
    writes `sigma` for all three; `mode` selects which, defaulting to the
    lineage's actual behaviour.

    tau = 0.01 is inherited from MMP/EOV-MMP. PAPER-GAP: METOR states the
    cosine and the sigmoid but never names a temperature; without one the
    logits sit in [-1, 1] and the sigmoid is nearly linear, which cannot
    produce the reported numbers.
    """

    def __init__(self, temperature: float = 0.01):
        super().__init__()
        self.temperature = temperature

    def forward(self, visual: torch.Tensor, text: torch.Tensor,
                mode: str = "sigmoid") -> torch.Tensor:
        visual = F.normalize(visual, dim=-1)
        text = F.normalize(text, dim=-1)
        logits = visual @ text.t() / self.temperature
        if mode == "sigmoid":
            return torch.sigmoid(logits)
        if mode == "softmax":
            return torch.softmax(logits, dim=-1)
        if mode == "logits":
            return logits
        raise ValueError(f"unknown mode {mode!r}")
