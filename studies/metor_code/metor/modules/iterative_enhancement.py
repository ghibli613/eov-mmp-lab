"""Iterative enhancement -- METOR Sec. 3.4, Eq. 11.

This is the paper's headline contribution: the thing that makes the framework
"mutual" rather than a cascade. One layer does two things in sequence.

    R_hat^(k)   = STTrans^(k)([O_s^(k-1); O_o^(k-1); H_CLS])
    O_s^(k)     = a * O_s^(k-1) + (1-a) * M_f^(k)(R_hat^(k))
    O_o^(k)     = a * O_o^(k-1) + (1-a) * M_f^(k)(R_hat^(k))

Objects produce a relationship representation; that relationship representation
is then mixed back into the objects. Stack `N_i` of these and the two keep
improving each other. `a = 0.9`, so each layer nudges the object features by
10% -- deliberately gentle, which is consistent with their Table 4 showing the
gain saturating at two layers (15.16 -> 16.04 -> 16.43 -> 16.25 novel mAP) and
turning over at three.

Note what the update rule implies, because it is easy to miss: `O_s` and `O_o`
receive the **same** additive term `M_f(R_hat)`. Subtracting the two lines gives

    O_s^(k) - O_o^(k) = a * (O_s^(k-1) - O_o^(k-1))

so after `N_i` layers the subject-object difference is the original one scaled
by `a^N_i` and never rotated. The module cannot make a subject look more or less
like its object; it can only translate the pair together, toward whatever the
relation says they both are. With `a = 0.9` and `N_i = 2` that difference is
also shrunk to 0.81 of its original size while the shared component grows --
which is to say the two object features are pushed toward each other. That is a
property of Eq. 11 exactly as printed, not an artefact of this reconstruction,
and it is worth knowing before attributing their +1.6 mAP to "mutual
enhancement". Nothing in the paper's ablations distinguishes this effect from a
smoothing prior over the pair.

Narrative walkthrough: docs/13_Iterative-enhancement.md

PAPER-GAPs in this file:
  * "spatio-temporal Transformer" is named but never specified -- see
    `SpatioTemporalTransformer`.
  * `M_f`'s initialisation is unstated -- see `IterativeEnhancementLayer`.
  * whether the blend carries a residual is unstated -- see `residual_mapping`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalTransformer(nn.Module):
    """`STTrans^(k)` -- factorised spatial-then-temporal attention.

    Input is the token triple `[subject; object; CLS]` at every timestep:
    `(P, T, 3, d)` for P subject-object pairs over T timesteps. It runs

      1. **spatial** attention across the 3 roles, independently per timestep --
         this is where the subject, the object and the global scene actually
         look at each other, and it is the only place the relation can be formed;
      2. **temporal** attention across the T timesteps, independently per role --
         this is where `walk_past` becomes distinguishable from `stand_left`.

    Factorised rather than joint attention over all 3*T tokens: it is what the
    same group used in EOV-MMP (`SpatialDecoder` then `TemporalDecoder`), it is
    the standard video-Transformer choice, and it is O(T*9 + 3*T^2) instead of
    O(9T^2). PAPER-GAP: the paper says only "spatio-temporal Transformer".

    A learned **role embedding** distinguishes the three input slots, exactly as
    EOV-MMP's `role_embeddings` does -- without it, one shared attention layer
    could not treat "subject" differently from "object".
    """

    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.1,
                 max_len: int = 512):
        super().__init__()
        self.role_embed = nn.Embedding(3, dim)      # subject / object / scene
        self.pos_embed = nn.Embedding(max_len, dim)
        self.max_len = max_len
        self.spatial = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True)
        self.temporal = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True)
        self.out = nn.Linear(dim * 3, dim)

    def forward(self, tokens: torch.Tensor, key_padding_mask=None):
        """tokens: (P, T, 3, d) -> relationship features (P, T, d)."""
        p, t, r, d = tokens.shape

        roles = self.role_embed(torch.arange(r, device=tokens.device))
        x = tokens + F.normalize(roles, dim=-1).view(1, 1, r, d)

        # --- spatial: attend across the 3 roles, per (pair, timestep)
        x = self.spatial(x.reshape(p * t, r, d)).reshape(p, t, r, d)

        # --- temporal: attend across T, per (pair, role)
        if t > self.max_len:
            raise ValueError(
                f"sequence of {t} exceeds max_len={self.max_len}; raise "
                f"--st_max_len (positional embeddings are learned, not sinusoidal)")
        pos = self.pos_embed(torch.arange(t, device=tokens.device))
        x = x + F.normalize(pos, dim=-1).view(1, t, 1, d)

        x = x.permute(0, 2, 1, 3).reshape(p * r, t, d)
        mask = None
        if key_padding_mask is not None:
            mask = key_padding_mask.repeat_interleave(r, dim=0)
        x = self.temporal(x, src_key_padding_mask=mask)
        x = x.reshape(p, r, t, d).permute(0, 2, 1, 3)       # (P, T, 3, d)

        # The relationship lives in the joint state of all three roles, so the
        # three are concatenated and projected rather than pooled -- pooling
        # would discard which role contributed what.
        return self.out(x.reshape(p, t, r * d))


class IterativeEnhancementLayer(nn.Module):
    """One `k` of Eq. 11: objects -> relationship -> objects.

    `M_f` is the paper's "relationship feature mapping layer". Its final linear
    is **zero-initialised** so an untrained layer contributes nothing beyond the
    `a` rescale, rather than injecting noise into COCO-pretrained object
    features on step 0. PAPER-GAP: initialisation is unstated.

    `residual_mapping` selects between two readings of the update:

      False (default, paper-literal)
          `O^(k) = a*O^(k-1) + (1-a)*M_f(R^(k))`, exactly Eq. 11. At
          initialisation this is `0.9 * O`, a pure rescale -- harmless, because
          every consumer of these features L2-normalises before the cosine of
          Eq. 4.
      True
          `O^(k) = a*O^(k-1) + (1-a)*(M_f(R^(k)) + O^(k-1))`, i.e. `M_f`
          predicts a *delta*. Starts as the exact identity and tends to train
          more stably, but it is not what the paper prints: it makes the
          subject-object difference exactly preserved rather than scaled by
          `a^k`. Offered because the distinction is invisible in the paper and
          may matter; it is a deviation and is labelled as one.
    """

    def __init__(self, dim: int, n_heads: int = 8, alpha: float = 0.9,
                 dropout: float = 0.1, max_len: int = 512,
                 residual_mapping: bool = False):
        super().__init__()
        self.alpha = alpha
        self.residual_mapping = residual_mapping
        self.st_transformer = SpatioTemporalTransformer(
            dim, n_heads, dropout, max_len)
        self.mapping = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)
        # Start as identity: with a zeroed final layer the mapping contributes
        # nothing on step 0, so an untrained IEM does not degrade the object
        # features it was handed. PAPER-GAP: initialisation unspecified.
        nn.init.zeros_(self.mapping[-1].weight)
        nn.init.zeros_(self.mapping[-1].bias)

    def forward(self, obj_s, obj_o, cls_feat, key_padding_mask=None):
        """obj_s, obj_o: (P, T, d). cls_feat: (T, d) or (P, T, d)."""
        p, t, d = obj_s.shape
        if cls_feat.dim() == 2:
            cls_feat = cls_feat.unsqueeze(0).expand(p, -1, -1)

        tokens = torch.stack([obj_s, obj_o, cls_feat], dim=2)    # (P, T, 3, d)
        rel = self.st_transformer(tokens, key_padding_mask)      # (P, T, d)
        rel = self.norm(rel)

        # Eq. 11's second and third lines. Both objects get the SAME mapped
        # relation feature -- see the module docstring for what that implies.
        mapped = self.mapping(rel)
        if self.residual_mapping:
            new_s = self.alpha * obj_s + (1.0 - self.alpha) * (mapped + obj_s)
            new_o = self.alpha * obj_o + (1.0 - self.alpha) * (mapped + obj_o)
        else:
            new_s = self.alpha * obj_s + (1.0 - self.alpha) * mapped
            new_o = self.alpha * obj_o + (1.0 - self.alpha) * mapped
        return new_s, new_o, rel


class IterativeEnhancementModule(nn.Module):
    """`Psi(.)` of Eq. 3 -- `N_i` stacked enhancement layers.

        (O_s_hat, O_o_hat, R_hat) = Psi(O, H_CLS)

    initialised with `O_s^(0) = O_s`, `O_o^(0) = O_o`, and returning the last
    layer's outputs. `n_iter=0` is a legal and meaningful setting: it is the
    paper's own ablation baseline (Table 4, "Iteration Number 0" = 15.16 novel
    mAP), where relationship features are produced once with no feedback into
    the objects. Comparing against it is the only clean measurement of what
    this module buys.
    """

    def __init__(self, dim: int, n_iter: int = 2, n_heads: int = 8,
                 alpha: float = 0.9, dropout: float = 0.1, max_len: int = 512,
                 residual_mapping: bool = False):
        super().__init__()
        if n_iter < 0:
            raise ValueError(f"n_iter must be >= 0, got {n_iter}")
        self.n_iter = n_iter
        self.alpha = alpha
        self.layers = nn.ModuleList([
            IterativeEnhancementLayer(dim, n_heads, alpha, dropout, max_len,
                                      residual_mapping)
            for _ in range(max(n_iter, 1))
        ])

    def forward(self, obj_s, obj_o, cls_feat, key_padding_mask=None):
        """obj_s, obj_o: (P, T, d); cls_feat: (T, d) -> enhanced s, o, rel."""
        rel = None
        if self.n_iter == 0:
            # Ablation baseline: form the relationship once, feed nothing back.
            _, _, rel = self.layers[0](obj_s, obj_o, cls_feat, key_padding_mask)
            return obj_s, obj_o, rel

        for layer in self.layers:
            obj_s, obj_o, rel = layer(obj_s, obj_o, cls_feat, key_padding_mask)
        return obj_s, obj_o, rel
