"""METOR -- the unified framework. Paper Sec. 3.2, Eq. 1-5.

The whole forward pass, in the paper's own notation:

    Eq.1  (H_CLS, H_patch, T_o, T_r, Q_hat, C_o, C_r) = Phi(V, Q)   contextual refinement
    Eq.2  O                                = Decoder(Q_hat, H_patch) transformer decoder
    Eq.3  (O_s_hat, O_o_hat, R_hat)        = Psi(O, H_CLS)           iterative enhancement
    Eq.4  S_s, S_o, S_r                    = sigma(cos(., T))        scoring
    Eq.5  tau_s, tau_o                     = M_b(O_.._hat)           trajectories

What makes this "unified" rather than a cascade is that there is exactly one
forward pass and one loss. EOV-MMP detects objects, freezes that answer, then
classifies relations from it; here Eq. 3 lets the relation representation flow
back into the object representation before either is scored, and Eq. 5 regresses
the trajectories from those same enhanced features rather than from a separate
tracker.

Two architectures are provided, and they are not interchangeable:

`--arch metor`
    The paper as written. 100 object queries, decoder initialised from
    MS-COCO Deformable-DETR, five losses, trained end-to-end for 30 epochs.
    This is the one whose numbers would be comparable to Table 1. It cannot
    load any released EOV-MMP checkpoint -- EOV uses 300 queries, so the
    shapes differ, not merely the weights.

`--arch metor-compat`
    METOR's two modules bolted onto the EOV-MMP stack that already runs, with
    EOV's 300 queries and its released checkpoints. Lets you measure what CRE
    and the IEM are worth *on a working baseline*, today, without training a
    detector from COCO. It is "EOV-MMP + METOR's modules", not METOR, and it
    will not reproduce 16.74 -- see metor/compat.py.

Read docs/30_Paper-gaps.md for every PAPER-GAP this file commits to.
The component/equation map is docs/10_Pipeline-overview.md.
"""
from __future__ import annotations

import json
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from metor.modules.contextual_encoding import ContextualRefinementEncoder
from metor.modules.criterion import METORCriterion
from metor.modules.heads import CosineScoreHead, TrajectoryRegressionHead
from metor.modules.iterative_enhancement import IterativeEnhancementModule


class VanillaQueryDecoder(nn.Module):
    """`Decoder(.)` of Eq. 2, as a plain Transformer decoder.

    The paper only says "a Transformer decoder", initialised from a COCO-
    pretrained detector. The COCO init is what matters for accuracy and it
    requires the Deformable-DETR variant (`build_deformable_decoder`); this
    plain version exists so the framework can be constructed, shape-checked and
    unit-tested without CUDA ops, and so `--arch metor` is runnable on CPU for
    development.

    Queries are decoded against the patch tokens of each frame independently,
    giving `O` of shape `(N_q, T, d)` -- N_q candidate objects tracked across T
    timesteps, which is what Eq. 3 consumes.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8, n_layers: int = 6,
                 patch_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.patch_proj = nn.Linear(patch_dim, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, queries: torch.Tensor, h_patch: torch.Tensor):
        """queries: (T, Nq, d) or (Nq, d); h_patch: (T, N, patch_dim) -> (Nq, T, d)."""
        t = h_patch.shape[0]
        if queries.dim() == 2:
            queries = queries.unsqueeze(0).expand(t, -1, -1)
        mem = self.patch_proj(h_patch.to(self.patch_proj.weight.dtype))
        out = self.decoder(queries, mem)            # (T, Nq, d)
        return self.norm(out).permute(1, 0, 2)      # (Nq, T, d)


class METOR(nn.Module):
    """The framework of Eq. 1-5.

    Parameters
    ----------
    clip_model      a loaded, frozen CLIP model (ViT-L/14@336px in the paper)
    obj_classnames  object category names, in dataset id order
    rel_classnames  predicate category names, in dataset id order
    n_queries       `N_q`, 100 in the paper (Sec. 4.2)
    n_iter          `N_i`, 2 for VidVRD and 3 for VidOR (Sec. 4.2)
    alpha           the blend of Eq. 11, 0.9 (Sec. 4.2)
    """

    def __init__(self, clip_model, obj_classnames: Sequence[str],
                 rel_classnames: Sequence[str], *,
                 n_queries: int = 100, n_iter: int = 2, alpha: float = 0.9,
                 d_model: int = 256, n_heads: int = 8, n_dec_layers: int = 6,
                 cre_mode: str = "inject", n_ctx_obj: int = 8, n_ctx_rel: int = 8,
                 n_prompt_vec: int = 8, rel_token_position: str = "end",
                 refine_queries: bool = True,
                 refine_text: bool = True, temperature: float = 0.01,
                 max_pair_objects: int = 10, decoder: Optional[nn.Module] = None,
                 dropout: float = 0.1, st_max_len: int = 512):
        super().__init__()
        self.n_queries = n_queries
        self.n_iter = n_iter
        self.max_pair_objects = max_pair_objects
        self.obj_classnames = list(obj_classnames)
        self.rel_classnames = list(rel_classnames)

        patch_dim = clip_model.visual.class_embedding.shape[0]
        text_dim = clip_model.text_projection.shape[1]
        self.patch_dim = patch_dim
        self.text_dim = text_dim
        self.d_model = d_model

        # --- Q, the learnable object queries of Eq. 1
        self.query_embed = nn.Embedding(n_queries, d_model)
        nn.init.normal_(self.query_embed.weight, std=0.02)

        # --- Phi, Eq. 1 / Sec. 3.3
        self.cre = ContextualRefinementEncoder(
            clip_model, obj_classnames, rel_classnames,
            query_dim=d_model, mode=cre_mode,
            n_ctx_obj=n_ctx_obj, n_ctx_rel=n_ctx_rel,
            n_prompt_vec=n_prompt_vec, n_heads=n_heads,
            rel_token_position=rel_token_position,
            refine_queries=refine_queries, refine_text=refine_text)

        # --- Decoder, Eq. 2
        self.decoder = decoder or VanillaQueryDecoder(
            d_model, n_heads, n_dec_layers, patch_dim, dropout)

        # --- Psi, Eq. 3 / Sec. 3.4
        self.iem = IterativeEnhancementModule(
            d_model, n_iter=n_iter, n_heads=n_heads, alpha=alpha,
            dropout=dropout, max_len=st_max_len)

        # --- Eq. 4: the visual side must land in CLIP's text space to be
        # compared against T_o / T_r by cosine.
        self.obj_to_text = nn.Linear(d_model, text_dim)
        self.rel_to_text = nn.Linear(d_model, text_dim)
        self.score_head = CosineScoreHead(temperature)

        # --- Eq. 5
        self.traj_head = TrajectoryRegressionHead(d_model)

        # H_CLS arrives in the ViT width and enters Eq. 11 in `d_model`.
        self.cls_proj = nn.Linear(patch_dim, d_model)

        # `objectness` ranks queries so pairs can be formed from the plausible
        # ones. PAPER-GAP: the paper says only "From the visual object features
        # O, we derive subject-object pairs" and never says how many or which.
        # All ordered pairs of 100 queries would be 9,900 per video.
        self.objectness = nn.Linear(d_model, 1)

    # ------------------------------------------------------------------ util
    @staticmethod
    def _pair_indices(n: int, device) -> torch.Tensor:
        """All ordered pairs (i, j), i != j -- a relation is not symmetric."""
        idx = torch.arange(n, device=device)
        s, o = torch.meshgrid(idx, idx, indexing="ij")
        keep = s != o
        return torch.stack([s[keep], o[keep]], dim=-1)

    def select_pairs(self, obj_feats: torch.Tensor, k: Optional[int] = None):
        """Rank queries by objectness, take the top k, pair them up.

        obj_feats: (Nq, T, d) -> (pairs, 2) index tensor and the kept ids.
        """
        k = k or self.max_pair_objects
        k = min(k, obj_feats.shape[0])
        scores = self.objectness(obj_feats.mean(dim=1)).squeeze(-1)   # (Nq,)
        keep = scores.topk(k).indices
        pairs = self._pair_indices(k, obj_feats.device)
        return pairs, keep

    # --------------------------------------------------------------- forward
    def forward(self, images=None, h_patch=None, h_cls=None,
                pair_index=None, keep_index=None):
        """One video through Eq. 1-5.

        images   (T, 3, H, W) pixel frames -- required for `cre_mode='inject'`
        h_patch  (T, N, patch_dim) cached patch tokens -- for `cre_mode='pool'`
        h_cls    (T, patch_dim) cached CLS tokens -- for `cre_mode='pool'`

        Returns a dict with the scores of Eq. 4, the trajectories of Eq. 5, and
        the intermediates the criterion needs.
        """
        # ---- Eq. 1
        queries = self.query_embed.weight
        fixed_o = fixed_r = None
        if not self.cre.refine_text:
            fixed_o, fixed_r = self.fixed_text_banks()
        enc_cls, enc_patch, t_o, t_r, q_hat, c_o, c_r = self.cre(
            queries, images=images, h_patch=h_patch,
            fixed_obj_text=fixed_o, fixed_rel_text=fixed_r)

        if enc_cls is None:
            if h_cls is None:
                raise ValueError("cre_mode='pool' needs cached `h_cls`")
            enc_cls = h_cls

        # ---- Eq. 2
        obj_feats = self.decoder(q_hat, enc_patch)          # (Nq, T, d)

        # ---- pair formation (see select_pairs for the PAPER-GAP)
        if pair_index is None:
            pair_index, keep_index = self.select_pairs(obj_feats)
        kept = obj_feats[keep_index] if keep_index is not None else obj_feats
        obj_s = kept[pair_index[:, 0]]                      # (P, T, d)
        obj_o = kept[pair_index[:, 1]]

        # ---- Eq. 3. H_CLS is the third token of Eq. 11, so it has to arrive in
        # `d_model` alongside the object features.
        cls_feat = self.cls_proj(enc_cls.to(self.cls_proj.weight.dtype))
        obj_s_hat, obj_o_hat, rel_hat = self.iem(obj_s, obj_o, cls_feat)

        # ---- Eq. 4. Object scores are video-level (a trajectory is one thing),
        # predicate scores stay per-timestep so the post-processor can associate
        # clips exactly as the rest of this benchmark's lineage does.
        vis_s = self.obj_to_text(obj_s_hat.mean(dim=1))
        vis_o = self.obj_to_text(obj_o_hat.mean(dim=1))
        vis_r = self.rel_to_text(rel_hat)

        p, t, _ = rel_hat.shape
        sbj_logits = self.score_head(vis_s, t_o, mode="logits")
        obj_logits = self.score_head(vis_o, t_o, mode="logits")
        rel_scores = self.score_head(vis_r.reshape(p * t, -1), t_r,
                                     mode="sigmoid").reshape(p, t, -1)

        # ---- Eq. 5
        traj_s = self.traj_head(obj_s_hat)                  # (P, T, 4)
        traj_o = self.traj_head(obj_o_hat)

        return {
            "rel_scores": rel_scores,
            "sbj_logits": sbj_logits,
            "obj_logits": obj_logits,
            "sbj_scores": torch.softmax(sbj_logits, dim=-1),
            "obj_scores": torch.softmax(obj_logits, dim=-1),
            "pred_sbj_boxes": traj_s,
            "pred_obj_boxes": traj_o,
            "ctx_obj": c_o.mean(dim=0) if c_o.dim() == 3 else c_o,
            "ctx_rel": c_r.mean(dim=0) if c_r.dim() == 3 else c_r,
            "text_obj": t_o,
            "text_rel": t_r,
            "obj_feats": obj_feats,
            "rel_feats": rel_hat,
            "pair_index": pair_index,
            "keep_index": keep_index,
        }

    def fixed_text_banks(self):
        """Unconditioned CLIP prompts -- the `--no_crt` fallback."""
        if not hasattr(self, "_fixed_banks"):
            raise RuntimeError(
                "fixed text banks were not built; pass refine_text=True or call "
                "build_fixed_text_banks() after construction")
        return self._fixed_banks


def build_metor(args, clip_model=None):
    """Construct METOR from parsed CLI args.

    Loads the dataset's category names in id order, so the classifier's rows
    line up with the label ids the evaluator uses -- getting this wrong
    produces a model that trains fine and scores near zero.
    """
    from utils import paths

    if clip_model is None:
        from vlm.backbones.clip import clip as clip_pkg
        clip_model, _ = clip_pkg.load(
            name=getattr(args, "clip_backbone", "ViT-L/14@336px"), device="cpu")
        clip_model = clip_model.float()
        for p in clip_model.parameters():
            p.requires_grad_(False)

    obj_split = json.load(open(paths.OBJ_SPLIT_INFO))
    rel_split = json.load(open(paths.PRED_SPLIT_INFO))
    obj_names = [obj_split["id2cls"][str(i)] for i in range(len(obj_split["id2cls"]))]
    rel_names = [rel_split["id2cls"][str(i)] for i in range(len(rel_split["id2cls"]))]

    model = METOR(
        clip_model, obj_names, rel_names,
        n_queries=getattr(args, "n_queries", 100),
        n_iter=getattr(args, "n_iter", 2),
        alpha=getattr(args, "alpha", 0.9),
        d_model=getattr(args, "d_model", 256),
        cre_mode=getattr(args, "cre_mode", "inject"),
        n_ctx_obj=getattr(args, "n_ctx_obj", 8),
        n_ctx_rel=getattr(args, "n_ctx_rel", 8),
        n_prompt_vec=getattr(args, "n_prompt_vec", 8),
        n_heads=getattr(args, "n_heads", 8),
        n_dec_layers=getattr(args, "n_dec_layers", 6),
        dropout=getattr(args, "dropout", 0.1),
        rel_token_position=getattr(args, "rel_token_position", "end"),
        refine_queries=not getattr(args, "no_crq", False),
        refine_text=not getattr(args, "no_crt", False),
        max_pair_objects=getattr(args, "max_pair_objects", 10),
        st_max_len=getattr(args, "st_max_len", 512),
    )

    n_base_rel = sum(1 for c in rel_split["cls2split"].values() if c == "base")
    criterion = METORCriterion(
        n_base_rel=n_base_rel,
        theta_traj=getattr(args, "theta_traj", 1.0),
        theta_ctx=getattr(args, "theta_ctx", 0.2),
        theta_cst=getattr(args, "theta_cst", 0.1),
    )
    return model, criterion
