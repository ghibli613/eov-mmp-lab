"""`--arch metor-compat` -- METOR's two modules on top of the EOV-MMP stack.

Narrative walkthrough: docs/13_Iterative-enhancement.md section 6

Why this exists
---------------
The paper-faithful `--arch metor` uses 100 object queries and a decoder
initialised from MS-COCO, and it is trained end-to-end for 30 epochs. EOV-MMP's
released detector uses **300** queries, so it cannot be loaded into a
METOR-shaped model -- the architectures differ, not just the weights -- and
EOV-MMP never released training code for its detector either. Training the
faithful model therefore means training an open-vocabulary video detector from
COCO init with no reference implementation to check against.

This module is the other end of that trade. It leaves the EOV-MMP stack exactly
as it is -- same 300 queries, same released checkpoints, same tracker, same
post-processing -- and inserts METOR's two contributions into the relationship
classifier, where they can be switched on and off against a baseline that is
already known to score 26.88 all / 15.64 novel on this machine's checkpoints.

What that buys, and what it does not
------------------------------------
It buys a **controlled measurement**: everything except CRE and the iterative
enhancement module is held fixed, so a delta is attributable to them. That is
strictly better evidence than the paper's own +1.70, which changes the module,
the query count, the epoch budget and the loss count all at once.

It does **not** reproduce 16.74. It cannot: the detector is a different one, and
METOR's trajectory regression (Eq. 5) does not exist here -- trajectories still
come from deep_sort + AFLink. Numbers from this path must be reported as
"EOV-MMP + METOR's modules", never as METOR.

Checkpoint compatibility
------------------------
Every parameter EOV-MMP ships keeps its name and shape, and the new modules are
added alongside. `load_state_dict(..., strict=False)` therefore loads a released
checkpoint cleanly, reporting the METOR additions as "missing" (they are new and
start from their own initialisation) and nothing as "unexpected". The
zero-initialised mapping in `IterativeEnhancementLayer` means a freshly built
compat model with `--n_iter 0` reproduces EOV-MMP's forward pass exactly -- which
is the equality `tests/test_compat_equivalence.py` pins.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from metor.modules.iterative_enhancement import IterativeEnhancementModule


class METORFeatEmbedding(nn.Module):
    """EOV-MMP's `FeatEmbedding` with the iterative enhancement module inserted.

    EOV represents each clip as **four** 768-d region vectors -- subject box,
    object box, union box, whole frame -- and its predicate classifier is four
    text embeddings side by side, so that structure cannot be disturbed without
    invalidating the released weights.

    The insertion respects it. The IEM is run over the roles that already mean
    subject / object / scene:

        subject region (slot 0)  ->  O_s of Eq. 11
        object region  (slot 1)  ->  O_o of Eq. 11
        frame region   (slot 3)  ->  H_CLS of Eq. 11

    and its outputs are written back in place: the enhanced object features
    return to slots 0 and 1, and the relationship feature `R_hat` is added into
    slot 2 -- the union region, which is where the relation already lives in
    EOV's design. Shapes are unchanged end to end, so the text side, the
    temporal decoder and the scoring are untouched.
    """

    def __init__(self, args, base: nn.Module):
        super().__init__()
        # Hold EOV's own submodules under their original names so their
        # parameter keys survive: featEmbedding.spatial_decoder.*, etc.
        self.enabled_streams = base.enabled_streams
        self.spatial_decoder = base.spatial_decoder
        self.temporal_decoder = base.temporal_decoder

        self.n_iter = getattr(args, "n_iter", 2)
        # `--n_iter 0` means "EOV-MMP, untouched" in compat mode: no module, no
        # parameters, no change to the forward pass. That is what makes it a
        # controlled baseline for the delta.
        #
        # Note this differs from the paper's own Table 4 "Iteration Number 0"
        # row, which is METOR-with-no-feedback -- relationship features are
        # still formed once from the object features. That ablation belongs to
        # `--arch metor` and is exercised by tests/test_metor_modules.py.
        # Conflating the two would make the compat delta unattributable.
        self.iem = None if self.n_iter == 0 else IterativeEnhancementModule(
            768,
            n_iter=self.n_iter,
            n_heads=getattr(args, "n_heads", 8),
            alpha=getattr(args, "alpha", 0.9),
            dropout=getattr(args, "dropout", 0.1),
            max_len=getattr(args, "st_max_len", 512),
            residual_mapping=getattr(args, "residual_mapping", False),
        )

    def _stream(self, inputs, name):
        x = inputs[name]
        return x if self.enabled_streams[name] else torch.zeros_like(x)

    def forward(self, inputs, seq_lens):
        bs, slen = inputs["clip_feat"].shape[0], inputs["clip_feat"].shape[1]

        img_feat = self._stream(inputs, "clip_feat")
        bbox_feat = self._stream(inputs, "bbox_feat")
        pos_feat = torch.cat((self._stream(inputs, "rel_feat"),
                              self._stream(inputs, "mot_feat")), dim=-1)

        pre_embs, sbj_embs, obj_embs = self.spatial_decoder(
            x_img=img_feat.view(bs * slen, 4, -1),
            x_box=bbox_feat.view(bs * slen, 4, -1),
            x_pos=pos_feat.view(bs * slen, -1),
        )

        pre_embs = pre_embs.view(bs, slen, 4, -1)

        # ---- METOR Eq. 11, in place over the four regions
        if self.iem is not None:
            o_s = pre_embs[:, :, 0, :]
            o_o = pre_embs[:, :, 1, :]
            h_cls = pre_embs[:, :, 3, :]
            o_s_hat, o_o_hat, rel_hat = self.iem(o_s, o_o, h_cls)
            pre_embs = torch.stack([
                F.normalize(o_s_hat, dim=-1),
                F.normalize(o_o_hat, dim=-1),
                F.normalize(pre_embs[:, :, 2, :] + rel_hat, dim=-1),
                pre_embs[:, :, 3, :],
            ], dim=2)
            # The video-level subject/object embeddings EOV scores against the
            # object text bank are re-derived from the enhanced features, so
            # the enhancement reaches the object scores and not only the
            # predicate ones. That feedback is the point of the module.
            sbj_embs = F.normalize(
                self.spatial_decoder.objEmb(o_s_hat.reshape(bs * slen, -1)), dim=-1)
            obj_embs = F.normalize(
                self.spatial_decoder.objEmb(o_o_hat.reshape(bs * slen, -1)), dim=-1)

        sbj_embs = torch.mean(sbj_embs.view(bs, slen, -1), dim=1)
        obj_embs = torch.mean(obj_embs.view(bs, slen, -1), dim=1)

        pre_embs, interactiveness = self.temporal_decoder(pre_embs, seq_lens)
        pre_embs = pre_embs.view(bs, slen, -1) / 2
        return pre_embs, sbj_embs, obj_embs, interactiveness


def attach_metor_modules(relation_model, args):
    """Swap METOR's modules into an already-built EOV-MMP relation classifier.

    Mutates and returns the model. Call **before** loading a checkpoint so the
    parameter set the loader sees is the final one.
    """
    relation_model.featEmbedding = METORFeatEmbedding(args, relation_model.featEmbedding)
    return relation_model


def build_compat_stack(args, load_components: bool = True):
    """EOV-MMP's full stack with METOR's modules attached to the relation model.

    Mirrors `cli/common.build_model_stack`, so the released checkpoints, the
    tracker and the evaluation path are all the ones already verified to work.
    """
    import torch as _torch

    from models.end2end_model import End2End_Model
    from models.methods import build_model
    from models.object_classifier import Classifier
    from models.relation_classifier import Model
    from models.tracking.aflink.AppFreeLink import PostLinker

    detector, criterion, postprocessors = build_model(args)
    relation = Model(args).cuda()
    relation = attach_metor_modules(relation, args).cuda()
    classifier = Classifier(args).cuda()

    if load_components:
        ckpt = _torch.load(args.detector_ckpt, weights_only=False)
        detector.load_state_dict(ckpt["model"])

        ckpt = _torch.load(args.relation_ckpt, weights_only=False)
        merged = relation.state_dict()
        merged.update({k: v for k, v in ckpt["state_dict"].items() if k in merged})
        relation.load_state_dict(merged)

        ckpt = _torch.load(args.obj_classifier_ckpt, weights_only=False)
        classifier.load_state_dict(ckpt["state_dict"])

    model = End2End_Model(args, detector, classifier, relation,
                          criterion, postprocessors).cuda()

    sort_model = PostLinker()
    sort_model.load_state_dict(_torch.load(args.path_AFLink, weights_only=False))
    return model, sort_model
