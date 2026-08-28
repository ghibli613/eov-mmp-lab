#!/usr/bin/env python
"""Tests H1's MECHANISM directly, on CPU, with no model inference at all.

    python pilot_analysis/scripts/clip_text_separability.py

Why this is a valid test of H1
-----------------------------
Traced through models/relation_classifier.py, the text side of the predicate
classifier is a concatenation of FOUR embeddings (split_text_embeddings, L465):

  1. sbj  "An image of a person or object {name} something."        frozen, .detach()
  2. obj  "An image of something {name} a person or object."        frozen, .detach()
  3. uni  "An image of the visual relation {name} between two entities."  frozen, .detach()
  4. rel  learned prompt (CoCoOp-style), instance-conditioned       LEARNED

Three of the four are frozen CLIP text embeddings of a hand-written template
(PredicateTextEncoder.build_clip_fixed_prompts, L401). The fourth is learned --
but what is learned is `self.ctx`, a set of context vectors SHARED ACROSS ALL
CLASSES, plus a meta_net conditioned on the image. The only class-specific
input anywhere is the tokenised class NAME.

So for a novel predicate the model has no learned per-class parameter at all:
its entire representation is CLIP's frozen text encoding of its name in those
templates. If CLIP maps two predicate names to nearly the same point, no
visual evidence can separate them. That is exactly H1's premise, and it is
checkable without running the model.

At inference on the novel split the model scores ONLY the 61 novel predicates
(Model.forward, L511: scores_[:, :, novel_pids] = pre_scores), so what matters
is separability WITHIN those 61 -- that is what this measures.

Caveat, stated because it is load-bearing: this measures the 3 frozen
components, not the learned 4th, and text-embedding separability is a
NECESSARY-condition proxy, not mAP. Poor separability here bounds what the
model can do; good separability here does not guarantee good mAP.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from vlm.backbones.clip import clip

TEMPLATES = {   # verbatim from PredicateTextEncoder.build_clip_fixed_prompts
    "sbj": "An image of a person or object {} something.",
    "obj": "An image of something {} a person or object.",
    "uni": "An image of the visual relation {} between two entities.",
}


def main():
    info = json.load(open("configs/VidVRD_pred_class_spilt_info_v2.json"))
    id2cls, cls2split = info["id2cls"], info["cls2split"]
    names = [id2cls[str(i)] for i in range(len(id2cls))]
    part = json.load(open("pilot_analysis/predicate_partition.json"))

    print("loading CLIP ViT-L/14@336px on CPU (text tower only is used)...")
    model, _ = clip.load(name="ViT-L/14@336px", device="cpu")
    model = model.eval()

    embs = {}
    with torch.no_grad():
        for key, tmpl in TEMPLATES.items():
            prompts = [tmpl.format(n.replace("_", " ")) for n in names]
            toks = clip.tokenize(prompts)
            e = model.encode_text(toks).float()
            e = e / e.norm(dim=-1, keepdim=True)
            embs[key] = e
    # the model concatenates the three (plus a learned 4th) -- concat then renorm
    cat = torch.cat([embs[k] for k in ("sbj", "obj", "uni")], dim=-1)
    cat = cat / cat.norm(dim=-1, keepdim=True)

    idx = {n: i for i, n in enumerate(names)}
    novel = [n for n in names if cls2split.get(n) == "novel"]
    base = [n for n in names if cls2split.get(n) == "base"]
    print(f"{len(names)} predicates ({len(base)} base, {len(novel)} novel)\n")

    # ---- separability WITHIN the 61 novel predicates (what inference actually scores)
    ni = [idx[n] for n in novel]
    S = (cat[ni] @ cat[ni].T).numpy()
    np.fill_diagonal(S, -1.0)
    nn_sim = S.max(axis=1)
    nn_of = [novel[j] for j in S.argmax(axis=1)]

    print("SEPARABILITY WITHIN THE 61 NOVEL PREDICATES (cosine, 3 frozen templates)")
    print(f"  mean nearest-neighbour similarity {nn_sim.mean():.4f}")
    print(f"  median                            {np.median(nn_sim):.4f}")
    print(f"  min / max                         {nn_sim.min():.4f} / {nn_sim.max():.4f}")
    print(f"  pairs above 0.95                  {int(((S > 0.95).sum())//2)}")
    print(f"  pairs above 0.99                  {int(((S > 0.99).sum())//2)}")

    print("\n  by partition group (H1 predicts APPEARANCE names separate worse):")
    print(f"  {'group':22} {'n':>3} {'mean NN sim':>12} {'median':>8}")
    for g in ("geometric_static", "geometric_dynamic", "appearance_static", "appearance_dynamic"):
        sel = [k for k, n in enumerate(novel) if part[n]["group"] == g]
        if not sel:
            print(f"  {g:22} {0:3d} {'--':>12} {'--':>8}")
            continue
        v = nn_sim[sel]
        print(f"  {g:22} {len(sel):3d} {v.mean():12.4f} {np.median(v):8.4f}")

    print("\n  10 least separable novel predicates (most confusable with another novel):")
    for k in np.argsort(-nn_sim)[:10]:
        print(f"    {novel[k]:16} ~ {nn_of[k]:16} sim {nn_sim[k]:.4f}  "
              f"[{part[novel[k]]['group']}]")

    # ---- does a novel name collapse onto a BASE name? (the recombination question)
    bi = [idx[n] for n in base]
    SB = (cat[ni] @ cat[bi].T).numpy()
    nb_sim = SB.max(axis=1)
    nb_of = [base[j] for j in SB.argmax(axis=1)]
    print(f"\nNOVEL -> NEAREST BASE PREDICATE")
    print(f"  mean {nb_sim.mean():.4f}   median {np.median(nb_sim):.4f}   "
          f"above 0.95: {(nb_sim > 0.95).sum()}/{len(novel)}")
    print("  10 highest (a novel predicate CLIP cannot tell from a base one):")
    for k in np.argsort(-nb_sim)[:10]:
        print(f"    {novel[k]:16} ~ {nb_of[k]:16} sim {nb_sim[k]:.4f}  "
              f"[{part[novel[k]]['group']}]")

    # ---- control: how separable are the OBJECT names, for scale?
    obj_info = json.load(open("configs/VidVRD_class_spilt_info.json"))
    onames = [o for o in obj_info["cls2split"] if o != "__background__"]
    with torch.no_grad():
        oe = model.encode_text(clip.tokenize([f"An image of a {o.replace('_',' ')}." for o in onames])).float()
        oe = oe / oe.norm(dim=-1, keepdim=True)
    OS = (oe @ oe.T).numpy()
    np.fill_diagonal(OS, -1.0)
    print(f"\nCONTROL -- the 35 OBJECT names, same encoder:")
    print(f"  mean nearest-neighbour similarity {OS.max(axis=1).mean():.4f}")
    print("  (objects are what CLIP is known to be good at; use it as the scale bar)")

    out = {
        "novel_within": {novel[k]: {"nearest_novel": nn_of[k], "sim": float(nn_sim[k]),
                                    "nearest_base": nb_of[k], "sim_base": float(nb_sim[k]),
                                    "group": part[novel[k]]["group"]}
                         for k in range(len(novel))},
        "object_control_mean_nn_sim": float(OS.max(axis=1).mean()),
    }
    json.dump(out, open("pilot_analysis/clip_text_separability.json", "w"), indent=1)
    print("\nwrote pilot_analysis/clip_text_separability.json")


if __name__ == "__main__":
    main()
