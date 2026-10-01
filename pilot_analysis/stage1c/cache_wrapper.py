#!/usr/bin/env python3
"""Stage 1 session 3, Tasks 2/3/6 -- the B6 feature cache, via forward hooks.

GPU host only. No model code is changed: the wrapper imports the repo's own
encoders, feature builder (gen_feats_test / FeatExtractor) and relation
classifier, loads modelC's weights from the end-to-end checkpoint, and captures
internal tensors with forward hooks.

    python pilot_analysis/stage1c/cache_wrapper.py --split test --videos V1 V2 --out CACHE
    python pilot_analysis/stage1c/cache_wrapper.py --split train --limit 3 --out CACHE
    python pilot_analysis/stage1c/cache_wrapper.py --split test --videos V1 V2 --fidelity-from-dump \
        pilot_analysis/preds_from_pod/full/segments_raw_all.json --out CACHE

Per video, one file CACHE/<split>/<video>.pt (resumable; SHA256 in CACHE/manifest.jsonl):
  per (pair, slot): logits over all 132 (fp32), interactiveness logit (fp32),
                    visual_pre_embeddings (3072, fp16), spatial-decoder tokens
                    (4x768, fp16), slot inputs clip/bbox/rel/mot (fp16)
  per pair:         conditioning mean (3072, fp16), meta_net bias (768, fp16),
                    subject/object embeddings, categories, scores, gt tids,
                    duration, slot boundaries, boxes, >40-slot window record
  per frame:        CLIP-L frame embedding (768); TagCLIP RoI mean per (track,
                    frame) and per (pair, frame) union (768, fp16)
Once: CACHE/once.pt -- ctx, meta_net, token_prefix/suffix, the three fixed
template embeddings for the 132 names, object text weights.

Logits are kept in fp32 (the brief says fp16): they are ranked by the
post-processing, fp16 rounding could reorder near-ties, and the cost is
~0.5 KB per slot.

Fidelity, checked on EVERY modelC call: the logits rebuilt from the hooked
tensors with the model's own methods must reproduce the scores the model
returns, i.e. sigmoid(logits) * sigmoid(int) == pre_scores.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, hashlib, json, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

MAX_SLOTS = 40
CKPT = "output/ckpt/baseline_fbce_vidvrd_bs1_lr1e-05_dim512_none_rel_mot_clip_bbox_end2end_base-001.pth"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
def build_modelC(ckpt_path):
    import torch
    from utils.parser_func import parse_args
    from models.relation_classifier import Model
    argv, sys.argv = sys.argv, [sys.argv[0], "--ckpt_path", ckpt_path, "--tgt_split", "all",
                                "--frame_stride", "1"]
    try:
        args = parse_args()
    finally:
        sys.argv = argv
    m = Model(args).cuda().eval()
    m.tgt_split = "all"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    sd = {k[len("modelC."):]: v for k, v in ck["state_dict"].items() if k.startswith("modelC.")}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    del ck
    return m, args, dict(missing=list(missing), unexpected=list(unexpected), loaded=len(sd))


def encode_frames(video, frame_dir, backbone="ViT-L/14@336px"):
    """Dataset_new.__getitem__ at frame_stride 1 (data_loading/dataset.py:88-119),
    keeping only what modelC and the cache use: global_proj and patch_proj.
    The detector input `patch_` is discarded as soon as it is produced."""
    import torch
    from PIL import Image
    from data_loading.dataset import _shared_encoder, _transform_resize
    pre = _transform_resize(336, 336)
    clipm, tag = _shared_encoder("clip", backbone), _shared_encoder("tagclip", backbone)
    gps, pps, size = [], [], None
    for name in sorted(os.listdir(os.path.join(frame_dir, video))):
        with torch.no_grad():
            img = Image.open(os.path.join(frame_dir, video, name)).convert("RGB")
            size = img.size
            x = pre(img).unsqueeze(0).cuda()
            _, gp = clipm.encode_image(x)
            p, pp = tag.encode_image_tagclip(x, 336, 336, attn_mask=1)
            del p
            gps.append(gp.cpu())
            pps.append(pp.cpu())
    return torch.cat(gps, 0), torch.cat(pps, 0), size        # (T, 768), (T, 576, 768), (w, h)


def pair_order(tracks):
    """The (subject, object) order gen_feats_test enumerates (models/gen_labels.py:655-659)."""
    out = []
    for i, s in enumerate(tracks):
        for j, o in enumerate(tracks):
            if i == j:
                continue
            if min(s["end_fid"], o["end_fid"]) - max(s["begin_fid"], o["begin_fid"]) < 10:
                continue
            out.append((i, j))
    return out


def roi_means(pp, frames, boxes, w, h, chunk=128):
    """The masked mean of models/gen_labels.py:_extract_clip_feat (unnormalised),
    for each (frame, box). Frames outside [0, T) -- the repo's 2-frame pre-roll
    below 0 -- are skipped and reported as absent."""
    import numpy as np, torch
    from models.gen_labels import create_mask
    keep = [(f, b) for f, b in zip(frames, boxes) if 0 <= f < pp.shape[0]]
    if not keep:
        return [], torch.zeros(0, pp.shape[-1], dtype=torch.float16)
    out = []
    for k in range(0, len(keep), chunk):
        part = keep[k:k + chunk]
        m = torch.tensor(np.stack([np.array(create_mask(24, 24, [b[0] / w * 24, b[1] / h * 24,
                                                                  b[2] / w * 24, b[3] / h * 24])).flatten()
                                   for _, b in part]), dtype=torch.float32).cuda()      # (n, 576)
        x = pp[[f for f, _ in part]].cuda().float()                                     # (n, 576, 768)
        s = torch.einsum("np,npd->nd", m, x) / m.sum(1, keepdim=True).clamp(min=1e-9)
        out.append(s.half().cpu())
    return [f for f, _ in keep], torch.cat(out, 0)


def run_modelC(m, feats, slen):
    """One modelC call with hooks. Returns per-slot / per-pair tensors and the
    fidelity residual max|sigmoid(logits)*sigmoid(int) - pre_scores|."""
    import torch, torch.nn.functional as F
    caps = {}
    hooks = [
        m.featEmbedding.spatial_decoder.register_forward_hook(lambda mod, i, o: caps.__setitem__("spatial", o)),
        m.featEmbedding.register_forward_hook(lambda mod, i, o: caps.__setitem__("fe", o)),
        m.pre_classifier.register_forward_hook(lambda mod, i, o: caps.__setitem__("text", o)),
        m.pre_classifier.prompt_learner.meta_net.register_forward_hook(lambda mod, i, o: caps.__setitem__("bias", o)),
    ]
    try:
        with torch.no_grad():
            pre_scores, _, _ = m(dict(feats), torch.tensor([slen]))
            pre_embs, sbj, obj, inter = caps["fe"][:4]
            vpe = F.normalize(pre_embs, dim=-1)                                   # rc:496
            text, _ = m.split_text_embeddings("all", pre_classifier_weights=caps["text"].squeeze(0))
            logits = torch.matmul(vpe, text.t()) / m.temperature                  # rc:504
            recon = torch.sigmoid(logits) * torch.sigmoid(inter).unsqueeze(-1)
            resid = (recon - pre_scores).abs().max().item()
    finally:
        for h in hooks:
            h.remove()
    x_rel = caps["spatial"][0]                                                    # (slen, 4, 768)
    return dict(logits=logits[0].float().cpu(), int_logit=inter[0].float().cpu(),
                vpe=vpe[0].half().cpu(), spatial=x_rel.half().cpu(),
                cond_mean=vpe[0].mean(0).half().cpu(), bias=caps["bias"][0].half().cpu(),
                sbj_emb=F.normalize(sbj, dim=-1)[0].half().cpu(), obj_emb=F.normalize(obj, dim=-1)[0].half().cpu(),
                pre_scores=pre_scores[0].float().cpu()), resid


def run_pair(m, item):
    """modelC over one gen_feats_test item, split into <= 40-slot windows."""
    import torch
    feats = {k: v.float() for k, v in item.items() if "feat" in k}
    slen = feats["clip_feat"].shape[1]
    wins = [(a, min(a + MAX_SLOTS, slen)) for a in range(0, slen, MAX_SLOTS)]
    outs, resid = [], 0.0
    for a, b in wins:
        o, r = run_modelC(m, {k: v[:, a:b] for k, v in feats.items()}, b - a)
        outs.append(o); resid = max(resid, r)
    cat = lambda k: torch.cat([o[k] for o in outs], 0)
    per_pair = {k: torch.stack([o[k] for o in outs], 0) for k in ("cond_mean", "bias", "sbj_emb", "obj_emb")}
    slot_in = {k: v[0].half() for k, v in feats.items()}
    return dict(logits=cat("logits"), int_logit=cat("int_logit"), vpe=cat("vpe"), spatial=cat("spatial"),
                pre_scores=cat("pre_scores"), slot_inputs=slot_in, windows=wins, **per_pair), resid


# ---------------------------------------------------------------------------
def cache_video(m, video, split, tracks, frame_dir, out_path):
    import torch
    from models.gen_labels import gen_feats_test
    from utils.eov_utils import gen_union_bbox
    t0 = time.time()
    gp, pp, (w, h) = encode_frames(video, frame_dir)
    t_enc = time.time() - t0
    items = gen_feats_test([video], tracks, split, pp.unsqueeze(0), gp.unsqueeze(0), w, h)
    order = pair_order(tracks)
    assert len(order) == len(items), (len(order), len(items))
    pairs, worst, split_pairs = [], 0.0, []
    for (si, oi), pid in zip(order, sorted(items)):
        it = items[pid]
        r, resid = run_pair(m, it)
        worst = max(worst, resid)
        pd = it["pair_data"][0]
        if len(r["windows"]) > 1:
            split_pairs.append((tracks[si]["tid"], tracks[oi]["tid"], len(r["logits"])))
        b0, b1 = pd["duration"]
        fr = list(range(b0, b1))
        uf, union = roi_means(pp, fr, [gen_union_bbox(s, o) for s, o in zip(pd["sbj_traj"], pd["obj_traj"])], w, h)
        pairs.append(dict(sub=tracks[si]["tid"], obj=tracks[oi]["tid"],
                          sub_gt=tracks[si].get("gt_tid"), obj_gt=tracks[oi].get("gt_tid"),
                          sbj_cls=pd["sbj_cls"], obj_cls=pd["obj_cls"], sbj_scr=pd["sbj_scr"],
                          obj_scr=pd["obj_scr"], duration=pd["duration"],
                          sbj_traj=pd["sbj_traj"], obj_traj=pd["obj_traj"],
                          union_frames=uf, union_roi=union, **{k: v for k, v in r.items() if k != "pre_scores"}))
    track_roi = {}
    for t in tracks:
        fs = list(range(t["begin_fid"], t["end_fid"]))
        f_, x = roi_means(pp, fs, [t["trajectory"][f] for f in fs], w, h)
        track_roi[t["tid"]] = dict(frames=f_, roi=x, category=t["category"], score=t["score"],
                                   gt_tid=t.get("gt_tid"), begin_fid=t["begin_fid"], end_fid=t["end_fid"],
                                   boxes=[t["trajectory"][f] for f in fs])
    obj = dict(video=video, split=split, size=(w, h), n_frames=gp.shape[0],
               global_proj=gp.half(), tracks=track_roi, pairs=pairs, split_pairs=split_pairs,
               fidelity_max_residual=worst)
    torch.save(obj, out_path)
    return dict(video=video, split=split, seconds=round(time.time() - t0, 1), encode_seconds=round(t_enc, 1),
                frames=int(gp.shape[0]), tracks=len(tracks), pairs=len(pairs),
                slots=int(sum(len(p["logits"]) for p in pairs)), split_pairs=split_pairs,
                fidelity_max_residual=worst, bytes=os.path.getsize(out_path), sha256=sha256(out_path))


def save_once(m, path):
    import torch
    pl = m.pre_classifier.prompt_learner
    te = m.pre_text_encoder
    torch.save(dict(ctx=pl.ctx.detach().cpu(), meta_net={k: v.cpu() for k, v in pl.meta_net.state_dict().items()},
                    token_prefix=pl.token_prefix.cpu(), token_suffix=pl.token_suffix.cpu(),
                    template_subject=te.sbj_classifier_weights.cpu(), template_object=te.obj_classifier_weights.cpu(),
                    template_relation=te.pre_classifier_weights.cpu(),
                    object_text=m.obj_text_encoder.classifier_weights.cpu(), temperature=m.temperature,
                    id2cls=te.id2cls), path)


# ---------------------------------------------------------------------------
def fidelity_from_dump(m, video, dump, frame_dir):
    """Recompute modelC on the DETECTED pairs stored in a segments_raw dump and
    compare with the dumped per-segment top-20 scores (all split)."""
    import torch
    from models.gen_labels import FeatExtractor, CLIP_LEN
    gp, pp, (w, h) = encode_frames(video, frame_dir)
    ann = json.load(open(os.path.join(ROOT, "data", "vidvrd", "anno", "test", f"{video}.json")))
    ann.pop("relation_instances", None)                # size only; labels stay with the evaluator
    anno = dict(video_id=video, width=ann["width"], height=ann["height"])
    fe = FeatExtractor(["rel_feat", "mot_feat", "clip_feat", "bbox_feat"])
    fe.load_frames(anno)
    id2 = m.pre_text_encoder.id2cls
    cls2id = {v: int(k) for k, v in id2.items()}
    worst, n_cmp, topset_same, n_seg = 0.0, 0, 0, 0
    for pr in dump[video]:
        b0, b1 = pr["duration"]
        n = (b1 - b0) // CLIP_LEN + (1 if (b1 - b0) % CLIP_LEN >= 10 else 0)
        pf = {k: [] for k in ("rel_feat", "mot_feat", "clip_feat", "bbox_feat")}
        for c in range(n):                                     # gen_labels.py:684-700
            last = c == n - 1
            clip = dict(sbj_id=0, obj_id=0, begin_fid=b0 + c * CLIP_LEN,
                        sbj_traj=pr["sbj_traj"][c * CLIP_LEN:] if last else pr["sbj_traj"][c * CLIP_LEN:(c + 1) * CLIP_LEN],
                        obj_traj=pr["obj_traj"][c * CLIP_LEN:] if last else pr["obj_traj"][c * CLIP_LEN:(c + 1) * CLIP_LEN],
                        end_fid=b1 if last else b0 + (c + 1) * CLIP_LEN)
            f = fe.gen_feats(clip, pp.unsqueeze(0), gp.unsqueeze(0), w, h)
            for k in pf:
                pf[k].append(f[k])
        item = {k: (torch.stack(v, 0).unsqueeze(0).float() if k == "clip_feat" else torch.tensor(v).unsqueeze(0))
                for k, v in pf.items()}
        r, _ = run_pair(m, item)
        ps = r["pre_scores"]                                    # (slen, 132)
        for sg in pr["segments"]:
            k = sg["segment_index"]
            n_seg += 1
            dumped = {c: s for c, s in sg["preds"]}
            re_top = [id2[str(int(i))] for i in torch.argsort(-ps[k])[:len(dumped)]]
            topset_same += set(re_top) == set(dumped)
            for c, s in dumped.items():
                worst = max(worst, abs(float(ps[k, cls2id[c]]) - s)); n_cmp += 1
    return dict(video=video, pairs=len(dump[video]), segments=n_seg, scores_compared=n_cmp,
                max_abs_diff=worst, top20_set_identical=f"{topset_same}/{n_seg}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "test"), required=True)
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--fidelity-from-dump")
    a = ap.parse_args()
    import torch
    from gt_tracks import load_tracks
    from utils import paths
    os.makedirs(os.path.join(a.out, a.split), exist_ok=True)
    m, args, load_info = build_modelC(os.path.join(ROOT, a.ckpt))
    json.dump(load_info, open(os.path.join(a.out, "modelC_load.json"), "w"), indent=1)
    if not os.path.exists(os.path.join(a.out, "once.pt")):
        save_once(m, os.path.join(a.out, "once.pt"))

    if a.fidelity_from_dump:
        dump = json.load(open(a.fidelity_from_dump))
        res = [fidelity_from_dump(m, v, dump, paths.FRAME_DIR) for v in a.videos]
        json.dump(res, open(os.path.join(a.out, "fidelity_from_dump.json"), "w"), indent=1)
        print(json.dumps(res, indent=1))
        return

    tracks = load_tracks(a.split)
    vids = a.videos or sorted(tracks)
    if a.limit:
        vids = vids[:a.limit]
    man_path = os.path.join(a.out, "manifest.jsonl")
    done = {}
    if os.path.exists(man_path):
        for line in open(man_path):
            r = json.loads(line); done[(r["split"], r["video"])] = r
    for v in vids:
        out = os.path.join(a.out, a.split, f"{v}.pt")
        prev = done.get((a.split, v))
        if prev and os.path.exists(out) and sha256(out) == prev["sha256"]:
            continue                                                    # resumable
        rec = cache_video(m, v, a.split, tracks[v], paths.FRAME_DIR, out)
        with open(man_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec))
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
