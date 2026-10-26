"""RPT: tracks -> EOV-MMP's relation head, unchanged (A28): gen_feats_test (models/gen_labels.py:642)
with whole-video features on demand, then session 3's run_pair (pilot_analysis/stage1c/
cache_wrapper.py:166-180), which runs modelC on windows of at most 40 clips and joins them; then the
repo's post-processing on each pair's joined clips (eval_from_cache.predict, :64-79).

Features: FrameFeatures encodes a frame only when gen_feats_test indexes it, exactly as session 3's
encode_frames does per frame (cache_wrapper.py:78-98: the repo's _shared_encoder and
_transform_resize(336, 336), batch 1, fp16 on CUDA), and memoizes it. Frame f is the f-th file of
the video's folder in sorted order (dataset.py:91). The GT range (dataset.py:120-135) is not used.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
STAGE1C = os.path.join(REPO, "pilot_analysis", "stage1c")
for p in (REPO, STAGE1C):
    if p not in sys.path:
        sys.path.insert(0, p)


class FrameFeatures:
    """CLIP-L global_proj and TagCLIP-L patch_proj of any frame of the whole video, on demand.

    gen_feats_test reads them as patch_proj[0][f:f+1, :, :] and global_proj[0][f:f+1, :]
    (gen_labels.py:106, :117); the two proxies answer exactly those reads. `log` keeps every read in
    order: per clip three patch reads (subject, object, union) and one global read, at its middle
    frame. `encoder(frame_path) -> (gp (1, 768), pp (1, 576, 768))` can be replaced in tests."""

    def __init__(self, video, frame_dir, encoder=None):
        self.video = video
        self.folder = os.path.join(frame_dir, video)
        self.files = sorted(os.listdir(self.folder))
        self.encoder = encoder or clip_encoder()
        self.cache, self.log = {}, []
        from PIL import Image
        self.size = Image.open(os.path.join(self.folder, self.files[0])).size      # (w, h)

    def encode(self, f):
        if f not in self.cache:
            self.cache[f] = self.encoder(os.path.join(self.folder, self.files[f]))
        return self.cache[f]

    @property
    def patch_proj(self):
        return _Proxy(self, "patch")

    @property
    def global_proj(self):
        return _Proxy(self, "global")


class _Proxy:
    def __init__(self, ff, kind):
        self.ff, self.kind = ff, kind

    def __getitem__(self, key):
        if isinstance(key, int) and key == 0:
            return self
        sl = key[0] if isinstance(key, tuple) else key
        rest = key[1:] if isinstance(key, tuple) else ()
        if not (isinstance(sl, slice) and sl.step is None and sl.stop == sl.start + 1
                and all(r == slice(None) for r in rest)):
            raise IndexError(f"FrameFeatures answers only gen_feats_test's one-frame reads, not {key!r}")
        f = sl.start
        if not 0 <= f < len(self.ff.files):
            raise IndexError(f"frame {f} is outside {self.ff.video} (0..{len(self.ff.files) - 1})")
        self.ff.log.append((self.kind, f))
        gp, pp = self.ff.encode(f)
        return pp if self.kind == "patch" else gp


def clip_encoder(backbone="ViT-L/14@336px"):
    """encode_frames' per-frame body (cache_wrapper.py:88-97), for one frame."""
    import torch
    from PIL import Image
    from data_loading.dataset import _shared_encoder, _transform_resize
    pre = _transform_resize(336, 336)
    clipm, tag = _shared_encoder("clip", backbone), _shared_encoder("tagclip", backbone)

    def encode(path):
        with torch.no_grad():
            img = Image.open(path).convert("RGB")
            x = pre(img).unsqueeze(0).cuda()
            _, gp = clipm.encode_image(x)
            p, pp = tag.encode_image_tagclip(x, 336, 336, attn_mask=1)
            del p
            return gp.cpu(), pp.cpu()
    return encode


def relation_head(video, tracks, split, ff, m, run_pair=None):
    """Tracks in the repo's format -> one record per ordered pair, in gen_feats_test's order:
    the pair data, modelC's joined per-clip logits and interactiveness logits, the windows, the
    slot inputs (fp16, as session 3 cached them) and the clips' middle frames."""
    from models.gen_labels import gen_feats_test
    from cache_wrapper import pair_order
    if run_pair is None:
        from cache_wrapper import run_pair
    w, h = ff.size
    items = gen_feats_test([video], tracks, split, ff.patch_proj, ff.global_proj, w, h)
    order = pair_order(tracks)
    assert len(order) == len(items), (len(order), len(items))
    mids = [f for kind, f in ff.log if kind == "global"]
    pairs, k = [], 0
    for (si, oi), pid in zip(order, sorted(items)):
        it = items[pid]
        n = int(it["clip_feat"].shape[1])
        r, resid = run_pair(m, it)
        pd = it["pair_data"][0]
        pairs.append(dict(sub=si, obj=oi, sub_tid=tracks[si]["tid"], obj_tid=tracks[oi]["tid"],
                          sbj_cls=pd["sbj_cls"], obj_cls=pd["obj_cls"], sbj_scr=pd["sbj_scr"], obj_scr=pd["obj_scr"],
                          duration=list(pd["duration"]), sbj_traj=pd["sbj_traj"], obj_traj=pd["obj_traj"],
                          logits=r["logits"], int_logit=r["int_logit"], windows=r["windows"],
                          slot_inputs=r["slot_inputs"], mid_frames=mids[k:k + n], fidelity_residual=resid))
        k += n
    assert k == len(mids), (k, len(mids))
    return pairs


def predictions(pairs, base_ids=None, mode="all"):
    """One video's pairs -> the repo's process_pred (top 20 per clip) -> association (per pair, on
    the joined clips) -> format_ (top 200), with modelC's own clip score sigmoid(logit) *
    sigmoid(interactiveness) and session 3's settings (eval_from_cache.POST, .slot_scores)."""
    from eval_from_cache import POST as args, meta, slot_scores
    from inference.post_process import association, format_, process_pred
    id2pre, obj2id, base = meta()
    rels = []
    for p in pairs:
        pd = dict(sbj_cls=p["sbj_cls"], obj_cls=p["obj_cls"], sbj_scr=p["sbj_scr"], obj_scr=p["obj_scr"],
                  sbj_traj=[list(b) for b in p["sbj_traj"]], obj_traj=[list(b) for b in p["obj_traj"]],
                  duration=list(p["duration"]))
        rels.extend(association(process_pred(args, id2pre, obj2id, None,
                                             slot_scores(p["logits"], p["int_logit"], mode, base_ids or base), pd)))
    return format_(args, rels)
