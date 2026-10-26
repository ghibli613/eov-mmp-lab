#!/usr/bin/env python3
"""RPT: SAM 3 on whole VidVRD videos, one text prompt at a time, every frame (A28). Runs in the SAM 3
environment and writes files only; the EOV-MMP side reads them (studies/rpt/adapter.py).

    ~/miniconda3/envs/sam3/bin/python studies/rpt/sam3_runner.py --checkpoint CKPT --sha256 SHA \
        --frames ov-vidvrd-lab/data/vidvrd/frames --out studies/rpt/g0/sam3_out VIDEO [VIDEO ...]
    ... --prompts bicycle --max-minutes 30        # the laptop check: one prompt, a wall-time cap
    ... --reuse-backbone                          # A28's runtime measurement only, never G0's output

Protocol (A28): SAM 3's own multi-prompt video protocol (Sam3VideoInference.forward,
sam3_video_inference.py:928-952) through the predictor API: one session per video on its JPEG
folder; per prompt add_prompt at frame 0, propagate_in_video forward from frame 0, reset_session.
build_sam3_video_predictor() with every default; the checkpoint is a local file whose SHA256 is
checked before loading.

Per frame SAM 3 returns, for each masklet it shows, a mask, a box and `out_probs` (the probability
of the detection that started the masklet). Whether a detection started or matched the masklet on
that frame is not in the output: Recorder reads it from SAM 3's own association step, read-only.

Output, per video: <out>/<video>.json.gz with, per prompt, the masklets SAM 3 output and, per output
frame, the COCO RLE of the returned mask, its tight box (torchvision masks_to_boxes, as SAM 3 boxes
its own masks), SAM 3's own box in pixels, out_probs, and the recorded detection and tracker scores;
and, per prompt, the raw detections SAM 3 discarded with probability in [0.1, 0.5) (`raw_low`:
frame, box in pixels, probability; before NMS).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OBJECT2ID = os.path.join(REPO, "data", "vidvrd", "data", "object2id.json")
sys.path.insert(0, HERE)
from no_test_labels import install as no_test_labels  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prompts(object2id_path=OBJECT2ID):
    """The 35 class names in id order, each with its text prompt: '_' becomes a space, nothing else."""
    o2i = json.load(open(object2id_path))
    return [(name, name.replace("_", " ")) for name, _ in sorted(o2i.items(), key=lambda kv: kv[1])]


RAW_LOW = (0.1, 0.5)            # A28: detections SAM 3 discards, recorded read-only (G2 decides their use)


class Recorder:
    """Read-only view of SAM 3's per-frame detection and association (A28). Three wrappers, each
    passing its call's arguments and return value through untouched:

    - the model's run_tracker_update_planning_phase (sam3_video_base.py:751-1100): per frame, which
      masklets a kept detection started (new_det_obj_ids) or matched (det_to_matched_trk_obj_ids),
      those detections' probabilities, the masklets removed, and the tracker's own score of each
      propagated masklet (sigmoid of its object-score logit, as sam3_video_base.py:510-516 stores it);
    - the detector's _build_multigpu_buffer_next_chunk (sam3_image.py:822): which frame the detector
      is on (one GPU: frame_idx_begin), without a GPU sync;
    - the detector's forward_grounding (sam3_image.py:441), as it returns and before NMS lowers the
      logits of every detection <= 0.5 in place (sam3_image.py:846-861): the raw detections with
      probability in [0.1, 0.5), box and probability.

    A frame processed twice (add_prompt and propagation both run frame 0) keeps its last record;
    starts are kept from every call."""

    def __init__(self, model):
        self.model, self.detector = model, model.detector
        self.orig = model.run_tracker_update_planning_phase
        self.orig_chunk = self.detector._build_multigpu_buffer_next_chunk
        self.orig_grounding = self.detector.forward_grounding
        self.frames, self.started, self.raw_low, self.chunk_frame = {}, {}, {}, None
        model.run_tracker_update_planning_phase = self.wrapped
        self.detector._build_multigpu_buffer_next_chunk = self.wrapped_chunk
        self.detector.forward_grounding = self.wrapped_grounding

    def wrapped_chunk(self, *args, **kwargs):
        self.chunk_frame = int(kwargs["frame_idx_begin"])
        try:
            return self.orig_chunk(*args, **kwargs)
        finally:
            self.chunk_frame = None

    def wrapped_grounding(self, *args, **kwargs):
        out = self.orig_grounding(*args, **kwargs)
        if self.chunk_frame is not None:
            p = out["pred_logits"].detach().float().sigmoid()[0, :, 0]
            sel = (p >= RAW_LOW[0]) & (p < RAW_LOW[1])
            boxes = out["pred_boxes_xyxy"].detach().float()[0][sel]
            self.raw_low[self.chunk_frame] = [b + [s] for b, s in zip(boxes.cpu().tolist(), p[sel].cpu().tolist())]
        return out

    def wrapped(self, *args, **kwargs):
        plan, meta_new = self.orig(*args, **kwargs)
        f = int(kwargs["frame_idx"])
        scores = kwargs["det_out"]["scores"].float().cpu().numpy()
        started = {int(o): float(scores[i]) for o, i in zip(plan["new_det_obj_ids"], plan["new_det_fa_inds"])}
        matched = defaultdict(list)
        for d, ids in plan["det_to_matched_trk_obj_ids"].items():
            for o in ids:
                matched[int(o)].append(float(scores[d]))
        trk_ids = kwargs["tracker_metadata_prev"]["obj_ids_all_gpu"]
        trk = kwargs["tracker_obj_scores_global"].float().sigmoid().cpu().tolist()
        self.frames[f] = dict(started=started, matched=dict(matched),
                              tracker={int(o): float(s) for o, s in zip(trk_ids, trk)},
                              removed=sorted(int(o) for o in plan["obj_ids_newly_removed"]),
                              kept_detections=int(len(scores)))
        for o, s in started.items():
            self.started.setdefault(o, (f, s))
        return plan, meta_new

    def reset(self):
        self.frames, self.started, self.raw_low = {}, {}, {}

    def close(self):                                               # the classes' own methods again
        del self.model.run_tracker_update_planning_phase
        del self.detector._build_multigpu_buffer_next_chunk
        del self.detector.forward_grounding


class BackboneCache:
    """Measurement only (A28; never used for G0's outputs): does reusing the image backbone across a
    video's prompts change SAM 3's output, and how much time does it save?

    Wraps the detector backbone's forward_image (vl_combiner.py:81), which depends on the image
    alone and is recomputed for every frame of every prompt. The first call for a frame runs it and
    keeps a CPU copy of the output; later calls for that frame get a GPU copy with the same tensor
    sharing. Position encodings bitwise equal to an earlier frame's are kept once. The frame comes
    from Recorder (one GPU); any other call passes through."""

    def __init__(self, detector, rec):
        self.backbone, self.rec = detector.backbone, rec
        self.orig = self.backbone.forward_image
        self.store, self.shared = {}, {}
        self.hits = self.misses = self.passthrough = 0
        self.backbone.forward_image = self.wrapped

    def wrapped(self, samples, *args, **kwargs):
        f = self.rec.chunk_frame
        if f is None or args or kwargs or samples.shape[0] != 1:
            self.passthrough += 1
            return self.orig(samples, *args, **kwargs)
        if f in self.store:
            self.hits += 1
            return _copy_tree(self.store[f], samples.device, {})
        out = self.orig(samples)
        self.misses += 1
        self.store[f] = self._stash(out)
        return out

    def _stash(self, out):
        import torch
        memo = {}

        def walk(v, path):
            if torch.is_tensor(v):
                if id(v) in memo:
                    return memo[id(v)]
                c = v.detach().to("cpu", copy=True)
                if "pos_enc" in path:
                    first = self.shared.setdefault(path, c)
                    if first is not c and torch.equal(first, c):
                        c = first
                memo[id(v)] = c
                return c
            if isinstance(v, (list, tuple)):
                return type(v)(walk(e, f"{path}[{i}]") for i, e in enumerate(v))
            if isinstance(v, dict):
                return {k: walk(e, f"{path}.{k}") for k, e in v.items()}
            return v
        return walk(out, "")

    def bytes_stored(self):
        seen, n = set(), 0

        def walk(v):
            nonlocal n
            if hasattr(v, "data_ptr"):
                if v.data_ptr() not in seen:
                    seen.add(v.data_ptr())
                    n += v.numel() * v.element_size()
            elif isinstance(v, (list, tuple)):
                for e in v:
                    walk(e)
            elif isinstance(v, dict):
                for e in v.values():
                    walk(e)
        for v in self.store.values():
            walk(v)
        return n

    def close(self):
        del self.backbone.forward_image
        self.store.clear()


def _copy_tree(v, device, memo):
    """A copy of a nested structure of tensors on `device`, keeping which entries are the same tensor."""
    import torch
    if torch.is_tensor(v):
        if id(v) not in memo:
            memo[id(v)] = v.to(device, copy=True)
        return memo[id(v)]
    if isinstance(v, (list, tuple)):
        return type(v)(_copy_tree(e, device, memo) for e in v)
    if isinstance(v, dict):
        return {k: _copy_tree(e, device, memo) for k, e in v.items()}
    return v


def frame_records(outputs, rec, width, height):
    """{obj_id: record} for one yielded frame; only masklets with a non-empty returned mask."""
    import numpy as np
    import torch
    from pycocotools import mask as mask_util
    from torchvision.ops import masks_to_boxes
    f = int(outputs["frame_index"])
    o = outputs["outputs"]
    out = {}
    r = rec.frames.get(f, {})
    for k, obj in enumerate(o["out_obj_ids"].tolist()):
        m = o["out_binary_masks"][k]
        if not m.any():
            continue
        x, y, w, h = (float(v) for v in o["out_boxes_xywh"][k])
        rle = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
        det = list(r.get("matched", {}).get(obj, []))
        if obj in r.get("started", {}):                    # the frame's own record has the start
            det.append(r["started"][obj])
        out[int(obj)] = dict(
            f=f, rle=rle["counts"].decode("ascii"), area=int(mask_util.area(rle)),
            box=[float(v) for v in masks_to_boxes(torch.from_numpy(m)[None])[0].tolist()],
            box_sam3=[x * width, y * height, (x + w) * width, (y + h) * height],
            prob=float(o["out_probs"][k]), started=rec.started.get(obj, (None,))[0] == f,
            matched=obj in r.get("matched", {}), det=det, trk=r.get("tracker", {}).get(obj))
    return out


def run_prompt(predictor, rec, session, prompt, n_frames, width, height, deadline=None):
    """One prompt over the whole video. -> (masklets, stats)."""
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    rec.reset()
    predictor.handle_request(dict(type="add_prompt", session_id=session, frame_index=0, text=prompt))
    masklets, seen, stopped = defaultdict(list), 0, False
    for out in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=session,
                                                    propagation_direction="forward", start_frame_index=0)):
        for obj, r in frame_records(out, rec, width, height).items():
            masklets[obj].append(r)
        seen += 1
        if deadline is not None and time.time() > deadline:
            stopped = True
            break
    torch.cuda.synchronize()
    seconds = time.time() - t0
    stats = dict(prompt=prompt, seconds=round(seconds, 3), frames_yielded=seen, frames=n_frames,
                 seconds_per_frame=round(seconds / max(seen, 1), 4), stopped_at_time_limit=stopped,
                 peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                 peak_gpu_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                 started=len(rec.started), removed_by_sam3=sorted({o for fr in rec.frames.values() for o in fr["removed"]}),
                 frames_with_detections=sum(1 for fr in rec.frames.values() if fr["kept_detections"] > 0),
                 raw_low_detections=sum(len(d) for d in rec.raw_low.values()))
    stats["raw_low"] = [[f] + [b[0] * width, b[1] * height, b[2] * width, b[3] * height, b[4]]
                        for f, ds in sorted(rec.raw_low.items()) for b in ds]       # [frame, x1, y1, x2, y2, p]
    if not stopped:
        predictor.handle_request(dict(type="reset_session", session_id=session))
    return masklets, stats


def run_video(predictor, video, frames_dir, names, deadline=None, reuse_backbone=False):
    import torch
    from PIL import Image
    folder = os.path.join(frames_dir, video)
    files = sorted(os.listdir(folder))
    width, height = Image.open(os.path.join(folder, files[0])).size
    t0 = time.time()
    session = predictor.handle_request(dict(type="start_session", resource_path=folder))["session_id"]
    load_s = time.time() - t0
    rec = Recorder(predictor.model)
    cache = BackboneCache(predictor.model.detector, rec) if reuse_backbone else None
    result = dict(video=video, frames=len(files), width=width, height=height, session_start_seconds=round(load_s, 3),
                  reuse_backbone=reuse_backbone, prompts=[])
    try:
        for name, prompt in names:
            masklets, stats = run_prompt(predictor, rec, session, prompt, len(files), width, height, deadline)
            stats["class"] = name
            result["prompts"].append(dict(stats, masklets=[dict(obj_id=o, frames=fr) for o, fr in sorted(masklets.items())]))
            print(json.dumps({k: v for k, v in stats.items() if k not in ("removed_by_sam3", "raw_low")}), flush=True)
            if stats["stopped_at_time_limit"]:
                break
    finally:
        if cache is not None:
            result["backbone_cache"] = dict(hits=cache.hits, misses=cache.misses, passthrough=cache.passthrough,
                                            stored_gib=cache.bytes_stored() / 2**30)
            cache.close()
        rec.close()
        predictor.handle_request(dict(type="close_session", session_id=session))
    result["seconds_total"] = round(sum(p["seconds"] for p in result["prompts"]), 3)
    result["peak_gpu_allocated_gib"] = max((p["peak_gpu_allocated_gib"] for p in result["prompts"]), default=0.0)
    result["peak_gpu_reserved_gib"] = max((p["peak_gpu_reserved_gib"] for p in result["prompts"]), default=0.0)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sha256", required=True, help="the checkpoint's SHA256, as A28 pins it")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default="all", help="'all' (the 35 names) or comma-separated class names")
    ap.add_argument("--max-minutes", type=float, default=None, help="wall-time cap over the whole run")
    ap.add_argument("--reuse-backbone", action="store_true",
                    help="measurement only (A28): reuse the image backbone across prompts")
    a = ap.parse_args()
    no_test_labels()
    import torch
    torch.manual_seed(0)
    names = prompts()
    if a.prompts != "all":
        want = a.prompts.split(",")
        unknown = [w for w in want if w not in dict(names)]
        if unknown:
            raise SystemExit(f"!! unknown class names: {unknown}")
        names = [(n, p) for n, p in names if n in want]
    got = sha256(a.checkpoint)
    if got != a.sha256:
        raise SystemExit(f"!! STOP: {a.checkpoint} hashes to {got}, not {a.sha256}")
    os.makedirs(a.out, exist_ok=True)
    deadline = None if a.max_minutes is None else time.time() + 60 * a.max_minutes
    t0 = time.time()
    from sam3.model_builder import build_sam3_video_predictor
    try:
        predictor = build_sam3_video_predictor(checkpoint_path=a.checkpoint)
    except torch.OutOfMemoryError as e:
        rep = dict(stage="model load", error="CUDA out of memory", message=str(e).splitlines()[0],
                   peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        json.dump(rep, open(os.path.join(a.out, "run_error.json"), "w"), indent=1)
        raise SystemExit(json.dumps(rep))
    print(json.dumps(dict(model_load_seconds=round(time.time() - t0, 1), torch=torch.__version__,
                          gpu=torch.cuda.get_device_name(0),
                          allocated_after_load_gib=torch.cuda.memory_allocated() / 2**30)), flush=True)
    for v in a.videos:
        try:
            res = run_video(predictor, v, a.frames, names, deadline, reuse_backbone=a.reuse_backbone)
        except torch.OutOfMemoryError as e:
            rep = dict(stage=f"video {v}", error="CUDA out of memory", message=str(e).splitlines()[0],
                       peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                       peak_gpu_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
            json.dump(rep, open(os.path.join(a.out, f"{v}.error.json"), "w"), indent=1)
            print(json.dumps(rep), flush=True)
            continue
        res.update(checkpoint_sha256=got, prompts_run=len(res["prompts"]), torch=torch.__version__,
                   gpu=torch.cuda.get_device_name(0))
        path = os.path.join(a.out, f"{v}.json.gz")
        with gzip.open(path, "wt") as f:
            json.dump(res, f)
        print(json.dumps(dict(video=v, frames=res["frames"], prompts=res["prompts_run"], seconds_total=res["seconds_total"],
                              peak_gpu_allocated_gib=res["peak_gpu_allocated_gib"], file=path, sha256=sha256(path))), flush=True)
    predictor.shutdown()


if __name__ == "__main__":
    main()
