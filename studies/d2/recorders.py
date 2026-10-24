"""D2 recorders (A25, "Recording"): read-only hooks on EOV-MMP's test-path front end.

Nothing in the pipeline is edited. `Recorder.install(model)` adds a forward hook on
`model.postprocessorA["bbox"]` and wraps four module-level functions the test branch of
`End2End_Model.forward` looks up at call time:

    models.end2end_model.run                        deep_sort (input detections, output rows)
    AFLink.link                                     tracklet linking (input, output rows)
    models.end2end_model.format_trajectories_test   the filter's input rows `temp`, its output
    models.gen_labels.interpolate_and_adjust_frames per-track filter calls

Each wrapper copies what it records *before* calling the original and returns the
original's own result object, so the forward sees exactly what it would without them.
`uninstall()` restores the originals.

Per video: `start(...)`, run the forward, `finish(model.traj.get(video, []))`, then
`arrays()` / `meta()` to save.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from d2.common import CUT, REC_FLOOR


def _selected(set_index_0) -> list[int]:
    """`set_index[0]` is a 1-d tensor of query ids, or `[0-d tensor]` for the fallback query."""
    import torch
    if torch.is_tensor(set_index_0):
        return [int(i) for i in set_index_0.reshape(-1).tolist()]
    return [int(i) for i in set_index_0]


def serialize_tracks(trajs) -> list[dict]:
    """Tracks as `model.traj` holds them -> plain lists, in the same form as the baseline cache
    (`sgdet_cache.py`: boxes over the sorted frames; begin/end; class; score)."""
    out = []
    for t in trajs:
        fs = sorted(int(f) for f in t["trajectory"])
        out.append(dict(tid=int(t["tid"]), category=t["category"], score=float(t["score"]),
                        begin_fid=int(t["begin_fid"]), end_fid=int(t["end_fid"]),
                        boxes=[[float(v) for v in t["trajectory"][f]] for f in fs]))
    return out


class Recorder:
    def __init__(self, floor: float = REC_FLOOR):
        self.floor = floor
        self.installed = False
        self.video = None
        self.cur: dict = {}

    # ------------------------------------------------------------------ install
    def install(self, model):
        import models.end2end_model as E
        import models.gen_labels as G
        from models.tracking.aflink import AppFreeLink as AF
        from models.util import box_ops
        if self.installed:
            raise RuntimeError("recorder already installed")
        if E.AFLink is not AF.AFLink:
            raise RuntimeError("end2end_model.AFLink is not AppFreeLink.AFLink; the wrapper would miss it")
        self._E, self._G, self._AF, self._box_ops = E, G, AF, box_ops
        self._orig = dict(run=E.run, fmt=E.format_trajectories_test, link=AF.AFLink.link,
                          interp=G.interpolate_and_adjust_frames)
        E.run = self._wrap_run(self._orig["run"])
        E.format_trajectories_test = self._wrap_fmt(self._orig["fmt"])
        AF.AFLink.link = self._wrap_link(self._orig["link"])
        G.interpolate_and_adjust_frames = self._wrap_interp(self._orig["interp"])
        self._hook = model.postprocessorA["bbox"].register_forward_hook(self._post_hook)
        self.installed = True
        return self

    def uninstall(self):
        if not self.installed:
            return
        E, G, AF = self._E, self._G, self._AF
        E.run = self._orig["run"]
        E.format_trajectories_test = self._orig["fmt"]
        AF.AFLink.link = self._orig["link"]
        G.interpolate_and_adjust_frames = self._orig["interp"]
        self._hook.remove()
        self.installed = False

    # ------------------------------------------------------------------ per video
    def start(self, video: str, begin_fid: int, end_fid: int, video_len: int, w: int, h: int):
        self.video = video
        self.cur = dict(video=video, begin_fid=int(begin_fid), end_fid=int(end_fid), video_len=int(video_len),
                        w=int(w), h=int(h), n_post=0, post_box_mismatch=0, post_cut_mismatch=0,
                        det=[], sel=[], fallback=[], ti=None, ds_rows=None, af_in=None, af_out=None,
                        temp=None, fmt_out=None, interp=[], final=None)

    def finish(self, final_trajs):
        c = self.cur
        c["final"] = serialize_tracks(final_trajs)
        n_frames = c["end_fid"] - c["begin_fid"]
        if c["n_post"] != n_frames:
            raise RuntimeError(f"{c['video']}: {c['n_post']} post-processor calls for {n_frames} frames")
        self.video = None
        return c

    # ------------------------------------------------------------------ hooks
    def _post_hook(self, module, inputs, output):
        """Every query's box (post-processor pixel coordinates), cut score and argmax class, for
        queries with cut score >= floor or selected by the cut. Checks that the selected set and
        boxes reproduce the post-processor's own output."""
        if self.video is None:
            return None
        outputs, target_sizes, _selected_id, classifier = inputs
        if classifier:
            return None
        import torch
        results, set_index, topk_boxes = output
        c = self.cur
        with torch.no_grad():
            prob = outputs["pred_logits"].sigmoid()
            max_prob, arg = prob.max(dim=-1)                      # post_process.py:194
            img_h, img_w = target_sizes[0].unbind(0)
            b = self._box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"][0].unsqueeze(0))[0]
            scale = torch.tensor([img_w, img_h, img_w, img_h]).to(b)
            boxes = b * scale                                     # as post_process.py:273-280
            sel = _selected(set_index[0])
            sel_t = torch.tensor(sel, dtype=torch.long, device=boxes.device)
            if not torch.equal(boxes[sel_t][topk_boxes[0].reshape(-1)], results[0]["boxes"].reshape(-1, 4)):
                c["post_box_mismatch"] += 1
            if module.pruning_threshold != CUT:
                raise RuntimeError(f"pruning_threshold is {module.pruning_threshold}, A25 fixes {CUT}")
            passing = torch.gt(max_prob[0], module.pruning_threshold).nonzero(as_tuple=True)[0].tolist()
            fallback = not passing
            if fallback:   # post_process.py:221-233: the frame's single best query
                ok = len(sel) == 1 and float(max_prob[0, sel[0]]) == float(max_prob[0].max())
            else:
                ok = sel == passing
            if not ok:
                c["post_cut_mismatch"] += 1
            keep = max_prob[0] >= self.floor
            keep[sel_t] = True
            q = keep.nonzero(as_tuple=True)[0]
            frame = c["begin_fid"] + c["n_post"]
            selmask = torch.zeros_like(keep)
            selmask[sel_t] = True
            c["det"].append(dict(frame=frame, q=q.cpu().numpy().astype(np.int16),
                                 box=boxes[q].float().cpu().numpy(),
                                 score=max_prob[0][q].float().cpu().numpy(),
                                 label=arg[0][q].cpu().numpy().astype(np.uint8),
                                 sel=selmask[q].cpu().numpy()))
            c["fallback"].append(bool(fallback))
            c["n_post"] += 1
        return None

    def _wrap_run(self, orig):
        rec = self

        def run(args, video_name, object_instances, *a, **kw):
            if rec.video is not None:
                n = len(object_instances)
                rec.cur["ti"] = dict(
                    frame=np.array([int(o["frame_id"]) for o in object_instances], dtype=np.int32),
                    box=(np.stack([o["box"].detach().cpu().numpy().astype(np.float32, copy=True)
                                   for o in object_instances]) if n else np.zeros((0, 4), np.float32)),
                    score=(np.stack([np.asarray(o["score"].detach().cpu().numpy(), dtype=np.float32).copy()
                                     for o in object_instances]) if n else np.zeros((0, 35), np.float32)))
            res = orig(args, video_name, object_instances, *a, **kw)
            if rec.video is not None:
                rec.cur["ds_rows"] = np.array(res, dtype=np.float64).reshape(-1, 10)
            return res
        run.__wrapped__ = orig
        return run

    def _wrap_link(self, orig):
        rec = self

        def link(linker):
            if rec.video is not None:
                rec.cur["af_in"] = np.array(linker.track, dtype=np.float64, copy=True)
            res = orig(linker)
            if rec.video is not None:
                rec.cur["af_out"] = np.array(res, dtype=np.float64, copy=True)
            return res
        link.__wrapped__ = orig
        return link

    def _wrap_fmt(self, orig):
        rec = self

        def format_trajectories_test(temp):
            if rec.video is not None:
                rec.cur["temp"] = (np.stack([np.asarray(t, dtype=np.float64).copy() for t in temp])
                                   if len(temp) else np.zeros((0, 45)))
            res = orig(temp)
            if rec.video is not None:
                rec.cur["fmt_out"] = serialize_tracks(copy.deepcopy(res))
            return res
        format_trajectories_test.__wrapped__ = orig
        return format_trajectories_test

    def _wrap_interp(self, orig):
        rec = self

        def interpolate_and_adjust_frames(track_frames):
            frames = sorted(float(f) for f in track_frames) if rec.video is not None else None
            res = orig(track_frames)
            if rec.video is not None:
                ks = sorted(res)
                rec.cur["interp"].append([len(frames), frames[0], frames[-1], len(res),
                                          ks[0] if ks else math.nan, ks[-1] if ks else math.nan])
            return res
        interpolate_and_adjust_frames.__wrapped__ = orig
        return interpolate_and_adjust_frames

    # ------------------------------------------------------------------ output
    def arrays(self) -> dict:
        """Flat numpy arrays for np.savez_compressed."""
        c = self.cur
        det = c["det"]
        cat = lambda k, dt, shape: (np.concatenate([d[k] for d in det]).astype(dt) if det
                                    else np.zeros(shape, dt))
        out = dict(
            det_frame=(np.concatenate([np.full(len(d["q"]), d["frame"], np.int32) for d in det])
                       if det else np.zeros(0, np.int32)),
            det_q=cat("q", np.int16, (0,)), det_box=cat("box", np.float32, (0, 4)),
            det_score=cat("score", np.float32, (0,)), det_label=cat("label", np.uint8, (0,)),
            det_sel=cat("sel", bool, (0,)), frame_fallback=np.array(c["fallback"], bool),
            interp=np.array(c["interp"], np.float64).reshape(-1, 6))
        ti = c["ti"] or dict(frame=np.zeros(0, np.int32), box=np.zeros((0, 4), np.float32),
                             score=np.zeros((0, 35), np.float32))
        out.update(ti_frame=ti["frame"], ti_box=ti["box"], ti_score=ti["score"])
        for k, cols in (("ds_rows", 10), ("af_in", 10), ("af_out", 10), ("temp", 45)):
            v = c[k]
            out[k] = np.zeros((0, cols)) if v is None else v.reshape(-1, v.shape[-1] if v.ndim > 1 else cols)
        return out

    def meta(self) -> dict:
        c = self.cur
        keys = ("video", "begin_fid", "end_fid", "video_len", "w", "h", "n_post", "post_box_mismatch",
                "post_cut_mismatch", "fmt_out", "final")
        return {k: c[k] for k in keys} | dict(floor=self.floor, n_fallback_frames=int(sum(c["fallback"])))
