"""Ground rule 3 on stubs: recorders on vs off give identical tracks.

The repo's own test branch of End2End_Model.forward runs on the CPU: the real OVPostProcess,
`filter`/unique_boxes, deep_sort, AFLink (with the real AFLink_epoch20.pth when present) and
format_trajectories_test. The detector (modelA) and classifier (modelB) are deterministic stubs;
gen_feats_test returns nothing so the relation head (modelC) is never called -- the same in every
run, so the comparison stays like for like.
"""
import copy
import math
import os

import numpy as np
import pytest
import torch
from torch import nn

from d2.baseline_tracks import compare
from d2.common import CODE_ROOT
from d2.recorders import Recorder, serialize_tracks

T, W, H, D, Q, C = 60, 640, 480, 16, 300, 35
VIDEO = "synthetic_d2_video"


def _logit(p):
    return math.log(p / (1 - p))


def _objects(f):
    """(xyxy, prob, class) of the scene's objects at frame f; q0..q3 in that order."""
    ax = 40 + 4 * f
    pa = 0.3 if f == 45 else 0.9                       # frame 45: nothing passes 0.35 -> fallback
    return [([ax, 60, ax + 120, 160], pa, 5),          # q0: object A
            ([ax + 2, 61, ax + 122, 161], 0.25 if f == 45 else 0.5, 5),   # q1: A's duplicate, IoU > 0.8
            ([400, 250 + 2 * f, 520, 350 + 2 * f], 0.7 if f < 40 else 0.2, 12),   # q2: B, below the cut from f=40
            ([100, 300, 200, 420], 0.1, 3)]            # q3: C, never passes


class StubDetector(nn.Module):
    """modelA: frame index read from patch_[f, 0, 0]; returns pred_logits / pred_boxes."""

    def forward(self, p_, targets=None):
        f = int(round(float(p_[0, 0, 0])))
        rng = np.random.default_rng(1000 + f)
        logits = np.full((Q, C), -9.0)
        boxes = np.zeros((Q, 4))
        objs = _objects(f)
        for q in range(Q):
            if q < len(objs):
                (x1, y1, x2, y2), p, c = objs[q]
            else:
                x1, y1 = rng.uniform(0, W - 60), rng.uniform(0, H - 60)
                x2, y2 = x1 + rng.uniform(20, 60), y1 + rng.uniform(20, 60)
                p, c = rng.uniform(0.001, 0.2), int(rng.integers(C))
            logits[q, c] = _logit(p)
            boxes[q] = [(x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H]
        out = {"pred_logits": torch.tensor(logits, dtype=torch.float32)[None],
               "pred_boxes": torch.tensor(boxes, dtype=torch.float32)[None]}
        return out, list(range(C))


class StubClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(7)
        self.w = torch.randn(C, D, generator=g)

    def forward(self, x):                                  # x: (1, D), as end2end_model.py:256 passes it
        return torch.softmax(x.reshape(1, D) @ self.w.T, -1)


def _inputs():
    g = torch.Generator().manual_seed(3)
    blocks = torch.randn(8, 8, D, generator=g)               # piecewise-constant appearance field
    field = blocks.repeat_interleave(3, 0).repeat_interleave(3, 1).reshape(576, D)
    patch_ = torch.zeros(1, T, 576, 4)
    patch_[0, :, 0, 0] = torch.arange(T, dtype=torch.float32)
    return dict(video_name=[VIDEO], begin_fid=torch.tensor([0]), end_fid=torch.tensor([T]),
                video_len=torch.tensor([T]), patch_=patch_, patch_proj=field.expand(T, 576, D)[None].clone(),
                global_proj=torch.zeros(1, T, 8), image_size=[torch.tensor([W]), torch.tensor([H])])


@pytest.fixture
def pipeline(args, cpu_cuda, monkeypatch):
    import models.end2end_model as E
    from models.methods.detectors.ov_prompt.post_process import OVPostProcess
    from models.tracking.aflink.model import PostLinker
    monkeypatch.setattr(E, "gen_feats_test", lambda *a, **k: {})
    post = OVPostProcess(num_queries=Q, dataset_file="open_vidvrd")
    model = E.End2End_Model(args, StubDetector(), StubClassifier(), nn.Identity(), None, {"bbox": post})
    model.eval()
    linker = PostLinker()
    ck = os.path.join(CODE_ROOT, "output", "ckpt", "AFLink_epoch20.pth")
    if os.path.exists(ck):
        linker.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False))
    else:
        torch.manual_seed(0)
    linker.eval()
    return model, linker


def _run(model, linker, rec=None):
    model.traj.clear()
    if rec:
        rec.start(VIDEO, 0, T, T, W, H)
    with torch.no_grad():
        res = model(_inputs(), linker)
    trajs = model.traj.get(VIDEO, [])
    out = serialize_tracks(copy.deepcopy(trajs))
    if rec:
        rec.finish(trajs)
    return out, res


def test_recorders_on_vs_off_identical(pipeline, repo_fns):
    import models.end2end_model as E
    import models.gen_labels as G
    from models.tracking.aflink import AppFreeLink as AF
    model, linker = pipeline
    originals = (E.run, E.format_trajectories_test, AF.AFLink.link, G.interpolate_and_adjust_frames)

    off1, res1 = _run(model, linker)
    rec = Recorder().install(model)
    on, res2 = _run(model, linker, rec)
    rec.uninstall()
    off2, res3 = _run(model, linker)

    assert len(off1) >= 2, "the scene must produce tracks, or the comparison is vacuous"
    assert compare(on, off1) == [] and compare(off2, off1) == []
    assert res1 == res2 == res3 == []
    assert (E.run, E.format_trajectories_test, AF.AFLink.link, G.interpolate_and_adjust_frames) == originals
    assert len(model.postprocessorA["bbox"]._forward_hooks) == 0

    # what was recorded is consistent with the pipeline
    c = rec.cur
    assert c["n_post"] == T and c["post_box_mismatch"] == 0 and c["post_cut_mismatch"] == 0
    assert c["fallback"][45] and sum(c["fallback"]) == 1
    sel_per_frame = [int(d["sel"].sum()) for d in c["det"]]
    assert sel_per_frame[0] == 3 and sel_per_frame[45] == 1 and sel_per_frame[50] == 2
    n_ti = len(c["ti"]["frame"])
    assert n_ti < sum(sel_per_frame), "the dedupe removed A's duplicate"
    assert np.all(c["det"][0]["score"][c["det"][0]["q"] == 3] < 0.35)           # C recorded below the cut
    assert c["temp"].shape[1] == 45 and len(c["temp"]) > 0
    assert len(c["interp"]) == len(np.unique(c["temp"][:, 1]))
    from d2.stages import replay_video
    _, _, G_ = repo_fns
    assert compare(replay_video(G_, c["temp"], T), on) == []
    assert compare(c["final"], on) == []
    a = rec.arrays()
    assert set(a) >= {"det_frame", "det_q", "det_box", "det_score", "det_label", "det_sel", "ti_frame",
                      "ti_box", "ti_score", "ds_rows", "af_in", "af_out", "temp", "interp", "frame_fallback"}
    assert a["det_frame"].shape == a["det_score"].shape and a["det_box"].shape[1] == 4
    assert np.all((a["det_score"] >= 0.01) | a["det_sel"])


def test_comparison_detects_a_changed_pipeline(pipeline, monkeypatch):
    """Not vacuous: a wrapper that moves the tracker input boxes by 1 px changes the tracks.
    (Moving a single frame-0 box changes nothing: that frame is in deep_sort's tentative phase,
    and with NSA at confidence 1 every later update snaps to its own detection.)"""
    import models.end2end_model as E
    model, linker = pipeline
    ref, _ = _run(model, linker)
    orig = E.run

    def bad_run(args, video_name, object_instances, **kw):
        for o in object_instances:
            o["box"] = o["box"] + torch.tensor([1.0, 0, 1.0, 0])
        return orig(args, video_name, object_instances, **kw)
    monkeypatch.setattr(E, "run", bad_run)
    changed, _ = _run(model, linker)
    assert compare(changed, ref) != []


def test_recorder_ignores_frames_outside_a_video(pipeline):
    """Between start() and finish() only; a forward with no video started records nothing."""
    model, linker = pipeline
    rec = Recorder().install(model)
    try:
        _run(model, linker)
        assert rec.cur == {}
    finally:
        rec.uninstall()
