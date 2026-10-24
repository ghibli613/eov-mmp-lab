"""run.py's per-video path on METOR's own End2End_Model, on the CPU.

The repo's test branch runs as on the machine: the real OVPostProcess, `filter`/unique_boxes, fusion,
deep_sort, AFLink (the real AFLink_epoch20.pth when present), format_trajectories_test, and the
repo's process_pred -> association -> format_. Stubs: the detector and classifier (D2's synthetic
scene), the relation head (sigmoid of a function of its input; novel = base columns zeroed, as the
real head), and gen_feats_test (the real one's pairing and clip rule, without the annotation file).
"""
import copy
import json
import math
import os

import numpy as np
import pytest
import torch
from torch import nn

from a26.common import CONFIGS, EOV_ROOT
from a26.run import predictions_for, run_configs
from a26.tracks import compare_tracks, serialize_tracks
from a26.wrappers import BothSplits, set_fusion

T, W, H, D, Q, C = 60, 640, 480, 16, 300, 35
GT = (10, 50)                         # the loader's GT-derived range in the synthetic video
VIDEO = "synthetic_a26_video"
CLIP_LEN = 30


def _logit(p):
    return math.log(p / (1 - p))


def _objects(f):
    """D2's scene: (xyxy, prob, class) at frame f; q0..q3."""
    ax = 40 + 4 * f
    pa = 0.3 if f == 45 else 0.9
    return [([ax, 60, ax + 120, 160], pa, 5),
            ([ax + 2, 61, ax + 122, 161], 0.25 if f == 45 else 0.5, 5),
            ([400, 250 + 2 * f, 520, 350 + 2 * f], 0.7 if f < 40 else 0.2, 12),
            ([100, 300, 200, 420], 0.1, 3)]


class StubDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.frames = []

    def forward(self, p_, targets=None):
        f = int(round(float(p_[0, 0, 0])))
        self.frames.append(f)
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
        return ({"pred_logits": torch.tensor(logits, dtype=torch.float32)[None],
                 "pred_boxes": torch.tensor(boxes, dtype=torch.float32)[None]}, list(range(C)))


class StubClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.randn(C, D, generator=torch.Generator().manual_seed(7))

    def forward(self, x):
        return torch.softmax(x.reshape(1, D) @ self.w.T, -1)


class StubRelation(nn.Module):
    def __init__(self, novel_pids):
        super().__init__()
        self.tgt_split = "all"
        self.novel_pids = list(novel_pids)

    def forward(self, feats, seq_lens, labels=None):
        x = feats["clip_feat"]
        pre = torch.sigmoid(x[..., :1] * 3 - x[..., 1:2] + torch.arange(132.0) / 40 - 2)
        if self.tgt_split == "novel":
            z = torch.zeros_like(pre)
            z[..., self.novel_pids] = pre[..., self.novel_pids]
            pre = z
        return pre, torch.zeros(1, 35), torch.zeros(1, 35)


def stub_gen_feats_test(video_name, trajectories, split, patch_proj, global_proj, w, h):
    """gen_labels.gen_feats_test's pairing, clip rule and pair_data, without its annotation file;
    clip_feat is a deterministic function of the boxes."""
    items, pair_id = {}, 0
    for si, s in enumerate(trajectories):
        for oi, o in enumerate(trajectories):
            if si == oi:
                continue
            b, e = max(s["begin_fid"], o["begin_fid"]), min(s["end_fid"], o["end_fid"])
            if e - b < 10:
                continue
            clip_num, tail = int((e - b) / CLIP_LEN), (e - b) % CLIP_LEN
            if tail >= 10:
                clip_num += 1
            elif 0 < tail < 10:
                e -= tail
            sb = [s["trajectory"][f] for f in range(int(b), int(e))]
            ob = [o["trajectory"][f] for f in range(int(b), int(e))]
            pair_data = dict(sbj_scr=s["score"], obj_scr=o["score"], sbj_cls=s["category"], obj_cls=o["category"],
                             sbj_traj=sb, obj_traj=ob, duration=[b, e])
            pair_id += 1
            cf = []
            for k in range(clip_num):
                ss, oo = sb[k * CLIP_LEN:(k + 1) * CLIP_LEN], ob[k * CLIP_LEN:(k + 1) * CLIP_LEN]
                cf.append([np.mean([x[0] for x in ss]) / W, np.mean([x[1] for x in ss]) / H,
                           np.mean([x[0] for x in oo]) / W, np.mean([x[1] for x in oo]) / H])
            items[pair_id] = {"clip_feat": torch.tensor(cf, dtype=torch.float32)[None], "vid": video_name[0],
                              "pair_data": [pair_data]}
    return items


def _batch():
    g = torch.Generator().manual_seed(3)
    field = torch.randn(8, 8, D, generator=g).repeat_interleave(3, 0).repeat_interleave(3, 1).reshape(576, D)
    patch_ = torch.zeros(1, T, 576, 4)
    patch_[0, :, 0, 0] = torch.arange(T, dtype=torch.float32)
    return dict(video_name=[VIDEO], begin_fid=torch.tensor([GT[0]]), end_fid=torch.tensor([GT[1]]),
                video_len=torch.tensor([T]), patch_=patch_, patch_proj=field.expand(T, 576, D)[None].clone(),
                global_proj=torch.zeros(1, T, 8), image_size=[torch.tensor([W]), torch.tensor([H])])


class _DS:
    """What run.py and _predict_split read from Dataset_new: the id maps and the prior."""

    def __init__(self):
        from utils import paths
        self.id2pre = json.load(open(os.path.join(paths.META_DIR, "id2predicate.json")))
        self.obj2id = json.load(open(os.path.join(paths.META_DIR, "object2id.json")))
        self.prior = None                                    # --use_prior is off (default)


def _novel_pids():
    from utils import paths
    info = json.load(open(paths.PRED_SPLIT_INFO))
    return [info["cls2id"][c] for c in info["cls2id"] if info["cls2split"][c] == "novel"]


@pytest.fixture
def stack(args, cpu_cuda, monkeypatch):
    import models.end2end_model as E
    from inference.post_process import association, format_, process_pred
    from models.methods.detectors.ov_prompt.post_process import OVPostProcess
    from models.tracking.aflink.model import PostLinker
    monkeypatch.setattr(E, "gen_feats_test", stub_gen_feats_test)
    inner = StubRelation(_novel_pids())
    model = E.End2End_Model(args, StubDetector(), StubClassifier(), BothSplits(inner), None,
                            {"bbox": OVPostProcess(num_queries=Q, dataset_file="open_vidvrd")})
    model.eval()
    linker = PostLinker()
    ck = os.path.join(EOV_ROOT, "output", "ckpt", "AFLink_epoch20.pth")
    if os.path.exists(ck):
        linker.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False))
    else:
        torch.manual_seed(0)
    linker.eval()
    return model, linker, inner, args, _DS(), (process_pred, association, format_)


def _norm(x):
    return json.loads(json.dumps(x))


def test_r0_equals_the_repos_own_predict_split(stack, monkeypatch):
    """R0 through run_configs (one pass, BothSplits) == cli/common.py:_predict_split, run once per
    split with modelC unwrapped, as A23's evaluation runs it; tracks equal too."""
    import cli.common as CC
    model, linker, inner, args, ds, fns = stack
    out = run_configs(model, linker, args, ds, _batch(), CONFIGS[:1], fns)["R0"]
    assert out["begin_fid"] == GT[0] and out["end_fid"] == GT[1] and out["video_len"] == T

    captured = {}
    monkeypatch.setattr(CC, "eval_relation_detection_openvoc",
                        lambda target_split_pred, prediction_results, rt_hit_infos: captured.__setitem__(
                            target_split_pred, copy.deepcopy(dict(prediction_results))) or (0.0, {50: 0.0, 100: 0.0}))
    plain = copy.copy(model)
    plain._modules = dict(model._modules, modelC=inner)
    tracks = {}
    for split in ("all", "novel"):
        plain.traj.clear()
        CC._predict_split(args, plain, linker, [_batch()], ds, split)
        tracks[split] = serialize_tracks(plain.traj[VIDEO])
    assert len(out["tracks"]) >= 2 and compare_tracks(out["tracks"], tracks["all"]) == []
    assert compare_tracks(tracks["novel"], tracks["all"]) == []
    for split in ("all", "novel"):
        assert len(out["predictions"][split]) > 0
        assert out["predictions"][split] == _norm(captured[split][VIDEO])
    novel = set(_novel_pids())
    names = ds.id2pre
    assert any(r["triplet"][1] in {names[i] for i in novel} for r in out["predictions"]["novel"])
    assert out["predictions"]["all"] != out["predictions"]["novel"]


def test_four_configurations(stack, monkeypatch):
    """Frame ranges, the fusion switch, restored state, and no leak between configurations."""
    import models.end2end_model as E
    model, linker, inner, args, ds, fns = stack
    calls = []
    real_fuse = E.fuse_scores
    monkeypatch.setattr(E, "fuse_scores", lambda sim, sc, uniform=False: calls.append(uniform) or real_fuse(sim, sc, uniform=uniform))
    det = model.modelA
    seen = {}
    for name, rng, uni in CONFIGS:
        det.frames.clear()
        calls.clear()
        r = run_configs(model, linker, args, ds, _batch(), [(name, rng, uni)], fns)[name]
        seen[name] = r
        assert det.frames == (list(range(*GT)) if rng == "gt" else list(range(T)))
        assert calls and set(calls) == {uni}
        assert (r["begin_fid"], r["end_fid"]) == (GT if rng == "gt" else (0, T))
        assert args.uniform_fusion is False                       # restored after each call
    assert seen["R0"]["tracks"] != seen["R0w"]["tracks"]                       # the range matters here
    assert [t["score"] for t in seen["R0"]["tracks"]] != [t["score"] for t in seen["R0u"]["tracks"]]

    together = run_configs(model, linker, args, ds, _batch(), CONFIGS, fns)
    backwards = run_configs(model, linker, args, ds, _batch(), CONFIGS[::-1], fns)
    for name, _, _ in CONFIGS:
        for d in (together, backwards):
            assert d[name]["tracks"] == seen[name]["tracks"]
            assert d[name]["predictions"] == seen[name]["predictions"]
    assert VIDEO not in model.traj and model.modelC.other == []


def test_fusion_restored_on_error(stack, monkeypatch):
    model, linker, inner, args, ds, fns = stack

    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(model, "forward", boom)
    with pytest.raises(RuntimeError):
        run_configs(model, linker, args, ds, _batch(), CONFIGS[1:2], fns)
    assert args.uniform_fusion is False


def test_whole_video_moves_the_length_rule(stack, monkeypatch):
    """min_l is end_fid - begin_fid (end2end_model.py:322): 40 frames in the GT range, 60 over the whole
    video. A 7-frame track passes 0.15 x 40 = 6 and fails 0.15 x 60 = 9."""
    import models.end2end_model as E
    model, linker, inner, args, ds, fns = stack

    def crafted(temp):
        return [dict(trajectory={f: [10.0, 10.0, 50.0, 50.0] for f in range(20, 27)}, category="dog", score=0.9,
                     begin_fid=20, end_fid=27),
                dict(trajectory={f: [60.0, 60.0, 90.0, 90.0] for f in range(10, 40)}, category="cat", score=0.8,
                     begin_fid=10, end_fid=40)]
    monkeypatch.setattr(E, "format_trajectories_test", crafted)
    r = run_configs(model, linker, args, ds, _batch(), [CONFIGS[0], CONFIGS[2]], fns)
    assert [t["category"] for t in r["R0"]["tracks"]] == ["dog", "cat"]
    assert [t["category"] for t in r["R0w"]["tracks"]] == ["cat"]


def test_predictions_for_refuses_a_desync():
    with pytest.raises(RuntimeError):
        predictions_for(None, None, [{}], [], None, None, None)
    assert predictions_for(None, None, [], [], None, None, None) == {"all": [], "novel": []}


def test_set_fusion_is_what_end2end_reads(args):
    m = type("M", (), {})()
    m.args = args
    set_fusion(m, True)
    assert getattr(args, "uniform_fusion") is True
