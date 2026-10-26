"""RPT G0 adapter unit tests (A28), on synthetic inputs, CPU only: no SAM 3, no CLIP, no checkpoint.

Run from studies/:  python -m pytest rpt/tests/test_adapter.py
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from pycocotools import mask as mask_util

RPT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(RPT))
for p in (RPT, REPO, os.path.join(REPO, "pilot_analysis", "stage1c")):
    if p not in sys.path:
        sys.path.insert(0, p)

import adapter as A  # noqa: E402
import relation as R  # noqa: E402
import sam3_runner as S  # noqa: E402

W, H = 200, 20


def rect(x1, y1, x2, y2):
    """A rectangular mask, inclusive corners (torchvision masks_to_boxes' convention)."""
    m = np.zeros((H, W), np.uint8)
    m[y1:y2 + 1, x1:x2 + 1] = 1
    return m


def rec(f, box, prob=0.8, started=False, matched=True):
    """One output frame of one masklet, as sam3_runner.frame_records writes it."""
    rle = mask_util.encode(np.asfortranarray(rect(*box)))
    return dict(f=f, rle=rle["counts"].decode("ascii"), area=int(mask_util.area(rle)), box=[float(v) for v in box],
                box_sam3=[float(v) for v in box], prob=prob, started=started, matched=matched,
                det=[prob] if (started or matched) else [], trk=0.9)


def video(prompts, n_frames=60):
    """A video as sam3_runner writes it: {class: {obj_id: [frame records]}}."""
    return dict(video="vid", frames=n_frames, width=W, height=H,
                prompts=[dict(prompt=c.replace("_", " "), **{"class": c},
                              masklets=[dict(obj_id=o, frames=fr) for o, fr in objs.items()])
                         for c, objs in prompts.items()])


@pytest.fixture
def cpu_cuda(monkeypatch):
    """The repo's .cuda() calls on the CPU (as a26/tests/conftest.py)."""
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self.clone())


# ---------------------------------------------------------------- 1. gaps, and the ends of a track
def test_internal_gap_is_filled_as_the_released_code_fills_it():
    """Where the released function's own coverage and trim rules do not fire, the two agree exactly."""
    from models.gen_labels import interpolate_and_adjust_frames
    boxes = {f: [10 + 0.37 * f, 5.0, 60 + 0.5 * f, 15 + 0.01 * f] for f in range(100) if not 40 <= f < 50}
    assert A.fill_gaps(boxes) == interpolate_and_adjust_frames(dict(boxes))
    small = A.fill_gaps({0: [0, 0, 10, 10], 3: [1, 0, 11, 10]})                    # 1/3 and 2/3 of a pixel
    assert small == {0: [0, 0, 10, 10], 1: [0, 0, 10, 10], 2: [1, 0, 11, 10], 3: [1, 0, 11, 10]}


def test_gap_filling_drops_nothing():
    """The coverage rule (< 65%) and the 15% gap trim are rule 8, replaced: every frame is kept."""
    from models.gen_labels import interpolate_and_adjust_frames
    boxes = {f: [f, 0, f + 10, 10] for f in list(range(10)) + list(range(30, 40))}   # 50% coverage
    assert interpolate_and_adjust_frames(dict(boxes)) == {}
    assert sorted(A.fill_gaps(boxes)) == list(range(40))


def test_leading_and_trailing_absence_are_not_filled():
    m = dict(obj_id=0, frames=[rec(f, (10, 2, 50, 12), started=(f == 12)) for f in range(12, 21) if f != 16])
    t = A.build_track(m, "dog", 9, W, H)
    assert (t["begin_fid"], t["end_fid"]) == (12, 21)                               # no pre-roll, nothing after
    assert sorted(t["trajectory"]) == list(range(12, 21))
    assert t["status"][16] == A.FILLED and t["status"][12] == A.DETECTED
    assert [f for f, s in t["status"].items() if s == A.FILLED] == [16]


# ---------------------------------------------------------------- 2. identity
def test_a_new_id_is_a_new_track():
    v = video({"dog": {0: [rec(f, (10, 2, 50, 12), started=(f == 0)) for f in range(10)],
                       1: [rec(f, (10, 2, 50, 12), started=(f == 10)) for f in range(10, 20)]}})
    out = A.tracks_from_sam3(v)
    assert [(t["obj_id"], t["begin_fid"], t["end_fid"]) for t in out["all"]] == [(0, 0, 10), (1, 10, 20)]
    assert len(out["repo"]) == 2 and [t["tid"] for t in out["repo"]] == [0, 1]
    assert out["duplicates"] == []                         # same prompt: never compared


# ---------------------------------------------------------------- 3. names
def test_names_map_underscores_to_spaces_and_back():
    o2i = A.object2id()
    assert len(o2i) == 35
    assert A.prompt_of("giant_panda") == "giant panda" and A.prompt_of("domestic_cat") == "domestic cat"
    assert all(A.class_of(A.prompt_of(n)) == n for n in o2i)
    assert S.prompts() == [(n, A.prompt_of(n)) for n in sorted(o2i, key=o2i.get)]   # both sides agree


@pytest.mark.parametrize("bad", ["panda", "Giant Panda", "cat"])
def test_an_unknown_name_raises(bad):
    with pytest.raises(KeyError):
        A.prompt_of(bad)
    with pytest.raises(KeyError):
        A.class_of(bad)


def test_an_unknown_category_fails_the_track_check():
    t = dict(tid=0, category="panda", score=0.9, begin_fid=0, end_fid=2, trajectory={0: [0, 0, 1, 1], 1: [0, 0, 1, 1]})
    with pytest.raises(ValueError, match="unknown category"):
        A.check_repo_tracks([t])
    with pytest.raises(ValueError, match="not dense"):
        A.check_repo_tracks([dict(t, category="dog", trajectory={0: [0, 0, 1, 1]})])


# ---------------------------------------------------------------- 4. duplicates across prompts
@pytest.mark.parametrize("shift,merged", [(17, True), (18, False)])      # IoU 83/117 = 0.709; 82/118 = 0.695
def test_duplicate_merge_just_above_and_below_0_7(shift, merged):
    a = [rec(f, (0, 0, 99, 9), prob=0.95, started=(f == 0)) for f in range(5)]
    b = [rec(f, (shift, 0, shift + 99, 9), prob=0.80, started=(f == 0)) for f in range(8)]   # 3 frames beyond a
    out = A.tracks_from_sam3(video({"red_panda": {0: a}, "giant_panda": {0: b}}))
    iou = (100 - shift) / (100 + shift)
    assert A.mean_mask_iou(*out["all"]) == pytest.approx(iou, abs=1e-6)
    if merged:
        assert [t["category"] for t in out["kept"]] == ["red_panda"]            # the higher score keeps its class
        (d,) = out["duplicates"]
        assert d["kept"][0] == "red_panda" and d["removed"][0] == "giant_panda"
        assert d["shared_frames"] == 5 and d["output_frames_outside_shared"] == 3
    else:
        assert sorted(t["category"] for t in out["kept"]) == ["giant_panda", "red_panda"]
        assert out["duplicates"] == []


# ---------------------------------------------------------------- 5. the score
def test_the_score_averages_detected_frames_only():
    frames = [rec(0, (0, 0, 9, 9), prob=0.8, started=True, matched=False),
              rec(1, (0, 0, 9, 9), prob=0.1, matched=False),                    # propagated: excluded
              rec(2, (0, 0, 9, 9), prob=0.9, matched=True),
              rec(4, (0, 0, 9, 9), prob=0.3, matched=False)]                    # frame 3 is filled
    t = A.build_track(dict(obj_id=3, frames=frames), "dog", 9, W, H)
    assert t["status"] == {0: A.DETECTED, 1: A.PROPAGATED, 2: A.DETECTED, 3: A.FILLED, 4: A.PROPAGATED}
    assert t["values"] == {0: 0.8, 1: 0.1, 2: 0.9, 4: 0.3}                     # stored per frame
    assert t["score"] == pytest.approx(0.85)


def test_a_masklet_without_a_detected_frame_has_no_score_and_is_not_paired():
    m = dict(obj_id=0, frames=[rec(f, (0, 0, 9, 9), matched=False) for f in range(12)])
    out = A.tracks_from_sam3(video({"dog": {0: m["frames"]}}))
    assert out["all"][0]["score"] is None and out["all"][0]["tier"] == "no detected frame"
    assert out["confident"] == [] and out["repo"] == []


# ---------------------------------------------------------------- 6. tiers
@pytest.mark.parametrize("score,tier", [(1.0, "confident"), (0.5, "confident"), (0.4999, "candidate"),
                                        (0.1, "candidate"), (0.0999, "dropped"), (0.0, "dropped")])
def test_tiers_at_the_cut_and_at_0_1(score, tier):
    assert A.tier(score) == tier


# ---------------------------------------------------------------- 7. windows of 40, joined before the merge
def test_a_95_clip_pair_is_windowed_40_40_15_and_joined_before_merging(monkeypatch):
    import cache_wrapper as CW
    from eval_from_cache import meta
    id2pre, _, _ = meta()
    seen = []

    def fake_modelC(m, feats, slen):
        """Logits keyed to each clip's global index (stored in clip_feat[..., 0, 0])."""
        idx = feats["clip_feat"][0, :, 0, 0].long()
        seen.append((int(idx[0]), int(idx[-1]) + 1, slen))
        lg = torch.full((slen, 132), -10.0)
        lg[:, :20] = 0.0                                   # the same 20 predicates in every clip ...
        lg[(idx >= 38) & (idx <= 42), 25] = 10.0           # ... and predicate 25 in clips 38-42 only
        z = lambda *s: torch.zeros(*s)
        return dict(logits=lg, int_logit=torch.full((slen,), 10.0), vpe=z(slen, 3072), spatial=z(slen, 4, 768),
                    cond_mean=z(3072), bias=z(768), sbj_emb=z(768), obj_emb=z(768), pre_scores=z(slen, 132)), 0.0

    monkeypatch.setattr(CW, "run_modelC", fake_modelC)
    clip = torch.zeros(1, 95, 4, 768)
    clip[0, :, 0, 0] = torch.arange(95, dtype=torch.float32)
    item = dict(clip_feat=clip, rel_feat=torch.zeros(1, 95, 42), mot_feat=torch.zeros(1, 95, 42),
                bbox_feat=torch.zeros(1, 95, 4, 24))
    r, _ = CW.run_pair(None, item)
    assert r["windows"] == [(0, 40), (40, 80), (80, 95)] and seen == [(0, 40, 40), (40, 80, 40), (80, 95, 15)]
    assert r["logits"].shape == (95, 132)

    b, n = 100, 95 * 30
    pair = dict(sbj_cls="dog", obj_cls="person", sbj_scr=0.9, obj_scr=0.8, duration=[b, b + n],
                sbj_traj=[[0, 0, 10, 10]] * n, obj_traj=[[5, 5, 15, 15]] * n, logits=r["logits"], int_logit=r["int_logit"])
    joined = [x for x in R.predictions([pair]) if x["triplet"][1] == id2pre[25]]
    assert [x["duration"] for x in joined] == [[b + 38 * 30, b + 43 * 30]]     # one relation across clip 40

    split = []                                                                  # the same, one window at a time
    for a, e in r["windows"]:
        win = dict(pair, duration=[b + 30 * a, b + 30 * e], sbj_traj=pair["sbj_traj"][30 * a:30 * e],
                   obj_traj=pair["obj_traj"][30 * a:30 * e], logits=r["logits"][a:e], int_logit=r["int_logit"][a:e])
        split += [x["duration"] for x in R.predictions([win]) if x["triplet"][1] == id2pre[25]]
    assert sorted(split) == [[b + 38 * 30, b + 40 * 30], [b + 40 * 30, b + 43 * 30]]   # what joining prevents


# ---------------------------------------------------------------- 8. a filled middle frame is flagged
def test_a_gap_on_a_clip_middle_frame_is_flagged(tmp_path, monkeypatch, cpu_cuda):
    from utils import paths
    vid = "vid"
    (tmp_path / "anno" / "train").mkdir(parents=True)
    json.dump(dict(video_id=vid, width=W, height=H), open(tmp_path / "anno" / "train" / f"{vid}.json", "w"))
    frames = tmp_path / "frames" / vid
    frames.mkdir(parents=True)
    from PIL import Image
    for i in range(60):
        Image.new("RGB", (W, H)).save(frames / f"{i + 1:06d}.jpg")
    monkeypatch.setattr(paths, "ANNO_DIR", str(tmp_path / "anno"))
    monkeypatch.setattr(paths, "FRAME_DIR", str(tmp_path / "frames"))

    sub = dict(obj_id=0, frames=[rec(f, (0, 0, 49, 9), started=(f == 0)) for f in range(60) if not 14 <= f <= 16])
    obj = dict(obj_id=0, frames=[rec(f, (100, 0, 149, 9), started=(f == 0)) for f in range(60)])
    out = A.tracks_from_sam3(video({"dog": {0: sub["frames"]}, "person": {0: obj["frames"]}}))
    calls = []
    ff = R.FrameFeatures(vid, str(tmp_path / "frames"),
                         encoder=lambda p: (calls.append(p), (torch.zeros(1, 768), torch.zeros(1, 576, 768)))[1])

    def fake_run_pair(m, it):
        n = it["clip_feat"].shape[1]
        return dict(logits=torch.zeros(n, 132), int_logit=torch.zeros(n), windows=[(0, n)], slot_inputs={}), 0.0

    pairs = R.relation_head(vid, out["repo"], "train", ff, None, run_pair=fake_run_pair)
    assert [(p["sub"], p["obj"], p["mid_frames"]) for p in pairs] == [(0, 1, [15, 45]), (1, 0, [15, 45])]
    assert A.gap_exposure(pairs, [t["status"] for t in out["confident"]]) == [[True, False], [True, False]]
    assert len(calls) == 2                                  # two distinct middle frames, each encoded once
    assert [k for k, _ in ff.log] == ["patch", "patch", "patch", "global"] * 4


def test_frame_features_answers_only_one_frame_reads(tmp_path):
    from PIL import Image
    (tmp_path / "v").mkdir()
    for i in range(3):
        Image.new("RGB", (8, 6)).save(tmp_path / "v" / f"{i + 1:06d}.jpg")
    ff = R.FrameFeatures("v", str(tmp_path), encoder=lambda p: (torch.zeros(1, 768), torch.zeros(1, 576, 768)))
    assert ff.size == (8, 6)
    assert ff.patch_proj[0][2:3, :, :].shape == (1, 576, 768) and ff.global_proj[0][2:3, :].shape == (1, 768)
    for bad in (slice(-2, -1), slice(3, 4)):
        with pytest.raises(IndexError):
            ff.global_proj[0][bad, :]
    with pytest.raises(IndexError):
        ff.patch_proj[0][0:2, :, :]


# ---------------------------------------------------------------- the SAM 3 side's records
def test_recorder_passes_through_and_records_starts_and_matches():
    plan = dict(new_det_obj_ids=np.array([5]), new_det_fa_inds=np.array([1]), obj_ids_newly_removed={2},
                det_to_matched_trk_obj_ids={0: np.array([3]), 2: np.array([3, 4])})
    meta = object()
    m = FakeModel(planning=lambda **kw: (plan, meta))
    rec_ = S.Recorder(m)
    kw = dict(frame_idx=7, det_out=dict(scores=torch.tensor([0.6, 0.9, 0.55])),
              tracker_metadata_prev=dict(obj_ids_all_gpu=np.array([3, 4])), tracker_obj_scores_global=torch.tensor([0.0, 2.0]))
    assert m.run_tracker_update_planning_phase(**kw) == (plan, meta)                  # untouched
    fr = rec_.frames[7]
    assert fr["started"] == {5: pytest.approx(0.9)} and fr["removed"] == [2]
    assert fr["matched"] == {3: [pytest.approx(0.6), pytest.approx(0.55)], 4: [pytest.approx(0.55)]}
    assert fr["tracker"] == {3: pytest.approx(0.5), 4: pytest.approx(1 / (1 + np.exp(-2.0)))}
    rec_.close()
    assert not {"run_tracker_update_planning_phase"} & set(vars(m))
    assert not {"_build_multigpu_buffer_next_chunk", "forward_grounding"} & set(vars(m.detector))


class FakeDetector:
    """The two detector methods Recorder wraps, as sam3_image.py:822-905 calls them: the chunk
    builder calls forward_grounding, then lowers the logits of detections <= 0.5 in place (NMS)."""

    def __init__(self, logits, boxes):
        self.logits, self.boxes, self.backbone = logits, boxes, None

    def forward_grounding(self, **kw):
        return dict(pred_logits=self.logits.clone(), pred_boxes_xyxy=self.boxes.clone())

    def _build_multigpu_buffer_next_chunk(self, frame_idx_begin, **kw):
        out = self.forward_grounding()
        out["pred_logits"][0, :, 0] -= 1e4 * (out["pred_logits"][0, :, 0].sigmoid() <= 0.5).float()
        self.last = out
        return None


class FakeModel:
    def __init__(self, planning=None, detector=None):
        self._planning = planning
        self.detector = detector or FakeDetector(torch.zeros(1, 1, 1), torch.zeros(1, 1, 4))

    def run_tracker_update_planning_phase(self, **kw):
        return self._planning(**kw)


def test_recorder_keeps_the_raw_detections_in_0_1_to_0_5_from_before_nms():
    probs = torch.tensor([0.05, 0.11, 0.3, 0.49, 0.51, 0.8])
    logits = torch.log(probs / (1 - probs)).reshape(1, 6, 1)
    boxes = torch.tensor([[[0.1 * i, 0.0, 0.1 * i + 0.1, 0.5] for i in range(6)]])
    det = FakeDetector(logits, boxes)
    rec_ = S.Recorder(FakeModel(detector=det))
    det._build_multigpu_buffer_next_chunk(frame_idx_begin=4, frame_idx_end=5)
    assert det.last["pred_logits"][0, 2, 0] < -1000                                     # NMS did lower them
    assert sorted(rec_.raw_low) == [4]
    got = rec_.raw_low[4]
    assert [round(d[4], 4) for d in got] == [0.11, 0.3, 0.49]                           # [0.1, 0.5) only
    assert got[0][:4] == pytest.approx([0.1, 0.0, 0.2, 0.5])
    assert rec_.chunk_frame is None
    det.forward_grounding()                                                              # outside a chunk: nothing
    assert sorted(rec_.raw_low) == [4]


def test_backbone_cache_returns_copies_with_the_same_sharing():
    calls = []

    class Backbone:
        def forward_image(self, samples):
            calls.append(samples)
            v = samples.mean() + torch.arange(4.0)
            return {"vision_features": v, "backbone_fpn": [v * 2, v], "vision_pos_enc": [torch.ones(3)],
                    "sam2_backbone_out": {"vision_pos_enc": [torch.ones(3)]}}

    det = FakeDetector(torch.zeros(1, 1, 1), torch.zeros(1, 1, 4))
    det.backbone = Backbone()
    rec_ = S.Recorder(FakeModel(detector=det))
    cache = S.BackboneCache(det, rec_)
    img = {0: torch.zeros(1, 3, 2, 2), 1: torch.ones(1, 3, 2, 2)}
    outs = []
    for f in (0, 1, 0, 1):
        rec_.chunk_frame = f
        outs.append(det.backbone.forward_image(img[f]))
    rec_.chunk_frame = None
    det.backbone.forward_image(img[0])                                                   # no frame: passes through
    assert (cache.misses, cache.hits, cache.passthrough) == (2, 2, 1) and len(calls) == 3
    first, again = outs[0], outs[2]
    assert torch.equal(first["vision_features"], again["vision_features"])
    assert torch.equal(first["backbone_fpn"][0], again["backbone_fpn"][0])
    assert again["backbone_fpn"][1] is again["vision_features"]                          # sharing kept
    assert again["vision_features"] is not first["vision_features"]                      # a copy, not the original
    assert cache.store[0]["vision_pos_enc"][0] is cache.store[1]["vision_pos_enc"][0]    # equal pos enc kept once
    cache.close()
    assert "forward_image" not in vars(det.backbone) and cache.store == {}


def test_sam3_compare_finds_identical_and_different_runs():
    sys.path.insert(0, os.path.join(RPT, "g0"))
    import sam3_compare as C
    fr = [rec(f, (0, 0, 9, 9)) for f in range(3)]
    run = dict(video="v", prompts=[dict(**{"class": "dog"}, seconds=10.0, raw_low=[[0, 1, 2, 3, 4, 0.2]],
                                        masklets=[dict(obj_id=0, frames=fr)])])
    fast = json.loads(json.dumps(run))
    fast["prompts"][0]["seconds"] = 4.0
    r = C.compare(run, fast)
    assert r["bitwise_identical"] and r["speedup_a_over_b"] == 2.5
    other = json.loads(json.dumps(fast))
    other["prompts"][0]["masklets"][0]["frames"][1]["prob"] = 0.81
    other["prompts"][0]["raw_low"] = []
    r = C.compare(run, other)
    assert not r["bitwise_identical"] and r["differences"] == {"frame.prob": 1, "prompt.raw_low": 1}


def test_frame_records_keep_non_empty_masks_with_their_tight_box():
    masks = np.zeros((2, H, W), bool)
    masks[0, 2:5, 10:21] = True                                                      # obj 3; obj 4 empty
    out = dict(frame_index=7, outputs=dict(out_obj_ids=np.array([3, 4]), out_probs=np.array([0.91, 0.8]),
                                           out_boxes_xywh=np.array([[0.05, 0.1, 0.05, 0.15], [0, 0, 0, 0]]),
                                           out_binary_masks=masks))

    class Rec:
        frames = {7: dict(started={}, matched={3: [0.7]}, tracker={3: 0.95})}
        started = {3: (2, 0.88)}
    r = S.frame_records(out, Rec(), W, H)
    assert list(r) == [3]
    assert r[3]["box"] == [10.0, 2.0, 20.0, 4.0] and r[3]["area"] == 33
    assert r[3]["box_sam3"] == pytest.approx([10.0, 2.0, 20.0, 5.0])
    assert r[3]["prob"] == pytest.approx(0.91) and r[3]["matched"] and r[3]["det"] == [0.7]
    assert r[3]["started"] is False                                     # started on frame 2, not 7
    Rec.started = {3: (7, 0.88)}                                        # started by add_prompt on this frame,
    r = S.frame_records(out, Rec(), W, H)                               # re-run as a match by propagation
    assert r[3]["started"] is True and r[3]["det"] == [0.7]             # the start is not counted twice


def test_no_test_labels_guard():
    import no_test_labels as G
    assert G.blocked("/x/data/vidvrd/anno/test/v.json") and G.blocked("/x/data/vidvrd/data/test_relation_gt.json")
    assert G.blocked("/s/cache_s3/test/v.pt") and not G.blocked("/s/cache_s3/train/v.pt")
    assert not G.blocked("/x/data/vidvrd/anno/train/v.json") and not G.blocked(3)
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import no_test_labels as G; G.install()\n"
            "try:\n    open('/nonexistent/anno/test/v.json')\nexcept PermissionError: print('blocked')\n")
    assert subprocess.run([sys.executable, "-c", code, RPT], capture_output=True, text=True).stdout.strip() == "blocked"
