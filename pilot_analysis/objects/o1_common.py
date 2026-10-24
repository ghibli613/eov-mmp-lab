"""Object check O1 (A22) -- shared definitions. Training videos only; no test file is read here.

GT tracks are A5's option-B runs (`stage1c/gt_tracks_train.json`, pinned): each annotated training
tid split where more than 30 consecutive frames lack a box, linearly interpolated inside a run. A
run's *annotated* frames are the frames with a real box in `anno/train` (interpolated frames carry
no GT box and are never used for localisation).

Repo functions only: frame IoU = `models.gen_labels.calculate_iou`; vIoU = the evaluator's `viou`.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import json, os, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "pilot_analysis", "stage2"))


class _Lazy:
    """stage 2's common.py (verify_frozen, env_info), imported on first use."""
    def __getattr__(self, name):
        import common
        return getattr(common, name)


C = _Lazy()


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

NOVEL = ("antelope", "bear", "cattle", "fox", "giant_panda", "horse", "lizard", "motorcycle", "red_panda", "watercraft")
OBJ_SPLIT = os.path.join(ROOT, "data", "vidvrd", "data", "openvoc_obj_class_spilt_info.json")
CATS = [json.load(open(OBJ_SPLIT))["id2cls"][str(i)] for i in range(35)]          # the score-vector order
GT_RUNS = os.path.join(ROOT, "pilot_analysis", "stage1c", "gt_tracks_train.json")   # A5, pinned
ANNO_TRAIN = os.path.join(ROOT, "data", "vidvrd", "anno", "train")
FRAG = os.path.join(ROOT, "data", "vidvrd", "data", "train_object_trajectories_gt.json")
TC_MISSED = os.path.join(ROOT, "pilot_analysis", "time_check", "data", "tc_missed.json")
THRESHOLDS = (0.35, 0.2, 0.1)          # 0.35 = the repo's cut-off; 0.2 / 0.1 = M5
LOC_IOU, VIOU_THR, MAJ = 0.5, 0.5, 0.5
OUT = os.path.join(HERE, "data")


def write_log(name, config, inputs, outputs, extra=None):
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    rel = lambda p: os.path.relpath(p, ROOT) if p.startswith(ROOT) else p
    rec = dict(step=name, time=time.strftime("%Y-%m-%d %H:%M:%S"), command=" ".join(sys.argv), config=config,
               env=C.env_info(), code={rel(os.path.join(HERE, f)): sha256(os.path.join(HERE, f))
                                       for f in sorted(os.listdir(HERE)) if f.endswith((".py", ".sh"))},
               inputs={rel(p): sha256(p) for p in inputs}, outputs={rel(p): sha256(p) for p in outputs})
    if extra:
        rec.update(extra)
    json.dump(rec, open(os.path.join(HERE, "logs", f"{name}.json"), "w"), indent=1, default=str)


def annotated_boxes(video):
    """{tid: {fid: [x1, y1, x2, y2]}} from the training annotation (trajectories only)."""
    d = json.load(open(os.path.join(ANNO_TRAIN, f"{video}.json")))
    raw = {}
    for fid, fr in enumerate(d["trajectories"]):
        for e in fr:
            b = e["bbox"]
            raw.setdefault(e["tid"], {})[fid] = [b["xmin"], b["ymin"], b["xmax"], b["ymax"]]
    return raw, {so["tid"]: so["category"] for so in d["subject/objects"]}


_RUNS = None


def gt_runs(video):
    """[{key, gt_tid, category, novel, begin, end, boxes (dense), ann: {fid: box}}] -- option-B runs."""
    global _RUNS
    if _RUNS is None:
        _RUNS = json.load(open(GT_RUNS))
    raw, _ = annotated_boxes(video)
    out = []
    for r in _RUNS[video]:
        b, e = r["begin_fid"], r["end_fid"]
        ann = {f: x for f, x in raw[r["gt_tid"]].items() if b <= f < e}
        out.append(dict(key=(video, r["tid"]), gt_tid=r["gt_tid"], category=r["category"], novel=r["category"] in NOVEL,
                        begin=b, end=e, boxes=r["boxes"], ann=ann))
    return out


def frame_iou(gt_box, boxes):
    """IoU of one GT box with an (N, 4) array of xyxy boxes -- the repo's calculate_iou."""
    import torch
    from models.gen_labels import calculate_iou
    if len(boxes) == 0:
        return np.zeros(0)
    return calculate_iou(torch.tensor([gt_box], dtype=torch.float64), torch.as_tensor(np.asarray(boxes, dtype=np.float64)))[0].numpy()


def viou_cut(track, run):
    """vIoU of a pipeline track, cut to the GT run's extent, with the run (A20's U1 convention).
    track: {begin, end, boxes (dense over [begin, end))}."""
    from inference.video_relation_detection_openvoc import viou
    lo, hi = max(track["begin"], run["begin"]), min(track["end"], run["end"])
    if hi <= lo:
        return 0.0
    return viou(track["boxes"][lo - track["begin"]:hi - track["begin"]], [lo, hi], run["boxes"], [run["begin"], run["end"]])


def viou_full(track, run):
    """vIoU of the whole pipeline track with the run (descriptive only)."""
    from inference.video_relation_detection_openvoc import viou
    return viou(track["boxes"], [track["begin"], track["end"]], run["boxes"], [run["begin"], run["end"]])


def boot_ci(values, groups, n=1000, seed=0):
    """95% percentile CI of a mean over items, resampling whole videos (groups)."""
    values, groups = np.asarray(values, dtype=float), np.asarray(groups)
    if len(values) == 0:
        return [None, None]
    vids = sorted(set(groups.tolist()))
    idx = {v: np.where(groups == v)[0] for v in vids}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        pick = np.concatenate([idx[vids[i]] for i in rng.integers(0, len(vids), len(vids))])
        bs.append(values[pick].mean())
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
