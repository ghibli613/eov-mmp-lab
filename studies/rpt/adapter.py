"""RPT adapter (A28): SAM 3's masklets -> the tracks gen_feats_test accepts.

Reads what studies/rpt/sam3_runner.py wrote; never imports SAM 3. The repo's own code is called,
never copied: round_and_positive (models/gen_labels.py:426) in the gap filling, gen_feats_test for
the pairs, session 3's run_pair for the windows (studies/rpt/relation.py).

The rules, as A28 fixes them:
- prompts: the 35 names of object2id.json, '_' -> ' ', nothing else; a track keeps its prompt's name;
- identity: one (prompt, SAM 3 ID) is one track; nothing is split;
- boxes: the tight box of SAM 3's returned mask on every frame it returned a non-empty mask for the
  ID (the "output frames"); SAM 3's own box is kept beside it;
- gaps (rule 1): frames between the first and the last output frame are filled as the released
  code does (gen_labels.py:416-423); nothing before or after (no pre-roll);
- status per frame: detected (a detection started or matched the masklet there), propagated
  (output, no detection), filled (a gap frame);
- score (rule 2): the mean of the per-frame value, SAM 3's out_probs, over detected frames;
- tiers (rule 5): confident >= 0.5, candidate >= 0.1, else dropped;
- duplicates across prompts: mean mask IoU over shared output frames > 0.7; the higher-scoring
  masklet is kept unchanged and the other removed, greedily by descending score.
"""
from __future__ import annotations

import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from models.gen_labels import round_and_positive  # noqa: E402
from utils import paths  # noqa: E402

CONFIDENT_CUT = 0.5          # rule 5: SAM 3 keeps a detection above 0.5
CANDIDATE_FLOOR = 0.1        # proposal §3.1 (D3)
DUPLICATE_IOU = 0.7          # proposal §3.1; G0 step 7
DETECTED, PROPAGATED, FILLED = "detected", "propagated", "filled"


# ---------------------------------------------------------------- names
def object2id(path=None) -> dict:
    return json.load(open(path or os.path.join(paths.META_DIR, "object2id.json")))


def prompt_of(name: str) -> str:
    """The text prompt of a VidVRD class: '_' becomes a space. An unknown name raises KeyError."""
    if name not in object2id():
        raise KeyError(f"not one of the 35 VidVRD classes: {name!r}")
    return name.replace("_", " ")


def class_of(prompt: str) -> str:
    """The VidVRD class of a text prompt. An unknown prompt raises KeyError."""
    back = {n.replace("_", " "): n for n in object2id()}
    if prompt not in back:
        raise KeyError(f"not one of the 35 prompts: {prompt!r}")
    return back[prompt]


# ---------------------------------------------------------------- one masklet -> one track
def fill_gaps(boxes: dict) -> dict:
    """Rule 1, gen_labels.py:416-423: each coordinate linearly interpolated over the output frames,
    on every frame of [first, last], then the repo's round_and_positive on every frame. The
    coverage rule, the gap trim and the length filter of the same function are not applied (rule 8)."""
    fs = sorted(int(f) for f in boxes)
    ids = np.arange(fs[0], fs[-1] + 1)
    cols = [np.interp(ids, fs, [boxes[f][c] for f in fs]) for c in range(4)]
    return {int(f): round_and_positive(b) for f, b in zip(ids, zip(*cols))}


def frame_status(record: dict) -> str:
    return DETECTED if (record["started"] or record["matched"]) else PROPAGATED


def track_score(values: dict, status: dict) -> float:
    """Rule 2: the mean per-frame value over detected frames only."""
    det = [values[f] for f in sorted(values) if status[f] == DETECTED]
    if not det:
        raise ValueError("no detected frame: the score is undefined (A28: reported, not paired)")
    return float(np.mean(det))


def tier(score: float) -> str:
    if score >= CONFIDENT_CUT:
        return "confident"
    if score >= CANDIDATE_FLOOR:
        return "candidate"
    return "dropped"


def build_track(masklet: dict, prompt_class: str, prompt_id: int, width: int, height: int) -> dict:
    """One SAM 3 masklet (sam3_runner's record) -> one track, with its per-frame status and values."""
    rec = {int(r["f"]): r for r in masklet["frames"]}
    traj = fill_gaps({f: r["box"] for f, r in rec.items()})
    status = {f: frame_status(rec[f]) if f in rec else FILLED for f in traj}
    values = {f: float(r["prob"]) for f, r in rec.items()}
    t = dict(prompt_id=prompt_id, obj_id=int(masklet["obj_id"]), category=prompt_class,
             begin_fid=min(traj), end_fid=max(traj) + 1, trajectory=traj, status=status, values=values,
             box_sam3={f: r["box_sam3"] for f, r in rec.items()}, masks={f: r["rle"] for f, r in rec.items()},
             det_probs={f: list(r["det"]) for f, r in rec.items()},
             tracker={f: r["trk"] for f, r in rec.items()}, size=(height, width))
    try:
        t["score"] = track_score(values, status)
        t["tier"] = tier(t["score"])
    except ValueError:
        t["score"], t["tier"] = None, "no detected frame"
    return t


# ---------------------------------------------------------------- duplicates across prompts
def _rle(counts: str, size) -> dict:
    return {"size": [int(size[0]), int(size[1])], "counts": counts.encode("ascii")}


def mean_mask_iou(a: dict, b: dict):
    """Mean per-frame mask IoU over the frames where both masklets have an output mask; None if none."""
    from pycocotools import mask as mask_util
    shared = sorted(set(a["masks"]) & set(b["masks"]))
    if not shared:
        return None
    ious = [mask_util.iou([_rle(a["masks"][f], a["size"])], [_rle(b["masks"][f], b["size"])], [0])[0][0]
            for f in shared]
    return float(np.mean(ious))


def deduplicate(tracks: list) -> tuple[list, list]:
    """Across prompts, a mean mask IoU > 0.7 makes two masklets duplicates. A28's reading of "merged,
    the higher score keeps its class": the higher-scoring masklet is kept unchanged, the other
    removed; greedily in descending score (ties: prompt id, then SAM 3 ID). Same-prompt masklets are
    SAM 3's own business and never compared. -> (kept, removed records)."""
    order = sorted((t for t in tracks if t["score"] is not None),
                   key=lambda t: (-t["score"], t["prompt_id"], t["obj_id"]))
    kept, removed = [], []
    for t in order:
        hit = None
        for k in kept:
            if k["prompt_id"] == t["prompt_id"]:
                continue
            iou = mean_mask_iou(k, t)
            if iou is not None and iou > DUPLICATE_IOU:
                hit = (k, iou)
                break
        if hit is None:
            kept.append(t)
            continue
        k, iou = hit
        shared = set(k["masks"]) & set(t["masks"])
        removed.append(dict(removed=(t["category"], t["obj_id"], t["score"]), kept=(k["category"], k["obj_id"], k["score"]),
                            mean_iou=iou, shared_frames=len(shared),
                            output_frames_outside_shared=len(set(t["masks"]) - shared)))
    kept += [t for t in tracks if t["score"] is None]
    return kept, removed


# ---------------------------------------------------------------- the repo's track format
def repo_tracks(tracks: list) -> list:
    """format_trajectories_test's schema (gen_labels.py:623-637), int frame keys; tid in list order."""
    out = []
    for tid, t in enumerate(tracks):
        out.append(dict(tid=tid, category=t["category"], score=float(t["score"]), begin_fid=int(t["begin_fid"]),
                        end_fid=int(t["end_fid"]), trajectory={int(f): list(b) for f, b in t["trajectory"].items()}))
    check_repo_tracks(out)
    return out


def check_repo_tracks(tracks: list) -> None:
    """What gen_feats_test needs (studies/rpt/code_check.md §1): a dense trajectory over [begin_fid,
    end_fid) with 4-number boxes, one of the 35 names, a float score. Raises ValueError otherwise."""
    names = object2id()
    for t in tracks:
        if t["category"] not in names:
            raise ValueError(f"track {t.get('tid')}: unknown category {t['category']!r}")
        if not isinstance(t["score"], float) or not np.isfinite(t["score"]):
            raise ValueError(f"track {t.get('tid')}: score {t['score']!r} is not a finite float")
        if not t["begin_fid"] < t["end_fid"] or set(t["trajectory"]) != set(range(t["begin_fid"], t["end_fid"])):
            raise ValueError(f"track {t.get('tid')}: trajectory is not dense over [{t['begin_fid']}, {t['end_fid']})")
        if any(len(b) != 4 for b in t["trajectory"].values()):
            raise ValueError(f"track {t.get('tid')}: a box is not [x1, y1, x2, y2]")


# ---------------------------------------------------------------- a whole video
def load_sam3(path: str) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def tracks_from_sam3(sam3_video: dict) -> dict:
    """All of one video's masklets -> tracks; duplicates; tiers. -> dict with every track (for the
    report), the removed duplicates, and the confident tracks in the repo's format (for pairing)."""
    o2i = object2id()
    W, H = sam3_video["width"], sam3_video["height"]
    tracks = []
    for p in sam3_video["prompts"]:
        name = class_of(p["prompt"])
        if name != p["class"]:
            raise ValueError(f"prompt {p['prompt']!r} maps to {name!r}, the runner wrote {p['class']!r}")
        for m in p["masklets"]:
            tracks.append(build_track(m, name, o2i[name], W, H))
    kept, removed = deduplicate(tracks)
    confident = sorted((t for t in kept if t["tier"] == "confident"), key=lambda t: (t["prompt_id"], t["obj_id"]))
    return dict(all=tracks, kept=kept, duplicates=removed, confident=confident, repo=repo_tracks(confident))


# ---------------------------------------------------------------- gap exposure
def gap_exposure(pairs: list, statuses: list) -> list:
    """Per clip, whether its middle frame is a filled frame of its subject or object.
    pairs: relation.relation_head's records (sub, obj indices and the clips' middle frames, as
    gen_feats_test requested them); statuses: per track, in the same order, {frame: status}."""
    flags = []
    for p in pairs:
        s, o = statuses[p["sub"]], statuses[p["obj"]]
        flags.append([s.get(f) == FILLED or o.get(f) == FILLED for f in p["mid_frames"]])
    return flags
