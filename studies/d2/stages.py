"""D2 analysis core (A25, "Definitions"): oracle links, pre-filter tracks, the filter replay,
the stage classifier and the video bootstrap. Pure functions; no file is read here.

vIoU is always the repo's `viou`, called as D1 calls it; per-frame IoU follows the repo's `iou`
(+1 convention). Both come from the pipeline's code root (`d2.common.use_code_root`).
"""
from __future__ import annotations

import collections

import numpy as np

from d2.common import EMPTY, FRAG_SHARE, N_BOOT, BOOT_SEED, THR, UNTRACKED_SHARE


def repo():
    """The repo's own viou / iou and the filter functions (gen_labels)."""
    from inference.video_relation_detection_openvoc import iou, viou
    import models.gen_labels as G
    return viou, iou, G


# --------------------------------------------------------------------------- geometry
def iou_many(boxes: np.ndarray, g) -> np.ndarray:
    """The repo's `iou` (inference/video_relation_detection_openvoc.py:48) of each box with g."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    g = np.asarray(g, dtype=np.float64)
    a1 = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)
    a2 = (g[2] - g[0] + 1) * (g[3] - g[1] + 1)
    ow = np.maximum(0, np.minimum(b[:, 2], g[2]) - np.maximum(b[:, 0], g[0]) + 1)
    oh = np.maximum(0, np.minimum(b[:, 3], g[3]) - np.maximum(b[:, 1], g[1]) + 1)
    ov = ow * oh
    return ov * 1.0 / (a1 + a2 - ov)


def clip_boxes(boxes: np.ndarray, w: float, h: float) -> np.ndarray:
    """end2end_model.py:224-232: x1, y1 below 0 -> 0; x2 above w -> w; y2 above h -> h."""
    b = np.asarray(boxes, dtype=np.float64).copy().reshape(-1, 4)
    b[b[:, 0] < 0, 0] = 0
    b[b[:, 1] < 0, 1] = 0
    b[b[:, 2] > w, 2] = w
    b[b[:, 3] > h, 3] = h
    return b


def is_empty(box) -> bool:
    return box is None or list(box) == EMPTY


# --------------------------------------------------------------------------- detection sets
class DetSets:
    """Per-frame detection sets of one video (A25): raw(s), cut, tracker input."""

    def __init__(self, arr: dict, w: float, h: float):
        fr = arr["det_frame"]
        order = np.lexsort((arr["det_q"], fr))       # by frame, then query index
        self._raw = {}
        for f, idx in _groups(fr[order], order):
            self._raw[f] = (clip_boxes(arr["det_box"][idx], w, h), arr["det_score"][idx].astype(np.float64),
                            arr["det_sel"][idx].astype(bool))
        self._ti = {}
        tfr = arr["ti_frame"]
        for f, idx in _groups(tfr, np.arange(len(tfr))):   # object_instances order
            self._ti[f] = np.asarray(arr["ti_box"][idx], dtype=np.float64)

    def raw(self, f, floor):
        if f not in self._raw:
            return None
        b, s, sel = self._raw[f]
        return b[(s >= floor) | sel]

    def raw_scores(self, f, floor):
        """Cut scores and cut selection of raw(f, floor), in the same order as raw()."""
        if f not in self._raw:
            return None, None
        _, s, sel = self._raw[f]
        keep = (s >= floor) | sel
        return s[keep], sel[keep]

    def cut(self, f):
        if f not in self._raw:
            return None
        b, _, sel = self._raw[f]
        return b[sel]

    def ti(self, f):
        return self._ti.get(f)


def _groups(keys, idx):
    """(key, indices) runs of a key array, in first-appearance order (stable)."""
    out = collections.OrderedDict()
    for k, i in zip(keys.tolist(), idx.tolist()):
        out.setdefault(int(k), []).append(i)
    return [(k, np.array(v, dtype=np.int64)) for k, v in out.items()]


# --------------------------------------------------------------------------- oracle and vIoU
def oracle_choice(get, gt_traj, gb):
    """A25's oracle link: per GT frame, the index (in the set's order) of the detection with the
    highest IoU to the GT box (ties: the first); None if the set is empty or that IoU is 0."""
    idx = []
    for k, g in enumerate(gt_traj):
        d = get(gb + k)
        if is_empty(g) or d is None or len(d) == 0:
            idx.append(None)
            continue
        ious = iou_many(d, g)
        j = int(np.argmax(ious))                      # first maximum
        idx.append(None if ious[j] <= 0 else j)
    return idx


def oracle_track(get, gt_traj, gb):
    """The oracle's track and chosen boxes (None where no box), from oracle_choice."""
    traj, chosen = [], []
    for k, j in enumerate(oracle_choice(get, gt_traj, gb)):
        if j is None:
            traj.append(EMPTY)
            chosen.append(None)
        else:
            box = [float(x) for x in get(gb + k)[j]]
            traj.append(box)
            chosen.append(box)
    return traj, chosen


def oracle_scores(ds, gt_traj, gb, floor):
    """A25c: the cut score (and cut selection) of the raw(floor) oracle's box in each GT frame;
    (None, None) where the oracle has no box."""
    out = []
    for k, j in enumerate(oracle_choice(lambda f: ds.raw(f, floor), gt_traj, gb)):
        if j is None:
            out.append((None, None))
        else:
            s, sel = ds.raw_scores(gb + k, floor)
            out.append((float(s[j]), bool(sel[j])))
    return out


def oracle_viou(get, gt_traj, gb, viou):
    traj, chosen = oracle_track(get, gt_traj, gb)
    ge = gb + len(gt_traj)
    return float(viou(traj, [gb, ge], gt_traj, [gb, ge])), chosen


def cut_viou(begin, boxes, gt_traj, gb, viou):
    """D1's convention (tc_run.py best_viou): the track cut to its overlap with the GT extent."""
    ge = gb + len(gt_traj)
    end = begin + len(boxes)
    lo, hi = max(gb, begin), min(ge, end)
    if hi <= lo:
        return 0.0
    return float(viou(boxes[lo - begin:hi - begin], [lo, hi], gt_traj, [gb, ge]))


def best_viou(tracks: dict, gt_traj, gb, viou) -> dict:
    """D1's best_viou: tid -> cut vIoU, for tracks given as {tid: dict(begin_fid, boxes)}."""
    return {tid: cut_viou(t["begin_fid"], t["boxes"], gt_traj, gb, viou) for tid, t in tracks.items()}


# --------------------------------------------------------------------------- pre-filter tracks
def pre_filter_tracks(temp: np.ndarray) -> dict:
    """A25: the rows of `temp` grouped by track id; x1y1x2y2 as in `temp`; missing frames inside
    filled by np.interp per coordinate; no rounding, no added frames."""
    out = {}
    if len(temp) == 0:
        return out
    for tid, idx in _groups(temp[:, 1].astype(np.int64), np.arange(len(temp))):
        rows = temp[idx]
        fr = rows[:, 0].astype(np.int64)
        o = np.argsort(fr, kind="stable")
        fr, bx = fr[o], rows[o, 2:6]
        full = np.arange(fr[0], fr[-1] + 1)
        interp = np.stack([np.interp(full, fr, bx[:, j]) for j in range(4)], axis=1)
        out[tid] = dict(rows=rows, frames=fr, raw_boxes=bx, begin_fid=int(fr[0]), end_fid=int(fr[-1]) + 1,
                        boxes=interp.tolist())
    return out


def replay_track(G, rows: np.ndarray, min_l: int) -> dict:
    """One track through the repo's own filter functions, in the order of
    format_trajectories_test (gen_labels.py:599-637) and end2end_model.py:319."""
    tf = {}
    for line in rows:
        tf[line[0]] = [line[2], line[3], line[4], line[5]]
    tf = G.add_initial_frames(tf)
    full = sorted(tf)
    out = G.interpolate_and_adjust_frames(tf)
    if not out:
        return dict(rule="F-cov", trimmed=None)
    ks = sorted(out)
    trimmed = bool(ks[0] != int(full[0]) or ks[-1] != int(full[-1]))
    if len(out) + 2 < 12:                              # len(track) >= 12 with 'category', 'score'
        return dict(rule="F-len", trimmed=trimmed, n_frames=len(out))
    if not len(out) > 0.15 * min_l:
        return dict(rule="F-range", trimmed=trimmed, n_frames=len(out))
    return dict(rule=None, trimmed=trimmed, n_frames=len(out), begin_fid=ks[0], end_fid=ks[-1] + 1,
                boxes=[[float(v) for v in out[f]] for f in ks])


def replay_video(G, temp: np.ndarray, min_l: int) -> list[dict]:
    """format_trajectories_test on the recorded rows, then the 15% rule (end2end_model.py:314-323)."""
    if len(temp) == 0:
        return []
    trajs = G.format_trajectories_test([row for row in temp])
    out, tid = [], 0
    for t in trajs:
        if len(t["trajectory"]) > 0.15 * min_l:
            fs = sorted(int(f) for f in t["trajectory"])
            out.append(dict(tid=tid, category=t["category"], score=float(t["score"]),
                            begin_fid=int(t["begin_fid"]), end_fid=int(t["end_fid"]),
                            boxes=[[float(v) for v in t["trajectory"][f]] for f in fs]))
            tid += 1
    return out


# --------------------------------------------------------------------------- fragmentation
def fragmentation(ti_chosen, gb, n_gt, pre: dict, gt_traj) -> dict:
    """A25 O2-frag: good frames = GT frames whose tracker-input oracle box has IoU >= 0.5 with the
    GT; each goes to the pre-filter track whose raw row box there has the highest IoU (>= 0.5) with
    the oracle box. Fragmented iff >= 2 tracks each hold >= 10% of the GT extent's frames."""
    rows_at = collections.defaultdict(list)
    for tid, t in pre.items():
        for f, b in zip(t["frames"].tolist(), t["raw_boxes"]):
            rows_at[f].append((tid, b))
    good, held = 0, collections.Counter()
    for k, box in enumerate(ti_chosen):
        if box is None or iou_many([box], gt_traj[k])[0] < THR:
            continue
        good += 1
        cand = rows_at.get(gb + k, [])
        if not cand:
            continue
        ious = iou_many(np.array([b for _, b in cand]), box)
        j = int(np.argmax(ious))
        if ious[j] >= THR:
            held[cand[j][0]] += 1
    big = [t for t, n in held.items() if n >= FRAG_SHARE * n_gt]
    assigned = sum(held.values())
    return dict(good_frames=good, assigned=assigned, tracks=dict(held), frag=len(big) >= 2,
                untracked=(good == 0) or (assigned < UNTRACKED_SHARE * good))


# --------------------------------------------------------------------------- classifier
ORDER = [("O1a", None), ("O1b", None),
         ("O2", "dedupe"), ("O2", "frag"), ("O2", "other-untracked"), ("O2", "other-tracked"),
         ("O3", "F-cov"), ("O3", "F-len"), ("O3", "F-range"), ("O3", "F-reshape-trim"),
         ("O3", "F-reshape-other"), ("O4", None)]
RANK = {k: i for i, k in enumerate(ORDER)}


def classify_side(o_raw, o_cut, o_ti, best_pre, frag=None, replay=None):
    """The first stage that applies, in A25's order. `best_pre` is the highest pre-filter vIoU,
    `frag` fragmentation()'s result (needed in O2), `replay` replay_track()'s result for the best
    pre-filter track plus its `final_viou` (needed in O3/O4)."""
    if o_raw < THR:
        return ("O1a", None)
    if o_cut < THR:
        return ("O1b", None)
    if best_pre < THR:
        if o_ti < THR:
            return ("O2", "dedupe")
        if frag["frag"]:
            return ("O2", "frag")
        return ("O2", "other-untracked" if frag["untracked"] else "other-tracked")
    if replay["rule"] is not None:
        return ("O3", replay["rule"])
    if replay["final_viou"] < THR:
        return ("O3", "F-reshape-trim" if replay["trimmed"] else "F-reshape-other")
    return ("O4", None)


def relation_label(side_labels: list):
    """The most upstream stage among the missed sides; S0 if none was missed."""
    if not side_labels:
        return ("S0", None)
    return min(side_labels, key=lambda k: RANK[k])


GROUPS = dict(link=("O2", "O3"), rescore=("O1b",), detector=("O1a",))
MODULE = dict(link="relation-aware track linking and keeping",
              rescore="relation-aware rescoring of detections before the cut",
              detector="a detector limit that relations can't recover; the clean-detector step moves up")


# --------------------------------------------------------------------------- bootstrap
def boot_share(mask, vids, seed=BOOT_SEED, n_boot=N_BOOT):
    """D1's boot_share (tc_run.py), unchanged: resample whole videos, percentile 95% CI."""
    mask, vids = np.asarray(mask, float), np.asarray(vids)
    uv = np.unique(vids)
    if not len(uv):
        return (None, None)
    by = {u: np.where(vids == u)[0] for u in uv}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        idx = np.concatenate([by[u] for u in rng.choice(uv, len(uv))])
        bs.append(mask[idx].mean())
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_diff(mask_a, mask_b, vids, seed=BOOT_SEED, n_boot=N_BOOT):
    """Paired: share(a) - share(b) on the same video resamples."""
    a, b, vids = np.asarray(mask_a, float), np.asarray(mask_b, float), np.asarray(vids)
    uv = np.unique(vids)
    if not len(uv):
        return (None, None)
    by = {u: np.where(vids == u)[0] for u in uv}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        idx = np.concatenate([by[u] for u in rng.choice(uv, len(uv))])
        bs.append(a[idx].mean() - b[idx].mean())
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
