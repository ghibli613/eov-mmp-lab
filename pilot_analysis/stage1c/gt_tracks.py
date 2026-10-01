#!/usr/bin/env python3
"""Stage 1 session 3, Task 1 -- whole GT tracks for train and test, and test ECC check.

    ~/miniconda3/envs/repro-next/bin/python pilot_analysis/stage1c/gt_tracks.py

CPU only. TEST: one track per annotated tid, densified with the repo's own
add_initial_frames + interpolate_and_adjust_frames (models/gen_labels.py:363,
373) -- the path detected trajectories take in format_trajectories_test.
TRAIN: option B (preregistration amendment A5) -- each tid split where more than
30 consecutive frames lack a box, every run its own track, gaps <= 30 linearly
interpolated, every annotated box kept, no pre-roll (optionB.runs_option_b).
The repo densifier rejects 33.5% of train tracks (< 65% coverage); its option-A
output is kept as gt_tracks_train_optionA.json for the record.
The fragment file train_object_trajectories_gt.json is not used.

Test annotations: BOXES AND CATEGORIES ONLY. `relation_instances` is deleted
from each parsed file before anything else touches it; relation labels are read
only by the official evaluator.

Output (gt_tracks_{train,test}.json): {video: [{tid, gt_tid, category, score,
begin_fid, end_fid, boxes}]}, `boxes` dense over [begin_fid, end_fid).
`load_tracks()` returns format_trajectories_test's schema with INT frame keys.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import glob, json, math, os
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from models.gen_labels import add_initial_frames, interpolate_and_adjust_frames  # noqa: E402
sys.path.insert(0, HERE)
from optionB import runs_option_b  # noqa: E402

CLIP_LEN, MIN_OVERLAP, MIN_TAIL, MAX_SLOTS = 30, 10, 10, 40
ANNO = {s: os.path.join(ROOT, "data", "vidvrd", "anno", s) for s in ("train", "test")}
ECC = {s: os.path.join(ROOT, "data", "vidvrd", "data", f"VidVRD_ECC_{s}.json") for s in ("train", "test")}
FRAMES = os.path.join(ROOT, "data", "vidvrd", "frames")


def read_boxes(path):
    d = json.load(open(path))
    d.pop("relation_instances", None)          # never used; test labels stay with the evaluator
    boxes = {}
    for fid, frame in enumerate(d["trajectories"]):
        for e in frame:
            b = e["bbox"]
            boxes.setdefault(e["tid"], {})[fid] = [b["xmin"], b["ymin"], b["xmax"], b["ymax"]]
    cats = {so["tid"]: so["category"] for so in d["subject/objects"]}
    return d["video_id"], boxes, cats


def densify(raw):
    """-> ({int fid: box}, stats). Mirrors format_trajectories_test (models/
    gen_labels.py:607-611): pre-roll, then interpolate unless a single frame."""
    t = add_initial_frames(dict(raw))
    t = interpolate_and_adjust_frames(t) if len(t) > 1 else {f: [max(0, round(c)) for c in b] for f, b in t.items()}
    return {int(f): b for f, b in t.items()}


def pair_slots(b, e):
    if e - b < MIN_OVERLAP:
        return 0
    n, tail = (e - b) // CLIP_LEN, (e - b) % CLIP_LEN
    return n + 1 if tail >= MIN_TAIL else n


def build(split, mode="B"):
    out, st = {}, Counter()
    long_pairs = []
    for f in sorted(glob.glob(os.path.join(ANNO[split], "*.json"))):
        vid, boxes, cats = read_boxes(f)
        tracks = []
        for tid in sorted(boxes):
            raw = boxes[tid]
            st["tracks_annotated"] += 1
            if split == "train" and mode == "B":
                for run, (lo, dense) in enumerate(runs_option_b(raw)):
                    hi = max(dense) + 1
                    tracks.append(dict(tid=len(tracks), gt_tid=tid, run=run, category=cats[tid], score=1.0,
                                       begin_fid=lo, end_fid=hi, boxes=[dense[x] for x in range(lo, hi)]))
                continue
            dense = densify(raw)
            if not dense:
                st["tracks_rejected_65pct_rule"] += 1
                continue
            lo, hi = min(dense), max(dense) + 1
            st["frames_dropped_by_gap_rule"] += len([x for x in raw if not lo <= x < hi])
            st["tracks_with_preroll_below_0"] += lo < 0
            tracks.append(dict(tid=len(tracks), gt_tid=tid, category=cats[tid], score=1.0,
                               begin_fid=lo, end_fid=hi, boxes=[dense[x] for x in range(lo, hi)]))
        out[vid] = tracks
        st["videos"] += 1
        st["tracks"] += len(tracks)
        for s in tracks:
            for o in tracks:
                if s is o:
                    continue
                n = pair_slots(max(s["begin_fid"], o["begin_fid"]), min(s["end_fid"], o["end_fid"]))
                if n:
                    st["pairs"] += 1
                    st["slots"] += n
                    if n > MAX_SLOTS:
                        st["pairs_over_40_slots"] += 1
                        long_pairs.append((vid, s["tid"], o["tid"], n))
    suffix = "_optionA" if (split == "train" and mode == "A") else ""
    json.dump(out, open(os.path.join(HERE, f"gt_tracks_{split}{suffix}.json"), "w"))
    return st, long_pairs


def load_tracks(split):
    """format_trajectories_test's schema, with int frame keys."""
    d = json.load(open(os.path.join(HERE, f"gt_tracks_{split}.json")))
    return {v: [dict(tid=t["tid"], gt_tid=t["gt_tid"], category=t["category"], score=t["score"],
                     begin_fid=t["begin_fid"], end_fid=t["end_fid"],
                     trajectory={t["begin_fid"] + i: b for i, b in enumerate(t["boxes"])})
                for t in ts] for v, ts in d.items()}


def check_ecc(split, video_ids):
    """Coverage and key convention of the ECC file; plus the box-residual test
    of stage 0c (which key offset best explains box motion) on BOXES only."""
    ecc = json.load(open(ECC[split]))
    missing = sorted(v for v in video_ids if v not in ecc)
    conv = Counter()
    for v in video_ids:
        if v in ecc:
            n = len(os.listdir(os.path.join(FRAMES, v)))
            ks = sorted(int(k) for k in ecc[v])
            conv[(ks[0], ks[-1] - n, len(ks) - n)] += 1
    res = {k: [] for k in ("none", "key t+1", "key t", "inverse key t+1")}
    for f in sorted(glob.glob(os.path.join(ANNO[split], "*.json")))[:120]:
        vid, boxes, _ = read_boxes(f)
        if vid not in ecc:
            continue
        d = json.load(open(f)); d.pop("relation_instances", None); W, H = d["width"], d["height"]; d.clear()
        diag = math.hypot(W, H)
        for tid, bx in boxes.items():
            for t in sorted(bx):
                if t - 1 not in bx:
                    continue
                c0 = np.array([(bx[t-1][0] + bx[t-1][2]) / 2, (bx[t-1][1] + bx[t-1][3]) / 2, 1.0])
                c1 = np.array([(bx[t][0] + bx[t][2]) / 2, (bx[t][1] + bx[t][3]) / 2])
                res["none"].append(np.linalg.norm(c1 - c0[:2]) / diag)
                for name, k, inv in (("key t+1", t + 1, False), ("key t", t, False), ("inverse key t+1", t + 1, True)):
                    m = ecc[vid].get(str(k))
                    if m is None:
                        continue
                    m = np.array(m)
                    if inv:
                        m = np.linalg.inv(m)
                    res[name].append(np.linalg.norm(c1 - (m @ c0)[:2]) / diag)
    return missing, dict(conv), {k: (float(np.median(v)) if v else None, len(v)) for k, v in res.items()}


def main():
    report = {}
    st_a, _ = build("train", mode="A")
    report_a = dict(stats=dict(st_a))
    for split in ("train", "test"):
        st, longp = build(split)
        vids = [os.path.basename(f)[:-5] for f in glob.glob(os.path.join(ANNO[split], "*.json"))]
        missing, conv, resid = check_ecc(split, vids)
        report[split] = dict(stats=dict(st), pairs_over_40=longp, ecc_missing=missing,
                             ecc_key_convention=[{"min_key": a, "max_key_minus_frames": b,
                                                  "n_keys_minus_frames": c, "videos": n}
                                                 for (a, b, c), n in conv.items()],
                             ecc_residual_median=resid)
    report["train_optionA_record"] = report_a
    json.dump(report, open(os.path.join(HERE, "gt_tracks_report.json"), "w"), indent=1)
    for s, r in ((k, v) for k, v in report.items() if k in ("train", "test")):
        print(s, json.dumps(r["stats"]), "| >40-slot pairs:", len(r["pairs_over_40"]),
              "| ECC missing:", len(r["ecc_missing"]), "| key convention:", r["ecc_key_convention"],
              "| residual medians:", {k: round(v[0] * 1000, 3) if v[0] else None for k, v in r["ecc_residual_median"].items()})
    return report


if __name__ == "__main__":
    main()
