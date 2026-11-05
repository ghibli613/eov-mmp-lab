#!/usr/bin/env python3
"""The D7 check's videos and guard (A34), on the laptop, from G1's committed files, before any rental.

    python studies/d7/check/select_videos.py      # -> studies/d7/check/video_list.json (+ .sha256)

Order: numpy.random.default_rng(20261103).permutation over the sorted HELD-A list (A31); the first 20
videos in that order that have G1 SAM 3 files; no length exclusion. The guard: the 35-name SAM 3's
class-agnostic object-recall gain over R0w (A29 run recall, points) on those 20; below +5 -> the first 40;
still below +5 on 40 (or <= 0) -> stop and report. Exit 0 with a list, exit 2 on the stop.
The HELD-B guard (and the test-label guard) is installed first.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
D7 = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(D7))
for p in (D7, os.path.join(REPO, "studies", "rpt"), os.path.join(REPO, "studies", "rpt", "g1"), REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
import held_b_guard  # noqa: E402
held_b_guard.install()

import numpy as np  # noqa: E402

SEED = 20261103
GUARD_POINTS = 5.0
SAM3_DIR, EOV_DIR = "studies/rpt/g1/sam3_out", "studies/rpt/g1/eov_out"
EXTRACTS = "studies/d7/extracts"
OPT_B = f"{EXTRACTS}/pilot_analysis__stage1c__gt_tracks_train.held_a.json"
HELD_GT = f"{EXTRACTS}/pilot_analysis__stage2__data__held_gt.held_a.json"
HELD_GT3 = f"{EXTRACTS}/studies__rpt__g1__held_gt_missing3.held_a.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ordered(held_a, has_file, seed=SEED):
    """-> the HELD-A videos with SAM 3 files, in the rng order (pure; tested)."""
    order = [str(x) for x in np.random.default_rng(seed).permutation(sorted(held_a))]
    return order, [v for v in order if has_file(v)]


def decide(gain20, gain40=None):
    """-> ('list20' | 'list40' | 'stop', reason) (pure; tested)."""
    if gain20 >= GUARD_POINTS:
        return "list20", f"gain on 20 = {gain20:+.2f} >= +5"
    if gain40 is None:
        return "need40", f"gain on 20 = {gain20:+.2f} < +5"
    if gain40 >= GUARD_POINTS and gain40 > 0:
        return "list40", f"gain on 20 = {gain20:+.2f} < +5; on 40 = {gain40:+.2f} >= +5"
    return "stop", f"gain on 20 = {gain20:+.2f} and on 40 = {gain40:+.2f}: still below +5 (or <= 0); the check cannot discriminate"


def agnostic_gain(videos, ctx):
    import adapter as A
    import metrics as M
    opt_b, rels, novel, anno_dir = ctx
    import gzip
    runs, rec_s, rec_e = {}, {}, {}
    for v in videos:
        anno = json.load(open(os.path.join(anno_dir, f"{v}.json")))
        runs[v] = M.runs_of(opt_b[v], anno)
        s3 = json.load(gzip.open(os.path.join(SAM3_DIR, f"{v}.json.gz"), "rt"))
        e = json.load(gzip.open(os.path.join(EOV_DIR, f"{v}.json.gz"), "rt"))
        rec_s[v] = M.video_record(M.tracks_from_list(A.tracks_from_sam3(s3)["repo"]), runs[v], rels[v])
        rec_e[v] = M.video_record(M.tracks_from_list(e["configs"]["R0w"]["tracks"]), runs[v], rels[v])
    ps, pe = M.pooled(rec_s, runs, rels, novel, videos), M.pooled(rec_e, runs, rels, novel, videos)
    a, b = ps["run_recall_agnostic"]["all"], pe["run_recall_agnostic"]["all"]
    return 100 * (a["recall"] - b["recall"]), dict(sam3=[a["hit"], a["n"]], R0w=[b["hit"], b["n"]])


def frames_of(videos):
    from utils import paths
    return {v: len(os.listdir(os.path.join(paths.FRAME_DIR, v))) for v in videos}


def main():
    os.chdir(REPO)
    from utils import paths
    pins = {r["extract"]: r["extract_sha256"] for r in json.load(open(f"{EXTRACTS}/manifest.json"))["extracts"]}
    for p in (OPT_B, HELD_GT, HELD_GT3):
        if sha256(p) != pins[p]:
            raise SystemExit(f"!! STOP: {p} differs from its manifest hash")
    held_a = json.load(open("studies/d7/held_split.json"))["held_a"]
    order, with_files = ordered(held_a, lambda v: os.path.exists(os.path.join(SAM3_DIR, f"{v}.json.gz")))
    rels = json.load(open(HELD_GT))
    rels.update(json.load(open(HELD_GT3)))
    sp = json.load(open(paths.OBJ_SPLIT_INFO))
    novel = {c for c, s in sp["cls2split"].items() if s == "novel"}
    ctx = (json.load(open(OPT_B)), rels, novel, paths.ANNO_TRAIN_DIR)
    g20, c20 = agnostic_gain(with_files[:20], ctx)
    status, why = decide(g20)
    g40 = c40 = None
    if status == "need40":
        g40, c40 = agnostic_gain(with_files[:40], ctx)
        status, why = decide(g20, g40)
    final = with_files[:20] if status == "list20" else with_files[:40] if status == "list40" else []
    fr = frames_of(final) if final else {}
    out = dict(label="A34: the D7 check's videos and guard", seed=SEED, order=order, with_sam3_files=with_files,
               guard=dict(gain_20_points=g20, counts_20=c20, gain_40_points=g40, counts_40=c40, decision=status,
                          reason=why),
               final=final, frames=fr, frames_total=sum(fr.values()))
    with open("studies/d7/check/video_list.json", "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    with open("studies/d7/check/video_list.json.sha256", "w") as f:
        f.write(f"{sha256('studies/d7/check/video_list.json')}  video_list.json\n")
    print(json.dumps({k: out[k] for k in ("guard", "final", "frames_total")}, indent=1))
    sys.exit(2 if status == "stop" else 0)


if __name__ == "__main__":
    main()
