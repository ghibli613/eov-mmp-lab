#!/usr/bin/env python3
"""A31 §6: FIT annotation counts (report only), from the FIT-only extract of
train_object_trajectories_gt.json. The HELD-B guard (and with it the test-label guard) is installed first.

    python studies/d7/fit_counts.py        # from the repo root; writes studies/d7/fit_counts.json

Sampled frames: f mod 30 == 0, f in [0, frame-folder count). A track is a (video, tid) with its
fragments merged; its span runs from its first to its last annotated frame.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import hashlib
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import held_b_guard  # noqa: E402
held_b_guard.install()

import numpy as np  # noqa: E402

EXTRACT = "studies/d7/extracts/data__vidvrd__data__train_object_trajectories_gt.fit.json"
STEP = 30


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tracks_of(entries: list) -> dict:
    """Fragments -> {tid: set of annotated frames}."""
    out = defaultdict(set)
    for e in entries:
        out[e["tid"]] |= {int(f) for f in e["trajectory"]}
    return out


def counts(videos: dict, frames: dict, step: int = STEP) -> dict:
    """videos: {video: [fragment entries]}; frames: {video: frame-folder count}. Pure; tested."""
    n_s = n_s_empty = n_s_inspan = n_s_inspan_empty = 0
    pairs = pairs_gap = 0
    cov = []
    for v, entries in videos.items():
        tr = tracks_of(entries)
        spans = {t: (min(fs), max(fs)) for t, fs in tr.items()}
        for t, fs in tr.items():
            b, e = spans[t]
            cov.append(len(fs) / (e - b + 1))
        for f in range(0, frames[v], step):
            has_box = any(f in fs for fs in tr.values())
            in_any = any(b <= f <= e for b, e in spans.values())
            n_s += 1
            n_s_empty += not has_box
            if in_any:
                n_s_inspan += 1
                n_s_inspan_empty += not has_box
            for t, (b, e) in spans.items():
                if b <= f <= e:
                    pairs += 1
                    pairs_gap += f not in tr[t]
    cov = np.array(cov)
    q = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
    return dict(
        videos=len(videos), tracks=int(cov.size), sampled_frames=n_s,
        no_box_share_all_sampled=n_s_empty / n_s,
        sampled_frames_inside_some_span=n_s_inspan,
        no_box_share_inside_spans=n_s_inspan_empty / n_s_inspan if n_s_inspan else None,
        frame_track_pairs_in_span=pairs, in_gap_share=pairs_gap / pairs if pairs else None,
        coverage_quantiles={str(k): float(np.quantile(cov, k)) for k in q},
        coverage_mean=float(cov.mean()), coverage_under_65_share=float((cov < 0.65).mean()),
        coverage_full_share=float((cov == 1.0).mean()),
        coverage_histogram={f"[{lo:.1f},{lo + 0.1:.1f})" if lo < 0.9 else "[0.9,1.0]": int(n) for lo, n in
                            zip(np.arange(0, 1.0, 0.1), np.histogram(cov, bins=np.linspace(0, 1, 11))[0])})


def main():
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    from utils import paths
    manifest = json.load(open("studies/d7/extracts/manifest.json"))
    pin = [r for r in manifest["extracts"] if r["extract"] == EXTRACT][0]["extract_sha256"]
    if sha256(EXTRACT) != pin:
        raise SystemExit("!! STOP: the FIT extract differs from its manifest hash")
    videos = json.load(open(EXTRACT))
    assert len(videos) == 640
    frames = {v: len(os.listdir(os.path.join(paths.FRAME_DIR, v))) for v in videos}
    out = dict(label="A31 §6: FIT annotation counts (report only)", extract=EXTRACT, extract_sha256=pin,
               sampling=f"f mod {STEP} == 0 over the frame folder", **counts(videos, frames))
    with open("studies/d7/fit_counts.json", "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
