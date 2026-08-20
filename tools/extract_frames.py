#!/usr/bin/env python
"""Decode the VidVRD videos into per-video JPEG frame directories.

`data_loading/dataset.py` reads frames, not videos:

    self.FEAT_ROOT = join(base_vidvrd, 'frames')
    frame_list = sorted(os.listdir(data_path))     # one dir per video
    video_len  = len(frame_list)

and stacks the CLIP encodings in that sorted order, so **filename order must
equal frame-id order**. Trajectory frame ids are 0-indexed and `end_fid` is
exclusive (verified: a trajectory with begin_fid=0, end_fid=90 has keys 0..89).

Filenames are **1-indexed**. `models/gen_labels.py:30` -- live code, reached
through `end2end_model.gen_feats` -- maps a trajectory frame id to a filename as

    self.frame_paths[fid] = '%06d.jpg' % (fid + 1)   # "picture No is named from 1"

so trajectory frame 0 is `000001.jpg`. Since 1-indexed names sort in the same
order, `sorted(os.listdir())` in dataset.py still yields frame 0 first, and both
readers agree.

Output:

    data/vidvrd/frames/<video_name>/000001.jpg   <- trajectory frame 0
                                   000002.jpg
                                   ...

Roughly 296k frames / ~11 GB for the full 1000 videos.

Resumable: a video whose directory already holds the expected number of frames
is skipped, so re-running after an interruption costs only the scan.

    python tools/extract_frames.py                 # all 1000
    python tools/extract_frames.py --limit 20      # a quick smoke run
    python tools/extract_frames.py --verify        # check counts, decode nothing
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = os.path.join(REPO, "data", "vidvrd", "videos")
ANNO_DIR = os.path.join(REPO, "data", "vidvrd", "anno")
FRAME_DIR = os.path.join(REPO, "data", "vidvrd", "frames")


def expected_counts() -> dict:
    """video name -> frame_count, taken from the annotations."""
    out = {}
    for split in ("train", "test"):
        d = os.path.join(ANNO_DIR, split)
        for fn in os.listdir(d):
            if fn.endswith(".json"):
                out[fn[:-5]] = json.load(open(os.path.join(d, fn)))["frame_count"]
    return out


def extract_one(video: str, n_expected: int, quality: int) -> tuple[int, str]:
    """Returns (frames_written, status)."""
    out_dir = os.path.join(FRAME_DIR, video)
    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) == n_expected:
        return n_expected, "skip"

    src = os.path.join(VIDEO_DIR, video + ".mp4")
    if not os.path.exists(src):
        return 0, "no-video"

    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return 0, "unopenable"

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # 1-INDEXED on purpose: models/gen_labels.py:30 (live, reached via
        # end2end_model.gen_feats) maps trajectory frame id -> '%06d.jpg' % (fid+1),
        # so trajectory frame 0 must be 000001.jpg. data_loading/dataset.py uses
        # sorted(os.listdir()) and is naming-agnostic, so this satisfies both.
        cv2.imwrite(os.path.join(out_dir, f"{i + 1:06d}.jpg"), frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        i += 1
    cap.release()
    return i, "ok" if i == n_expected else f"count {i} != {n_expected}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="only the first N videos")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality (default 95)")
    ap.add_argument("--verify", action="store_true",
                    help="report per-video frame counts against the annotations, decode nothing")
    args = ap.parse_args()

    counts = expected_counts()
    videos = sorted(counts)
    if args.limit:
        videos = videos[:args.limit]

    if args.verify:
        missing, wrong, ok = [], [], 0
        for v in videos:
            d = os.path.join(FRAME_DIR, v)
            if not os.path.isdir(d):
                missing.append(v); continue
            n = len(os.listdir(d))
            if n != counts[v]:
                wrong.append((v, n, counts[v]))
            else:
                ok += 1
        print(f"  complete : {ok}/{len(videos)}")
        print(f"  missing  : {len(missing)}")
        print(f"  wrong    : {len(wrong)}")
        for w in wrong[:10]:
            print(f"     {w[0]}: {w[1]} frames, expected {w[2]}")
        return 0 if not missing and not wrong else 1

    os.makedirs(FRAME_DIR, exist_ok=True)
    problems, written, skipped = [], 0, 0
    for v in tqdm(videos, desc="extracting"):
        n, status = extract_one(v, counts[v], args.quality)
        if status == "skip":
            skipped += 1
        elif status == "ok":
            written += n
        else:
            problems.append((v, status))

    print(f"\n  videos    : {len(videos)}")
    print(f"  skipped   : {skipped} (already complete)")
    print(f"  frames    : {written} newly written")
    print(f"  problems  : {len(problems)}")
    for v, s in problems[:20]:
        print(f"     {v}: {s}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
