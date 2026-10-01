#!/usr/bin/env python3
"""Helpers for setup_session3.sh (stage 1, session 3, on the rented box).

    python pilot_analysis/stage1c/session3_tasks.py verify-hashes
    python pilot_analysis/stage1c/session3_tasks.py fetch-test-frames --shards URL
    python pilot_analysis/stage1c/session3_tasks.py frame-provenance --videos V1 V2
    python pilot_analysis/stage1c/session3_tasks.py select-videos
    python pilot_analysis/stage1c/session3_tasks.py summarize --cache DIR

No model code, no labels. Test annotations are read for frame counts only.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse, glob, json, os, re, tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

TOTAL_FRAMES = 244_100 + 52_104          # train + test frames (stage 1 audit, B5)


def verify_hashes(_):
    """Every `path` | `sha256` row of the preregistration (including A5) must match."""
    import hashlib
    text = open(os.path.join(ROOT, "pilot_analysis", "stage1b", "preregistration.md")).read()
    rows = re.findall(r"`(pilot_analysis/[^`]+?)`[^|\n]*\|\s*`([0-9a-f]{64})`", text)
    bad = []
    for path, h in rows:
        got = hashlib.sha256(open(os.path.join(ROOT, path), "rb").read()).hexdigest()
        print(f"  {'OK ' if got == h else 'BAD'}  {path}")
        if got != h:
            bad.append(path)
    if not rows or bad:
        sys.exit(f"!! pinned-hash check failed: {bad or 'no rows found'}")
    print(f"  {len(rows)} pinned artefacts verified")


def fetch_test_frames(a):
    """The test frames the pilot ran on: per-video tars from the private bundle,
    sha256-checked and unpacked into the frame directory (as dump_predictions does)."""
    from tools.hugging_download import download, load_manifest, sha256
    from utils import paths
    idx = load_manifest(a.shards)
    man = load_manifest(a.shards.rsplit("/", 1)[0] + "/MANIFEST.json")
    shards = [s for s in idx["shards"] if s["split"] == "test"
              and (not a.videos or s["video_id"] in a.videos)]
    os.makedirs(paths.FRAME_DIR, exist_ok=True)
    staging = os.path.join(ROOT, "data", "vidvrd", "_staging")
    os.makedirs(staging, exist_ok=True)
    done = 0
    for sh in shards:
        vdir = os.path.join(paths.FRAME_DIR, sh["video_id"])
        need = range(sh["frame_start"], sh["frame_end"] + 1)          # 1-based, inclusive (= file names)
        if os.path.isdir(vdir) and all(os.path.exists(os.path.join(vdir, f"{i:06d}.jpg")) for i in need):
            continue
        dst = os.path.join(staging, sh["tar"])
        download(f"{man['base_url']}/{sh['tar']}", dst)
        if sha256(dst) != sh["sha256"]:
            sys.exit(f"!! sha256 mismatch on {sh['tar']}")
        with tarfile.open(dst) as tf:
            try:
                tf.extractall(paths.FRAME_DIR, filter="data")
            except TypeError:
                tf.extractall(paths.FRAME_DIR)
        os.remove(dst)
        done += 1
    vids = {s["video_id"] for s in shards}
    print(f"  test frames: {len(vids)} videos, {done} shards fetched now")


def decode(a):
    """Decode only the named videos, with tools/extract_frames.py's own function
    (cv2, JPEG quality 95, 1-indexed names), so Task 2 needs a few hundred MB of
    frames instead of all 42 GB."""
    import tools.extract_frames as X
    for v in a.videos:
        n, status = X.extract_one(v, frame_count(v), quality=95)
        print(f"  {v}: {n} frames, {status}")
        if status not in ("ok", "skip"):
            sys.exit(f"!! decode failed for {v}: {status}")


def frame_count(v):
    """Frame count from the annotation file (metadata only; labels untouched)."""
    for split in ("train", "test"):
        p = os.path.join(ROOT, "data", "vidvrd", "anno", split, f"{v}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            d.pop("relation_instances", None)
            return d["frame_count"]
    raise KeyError(v)


def frame_provenance(a):
    """Are frames decoded on this box identical to the bundle's? Decode the mp4
    exactly as tools/extract_frames.py does (cv2, JPEG quality 95) in memory and
    compare with the bundle's files. Report only."""
    import cv2, numpy as np
    from utils import paths
    out = {}
    for v in a.videos:
        cap = cv2.VideoCapture(os.path.join(paths.VIDEO_DIR, v + ".mp4"))
        same = n = 0
        maxdiff = 0
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            ref_path = os.path.join(paths.FRAME_DIR, v, f"{i:06d}.jpg")
            if not os.path.exists(ref_path):
                continue
            ref = open(ref_path, "rb").read()
            n += 1
            same += enc.tobytes() == ref
            d = np.abs(cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(int)
                       - cv2.imread(ref_path, cv2.IMREAD_COLOR).astype(int)).max()
            maxdiff = max(maxdiff, int(d))
        cap.release()
        out[v] = dict(frames_compared=n, byte_identical=same, max_pixel_diff=maxdiff)
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(a.out, "frame_provenance.json"), "w"), indent=1)


def select_videos(a):
    """3 train + 2 test, deterministic: train at the 25th/50th/75th percentile of
    frame count; test at the 50th/75th among videos with >= 2 GT pairs."""
    from gt_tracks import load_tracks

    n_frames = frame_count

    def n_pairs(ts):
        return sum(1 for s in ts for o in ts if s is not o and
                   min(s["end_fid"], o["end_fid"]) - max(s["begin_fid"], o["begin_fid"]) >= 10)

    pick = {}
    for split, qs, min_pairs in (("train", (0.25, 0.5, 0.75), 1), ("test", (0.5, 0.75), 2)):
        tr = load_tracks(split)
        cand = sorted((n_frames(v), v) for v, ts in tr.items() if n_pairs(ts) >= min_pairs)
        pick[split] = [cand[int(q * (len(cand) - 1))][1] for q in qs]
    print(json.dumps(pick))
    json.dump(pick, open(os.path.join(a.out, "selected_videos.json"), "w"), indent=1)


def summarize(a):
    recs = [json.loads(l) for l in open(os.path.join(a.cache, "manifest.jsonl"))]
    load = json.load(open(os.path.join(a.cache, "modelC_load.json")))
    fid_path = os.path.join(a.cache, "fidelity_from_dump.json")
    fid = json.load(open(fid_path)) if os.path.exists(fid_path) else None
    enc = sum(r["encode_seconds"] for r in recs) / sum(r["frames"] for r in recs)
    pairs = sum(r["pairs"] for r in recs)
    rest = sum(r["seconds"] - r["encode_seconds"] for r in recs) / max(pairs, 1)
    total_pairs = 8_926 + 1_392                       # train (option B) + test
    proj_h = (enc * TOTAL_FRAMES + rest * total_pairs) / 3600
    bytes_per_frame = sum(r["bytes"] for r in recs) / sum(r["frames"] for r in recs)
    out = dict(videos=recs, modelC_load=dict(missing=load["missing"], unexpected=load["unexpected"],
                                             loaded=load["loaded"]),
               fidelity_max_residual=max(r["fidelity_max_residual"] for r in recs),
               dump_fidelity=fid,
               rates=dict(encode_seconds_per_frame=enc, other_seconds_per_pair=rest),
               projection=dict(hours_all_1000_videos=proj_h,
                               cache_gib_all=bytes_per_frame * TOTAL_FRAMES / 2 ** 30))
    json.dump(out, open(os.path.join(a.cache, "session3_task2_summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "videos"}, indent=1))
    for r in recs:
        print(f"  {r['split']:5} {r['video']}  {r['frames']:5d} frames  {r['pairs']:3d} pairs  "
              f"{r['slots']:4d} slots  {r['seconds']:7.1f}s (encode {r['encode_seconds']:.1f}s)  "
              f"{r['bytes'] / 2**20:7.1f} MiB  residual {r['fidelity_max_residual']:.2e}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify-hashes")
    p = sub.add_parser("fetch-test-frames"); p.add_argument("--shards", required=True); p.add_argument("--videos", nargs="*")
    p = sub.add_parser("decode"); p.add_argument("--videos", nargs="+", required=True)
    p = sub.add_parser("frame-provenance"); p.add_argument("--videos", nargs="+", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("select-videos"); p.add_argument("--out", required=True)
    p = sub.add_parser("summarize"); p.add_argument("--cache", required=True)
    a = ap.parse_args()
    dict(verify_hashes=verify_hashes, fetch_test_frames=fetch_test_frames, decode=decode,
         frame_provenance=frame_provenance,
         select_videos=select_videos, summarize=summarize)[a.cmd.replace("-", "_")](a)


if __name__ == "__main__":
    main()
