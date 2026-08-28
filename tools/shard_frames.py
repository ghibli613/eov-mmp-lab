#!/usr/bin/env python
"""Pack extracted frames into per-video tars for HuggingFace, plus a shard index.

    # plan first -- writes nothing
    python tools/shard_frames.py --split test --dry-run

    # build the bundle
    python tools/shard_frames.py --split test --bundle ../_frames_bundle_test

    # then the existing uploader takes over (private by default)
    python tools/hugging_upload.py --repo <user>/ov-vidvrd-frames \\
        --repo-type dataset --bundle ../_frames_bundle_test \\
        --dest data/vidvrd/_frame_shards --needed-for eval \\
        --title "VidVRD test frames, per-video tars"

    # verify a built bundle without rebuilding
    python tools/shard_frames.py --split test --bundle ../_frames_bundle_test --verify

    # print a download plan for a disk budget (for the Colab fetch loop)
    python tools/shard_frames.py --split test --bundle ../_frames_bundle_test \\
        --plan-batches 1.0

Why tars: 52,104 loose frames would be 52,104 HTTP requests. One tar per video
(chunked for the long ones) brings the test split to 231 files.

Why chunks: one train video is 5,492 frames / 1.8 GiB. A dropped download would
cost the whole 1.8 GiB; at 512 frames per chunk the retry unit is ~170 MiB. Over
80% of videos fit in a single chunk and are not split at all. A chunk is purely
a transport unit -- the dataset reads every frame of a video in one __getitem__,
so all of a video's chunks must be present before it can be used.

Interop with the existing tools:
  * The bundle is FLAT, because tools/hugging_upload.py enumerates it with
    os.listdir + isfile and ignores subdirectories.
  * That uploader writes its own MANIFEST.json (name/dest/needed_for/bytes/
    sha256) and would drop any extra fields, so the per-shard metadata this
    script needs to carry -- video_id, chunk index, frame range -- goes in a
    SHARDS.json sidecar that is uploaded alongside as an ordinary file.
    The Colab fetch loop reads SHARDS.json to decide WHAT to pull and
    MANIFEST.json for the URL and hash of each file.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import tarfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from utils import paths

CHUNK_FRAMES = 512
#: maps --split onto hugging_download.py's --only vocabulary, so
#: `--only eval` fetches exactly the test frames.
NEEDED_FOR = {"test": "eval", "train": "train"}


def sha256(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def video_ids(split: str) -> dict:
    """video_id -> annotation dict, for the requested split."""
    d = paths.ANNO_TEST_DIR if split == "test" else paths.ANNO_TRAIN_DIR
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        a = json.load(open(f))
        out[a["video_id"]] = a
    return out


def plan(split: str, chunk_frames: int):
    """-> (entries, problems). Entries describe every tar that would be written."""
    annos = video_ids(split)
    entries, problems = [], []
    for vid in sorted(annos):
        vdir = os.path.join(paths.FRAME_DIR, vid)
        if not os.path.isdir(vdir):
            problems.append(f"{vid}: no frame directory at {vdir}")
            continue
        frames = sorted(os.listdir(vdir))
        n = len(frames)
        a = annos[vid]
        if n != a["frame_count"]:
            problems.append(f"{vid}: {n} frames on disk but frame_count={a['frame_count']}")
        # frames are 1-indexed %06d.jpg (docs/03_Data.md); verify contiguity now,
        # because a hole would surface later as a KeyError deep in the model
        expect = [f"{i:06d}.jpg" for i in range(1, n + 1)]
        if frames != expect:
            missing = set(expect) - set(frames)
            problems.append(f"{vid}: frame names are not a contiguous 1..{n} run "
                            f"({len(missing)} missing, e.g. {sorted(missing)[:3]})")
            continue
        n_chunks = max(1, -(-n // chunk_frames))
        for ci in range(n_chunks):
            lo = ci * chunk_frames
            hi = min(lo + chunk_frames, n)
            members = frames[lo:hi]
            entries.append({
                "tar": f"{split}_{vid}_{ci:03d}.tar",
                "video_id": vid,
                "split": split,
                "chunk_index": ci,
                "n_chunks": n_chunks,
                "frame_start": lo + 1,          # 1-indexed, inclusive
                "frame_end": hi,                # 1-indexed, inclusive
                "n_frames": len(members),
                "frame_count": a["frame_count"],
                "width": a["width"],
                "height": a["height"],
                "fps": a["fps"],
                "_members": [os.path.join(vdir, m) for m in members],
                "_raw_bytes": sum(os.path.getsize(os.path.join(vdir, m)) for m in members),
            })
    return entries, problems


def batches(entries, budget_bytes):
    """Group shards into download batches under a byte budget, keeping every
    chunk of a video together -- a video is unusable until all its chunks are
    present, so a batch boundary must never fall inside one."""
    by_video = {}
    for e in entries:
        by_video.setdefault(e["video_id"], []).append(e)
    out, cur, cur_b = [], [], 0
    for vid in sorted(by_video):
        es = sorted(by_video[vid], key=lambda e: e["chunk_index"])
        sz = sum(e.get("bytes", e.get("_raw_bytes", 0)) for e in es)
        if cur and cur_b + sz > budget_bytes:
            out.append((cur, cur_b))
            cur, cur_b = [], 0
        cur.extend(es)
        cur_b += sz
    if cur:
        out.append((cur, cur_b))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "train"], required=True)
    ap.add_argument("--bundle", default=None,
                    help="flat output directory for the tars + SHARDS.json")
    ap.add_argument("--chunk-frames", type=int, default=CHUNK_FRAMES)
    ap.add_argument("--dry-run", action="store_true", help="report the plan, write nothing")
    ap.add_argument("--verify", action="store_true",
                    help="re-hash an existing bundle against its SHARDS.json")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N videos (for testing the format)")
    ap.add_argument("--plan-batches", type=float, default=None, metavar="GIB",
                    help="print a download plan for this disk budget, in GiB")
    a = ap.parse_args()

    entries, problems = plan(a.split, a.chunk_frames)
    if a.limit:
        keep = sorted({e["video_id"] for e in entries})[:a.limit]
        entries = [e for e in entries if e["video_id"] in set(keep)]
    if problems:
        print(f"!! {len(problems)} problem(s) found in the frame directories:")
        for p in problems[:12]:
            print("   " + p)
        if len(problems) > 12:
            print(f"   ... and {len(problems)-12} more")
        if not a.dry_run:
            return 1
    raw = sum(e["_raw_bytes"] for e in entries)
    nvid = len({e["video_id"] for e in entries})
    split_ct = sum(1 for e in entries if e["n_chunks"] > 1)
    print(f"{a.split}: {nvid} videos -> {len(entries)} tar(s), {raw/1024**3:.2f} GiB of frames")
    print(f"  videos needing more than one chunk: "
          f"{len({e['video_id'] for e in entries if e['n_chunks']>1})}"
          f"  ({split_ct} such chunks)")
    biggest = max(entries, key=lambda e: e["_raw_bytes"])
    print(f"  largest tar: {biggest['tar']}  {biggest['_raw_bytes']/1024**2:.0f} MiB")

    if a.plan_batches:
        bs = batches(entries, int(a.plan_batches * 1024**3))
        print(f"\ndownload plan at {a.plan_batches:.2f} GiB per batch: {len(bs)} batches")
        for i, (es, b) in enumerate(bs[:5]):
            print(f"  batch {i:02d}: {len({e['video_id'] for e in es}):3d} videos, "
                  f"{len(es):3d} tars, {b/1024**3:.2f} GiB")
        if len(bs) > 5:
            print(f"  ... and {len(bs)-5} more")

    if a.dry_run or (a.plan_batches and not a.bundle):
        return 0
    if not a.bundle:
        raise SystemExit("--bundle is required unless --dry-run")

    index_path = os.path.join(a.bundle, "SHARDS.json")

    if a.verify:
        if not os.path.exists(index_path):
            raise SystemExit(f"no SHARDS.json in {a.bundle}")
        idx = json.load(open(index_path))
        bad = 0
        for e in idx["shards"]:
            p = os.path.join(a.bundle, e["tar"])
            if not os.path.exists(p):
                print(f"  MISSING {e['tar']}"); bad += 1; continue
            if os.path.getsize(p) != e["bytes"] or sha256(p) != e["sha256"]:
                print(f"  CORRUPT {e['tar']}"); bad += 1
        print(f"\nverified {len(idx['shards'])} tar(s); {bad} problem(s)")
        return 1 if bad else 0

    os.makedirs(a.bundle, exist_ok=True)
    print(f"\nwriting {len(entries)} tar(s) to {a.bundle}")
    for i, e in enumerate(entries, 1):
        out = os.path.join(a.bundle, e["tar"])
        tmp = out + ".tmp"
        with tarfile.open(tmp, "w") as tf:
            for m in e["_members"]:
                # arcname keeps the video folder, so untarring into
                # data/vidvrd/frames/ reconstructs the layout the dataset expects
                tf.add(m, arcname=os.path.join(e["video_id"], os.path.basename(m)))
        os.replace(tmp, out)
        e["bytes"] = os.path.getsize(out)
        e["sha256"] = sha256(out)
        if i % 25 == 0 or i == len(entries):
            print(f"  {i}/{len(entries)}")

    for e in entries:
        e.pop("_members", None)
        e.pop("_raw_bytes", None)
    index = {
        "version": 1,
        "source": "ImageNet-VidVRD extracted frames",
        "split": a.split,
        "chunk_frames": a.chunk_frames,
        "frame_naming": "%06d.jpg, 1-INDEXED (see docs/03_Data.md)",
        "untar_into": "data/vidvrd/frames",
        "needed_for": NEEDED_FOR[a.split],
        "n_videos": nvid,
        "n_shards": len(entries),
        "total_bytes": sum(e["bytes"] for e in entries),
        "shards": entries,
    }
    with open(index_path, "w") as fh:
        json.dump(index, fh, indent=1)
    print(f"\nwrote {index_path}")
    print(f"  {len(entries)} tars, {index['total_bytes']/1024**3:.2f} GiB\n")
    print("next:")
    print(f"  python tools/hugging_upload.py --repo <user>/ov-vidvrd-frames \\")
    print(f"      --repo-type dataset --bundle {a.bundle} \\")
    print(f"      --dest data/vidvrd/_frame_shards --needed-for {NEEDED_FOR[a.split]} \\")
    print(f"      --title \"VidVRD {a.split} frames, per-video tars\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
