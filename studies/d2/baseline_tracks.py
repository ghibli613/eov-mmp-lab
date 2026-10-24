#!/usr/bin/env python3
"""The baseline's final SGDet tracks, and ground rule 3's comparison (A25).

    python -m d2.baseline_tracks            # laptop: cache_s3_sgdet -> d2/data/baseline_tracks.json

Reads only the detected tracks of session 3's SGDet cache (each file verified against
SHA256SUMS_session3.txt first). These are detector outputs, not test labels.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

from d2.common import CACHE_SGDET, CACHE_SUMS, DATA, sha256, verify_pins, write_log

OUT = os.path.join(DATA, "baseline_tracks.json")


def _same_float(a, b) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == b


def compare(rec: list, base: list) -> list:
    """Ground rule 3: same tracks (ids in order), classes, begin/end, every box exactly, and the
    score exactly (NaN equals NaN). Returns the differences; empty means identical."""
    diffs = []
    if len(rec) != len(base):
        diffs.append(f"track count {len(rec)} vs baseline {len(base)}")
    for r, b in zip(rec, base):
        for k in ("tid", "category", "begin_fid", "end_fid"):
            if r[k] != b[k]:
                diffs.append(f"tid {b['tid']}: {k} {r[k]!r} vs baseline {b[k]!r}")
        if not _same_float(float(r["score"]), float(b["score"])):
            diffs.append(f"tid {b['tid']}: score {r['score']!r} vs baseline {b['score']!r}")
        if r["boxes"] != b["boxes"]:
            n = sum(x != y for x, y in zip(r["boxes"], b["boxes"])) + abs(len(r["boxes"]) - len(b["boxes"]))
            diffs.append(f"tid {b['tid']}: {n} boxes differ")
    return diffs


def load(path: str = OUT) -> dict:
    with open(path) as f:
        return json.load(f)


def extract(out: str = OUT) -> dict:
    import torch
    verify_pins({"session3_box_copy/SHA256SUMS_session3.txt": "2cc2d0710f6a92d68ae27ecf74973a9dd122dc1594b9788cff4b0c560567954b"})
    sums = dict(reversed(l.strip().split(None, 1)) for l in open(CACHE_SUMS))
    res = {}
    files = sorted(glob.glob(os.path.join(CACHE_SGDET, "test", "*.pt")))
    if len(files) != 200:
        raise SystemExit(f"!! STOP: {len(files)} SGDet cache files, expected 200")
    for f in files:
        rel = os.path.relpath(f, os.path.dirname(CACHE_SGDET))
        if sha256(f) != sums[rel]:
            raise SystemExit(f"!! STOP: {rel} does not match SHA256SUMS_session3.txt")
        d = torch.load(f, weights_only=False, map_location="cpu")
        tracks = d["tracks"]
        res[d["video"]] = [dict(tid=int(tid), category=t["category"], score=float(t["score"]),
                                begin_fid=int(t["begin_fid"]), end_fid=int(t["end_fid"]),
                                boxes=[[float(v) for v in b] for b in t["boxes"]])
                           for tid, t in sorted(tracks.items())]
        for t in res[d["video"]]:
            if len(t["boxes"]) != t["end_fid"] - t["begin_fid"]:
                raise SystemExit(f"!! STOP: {d['video']} tid {t['tid']}: boxes do not cover begin..end")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh)
    write_log("baseline_tracks", dict(source=CACHE_SGDET), [CACHE_SUMS], [out],
              extra=dict(videos=len(res), tracks=sum(len(v) for v in res.values())))
    return res


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    r = extract()
    print(f"{len(r)} videos, {sum(len(v) for v in r.values())} tracks -> {OUT} ({sha256(OUT)})")
