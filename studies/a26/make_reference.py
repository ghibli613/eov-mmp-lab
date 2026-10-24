#!/usr/bin/env python3
"""Laptop: session 3's R0 predictions for the two gate videos -> a26/data/r0_reference.json.

    cd METOR/studies && python -m a26.make_reference

The gate videos are the first and last of Dataset_new('val').path_list, as D2's 2-video check ran
them (d2/logs/box/run_first_last.log). Reads only session 3's prediction files, each checked against
SHA256SUMS_session3.txt (itself pinned). No label file is opened.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from a26.common import DATA, REFERENCE, S3_DIR, S3_SUMS, S3_SUMS_SHA  # noqa: E402
from a26.run import sha256  # noqa: E402

GATE_VIDEOS = ["ILSVRC2015_val_00046001", "ILSVRC2015_train_00047020"]


def main():
    if sha256(S3_SUMS) != S3_SUMS_SHA:
        raise SystemExit("!! STOP: SHA256SUMS_session3.txt differs from its pin")
    sums = dict(reversed(l.strip().split(None, 1)) for l in open(S3_SUMS))
    out = dict(videos=GATE_VIDEOS, source={}, predictions={v: {} for v in GATE_VIDEOS})
    for split in ("all", "novel"):
        rel = f"cache_s3_sgdet/sgdet_{split}.json"
        path = os.path.join(S3_DIR, f"sgdet_{split}.json")
        if sha256(path) != sums[rel]:
            raise SystemExit(f"!! STOP: {rel} does not match SHA256SUMS_session3.txt")
        out["source"][rel] = sums[rel]
        preds = json.load(open(path))
        for v in GATE_VIDEOS:
            out["predictions"][v][split] = preds.get(v, [])
        del preds
    os.makedirs(DATA, exist_ok=True)
    with open(REFERENCE, "w") as f:
        json.dump(out, f)
    n = {v: {s: len(p) for s, p in d.items()} for v, d in out["predictions"].items()}
    print(f"{REFERENCE}: {n} sha256={sha256(REFERENCE)}")


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
