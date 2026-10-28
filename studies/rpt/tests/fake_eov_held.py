#!/usr/bin/env python3
"""A stand-in for studies/rpt/g1/eov_held.py in the parallel tests: writes one file per video with a
timing field that differs per run; FAKE_EOV_GPU_DIFF=1 also writes the GPU into a compared field."""
import argparse
import gzip
import json
import os
import random

ap = argparse.ArgumentParser()
ap.add_argument("--sam3")
ap.add_argument("--out")
ap.add_argument("--tag", default="")
ap.add_argument("--no-sam3", default="")
ap.add_argument("--videos", nargs="*")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
for v in a.videos:
    rec = dict(video=v, configs=dict(R0w=dict(tracks=[1, 2], seconds=random.random())),
               sensitivity_seconds=random.random(), sensitivity=[dict(k=0, x=1.5)])
    if v in a.no_sam3.split(","):
        rec["sensitivity"] = None
    if os.environ.get("FAKE_EOV_GPU_DIFF"):
        rec["configs"]["R0w"]["tracks"].append(gpu)
    json.dump(rec, gzip.open(os.path.join(a.out, f"{v}.json.gz"), "wt"))
    with open(os.path.join(a.out, f"manifest_{a.tag}.jsonl"), "a") as f:
        f.write(json.dumps(dict(video=v, gpu=gpu, sensitivity=rec["sensitivity"] is not None)) + "\n")
