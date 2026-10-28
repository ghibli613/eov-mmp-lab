#!/usr/bin/env python3
"""A stand-in for studies/rpt/g1/sam3_worker.py in the driver tests: same protocol, no SAM 3.

Smoke video 1 gets G0's reference output, copied; any other video a small file. It reports
wall_seconds = frames x 35 x FAKE_RATE (default 0.094, G0's rate) and sleeps frames x FAKE_SLEEP
(default 0.0003 s), so jobs overlap in the order a real run would see.
FAKE_DIFF (comma list): a job whose output folder contains one of these makes smoke video 1's output
differ (one out_probs changed). FAKE_FAIL (comma list of videos): those jobs fail."""
import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import time

RPT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(RPT, "g0", "sam3_out", "ILSVRC2015_train_00008005.json.gz")

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint")
ap.add_argument("--sha256")
ap.add_argument("--frames")
ap.add_argument("--compile", action="store_true")
a = ap.parse_args()
diff = [x for x in os.environ.get("FAKE_DIFF", "").split(",") if x]
fail = [x for x in os.environ.get("FAKE_FAIL", "").split(",") if x]
rate = float(os.environ.get("FAKE_RATE", "0.094"))
nap = float(os.environ.get("FAKE_SLEEP", "0.0003"))
gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
print("READY " + json.dumps(dict(model_load_seconds=0.0, compile=a.compile, gpu=gpu)), flush=True)
for line in sys.stdin:
    if not line.strip():
        break
    job = json.loads(line)
    v, out = job["video"], job["out"]
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{v}.json.gz")
    n = len(os.listdir(os.path.join(a.frames, v)))
    time.sleep(n * nap)
    if v in fail:
        print("RESULT " + json.dumps(dict(video=v, ok=False, error="fake failure")), flush=True)
        continue
    if v == "ILSVRC2015_train_00008005":
        if any(d in out for d in diff):
            d = json.load(gzip.open(REF, "rt"))
            m = next(p for p in d["prompts"] if p["masklets"])["masklets"][0]["frames"][0]
            m["prob"] += 1e-6
            json.dump(d, gzip.open(path, "wt"))
        else:
            shutil.copy(REF, path)
    else:
        json.dump(dict(video=v, job=job, gpu=gpu), gzip.open(path, "wt"))
    print("RESULT " + json.dumps(dict(video=v, ok=True, frames=n, prompts=35, seconds_total=n * 35 * rate,
                                      wall_seconds=n * 35 * rate, peak_gpu_allocated_gib=7.0,
                                      peak_gpu_reserved_gib=8.0, file=path, gpu=gpu,
                                      sha256=hashlib.sha256(open(path, "rb").read()).hexdigest())), flush=True)
