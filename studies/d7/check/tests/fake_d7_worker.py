#!/usr/bin/env python3
"""A stand-in for studies/d7/check/d7_worker.py in the driver tests: G1's fake worker
(studies/rpt/tests/fake_sam3_worker.py) adapted to prompts given per job. No SAM 3.

The check-0 video with "all35" gets G0's reference output, copied (or, if FAKE_DIFF names a folder in the
job's output path, with one out_probs changed). Any other job gets a small file with one entry per prompt
and no masklet. FAKE_OOM (comma list of videos): those jobs report CUDA out of memory.
The RESULT line carries the per-pass fields d7_run.py records."""
import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys

D7 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(os.path.dirname(D7))
CHECK0 = "ILSVRC2015_train_00010024"
REF = os.path.join(REPO, "studies", "rpt", "g0", "sam3_out", f"{CHECK0}.json.gz")

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint")
ap.add_argument("--sha256")
ap.add_argument("--frames")
a = ap.parse_args()
diff = [x for x in os.environ.get("FAKE_DIFF", "").split(",") if x]
oom = [x for x in os.environ.get("FAKE_OOM", "").split(",") if x]
print("READY " + json.dumps(dict(model_load_seconds=0.0, gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))), flush=True)
for line in sys.stdin:
    if not line.strip():
        break
    job = json.loads(line)
    v, out = job["video"], job["out"]
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{v}.json.gz")
    n = len(os.listdir(os.path.join(a.frames, v)))
    if v in oom:
        print("RESULT " + json.dumps(dict(video=v, ok=False, error="CUDA out of memory")), flush=True)
        continue
    if v == CHECK0 and job["prompts"] == "all35":
        d = json.load(gzip.open(REF, "rt"))
        if any(x in out for x in diff):
            m = next(p for p in d["prompts"] if p["masklets"])["masklets"][0]["frames"][0]
            m["prob"] += 1e-6
        json.dump(d, gzip.open(path, "wt"))
    else:
        names = [["all35", "all35"]] if job["prompts"] == "all35" else job["prompts"]
        d = dict(video=v, frames=n, width=640, height=360, prompts=[
            dict(**{"class": nm}, prompt=t, seconds=n * 0.1, frames=n, seconds_per_frame=0.1, masklets=[],
                 peak_gpu_allocated_gib=7.0, peak_gpu_reserved_gib=8.0) for nm, t in names])
        json.dump(d, gzip.open(path, "wt"))
    passes = [dict(name=p["class"], prompt=p["prompt"], seconds=p["seconds"], frames=p["frames"],
                   seconds_per_frame=p["seconds_per_frame"], masklets=len(p["masklets"]),
                   peak_gpu_allocated_gib=p["peak_gpu_allocated_gib"], peak_gpu_reserved_gib=p["peak_gpu_reserved_gib"])
              for p in d["prompts"]]
    print("RESULT " + json.dumps(dict(video=v, ok=True, frames=n, prompts=len(passes), seconds_total=n * 0.1 * len(passes),
                                      wall_seconds=0.0, peak_gpu_allocated_gib=7.0, peak_gpu_reserved_gib=8.0,
                                      passes=passes, file=path,
                                      sha256=hashlib.sha256(open(path, "rb").read()).hexdigest())), flush=True)
