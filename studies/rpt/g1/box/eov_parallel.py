#!/usr/bin/env python3
"""RPT G1, the EOV-MMP side on several GPUs (A29a): studies/rpt/g1/eov_held.py, one process per GPU
(CUDA_VISIBLE_DEVICES), the videos split by frames, longest first, onto the least-loaded GPU.

    python studies/rpt/g1/box/eov_parallel.py preflight   # 2 smoke videos on GPU 0 and on GPU 1, compared
    python studies/rpt/g1/box/eov_parallel.py full        # the 160 HELD videos, after the 160 SAM 3 files

preflight: both GPUs run the same 2 videos with G0's SAM 3 files; every output must be identical
apart from the timing fields (`seconds`, `sensitivity_seconds`): a video's result must not depend on
the GPU that ran it. Exit 8 otherwise (stop and ask). Before the cap's clock (set-up).
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import gzip
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
G1 = os.path.dirname(HERE)
RPT = os.path.dirname(G1)
REPO = os.path.abspath(os.path.join(RPT, "..", ".."))
HOME = os.path.expanduser("~")
OUT = os.environ.get("G1_OUT", os.path.join(HOME, "rpt_g1_out"))
BOX = os.environ.get("G1_BOX", os.path.join(HOME, "rpt_g1_box"))
PY = os.environ.get("G1_EOVPY", "/venv/main/bin/python")
EOV = os.environ.get("G1_EOV", os.path.join(G1, "eov_held.py"))      # G1_EOV: the laptop test only
PREFLIGHT = ["ILSVRC2015_train_00010024", "ILSVRC2015_train_00008005"]
TIMING = {"seconds", "sensitivity_seconds"}


def n_gpus():
    if os.environ.get("G1_GPUS"):
        return int(os.environ["G1_GPUS"])
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    return sum(1 for line in out.splitlines() if line.startswith("GPU "))


def split(videos: list, frames: dict, n: int) -> list:
    """Longest first, each onto the GPU with the fewest frames so far (ties: lowest index)."""
    load, parts = [0] * n, [[] for _ in range(n)]
    for v in sorted(videos, key=lambda v: (-frames[v], v)):
        g = min(range(n), key=lambda i: (load[i], i))
        parts[g].append(v)
        load[g] += frames[v]
    return parts


def launch(g, videos, sam3, out, omp, tag, no_sam3=()):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), OMP_NUM_THREADS=str(omp), MKL_NUM_THREADS=str(omp))
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(BOX, f"eov_{os.path.basename(out)}_{tag or 'g' + str(g)}.log"), "a")
    return subprocess.Popen([PY, "-B", EOV, "--sam3", sam3, "--out", out, "--tag", tag,
                             "--no-sam3", ",".join(no_sam3), "--videos", *videos],
                            cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)


def strip(v):
    if isinstance(v, dict):
        return {k: strip(x) for k, x in v.items() if k not in TIMING}
    if isinstance(v, list):
        return [strip(x) for x in v]
    return v


def same_except_timing(a_path, b_path) -> bool:
    with gzip.open(a_path, "rt") as f:
        a = json.load(f)
    with gzip.open(b_path, "rt") as f:
        b = json.load(f)
    return strip(a) == strip(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["preflight", "full"])
    a = ap.parse_args()
    os.makedirs(BOX, exist_ok=True)
    n = n_gpus()
    cores = int(os.environ.get("G1_CORES") or len(os.sched_getaffinity(0)))
    if a.what == "preflight":
        sam3 = os.path.join(RPT, "g0", "sam3_out")
        omp = max(1, min(8, cores // 2))
        outs = [os.path.join(OUT, "preflight_eov", f"g{g}") for g in (0, 1 % n)]
        ps = [launch(g, PREFLIGHT, sam3, o, omp, f"g{g}") for g, o in zip((0, 1 % n), outs)]
        codes = [p.wait() for p in ps]
        if any(codes):
            raise SystemExit(f"!! preflight: eov_held exited {codes}")
        res = {v: same_except_timing(*(os.path.join(o, f"{v}.json.gz") for o in outs)) for v in PREFLIGHT}
        json.dump(dict(gpus=[0, 1 % n], identical=res), open(os.path.join(BOX, "eov_preflight.json"), "w"), indent=1)
        print(f"EOV preflight on GPUs 0 and {1 % n}: {res}", flush=True)
        if not all(res.values()):
            json.dump(res, open(os.path.join(BOX, "STOP_eov_gpu_dependence.json"), "w"), indent=1)
            print("!! STOP: a video's EOV-side result depends on the GPU -- ask the author", flush=True)
            sys.exit(8)
        return
    if not os.path.exists(os.path.join(BOX, "sam3_done.json")):
        raise SystemExit("!! the 160 SAM 3 files are not done")
    held = sorted(json.load(open(os.path.join(REPO, "pilot_analysis", "stage1", "train_split.json")))["held"])
    frames = {v: len(os.listdir(os.path.join(REPO, "data", "vidvrd", "frames", v))) for v in held}
    parts = split(held, frames, n)
    json.dump(dict(gpus=n, parts=parts, frames=[sum(frames[v] for v in p) for p in parts]),
              open(os.path.join(BOX, "eov_split.json"), "w"), indent=1)
    omp = max(1, min(8, cores // n))
    done = json.load(open(os.path.join(BOX, "sam3_done.json")))
    excluded = done.get("excluded", []) + done.get("moved_to_l40", [])
    print(f"EOV side: all 160 videos for R0w and R0; no box sensitivity for {excluded} (no SAM 3 file)", flush=True)
    ps = [launch(g, parts[g], os.path.join(OUT, "sam3"), os.path.join(OUT, "eov"), omp, f"g{g}", excluded)
          for g in range(n) if parts[g]]
    codes = [p.wait() for p in ps]
    if any(codes):
        raise SystemExit(f"!! eov_held exited {codes}: see {BOX}/eov_eov_g*.log")
    print(f"EOV side done on {n} GPUs", flush=True)


if __name__ == "__main__":
    main()
