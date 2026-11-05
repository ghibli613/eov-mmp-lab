#!/usr/bin/env python3
"""The D7 check's SAM 3 worker (A34): one long-lived SAM 3 process, G1's protocol (studies/rpt/g1/sam3_worker.py),
with the prompts given per job. Runs in the SAM 3 environment, at SAM 3's defaults (no offload, A1, A2 or A3).
The per-video work is the committed runner's, unchanged (studies/rpt/sam3_runner.py: run_video, prompts).

    ~/sam3env/bin/python -B studies/d7/check/d7_worker.py --checkpoint CKPT --sha256 SHA --frames FRAMES
    # then one JSON job per line on stdin:
    {"video": V, "out": DIR, "prompts": "all35" | [[name, text], ...]}

Writes DIR/V.json.gz (atomically) and answers one line starting with "RESULT ", which carries, per pass
(prompt): seconds, frames, seconds per frame, masklets, peak GPU memory. An empty line or EOF ends it.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RPT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "rpt")
sys.path.insert(0, RPT)
import sam3_runner as R  # noqa: E402


def names_of(prompts):
    if prompts == "all35":
        return R.prompts()
    out = [(str(n), str(t)) for n, t in prompts]
    if not out or len({n for n, _ in out}) != len(out):
        raise ValueError(f"bad prompt list: {prompts}")
    return out


def passes(res):
    return [dict(name=p["class"], prompt=p["prompt"], seconds=p["seconds"], frames=p["frames"],
                 seconds_per_frame=p["seconds_per_frame"], masklets=len(p["masklets"]),
                 peak_gpu_allocated_gib=p["peak_gpu_allocated_gib"], peak_gpu_reserved_gib=p["peak_gpu_reserved_gib"])
            for p in res["prompts"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--frames", required=True)
    a = ap.parse_args()
    R.no_test_labels()
    import torch
    torch.manual_seed(0)
    got = R.sha256(a.checkpoint)
    if got != a.sha256:
        raise SystemExit(f"!! STOP: {a.checkpoint} hashes to {got}, not {a.sha256}")
    t0 = time.time()
    from sam3.model_builder import build_sam3_video_predictor
    predictor = build_sam3_video_predictor(checkpoint_path=a.checkpoint)
    print("READY " + json.dumps(dict(model_load_seconds=round(time.time() - t0, 1), torch=torch.__version__,
                                     cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
                                     pid=os.getpid())), flush=True)
    for line in sys.stdin:
        if not line.strip():
            break
        job = json.loads(line)
        v, out_dir = job["video"], job["out"]
        os.makedirs(out_dir, exist_ok=True)
        tv = time.time()
        try:
            res = R.run_video(predictor, v, a.frames, names_of(job["prompts"]), reuse_backbone=False)
        except torch.OutOfMemoryError as e:
            rep = dict(video=v, ok=False, error="CUDA out of memory", message=str(e).splitlines()[0],
                       peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
            json.dump(rep, open(os.path.join(out_dir, f"{v}.error.json"), "w"), indent=1)
            print("RESULT " + json.dumps(rep), flush=True)
            torch.cuda.empty_cache()
            continue
        res.update(checkpoint_sha256=got, prompts_run=len(res["prompts"]), torch=torch.__version__,
                   gpu=torch.cuda.get_device_name(0))
        path = os.path.join(out_dir, f"{v}.json.gz")
        with gzip.open(path + ".part", "wt") as f:
            json.dump(res, f)
        os.replace(path + ".part", path)
        torch.cuda.empty_cache()
        print("RESULT " + json.dumps(dict(video=v, ok=True, frames=res["frames"], prompts=res["prompts_run"],
                                          seconds_total=res["seconds_total"], wall_seconds=round(time.time() - tv, 2),
                                          peak_gpu_allocated_gib=res["peak_gpu_allocated_gib"],
                                          peak_gpu_reserved_gib=res["peak_gpu_reserved_gib"], passes=passes(res),
                                          file=path, sha256=R.sha256(path))), flush=True)
    predictor.shutdown()


if __name__ == "__main__":
    main()
