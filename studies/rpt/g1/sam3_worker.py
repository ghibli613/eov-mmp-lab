#!/usr/bin/env python3
"""RPT G1 (A29): one long-lived SAM 3 process that runs videos one at a time, as G1's driver sends
them. Runs in the SAM 3 environment. The per-video work is the committed runner's, unchanged
(studies/rpt/sam3_runner.py at b249c1d: run_video, Recorder, BackboneCache, prompts); this file only
builds the predictor with A29's implementation options and reads jobs.

    ~/sam3env/bin/python -B studies/rpt/g1/sam3_worker.py --checkpoint CKPT --sha256 SHA \
        --frames FRAMES [--compile]          # then one JSON job per line on stdin

A job: {"video": V, "out": DIR, "reuse_backbone": bool, "offload_video_to_cpu": bool}. The worker
writes DIR/V.json.gz (atomically, via a .part file) and answers one JSON line on stdout that starts
with "RESULT ". An empty line or EOF ends the worker.

Options (A29, each used only after its bitwise check):
- A1 `reuse_backbone`: the runner's own BackboneCache (G0's measurement);
- A3 `--compile`: SAM 3's own `compile=True` (build_sam3_video_predictor -> build_sam3_video_model);
- `offload_video_to_cpu`: SAM 3's own start_session option (sam3_base_predictor.py:140-147), passed
  by adding it to the runner's start_session request;
- A2 is several workers at once; nothing in this file.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RPT = os.path.dirname(HERE)
sys.path.insert(0, RPT)
import sam3_runner as R  # noqa: E402


class _Offload:
    """The predictor, with offload_video_to_cpu added to start_session requests while `on`."""

    def __init__(self, predictor):
        self.p, self.on = predictor, False

    def handle_request(self, request):
        if self.on and request.get("type") == "start_session":
            request = dict(request, offload_video_to_cpu=True)
        return self.p.handle_request(request)

    def __getattr__(self, name):
        return getattr(self.p, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--compile", action="store_true", help="A3: SAM 3's compile=True")
    a = ap.parse_args()
    R.no_test_labels()
    import torch
    torch.manual_seed(0)
    got = R.sha256(a.checkpoint)
    if got != a.sha256:
        raise SystemExit(f"!! STOP: {a.checkpoint} hashes to {got}, not {a.sha256}")
    names = R.prompts()
    t0 = time.time()
    from sam3.model_builder import build_sam3_video_predictor
    predictor = build_sam3_video_predictor(checkpoint_path=a.checkpoint, compile=a.compile)
    pred = _Offload(predictor)
    print("READY " + json.dumps(dict(model_load_seconds=round(time.time() - t0, 1), compile=a.compile,
                                     torch=torch.__version__, cudnn=torch.backends.cudnn.version(),
                                     gpu=torch.cuda.get_device_name(0), pid=os.getpid())), flush=True)
    for line in sys.stdin:
        if not line.strip():
            break
        job = json.loads(line)
        v, out_dir = job["video"], job["out"]
        os.makedirs(out_dir, exist_ok=True)
        pred.on = bool(job.get("offload_video_to_cpu"))
        tv = time.time()
        try:
            res = R.run_video(pred, v, a.frames, names, reuse_backbone=bool(job.get("reuse_backbone")))
        except torch.OutOfMemoryError as e:
            rep = dict(video=v, ok=False, error="CUDA out of memory", message=str(e).splitlines()[0],
                       peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
            json.dump(rep, open(os.path.join(out_dir, f"{v}.error.json"), "w"), indent=1)
            print("RESULT " + json.dumps(rep), flush=True)
            torch.cuda.empty_cache()
            continue
        finally:
            pred.on = False
        res.update(checkpoint_sha256=got, prompts_run=len(res["prompts"]), torch=torch.__version__,
                   gpu=torch.cuda.get_device_name(0), compile=a.compile,
                   offload_video_to_cpu=bool(job.get("offload_video_to_cpu")))
        path = os.path.join(out_dir, f"{v}.json.gz")
        with gzip.open(path + ".part", "wt") as f:
            json.dump(res, f)
        os.replace(path + ".part", path)
        torch.cuda.empty_cache()
        print("RESULT " + json.dumps(dict(video=v, ok=True, frames=res["frames"], prompts=res["prompts_run"],
                                          seconds_total=res["seconds_total"], wall_seconds=round(time.time() - tv, 2),
                                          peak_gpu_allocated_gib=res["peak_gpu_allocated_gib"],
                                          peak_gpu_reserved_gib=res["peak_gpu_reserved_gib"],
                                          backbone_cache=res.get("backbone_cache"), file=path,
                                          sha256=R.sha256(path))), flush=True)
    predictor.shutdown()


if __name__ == "__main__":
    main()
