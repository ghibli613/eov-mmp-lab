#!/usr/bin/env python3
"""RPT G1 on the 48 GB machine: A29b's `...00280000` and A29c's videos moved off the 8x machine
(studies/rpt/g1/logs/box/moved_to_l40.json), longest first. Runs in /venv/main.

    python studies/rpt/g1/box48/l40_run.py check0      # smoke video 1 at defaults, bitwise against G0
    python studies/rpt/g1/box48/l40_run.py video       # each video: its own process, defaults, no offload/A1/A2/A3
    python studies/rpt/g1/box48/l40_run.py preflight   # the EOV-MMP preflight, against the 8x machine's
    python studies/rpt/g1/box48/l40_run.py eov         # R0w, R0 and (if the preflight matched) box sensitivity

check0 failing writes ~/rpt_g1_box/STOP_check0.json and exits 3 (A29b: stop and ask); any video failing
writes STOP_video.json and exits 5 (A29a; there is no further machine to move it to). The SAM 3 side
uses the 8x machine's driver code (studies/rpt/g1/box/g1_driver.py: Worker, run_check); the EOV-MMP
side its eov_held.py and eov_parallel.py's comparison. The bars' R0w and R0 tracks stay the 8x
machine's (A29b); the ones computed here serve box sensitivity only.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import json
import os
import queue
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
G1 = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(G1, "box"), G1]
import g1_driver as D      # noqa: E402
import eov_parallel as EP  # noqa: E402

VIDEO = "ILSVRC2015_train_00280000"
MOVED = os.path.join(G1, "logs", "box", "moved_to_l40.json")   # A29c, from the 8x machine's copy-back
REF8 = os.path.join(D.REPO, "studies", "rpt", "g1", "box_out", "preflight_eov", "g0")   # the 8x machine's, uploaded


def videos():
    vs = [VIDEO] + [m["video"] for m in json.load(open(MOVED))]
    return sorted(vs, key=lambda v: (-len(os.listdir(os.path.join(D.FRAMES, v))), v))


def check0():
    os.makedirs(D.BOX, exist_ok=True)
    if not os.path.exists(D.T0):
        open(D.T0, "w").write(f"{time.time()}\n")
    print(f"videos for this machine: {videos()}", flush=True)       # their frames are here
    D.NGPU = 1
    r = D.run_check("c0_default", [0], 1, False, False, False, main=False, omp=8)
    D.save("checks_l40.json", r)
    if not r["passed"]:
        D.stop(3, "check0", dict(runs=r["runs"]))
    print("check 0 on the 48 GB GPU: identical to G0", flush=True)


def video():
    if not json.load(open(os.path.join(D.BOX, "checks_l40.json")))["passed"]:
        raise SystemExit("!! check 0 has not passed")
    D.NGPU = 1
    files = {}
    if os.path.exists(D.MANIFEST):
        files = {r["video"]: r["sha256"] for r in map(json.loads, open(D.MANIFEST))}
    for i, v in enumerate(videos()):
        if v in files:
            continue
        q = queue.Queue()
        w = D.Worker(f"l40_w{i}", 0, False, q, 8)
        D.wait_ready([w], main=False)
        w.send(dict(video=v, out=os.path.join(D.OUT, "sam3"), reuse_backbone=False, offload_video_to_cpu=False))
        while True:
            try:
                _, r = q.get(timeout=30)
                break
            except queue.Empty:
                if w.dead():
                    D.stop(5, "video", dict(video=v, error="worker died"))
        w.close()
        if not r.get("ok"):
            D.stop(5, "video", dict(video=v, result=r))
        D.append(D.MANIFEST, dict(r, gpu=0, options=dict(reuse_backbone=False, offload_video_to_cpu=False, alone=True),
                                  group="l40", machine="48 GB (A29b, A29c)", t_end=round(time.time(), 2)))
        files[v] = r["sha256"]
        print(f"{v}: {r['seconds_total']:.0f} s, peak {r['peak_gpu_allocated_gib']:.1f} GiB", flush=True)
    D.save("sam3_done.json", dict(videos=len(files), files=files, machine="48 GB (A29b, A29c)"))


def preflight():
    out = os.path.join(D.OUT, "preflight_eov", "l40")
    sam3 = os.path.join(D.REPO, "studies", "rpt", "g0", "sam3_out")
    p = EP.launch(0, EP.PREFLIGHT, sam3, out, 8, "l40")
    if p.wait():
        raise SystemExit("!! preflight: eov_held failed")
    res = {v: EP.same_except_timing(os.path.join(out, f"{v}.json.gz"), os.path.join(REF8, f"{v}.json.gz"))
           for v in EP.PREFLIGHT}
    D.save("eov_preflight_l40.json", dict(identical_to_8x=res))
    print(f"EOV preflight vs the 8x machine: {res}", flush=True)


def eov():
    ok = all(json.load(open(os.path.join(D.BOX, "eov_preflight_l40.json")))["identical_to_8x"].values())
    vs = videos()
    no_sam3 = [] if ok else vs
    print(f"box sensitivity for {vs}: {'yes' if ok else 'no (preflight differs; A29b: judged without them)'}",
          flush=True)
    p = EP.launch(0, vs, os.path.join(D.OUT, "sam3"), os.path.join(D.OUT, "eov"), 8, "l40", no_sam3)
    if p.wait():
        raise SystemExit("!! eov_held failed")


if __name__ == "__main__":
    dict(check0=check0, video=video, preflight=preflight, eov=eov)[sys.argv[1]]()
