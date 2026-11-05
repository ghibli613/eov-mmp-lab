#!/usr/bin/env python3
"""The D7 check on the 48 GB machine (A34). One SAM 3 process on one GPU; no skips.

    python studies/d7/check/d7_run.py check0   # G0's smoke video 2 (...00010024, HELD-A), 35 names, against G0
    python studies/d7/check/d7_run.py ref35    # only if check 0 is not bitwise: the 35 names on the final list
    python studies/d7/check/d7_run.py arms     # arm A ("object") and arm B (person, animal, vehicle, object)

G1's Worker (studies/rpt/g1/box/g1_driver.py) runs studies/d7/check/d7_worker.py through its own overrides
(G1_WORKER, G1_BOX, G1_OUT); G0's comparison (studies/rpt/g0/sam3_compare.py) decides check 0. A failed
job of any kind (out of memory included) writes ~/d7_box/STOP_<stage>.json and exits 5: no video is
skipped. Rerunning a stage skips the (stage, video) pairs already in the manifest with a matching file.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import hashlib
import json
import os
import queue
import time

HERE = os.path.dirname(os.path.abspath(__file__))
D7 = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(D7))
HOME = os.path.expanduser("~")
os.environ.setdefault("G1_WORKER", os.path.join(HERE, "d7_worker.py"))
os.environ.setdefault("G1_BOX", os.path.join(HOME, "d7_box"))
os.environ.setdefault("G1_OUT", os.path.join(HOME, "d7_out"))
sys.path[:0] = [os.path.join(REPO, "studies", "rpt", "g1", "box")]
import g1_driver as D      # noqa: E402
import sam3_compare        # noqa: E402

CHECK0 = "ILSVRC2015_train_00010024"            # G0's smoke video 2: the first G0 smoke video in HELD-A
REF0 = os.path.join(REPO, "studies", "rpt", "g0", "sam3_out", f"{CHECK0}.json.gz")
REF0_SHA = "a2929790527dea7165b1abf3a59ac4d314bd16389799e5743ecf6faf8fdddf3f"   # G0's committed output
LIST = os.environ.get("D7_LIST", os.path.join(HERE, "video_list.json"))   # D7_LIST: the laptop's driver tests only
ARM_A = [["object", "object"]]
ARM_B = [["person", "person"], ["animal", "animal"], ["vehicle", "vehicle"], ["object", "object"]]
MANIFEST = os.path.join(D.OUT, "manifest.jsonl")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def final_list():
    want = open(LIST + ".sha256").read().split()[0]
    if sha256(LIST) != want:
        raise SystemExit("!! STOP: video_list.json differs from its .sha256")
    vl = json.load(open(LIST))
    if not vl["final"]:
        raise SystemExit("!! STOP: the guard stopped the check; there is no list")
    return vl["final"]


def done():
    out = set()
    if os.path.exists(MANIFEST):
        for r in map(json.loads, open(MANIFEST)):
            p = os.path.join(D.OUT, r["stage"], f"{r['video']}.json.gz")
            if os.path.exists(p) and sha256(p) == r["sha256"]:
                out.add((r["stage"], r["video"]))
    return out


def run_jobs(stage, jobs):
    """jobs: [(video, prompts)] in order, all in one worker on GPU 0. Any failure stops the check."""
    os.makedirs(D.BOX, exist_ok=True)
    have = done()
    todo = [(v, p) for v, p in jobs if (stage, v) not in have]
    if not todo:
        return
    q = queue.Queue()
    w = D.Worker(f"{stage}_w0", 0, False, q, 8)
    D.wait_ready([w], main=False)
    for v, prompts in todo:
        w.send(dict(video=v, out=os.path.join(D.OUT, stage), prompts=prompts))
        while True:
            try:
                _, r = q.get(timeout=30)
                break
            except queue.Empty:
                if w.dead():
                    D.stop(5, stage, dict(video=v, error="worker died"))
        if not r.get("ok"):
            D.stop(5, stage, dict(video=v, result=r))
        D.append(MANIFEST, dict(r, stage=stage, t_end=round(time.time(), 2)))
        D.log(f"{stage} {v}: {r['seconds_total']:.0f} s, peak {r['peak_gpu_allocated_gib']:.1f} GiB, "
              f"masklets {[p['masklets'] for p in r['passes']]}")
    w.close()


def check0():
    if sha256(REF0) != REF0_SHA:
        raise SystemExit(f"!! STOP: {REF0} differs from its pin")
    run_jobs("check0", [(CHECK0, "all35")])
    got = os.path.join(D.OUT, "check0", f"{CHECK0}.json.gz")
    c = sam3_compare.compare(sam3_compare.load(REF0), sam3_compare.load(got))
    identical = bool(c["bitwise_identical"]) and c["prompts_compared"] == 35
    res = dict(video=CHECK0, identical=identical, compare=c, reference=REF0, reference_sha256=REF0_SHA)
    if not identical:
        import sam3_diff
        res["difference_size"] = sam3_diff.size(sam3_compare.load(REF0), sam3_compare.load(got))
    D.save("check0.json", res)
    D.log(f"check 0: {'bitwise identical to G0: G1 files are the reference' if identical else 'NOT identical: the 35-name reference is rerun here'}")


def check0_identical():
    p = os.path.join(D.BOX, "check0.json")
    if not os.path.exists(p):
        raise SystemExit("!! check 0 has not run")
    return json.load(open(p))["identical"]


def ref35():
    if check0_identical():
        D.log("ref35: not needed (check 0 identical)")
        return
    run_jobs("ref35", [(v, "all35") for v in final_list()])


def arms():
    check0_identical()                                  # check 0 must have run
    jobs_a = [(v, ARM_A) for v in final_list()]
    jobs_b = [(v, ARM_B) for v in final_list()]
    run_jobs("armA", jobs_a)
    run_jobs("armB", jobs_b)
    D.save("d7_done.json", dict(videos=final_list(), stages=sorted({s for s, _ in done()})))


if __name__ == "__main__":
    D.NGPU = 1
    dict(check0=check0, ref35=ref35, arms=arms)[sys.argv[1]]()
