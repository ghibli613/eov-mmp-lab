#!/usr/bin/env python3
"""RPT G1 on the machine (A29): the first-hour checks 0-5, the first 20 videos, the projection, then
the other 140; every stop rule A29 names. Runs in /venv/main (no torch needed); SAM 3 runs in
studies/rpt/g1/sam3_worker.py processes in ~/sam3env.

    python studies/rpt/g1/box/g1_driver.py checks     # checks 0-5 -> ~/rpt_g1_box/g1_config.json
    python studies/rpt/g1/box/g1_driver.py videos     # the first 20, the projection, the other 140
    python studies/rpt/g1/box/g1_driver.py all        # both

The cap's clock starts when the checks start (~/rpt_g1_box/g1_t0); billed hours = wall clock x 1 GPU.
Exit codes (each writes ~/rpt_g1_box/STOP_*.json; then stop and ask the author):
  3  check 0: the default run does not reproduce G0 bitwise
  4  the cap: actual SAM 3 hours reached 35 before the 160 videos were done
  5  a video failed to run (A29 names the first 20; any other failure also stops)
  6  the projection after the first 20 exceeds 35 hours
  7  offload did not pass (check 4 or check 5): stopped before ...00280000
Restartable: a video whose output is in the manifest with its SHA256 is not run again.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import os
import queue
import subprocess
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
G1 = os.path.dirname(HERE)
RPT = os.path.dirname(G1)
REPO = os.path.abspath(os.path.join(RPT, "..", ".."))
sys.path[:0] = [G1, os.path.join(RPT, "g0")]

import schedule as S      # noqa: E402
import sam3_compare       # noqa: E402  (G0's bitwise comparison)
import sam3_diff          # noqa: E402

HOME = os.path.expanduser("~")
ENV = os.environ.get                             # G1_* overrides: the laptop's driver test only
BOX, OUT = ENV("G1_BOX", os.path.join(HOME, "rpt_g1_box")), ENV("G1_OUT", os.path.join(HOME, "rpt_g1_out"))
SAM3PY = ENV("G1_SAM3PY", os.path.join(HOME, "sam3env", "bin", "python"))
WORKER = ENV("G1_WORKER", os.path.join(G1, "sam3_worker.py"))
CKPT = os.path.join(HOME, "sam3_ckpt", "sam3.pt")
CKPT_SHA = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
FRAMES = os.path.join(REPO, "data", "vidvrd", "frames")
SPLIT = os.path.join(REPO, "pilot_analysis", "stage1", "train_split.json")
V1 = "ILSVRC2015_train_00008005"
REF = os.path.join(RPT, "g0", "sam3_out", f"{V1}.json.gz")
REF_SHA = "aed5945d1973b8de25876a7069fd5fcfa92ddd271f97ca24abf48380ca705db4"
T0 = os.path.join(BOX, "g1_t0")
CONFIG = os.path.join(BOX, "g1_config.json")
MANIFEST = os.path.join(BOX, "sam3_manifest.jsonl")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {msg}"
    print(line, flush=True)
    with open(os.path.join(BOX, "g1_driver.log"), "a") as f:
        f.write(line + "\n")


def save(name, obj):
    with open(os.path.join(BOX, name), "w") as f:
        json.dump(obj, f, indent=1)


def stop(code, name, obj):
    save(f"STOP_{name}.json", obj)
    log(f"!! STOP ({name}): {json.dumps(obj)[:400]} -- ask the author")
    Worker.kill_all()
    sys.exit(code)


def hours():
    return (time.time() - float(open(T0).read())) / 3600


def cap_guard():
    h = hours()
    if h >= S.CAP_HOURS:
        stop(4, "cap", dict(billed_hours=h, cap=S.CAP_HOURS, done=len(done_videos())))


# ---------------------------------------------------------------- SAM 3 workers
class Worker:
    live: list = []
    results: "queue.Queue" = queue.Queue()

    def __init__(self, name, compile_=False):
        self.name, self.compile, self.job, self.ready = name, compile_, None, None
        os.makedirs(os.path.join(BOX, "workers"), exist_ok=True)
        self.err = open(os.path.join(BOX, "workers", f"{name}.log"), "a")
        cmd = [SAM3PY, "-B", WORKER, "--checkpoint", CKPT, "--sha256", CKPT_SHA,
               "--frames", FRAMES] + (["--compile"] if compile_ else [])
        self.p = subprocess.Popen(cmd, cwd=REPO, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err,
                                  text=True, bufsize=1)
        self.t_start = time.time()
        threading.Thread(target=self._read, daemon=True).start()
        Worker.live.append(self)

    def _read(self):
        for line in self.p.stdout:
            if line.startswith("READY "):
                self.ready = json.loads(line[6:])
                self.ready["seconds_to_ready"] = round(time.time() - self.t_start, 1)
            elif line.startswith("RESULT "):
                Worker.results.put((self, json.loads(line[7:])))
            self.err.write(line)
            self.err.flush()

    def send(self, job):
        self.job, self.t_job = job, time.time()
        self.p.stdin.write(json.dumps(job) + "\n")
        self.p.stdin.flush()

    def dead(self):
        return self.p.poll() is not None

    def close(self):
        if not self.dead():
            self.p.stdin.write("\n")
            self.p.stdin.flush()
            self.p.wait(timeout=600)
        Worker.live.remove(self)

    @classmethod
    def kill_all(cls):
        for w in list(cls.live):
            if not w.dead():
                w.p.kill()
        cls.live.clear()


def wait_ready(workers):
    while not all(w.ready for w in workers):
        if any(w.dead() for w in workers):
            raise RuntimeError(f"a worker died before READY: {[w.name for w in workers if w.dead()]}")
        time.sleep(2)


def run_check(label, n, compile_, reuse, offload):
    """n workers, each runs smoke video 1 at the same moment; every output compared with G0's."""
    ws = [Worker(f"{label}_w{i}", compile_) for i in range(n)]
    wait_ready(ws)
    t = time.time()
    for i, w in enumerate(ws):
        w.send(dict(video=V1, out=os.path.join(OUT, "checks", label, f"w{i}"), reuse_backbone=reuse,
                    offload_video_to_cpu=offload))
    res = {}
    while len(res) < n:
        try:
            w, r = Worker.results.get(timeout=10)
            res[w.name] = r
        except queue.Empty:
            if any(w.dead() and w.name not in res for w in ws):
                res.update({w.name: dict(ok=False, error="worker died") for w in ws if w.dead() and w.name not in res})
        cap_guard()
    wall = time.time() - t
    for w in ws:
        w.close()
    ref = sam3_compare.load(REF)
    rows = []
    for name, r in sorted(res.items()):
        row = dict(worker=name, **{k: r.get(k) for k in ("ok", "error", "seconds_total", "wall_seconds",
                                                         "peak_gpu_allocated_gib", "peak_gpu_reserved_gib",
                                                         "backbone_cache", "sha256")})
        if r.get("ok"):
            got = sam3_compare.load(r["file"])
            c = sam3_compare.compare(ref, got)
            row.update(identical=bool(c["bitwise_identical"]) and c["prompts_compared"] == S.PROMPTS,
                       compare=c)
            if not row["identical"]:
                row["difference_size"] = sam3_diff.size(ref, got)
        else:
            row["identical"] = False
        rows.append(row)
    out = dict(label=label, workers=n, compile=compile_, reuse_backbone=reuse, offload_video_to_cpu=offload,
               wall_seconds=round(wall, 2), passed=all(r["identical"] for r in rows), runs=rows,
               worker_ready=[w.ready for w in ws], billed_hours_so_far=round(hours(), 3))
    log(f"check {label}: {'PASS' if out['passed'] else 'NOT IDENTICAL'} wall {wall:.0f}s "
        f"peaks {[r.get('peak_gpu_allocated_gib') for r in rows]}")
    return out


def checks():
    os.makedirs(BOX, exist_ok=True)
    if sha256(REF) != REF_SHA:
        raise SystemExit("!! G0's reference output does not match its SHA256")
    if not os.path.exists(T0):
        open(T0, "w").write(f"{time.time()}\n")
    log("checks start (the cap's clock runs from here)")
    res = {}
    res["c0"] = run_check("c0_default", 1, False, False, False)
    save("checks.json", res)
    if not res["c0"]["passed"]:
        stop(3, "check0", res["c0"])
    res["c1"] = run_check("c1_A1", 1, False, True, False)
    save("checks.json", res)
    res["c2"] = {}
    a2 = 1
    for n in (2, 3):
        res["c2"][n] = c = run_check(f"c2_A2_{n}", n, False, False, False)
        c["throughput_vs_c0"] = round(n * res["c0"]["wall_seconds"] / c["wall_seconds"], 3)
        save("checks.json", res)
        if not c["passed"]:
            break
        a2 = n
    res["c3"] = run_check("c3_A3", 1, True, False, False)
    save("checks.json", res)
    res["c4"] = run_check("c4_offload", 1, False, False, True)
    save("checks.json", res)
    passing = [o for o, ok in (("A1", res["c1"]["passed"]), ("A2", a2 > 1), ("A3", res["c3"]["passed"]),
                               ("offload", res["c4"]["passed"])) if ok]
    res["c5"] = []
    opts = list(passing)
    while opts:
        c = run_check("c5_" + "+".join(opts), a2 if "A2" in opts else 1, "A3" in opts, "A1" in opts, "offload" in opts)
        c["options"] = list(opts)
        res["c5"].append(c)
        save("checks.json", res)
        if c["passed"]:
            break
        opts = S.drop_last(opts)
    r0 = res["c0"]["runs"][0]
    r4 = res["c4"]["runs"][0]
    cfg = dict(A1="A1" in opts, A2=a2 if "A2" in opts else 1, A3="A3" in opts, offload="offload" in opts,
               passing_alone=passing, check5_options=opts,
               r_check0=S.rate(r0["wall_seconds"], 130), r_check4=S.rate(r4["wall_seconds"], 130) if r4.get("ok") else None,
               billed_hours_after_checks=round(hours(), 3))
    save("g1_config.json", cfg)
    log(f"checks done: {json.dumps(cfg)}")
    return cfg


# ---------------------------------------------------------------- the videos
def held_frames():
    held = sorted(json.load(open(SPLIT))["held"])
    assert len(held) == 160
    return held, {v: len(os.listdir(os.path.join(FRAMES, v))) for v in held}


def done_videos():
    out = {}
    if os.path.exists(MANIFEST):
        for line in open(MANIFEST):
            r = json.loads(line)
            p = os.path.join(OUT, "sam3", f"{r['video']}.json.gz")
            if os.path.exists(p) and sha256(p) == r["sha256"]:
                out[r["video"]] = r
    return out


def run_videos(todo, frames, cfg, workers, phase):
    """Runs `todo` in order under the scheduling rule; returns the phase's wall seconds."""
    pending = list(todo)
    running = {}
    t = time.time()
    while pending or running:
        cap_guard()
        while pending:
            v = pending[0]
            o = S.job_options(v, frames[v], cfg)
            cand = dict(frames=frames[v], **o)
            free = [w for w in workers if w.job is None]
            if not free or not S.can_start(cand, [running[w.name][1] for w in workers if w.name in running],
                                           len(workers)):
                break
            w = free[0]
            w.send(dict(video=v, out=os.path.join(OUT, "sam3"), reuse_backbone=o["reuse_backbone"],
                        offload_video_to_cpu=o["offload_video_to_cpu"]))
            running[w.name] = (v, cand, time.time())
            pending.pop(0)
            log(f"[{phase}] start {v} ({frames[v]} frames) on {w.name} {o}")
        try:
            w, r = Worker.results.get(timeout=10)
        except queue.Empty:
            for w in workers:
                if w.name in running and w.dead():
                    stop(5, "video", dict(phase=phase, video=running[w.name][0], error="worker died"))
            continue
        v, cand, ts = running.pop(w.name)
        w.job = None
        if not r.get("ok"):
            stop(5, "video", dict(phase=phase, video=v, result=r))
        row = dict(r, phase=phase, options=cand, billed_hours_at_end=round(hours(), 4))
        with open(MANIFEST, "a") as f:
            f.write(json.dumps(row) + "\n")
        log(f"[{phase}] done {v}: {r['seconds_total']:.0f}s in-run, peak {r['peak_gpu_allocated_gib']:.1f} GiB, "
            f"billed {hours():.2f} h")
    return time.time() - t


def videos(cfg):
    held, frames = held_frames()
    first20, rest = held[:20], held[20:]
    assert max(frames[v] for v in first20) <= S.LONG
    done = done_videos()
    t20 = time.time()                                # billed: the workers' start-up counts
    workers = [Worker(f"g1_w{i}", cfg["A3"]) for i in range(cfg["A2"])]
    wait_ready(workers)
    todo20 = [v for v in first20 if v not in done]
    wall20 = run_videos(todo20, frames, cfg, workers, "first20")
    proj_path = os.path.join(BOX, "projection.json")
    if os.path.exists(proj_path):
        proj = json.load(open(proj_path))
        log("projection already made: " + json.dumps(proj))
    else:
        if len(todo20) != 20:
            stop(6, "projection", dict(error="the first 20 were not all run in this phase; the rate cannot be measured"))
        r20 = S.rate(time.time() - t20, sum(frames[v] for v in first20))
        # without offload, ...00280000 is not run on this GPU (A29 check 4), so not projected here
        proj_rest = rest if cfg["offload"] else [v for v in rest if v != S.OFFLOAD_VIDEO]
        proj = S.projection(hours(), proj_rest, frames, r20, cfg["r_check0"], cfg["r_check4"])
        proj["first20_wall_seconds_in_run"] = round(wall20, 1)
        save("projection.json", proj)
        log("projection: " + json.dumps(proj))
        if proj["stop"]:
            stop(6, "projection", proj)
    order = S.order_rest(rest, frames)
    if not cfg["offload"]:
        order = [v for v in order if v != S.OFFLOAD_VIDEO]
    run_videos([v for v in order if v not in done], frames, cfg, workers, "rest")
    if not cfg["offload"]:
        stop(7, "offload", dict(error="offload did not pass; ...00280000 is not run on this GPU",
                                done=len(done_videos())))
    for w in list(workers):
        w.close()
    d = done_videos()
    assert sorted(d) == held, sorted(set(held) - set(d))
    save("sam3_done.json", dict(videos=len(d), billed_hours=round(hours(), 3),
                                files={v: d[v]["sha256"] for v in sorted(d)}))
    log(f"all 160 SAM 3 files written; billed SAM 3 hours {hours():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["checks", "videos", "all"])
    a = ap.parse_args()
    os.makedirs(BOX, exist_ok=True)
    threading.Thread(target=gpu_monitor, daemon=True).start()
    if a.what in ("checks", "all"):
        cfg = checks()
    else:
        cfg = json.load(open(CONFIG))
    if a.what in ("videos", "all"):
        videos(cfg)


def gpu_monitor():
    path = os.path.join(BOX, "gpu_memory.log")
    while True:
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=30).stdout.strip()
            with open(path, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {q} | ram_avail_gib "
                        f"{int(open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0]) / 2**20:.1f}\n")
        except Exception:
            pass
        time.sleep(60)


if __name__ == "__main__":
    main()
