#!/usr/bin/env python3
"""RPT G1 on the machine (A29, A29a): check 0 on every GPU, checks 1-4 in parallel on separate GPUs,
check 5, then the 160 videos longest first over the GPUs; every stop rule. Runs in /venv/main (no
torch needed); SAM 3 runs in studies/rpt/g1/sam3_worker.py processes in ~/sam3env, each pinned to
one GPU by CUDA_VISIBLE_DEVICES.

    python studies/rpt/g1/box/g1_driver.py checks     # checks 0-5 -> ~/rpt_g1_box/g1_config.json
    python studies/rpt/g1/box/g1_driver.py videos     # the long videos and the first 20, the projection, the rest
    python studies/rpt/g1/box/g1_driver.py all        # both

The cap's clock starts with check 0 (~/rpt_g1_box/g1_t0); billed GPU-hours = wall clock x GPUs.
Exit codes (each writes ~/rpt_g1_box/STOP_*.json; then stop and ask the author):
  3  check 0: some GPU's default run does not reproduce G0 bitwise
  4  the cap: 48 billed GPU-hours reached before the 160 videos were done
  5  a video failed to run
  6  the projection after the first 20 exceeds 48 GPU-hours
  7  offload did not pass (check 4 or check 5): the other 159 ran, ...00280000 did not
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
ENV = os.environ.get                             # G1_* overrides: the laptop's driver tests only
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
EVENTS = os.path.join(BOX, "schedule_events.jsonl")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def n_gpus():
    if ENV("G1_GPUS"):
        return int(ENV("G1_GPUS"))
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    return sum(1 for line in out.splitlines() if line.startswith("GPU "))


def ram_gb():
    """The instance's RAM: /proc/meminfo, or the container's cgroup limit if lower (Vast.ai
    splits a machine's RAM by GPU through it). GB = 1e9 bytes."""
    if ENV("G1_RAM_GB"):
        return float(ENV("G1_RAM_GB"))
    total = int(open("/proc/meminfo").read().split("MemTotal:")[1].split()[0]) * 1024
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
            if v.isdigit():
                total = min(total, int(v))
        except OSError:
            pass
    return total / 1e9


def cores():
    return int(ENV("G1_CORES") or len(os.sched_getaffinity(0)))


def available_gb():
    """A29c's live guard: the lower of /proc/meminfo's MemAvailable and the memory cgroup's limit minus
    its usage, inactive file cache counted as free. GB = 1e9 bytes."""
    if ENV("G1_AVAIL_GB"):
        return float(ENV("G1_AVAIL_GB"))
    avail = int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) * 1024
    try:
        mx = open("/sys/fs/cgroup/memory.max").read().strip()
        if mx.isdigit():
            cur = int(open("/sys/fs/cgroup/memory.current").read())
            stat = dict(line.split() for line in open("/sys/fs/cgroup/memory.stat"))
            avail = min(avail, int(mx) - cur + int(stat.get("inactive_file", 0)))
    except OSError:
        pass
    return avail / 1e9


def oom_kills():
    """The memory cgroup's oom_kill count (A29c: a kernel oom_kill of a SAM 3 process)."""
    try:
        for line in open(ENV("G1_OOM_EVENTS") or "/sys/fs/cgroup/memory.events"):
            k, v = line.split()
            if k == "oom_kill":
                return int(v)
    except OSError:
        pass
    return 0


NGPU = None                                      # set in main()


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {msg}"
    print(line, flush=True)
    with open(os.path.join(BOX, "g1_driver.log"), "a") as f:
        f.write(line + "\n")


def save(name, obj):
    with open(os.path.join(BOX, name), "w") as f:
        json.dump(obj, f, indent=1)


def append(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def stop(code, name, obj):
    save(f"STOP_{name}.json", obj)
    log(f"!! STOP ({name}): {json.dumps(obj)[:400]} -- ask the author")
    Worker.kill_all()
    os._exit(code)                               # also from a worker thread's caller; no cleanup needed


def gpu_hours():
    return NGPU * (time.time() - float(open(T0).read())) / 3600


def cap_guard():
    h = gpu_hours()
    if h >= S.CAP_GPU_HOURS:
        stop(4, "cap", dict(billed_gpu_hours=h, cap=S.CAP_GPU_HOURS, gpus=NGPU, done=len(done_videos())))


# ---------------------------------------------------------------- SAM 3 workers
class Worker:
    live: list = []
    lock = threading.Lock()

    def __init__(self, name, gpu, compile_, results, omp=8):
        self.name, self.gpu, self.compile, self.job, self.ready = name, gpu, compile_, None, None
        self.results = results
        os.makedirs(os.path.join(BOX, "workers"), exist_ok=True)
        self.err = open(os.path.join(BOX, "workers", f"{name}.log"), "a")
        cmd = [SAM3PY, "-B", WORKER, "--checkpoint", CKPT, "--sha256", CKPT_SHA,
               "--frames", FRAMES] + (["--compile"] if compile_ else [])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS=str(omp), MKL_NUM_THREADS=str(omp))
        self.p = subprocess.Popen(cmd, cwd=REPO, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err,
                                  text=True, bufsize=1, env=env)
        self.t_start = time.time()
        threading.Thread(target=self._read, daemon=True).start()
        with Worker.lock:
            Worker.live.append(self)

    def _read(self):
        for line in self.p.stdout:
            if line.startswith("READY "):
                self.ready = json.loads(line[6:])
                self.ready["seconds_to_ready"] = round(time.time() - self.t_start, 1)
            elif line.startswith("RESULT "):
                self.results.put((self, json.loads(line[7:])))
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
        with Worker.lock:
            if self in Worker.live:
                Worker.live.remove(self)

    @classmethod
    def kill_all(cls):
        with cls.lock:
            for w in cls.live:
                if not w.dead():
                    w.p.kill()
            cls.live.clear()


def wait_ready(workers, main=True):
    while not all(w.ready for w in workers):
        if any(w.dead() for w in workers):
            raise RuntimeError(f"a worker died before READY: {[w.name for w in workers if w.dead()]}")
        if main:
            cap_guard()
        time.sleep(1)


# ---------------------------------------------------------------- checks
def omp_for(concurrent):
    """A29a: OMP_NUM_THREADS = CPU cores / concurrent processes, at most 8."""
    return max(1, min(8, cores() // concurrent))


def run_check(label, gpus, n, compile_, reuse, offload, main=True, omp=None):
    """n workers on each GPU in `gpus`; each runs smoke video 1 at the same moment; every output is
    compared with G0's."""
    res_q = queue.Queue()
    omp = omp or omp_for(len(gpus) * n)
    ws = [Worker(f"{label}_g{g}_w{i}", g, compile_, res_q, omp) for g in gpus for i in range(n)]
    wait_ready(ws, main)
    t = time.time()
    for w in ws:
        w.send(dict(video=V1, out=os.path.join(OUT, "checks", label, f"g{w.gpu}_{w.name.rsplit('_', 1)[1]}"),
                    reuse_backbone=reuse, offload_video_to_cpu=offload))
    res = {}
    while len(res) < len(ws):
        try:
            w, r = res_q.get(timeout=5)
            res[w.name] = (w, r)
        except queue.Empty:
            for w in ws:
                if w.dead() and w.name not in res:
                    res[w.name] = (w, dict(ok=False, error="worker died"))
        if main:
            cap_guard()
    wall = time.time() - t
    for w in ws:
        w.close()
    ref = sam3_compare.load(REF)
    rows = []
    for name, (w, r) in sorted(res.items()):
        row = dict(worker=name, gpu=w.gpu, **{k: r.get(k) for k in ("ok", "error", "seconds_total", "wall_seconds",
                                                                    "peak_gpu_allocated_gib", "peak_gpu_reserved_gib",
                                                                    "backbone_cache", "sha256")})
        if r.get("ok"):
            got = sam3_compare.load(r["file"])
            c = sam3_compare.compare(ref, got)
            row.update(identical=bool(c["bitwise_identical"]) and c["prompts_compared"] == S.PROMPTS, compare=c)
            if not row["identical"]:
                row["difference_size"] = sam3_diff.size(ref, got)
        else:
            row["identical"] = False
        rows.append(row)
    out = dict(label=label, gpus=list(gpus), workers_per_gpu=n, omp_num_threads=omp, compile=compile_, reuse_backbone=reuse,
               offload_video_to_cpu=offload, wall_seconds=round(wall, 2), passed=all(r["identical"] for r in rows),
               runs=rows, worker_ready=[w.ready for w in ws])
    log(f"check {label} on GPUs {list(gpus)}: {'PASS' if out['passed'] else 'NOT IDENTICAL'} wall {wall:.0f}s "
        f"peaks {[r.get('peak_gpu_allocated_gib') for r in rows]}")
    return out


def check_a2(gpu, c0_wall, main, omp):
    """Check 2 on one GPU: 2 processes, then 3 if 2 passed."""
    out, a2 = {}, 1
    for n in (2, 3):
        out[n] = c = run_check(f"c2_A2_{n}", [gpu], n, False, False, False, main, omp)
        c["throughput_vs_c0"] = round(n * c0_wall / c["wall_seconds"], 3)
        if not c["passed"]:
            break
        a2 = n
    return out, a2


def checks():
    os.makedirs(BOX, exist_ok=True)
    if sha256(REF) != REF_SHA:
        raise SystemExit("!! G0's reference output does not match its SHA256")
    if not os.path.exists(T0):
        open(T0, "w").write(f"{time.time()}\n")
    log(f"checks start on {NGPU} GPUs (the cap's clock runs from here)")
    res = dict(machine=dict(gpus=NGPU, ram_gb=ram_gb(), cores=cores()))
    res["c0"] = run_check("c0_default", list(range(NGPU)), 1, False, False, False)
    save("checks.json", res)
    if not res["c0"]["passed"]:
        stop(3, "check0", dict(failed_gpus=[r["gpu"] for r in res["c0"]["runs"] if not r["identical"]],
                               runs=res["c0"]["runs"]))
    c0_wall = max(r["wall_seconds"] for r in res["c0"]["runs"])
    omp = omp_for(6 if NGPU >= 4 else 3)              # checks 1-4 at once: at most 6 processes
    jobs = dict(c1=lambda m: run_check("c1_A1", [0], 1, False, True, False, m, omp),
                c2=lambda m: check_a2(1 % NGPU, c0_wall, m, omp),
                c3=lambda m: run_check("c3_A3", [2 % NGPU], 1, True, False, False, m, omp),
                c4=lambda m: run_check("c4_offload", [3 % NGPU], 1, False, False, True, m, omp))
    if NGPU >= 4:                                     # A29a: checks 1-4 at once, each on its own GPU
        got = {}
        threads = {k: threading.Thread(target=lambda k=k, f=f: got.__setitem__(k, f(False)), daemon=True)
                   for k, f in jobs.items()}
        for t in threads.values():
            t.start()
        while any(t.is_alive() for t in threads.values()):
            cap_guard()
            time.sleep(2)
        if set(got) != set(jobs):
            raise RuntimeError(f"a check thread failed: {sorted(set(jobs) - set(got))}")
    else:
        got = {k: f(True) for k, f in jobs.items()}
    res["c1"], (res["c2"], a2), res["c3"], res["c4"] = got["c1"], got["c2"], got["c3"], got["c4"]
    save("checks.json", res)
    passing = [o for o, ok in (("A1", res["c1"]["passed"]), ("A2", a2 > 1), ("A3", res["c3"]["passed"]),
                               ("offload", res["c4"]["passed"])) if ok]
    res["c5"] = []
    opts = list(passing)
    while opts:
        n5 = a2 if "A2" in opts else 1                # with the videos' own thread count (A29a)
        c = run_check("c5_" + "+".join(opts), [0], n5, "A3" in opts, "A1" in opts, "offload" in opts,
                      omp=omp_for(NGPU * n5))
        c["options"] = list(opts)
        res["c5"].append(c)
        save("checks.json", res)
        if c["passed"]:
            break
        opts = S.drop_last(opts)
    r4 = res["c4"]["runs"][0]
    cfg = dict(A1="A1" in opts, A2=a2 if "A2" in opts else 1, A3="A3" in opts, offload="offload" in opts,
               passing_alone=passing, check5_options=opts, gpus=NGPU, ram_gb=res["machine"]["ram_gb"],
               a1_limit_gb=S.a1_limit_gb(res["machine"]["ram_gb"]), cores=res["machine"]["cores"],
               r_check0=S.rate(c0_wall, 130), r_check4=S.rate(r4["wall_seconds"], 130) if r4.get("ok") else None,
               billed_gpu_hours_after_checks=round(gpu_hours(), 3))
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


class Pool:
    """The SAM 3 workers, per GPU. A GPU that runs a long video alone has one worker; when that video
    ends it gets A2's other workers (A29a)."""

    def __init__(self, cfg, long_gpus):
        self.cfg, self.slots, self.results = cfg, cfg["A2"], queue.Queue()
        self.omp = omp_for(NGPU * self.slots)
        self.workers = {g: [] for g in range(NGPU)}
        for g in range(NGPU):
            for _ in range(1 if g in long_gpus else self.slots):
                self.add(g)

    def add(self, g):
        w = Worker(f"g1_g{g}_w{len(self.workers[g])}", g, self.cfg["A3"], self.results, self.omp)
        self.workers[g].append(w)

    def idle(self, g):
        return [w for w in self.workers[g] if w.ready and w.job is None and not w.dead()]

    def close(self):
        for ws in self.workers.values():
            for w in ws:
                w.close()


def move_to_l40(v, reason, detail, moved):
    """A29c: a video failing only from out-of-memory goes to the 48 GB run (A29b); partial output
    discarded; more than 3 such videos stop G1."""
    moved.append(dict(video=v, reason=reason, detail=detail, t=round(time.time(), 2)))
    save("moved_to_l40.json", moved)
    os.makedirs(os.path.join(BOX, "oom_failed"), exist_ok=True)
    for suffix in (".json.gz.part", ".error.json"):
        p = os.path.join(OUT, "sam3", v + suffix)
        if os.path.exists(p):
            os.replace(p, os.path.join(BOX, "oom_failed", v + suffix))
    log(f"!! {v}: {reason}; moved to the 48 GB run (A29c), {len(moved)} so far")
    if len(moved) > 3:
        stop(5, "oom_count", dict(moved=moved, error="more than 3 videos failed from out-of-memory (A29c)"))


def run_videos(cfg, held, frames, groups, done):
    a29c = cfg.get("a29c", False)
    moved = json.load(open(os.path.join(BOX, "moved_to_l40.json"))) if os.path.exists(
        os.path.join(BOX, "moved_to_l40.json")) else []
    oom_base = oom_kills()
    pool = Pool(cfg, long_gpus=set(range(min(len(groups["long"]), NGPU))))
    running = {}                                      # worker name -> (video, job, gpu, start time, worker)
    gone = set(done) | {m["video"] for m in moved}
    queued = [v for v in groups["long"] + groups["first20"] if v not in gone]
    rest = [v for v in groups["rest"] if v not in gone]
    first20 = set(groups["first20"])
    completed = set(done)
    seq = sum(1 for _ in open(EVENTS)) if os.path.exists(EVENTS) else 0
    released_rest = os.path.exists(os.path.join(BOX, "projection.json"))
    if released_rest:
        queued += rest
    while queued or running:
        cap_guard()
        if not released_rest and not (first20 - completed):
            proj = make_projection(cfg, frames, running, rest)
            if proj["stop"]:
                stop(6, "projection", proj)
            queued += rest
            released_rest = True
        on_gpu = [[r[1] for r in running.values() if r[2] == g] for g in range(NGPU)]
        for v in list(queued):
            o = S.job_options(v, frames[v], cfg)
            cand = dict(frames=frames[v], **o)
            ok = [g for g in range(NGPU) if pool.idle(g) and S.fits_gpu(cand, on_gpu[g], len(pool.workers[g]))]
            if not ok:
                continue
            g = min(ok, key=lambda x: (len(on_gpu[x]), x))
            if a29c:
                cand["reuse_backbone"] = S.use_a1(cand, on_gpu, cfg["a1_limit_gb"], S.A1_MB_PER_FRAME_A29C,
                                                  available_gb())
            else:
                cand["reuse_backbone"] = S.use_a1(cand, on_gpu, cfg["a1_limit_gb"])
            w = pool.idle(g)[0]
            w.send(dict(video=v, out=os.path.join(OUT, "sam3"), reuse_backbone=cand["reuse_backbone"],
                        offload_video_to_cpu=cand["offload_video_to_cpu"]))
            running[w.name] = (v, cand, g, time.time(), w)
            on_gpu[g].append(cand)
            seq += 1
            waiting = queued[:queued.index(v)]          # longer queued videos that fit on no GPU now
            append(EVENTS, dict(seq=seq, t=round(time.time(), 2), video=v, gpu=g, frames=frames[v],
                                waiting_longer=len(waiting),
                                alone=cand["alone"], a1=cand["reuse_backbone"], offload=cand["offload_video_to_cpu"],
                                gpu_frames_running=[j["frames"] for j in on_gpu[g]],
                                gpu_predicted_gib=round(sum(S.predicted_peak_gib(j["frames"]) for j in on_gpu[g]), 4),
                                a1_total_gb=round(sum(S.a1_ram_gb(j["frames"], cfg.get("a1_mb_per_frame", S.A1_MB_PER_FRAME))
                                                      for jj in on_gpu for j in jj if j["reuse_backbone"]), 1),
                                a1_limit_gb=cfg["a1_limit_gb"], group="first20" if v in first20 else
                                ("long" if v in groups["long"] else "rest")))
            queued.remove(v)
            log(f"start {v} ({frames[v]} frames) on GPU {g} ({w.name}) a1={cand['reuse_backbone']} "
                f"offload={cand['offload_video_to_cpu']}")
        for name, (v, cand, g, ts, ww) in list(running.items()):
            if ww.dead() and pool.results.empty():
                kills = oom_kills()
                if a29c and kills > oom_base:
                    oom_base = kills
                    running.pop(name)
                    pool.workers[g].remove(ww)
                    pool.add(g)                         # a replacement process on that GPU
                    move_to_l40(v, "kernel oom_kill of its SAM 3 process", dict(oom_kill=kills), moved)
                else:
                    stop(5, "video", dict(video=v, error="worker died"))
        try:
            w, r = pool.results.get(timeout=5)
        except queue.Empty:
            continue
        v, cand, g, ts, _ = running.pop(w.name)
        w.job = None
        if not r.get("ok"):
            if a29c and r.get("error") == "CUDA out of memory":
                move_to_l40(v, "CUDA out of memory", r, moved)
                continue
            stop(5, "video", dict(video=v, result=r))
        completed.add(v)
        append(MANIFEST, dict(r, gpu=g, omp_num_threads=pool.omp, options=cand, group="first20" if v in first20 else
                              ("long" if v in groups["long"] else "rest"),
                              billed_gpu_hours_at_end=round(gpu_hours(), 4), t_end=round(time.time(), 2)))
        log(f"done {v} on GPU {g}: {r.get('seconds_total', 0):.0f}s in-run, billed {gpu_hours():.2f} GPU-h")
        if cand["alone"]:
            for _ in range(pool.slots - len(pool.workers[g])):
                pool.add(g)
    pool.close()


def make_projection(cfg, frames, running, rest):
    """A29a: the remaining schedule simulated with the driver's own placement rule."""
    rows = [r for r in done_videos().values() if r.get("group") == "first20"]
    r_short = S.rate(sum(r["wall_seconds"] for r in rows), sum(frames[r["video"]] for r in rows))
    dur = lambda v: S.duration(v, frames[v], r_short, cfg["r_check0"], cfg["r_check4"] or float("inf"))
    now = time.time()
    run = [(g, dur(v) - (now - ts), dict(frames=cand["frames"], alone=cand["alone"]))
           for v, cand, g, ts, _ in running.values()]
    q = [(dur(v), dict(frames=frames[v], alone=frames[v] > S.LONG)) for v in rest]
    elapsed = now - float(open(T0).read())
    proj = S.projection(elapsed, NGPU, cfg["A2"], run, q)
    proj.update(r_short=r_short, r_check0=cfg["r_check0"], r_check4=cfg["r_check4"],
                running=[v for v, *_ in running.values()], n_queued=len(q))
    save("projection.json", proj)
    log("projection: " + json.dumps(proj))
    return proj


def videos(cfg, exclude=(), a29c=False):
    """`exclude`: videos the plan author has taken off this machine (stop 1: `...00280000`, decision D1),
    logged and recorded in plan.json and sam3_done.json. The first 20 are A29's, from all 160."""
    held, frames = held_frames()
    exclude = sorted(set(exclude))
    assert set(exclude) <= set(held), sorted(set(exclude) - set(held))
    if exclude:
        log(f"excluded by the plan author's decision: {exclude}")
    if a29c:                                         # A29c: A2 off; A1 at 112 MB/frame, RAM rule, live guard
        cfg = dict(cfg, A2=1, a29c=True, a1_mb_per_frame=S.A1_MB_PER_FRAME_A29C,
                   a1_limit_gb=S.a1_limit_gb_a29c(cfg["ram_gb"], NGPU))
        save("a29c_config.json", cfg)
        log(f"A29c: A2 off (8 processes), A1 limit {cfg['a1_limit_gb']:.1f} GB at 112 MB/frame, live guard on, "
            f"out-of-memory failures move to the 48 GB run")
    groups = {k: [v for v in vs if v not in exclude] for k, vs in S.plan(held, frames, cfg).items()}
    save("plan.json", dict(groups, excluded=exclude))
    run_videos(cfg, held, frames, groups, done_videos())
    d = done_videos()
    if not cfg["offload"]:
        stop(7, "offload", dict(error="offload did not pass; ...00280000 is not run on this GPU",
                                done=len(d)))
    moved = [m["video"] for m in json.load(open(os.path.join(BOX, "moved_to_l40.json")))] if os.path.exists(
        os.path.join(BOX, "moved_to_l40.json")) else []
    want = [v for v in held if v not in exclude and v not in moved]
    assert sorted(d) == want, sorted(set(want) - set(d))
    save("sam3_done.json", dict(videos=len(d), excluded=exclude, moved_to_l40=moved,
                                billed_gpu_hours=round(gpu_hours(), 3), gpus=NGPU,
                                files={v: d[v]["sha256"] for v in sorted(d)}))
    log(f"all {len(d)} SAM 3 files written (excluded: {exclude}; moved to the 48 GB run: {moved}); "
        f"billed SAM 3 GPU-hours {gpu_hours():.2f}")


def gpu_monitor():
    path = os.path.join(BOX, "gpu_memory.log")
    while True:
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout
            with open(path, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {' | '.join(q.split(chr(10)))} | ram_avail_gib "
                        f"{int(open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0]) / 2**20:.1f}\n")
        except Exception:
            pass
        time.sleep(60)


def main():
    global NGPU
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["checks", "videos", "all"])
    ap.add_argument("--exclude", default="", help="comma-separated videos taken off this machine by a decision")
    ap.add_argument("--a29c", action="store_true", help="A29c's rules for the rest of D1")
    a = ap.parse_args()
    os.makedirs(BOX, exist_ok=True)
    NGPU = n_gpus()
    threading.Thread(target=gpu_monitor, daemon=True).start()
    cfg = checks() if a.what in ("checks", "all") else json.load(open(CONFIG))
    if cfg["gpus"] != NGPU:
        raise SystemExit(f"!! the config was made on {cfg['gpus']} GPUs, this machine has {NGPU}")
    if a.what in ("videos", "all"):
        videos(cfg, [v for v in a.exclude.split(",") if v], a.a29c)


if __name__ == "__main__":
    main()
