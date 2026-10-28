"""RPT G1 (A29, A29a): the scheduling rule, the cost cap's projection and check 5's drop order, as
pure functions (the machine's driver, studies/rpt/g1/box/g1_driver.py, calls them; the tests check
them). One machine of G identical RTX 4090s (A29a: 8, or 4 as fallback; G = 1 is A29's case).

A29a, scheduling: longest videos first. `...00280000` (with offload), `...00237001` and `...00896000`
start at once, each alone on its own GPU, without A1; the first 20 spread over the other GPUs; after
the projection, the rest, longest first. Per GPU, A29's rules: the predicted peaks (6.5 GiB + 6.3 MiB
x frames) of the processes on it sum under 21 GiB, and A2's process count only if A2 passed.
Machine-wide: the A1 caches (106 MB per frame) of all running processes stay at most RAM - 64 GB; a
video that would exceed it runs without A1 (same output).
A29a, the cap: 48 billed GPU-hours (wall clock x GPUs) for the SAM 3 phase. The projection after the
first 20 simulates the remaining schedule with this module's own placement rule.
"""
from __future__ import annotations

PROMPTS = 35
CAP_GPU_HOURS = 48.0
LONG = 1000                                  # frames: above this, alone on a GPU and without A1
OFFLOAD_VIDEO = "ILSVRC2015_train_00280000"
GPU_SUM_GIB = 21.0
A1_MB_PER_FRAME = 106.0
RAM_RESERVE_GB = 64.0
CHECK_ORDER = ("A1", "A2", "A3", "offload")  # checks 1-4; check 5 drops the last added first


def predicted_peak_gib(frames: int) -> float:
    return 6.5 + 6.3 / 1024 * frames


def a1_ram_gb(frames: int) -> float:
    return A1_MB_PER_FRAME * frames / 1000


def a1_limit_gb(ram_gb: float) -> float:
    return ram_gb - RAM_RESERVE_GB


def job_options(video: str, frames: int, cfg: dict) -> dict:
    """What a video asks for, from the combination that passed check 5 (`cfg`: A1, A3, offload
    booleans) and the scheduling rule. Whether A1 is then used depends on the RAM limit at start."""
    return dict(reuse_backbone=bool(cfg["A1"]) and frames <= LONG,
                offload_video_to_cpu=video == OFFLOAD_VIDEO,
                alone=frames > LONG)


def fits_gpu(cand: dict, on_gpu: list, slots: int) -> bool:
    """cand / on_gpu entries: dict(frames, alone). A29's per-GPU rule."""
    if len(on_gpu) >= slots:
        return False
    if cand["alone"]:
        return not on_gpu
    if any(r["alone"] for r in on_gpu):
        return False
    return sum(predicted_peak_gib(r["frames"]) for r in on_gpu) + predicted_peak_gib(cand["frames"]) < GPU_SUM_GIB


def place(cand: dict, gpus: list, slots: int):
    """The GPU a candidate starts on now: among those it fits, the one with the fewest running
    processes, then the lowest index. None if it fits nowhere. `gpus`: per GPU, its running jobs."""
    ok = [g for g in range(len(gpus)) if fits_gpu(cand, gpus[g], slots)]
    return min(ok, key=lambda g: (len(gpus[g]), g)) if ok else None


def use_a1(cand: dict, gpus: list, limit_gb: float) -> bool:
    """A1 for this start only if asked for and all running A1 caches plus this one stay <= the limit."""
    if not cand["reuse_backbone"]:
        return False
    running = sum(a1_ram_gb(r["frames"]) for g in gpus for r in g if r["reuse_backbone"])
    return running + a1_ram_gb(cand["frames"]) <= limit_gb


def longest_first(videos: list, frames: dict) -> list:
    return sorted(videos, key=lambda v: (-frames[v], v))


def plan(held: list, frames: dict, cfg: dict) -> dict:
    """A29a's three groups, each in start order: the long videos at once, the first 20 (A29: the
    first 20 sorted IDs), then the rest; longest first within each. Without offload, `...00280000` is
    left out (A29 check 4: stop before it)."""
    first20 = sorted(held)[:20]
    long_ = [v for v in held if frames[v] > LONG and (v != OFFLOAD_VIDEO or cfg["offload"])]
    rest = [v for v in held if v not in first20 and frames[v] <= LONG]
    assert not set(long_) & set(first20)
    return dict(long=longest_first(long_, frames), first20=longest_first(first20, frames),
                rest=longest_first(rest, frames))


def rate(seconds: float, frames: int) -> float:
    """Seconds per frame per prompt."""
    return seconds / (frames * PROMPTS)


def duration(video: str, frames: int, r_short: float, r_check0: float, r_check4: float) -> float:
    """A29's per-video rates: the first 20's for <= 1,000 frames, check 0's for the other long
    videos, check 4's for `...00280000`."""
    r = r_check4 if video == OFFLOAD_VIDEO else (r_check0 if frames > LONG else r_short)
    return frames * PROMPTS * r


def simulate(running: list, queue: list, n_gpus: int, slots: int) -> float:
    """Makespan in seconds of the remaining schedule. running: [(gpu, remaining_s, job)]; queue:
    [(duration_s, job)] in start order; job = dict(frames, alone). The driver's own rule: at every
    event, each queued job in order starts on place()'s GPU if it fits."""
    gpus = [[] for _ in range(n_gpus)]
    ends = []                                         # (end time, gpu, job)
    for g, rem, job in running:
        gpus[g].append(job)
        ends.append((max(0.0, rem), g, job))
    t, pending = 0.0, list(queue)
    while pending or ends:
        for item in list(pending):
            g = place(item[1], gpus, slots)
            if g is not None:
                gpus[g].append(item[1])
                ends.append((t + item[0], g, item[1]))
                pending.remove(item)
        if not ends:
            raise RuntimeError("a queued job fits on no GPU")
        ends.sort(key=lambda e: e[0])
        t, g, job = ends.pop(0)
        gpus[g].remove(job)
    return t


def projection(elapsed_s: float, n_gpus: int, slots: int, running: list, queue: list) -> dict:
    """A29a: projected billed GPU-hours = GPUs x (wall clock so far + simulated remaining wall)."""
    rem = simulate(running, queue, n_gpus, slots)
    gpu_h = n_gpus * (elapsed_s + rem) / 3600
    return dict(elapsed_wall_hours=elapsed_s / 3600, remaining_wall_hours=rem / 3600, gpus=n_gpus,
                projected_gpu_hours=gpu_h, cap_gpu_hours=CAP_GPU_HOURS, stop=gpu_h > CAP_GPU_HOURS)


def drop_last(opts: list) -> list:
    """Check 5: on a mismatch, drop the option added last (order of checks 1-4)."""
    present = [o for o in CHECK_ORDER if o in opts]
    return present[:-1]
