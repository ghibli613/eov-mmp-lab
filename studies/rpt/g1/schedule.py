"""RPT G1 (A29): the cost cap's projection, the scheduling rule and check 5's drop order, as pure
functions (the machine's driver, studies/rpt/g1/box/g1_driver.py, calls them; the tests check them).

A29, "The cost cap": billed hours = wall clock x 1 GPU. Projection after the first 20 videos =
hours spent + sum over the 140 remaining videos of frames x 35 prompts x r(video), with r the billed
seconds per frame per prompt of: the first 20 (videos <= 1,000 frames), check 0 (the two videos of
1,001-2,300 frames, alone, without A1), check 4 (`...00280000`, alone, offload).
A29, "Scheduling": A1 only for videos <= 1,000 frames, concurrent A1 caches below about 150 GB
(106 MB per frame); predicted GPU peaks 6.5 GiB + 6.3 MiB x frames, summed under 21 GiB; videos over
1,000 frames alone, without A1; only `...00280000` with offload.
"""
from __future__ import annotations

PROMPTS = 35
CAP_HOURS = 35.0
LONG = 1000                                  # frames: above this, alone and without A1
OFFLOAD_VIDEO = "ILSVRC2015_train_00280000"
GPU_SUM_GIB = 21.0
A1_RAM_GB = 150.0
A1_MB_PER_FRAME = 106.0
CHECK_ORDER = ("A1", "A2", "A3", "offload")  # checks 1-4; check 5 drops the last added first


def predicted_peak_gib(frames: int) -> float:
    return 6.5 + 6.3 / 1024 * frames


def a1_ram_gb(frames: int) -> float:
    return A1_MB_PER_FRAME * frames / 1000


def job_options(video: str, frames: int, cfg: dict) -> dict:
    """The options a video runs with, from the combination that passed check 5 (`cfg`: A1, A3,
    offload booleans; A2 an integer >= 1) and the scheduling rule."""
    return dict(reuse_backbone=bool(cfg["A1"]) and frames <= LONG,
                offload_video_to_cpu=video == OFFLOAD_VIDEO,
                alone=frames > LONG)


def can_start(cand: dict, running: list, n_workers: int) -> bool:
    """cand / running: dict(frames, reuse_backbone, alone). True if the candidate may start now."""
    if len(running) >= n_workers:
        return False
    if cand["alone"]:
        return not running
    if any(r["alone"] for r in running):
        return False
    if sum(predicted_peak_gib(r["frames"]) for r in running) + predicted_peak_gib(cand["frames"]) >= GPU_SUM_GIB:
        return False
    a1 = [r["frames"] for r in running if r["reuse_backbone"]] + ([cand["frames"]] if cand["reuse_backbone"] else [])
    return sum(a1_ram_gb(f) for f in a1) < A1_RAM_GB


def order_rest(videos: list, frames: dict) -> list:
    """The 140 after the first 20: those <= 1,000 frames in sorted order, then the longer ones
    (alone) in sorted order, `...00280000` last."""
    short = sorted(v for v in videos if frames[v] <= LONG)
    long_ = sorted(v for v in videos if frames[v] > LONG and v != OFFLOAD_VIDEO)
    return short + long_ + ([OFFLOAD_VIDEO] if OFFLOAD_VIDEO in videos else [])


def rate(seconds: float, frames: int) -> float:
    """Billed seconds per frame per prompt."""
    return seconds / (frames * PROMPTS)


def projection(spent_hours: float, remaining: list, frames: dict, r_first20: float, r_check0: float,
               r_check4: float) -> dict:
    """A29's frame-weighted projection over the remaining videos."""
    rows = []
    for v in remaining:
        f = frames[v]
        r = r_check4 if v == OFFLOAD_VIDEO else (r_check0 if f > LONG else r_first20)
        rows.append((v, f, r, f * PROMPTS * r / 3600))
    hours = spent_hours + sum(h for *_, h in rows)
    return dict(spent_hours=spent_hours, remaining_hours=hours - spent_hours, projected_hours=hours,
                cap_hours=CAP_HOURS, stop=hours > CAP_HOURS, r_first20=r_first20, r_check0=r_check0,
                r_check4=r_check4, n_remaining=len(rows))


def drop_last(opts: list) -> list:
    """Check 5: on a mismatch, drop the option added last (order of checks 1-4)."""
    present = [o for o in CHECK_ORDER if o in opts]
    return present[:-1]
