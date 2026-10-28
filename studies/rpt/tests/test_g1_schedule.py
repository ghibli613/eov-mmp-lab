"""RPT G1 schedule and driver tests (A29, A29a), CPU only: the placement rule, the A1 RAM limit, the
plan, the simulated projection, check 5's drop order, and the machine driver end to end on 4 and 8
fake GPUs with a fake SAM 3 worker (no GPU, no SAM 3).

Run from studies/:  python -m pytest rpt/tests/test_g1_schedule.py
"""
import json
import os
import subprocess
import sys
import time

import pytest

TESTS = os.path.dirname(os.path.abspath(__file__))
RPT = os.path.dirname(TESTS)
REPO = os.path.dirname(os.path.dirname(RPT))
sys.path.insert(0, os.path.join(RPT, "g1"))

import schedule as S  # noqa: E402

DRIVER = os.path.join(RPT, "g1", "box", "g1_driver.py")
FRAMES = os.path.join(REPO, "data", "vidvrd", "frames")
LONG3 = [S.OFFLOAD_VIDEO, "ILSVRC2015_train_00237001", "ILSVRC2015_train_00896000"]


# ---------------------------------------------------------------- pure functions
def job(frames, a1=True, alone=False):
    return dict(frames=frames, reuse_backbone=a1, alone=alone)


def test_per_gpu_memory_rule():
    assert S.predicted_peak_gib(130) == pytest.approx(7.3, abs=0.05)
    assert S.fits_gpu(job(130), [job(130)], 3)
    assert not S.fits_gpu(job(130), [job(130), job(130)], 3)           # 21.9 GiB
    assert S.fits_gpu(job(30), [job(30), job(30)], 3)                  # 20.1 GiB
    assert not S.fits_gpu(job(981), [job(500)], 3)
    assert not S.fits_gpu(job(30), [job(30)], 1)                       # A2 not passed: one process
    assert not S.fits_gpu(job(1080, False, True), [job(30)], 2)        # long: alone
    assert S.fits_gpu(job(1080, False, True), [], 2)
    assert not S.fits_gpu(job(30), [job(1080, False, True)], 2)


def test_place_fewest_running_then_lowest_index():
    gpus = [[job(130)], [], [job(30)], []]
    assert S.place(job(30), gpus, 2) == 1
    assert S.place(job(1080, False, True), [[job(30)], [job(30)]], 2) is None


def test_machine_wide_a1_limit():
    lim = S.a1_limit_gb(256)
    assert lim == 192
    gpus = [[job(900)], [job(800, a1=False)], []]                      # 95.4 GB of A1 running
    assert S.use_a1(job(900), gpus, lim)                               # 190.8 <= 192
    assert not S.use_a1(job(920), gpus, lim)                           # 95.4 + 97.5 = 192.9 > 192
    assert S.a1_ram_gb(900) + S.a1_ram_gb(920) > lim
    assert not S.use_a1(job(100, a1=False), gpus, lim)                 # not asked for


def test_plan_longest_first_groups():
    frames = {f"ILSVRC2015_train_000000{i:02d}": 10 + i for i in range(25)} | {v: f for v, f in zip(LONG3, (2947, 1470, 1080))}
    held = sorted(frames)
    p = S.plan(held, frames, dict(offload=True))
    assert p["long"] == LONG3
    assert p["first20"] == sorted(sorted(held)[:20], key=lambda v: (-frames[v], v))
    assert all(frames[a] >= frames[b] for a, b in zip(p["rest"], p["rest"][1:]))
    assert S.OFFLOAD_VIDEO not in S.plan(held, frames, dict(offload=False))["long"]


def test_simulate_longest_first_makespan():
    # 2 GPUs, one slot each: a 100 s job running with 40 s left, then 3 queued jobs of 50 s
    run = [(0, 40.0, dict(frames=500, alone=False))]
    q = [(50.0, dict(frames=100, alone=False))] * 3
    assert S.simulate(run, q, 2, 1) == pytest.approx(100.0)            # gpu1: 0-50, 50-100; gpu0: 40-90
    p = S.projection(3600.0, 2, 1, run, q)
    assert p["projected_gpu_hours"] == pytest.approx(2 * (3600 + 100) / 3600) and not p["stop"]
    assert S.projection(24 * 3600.0, 2, 1, run, q)["stop"]             # 48.06 GPU-hours


def test_simulate_respects_alone_and_memory():
    q = [(10.0, dict(frames=1080, alone=True)), (10.0, dict(frames=981, alone=False)),
         (10.0, dict(frames=981, alone=False))]
    assert S.simulate([], q, 1, 3) == pytest.approx(30.0)              # nothing shares the GPU


def test_durations_and_drop_last():
    assert S.duration(S.OFFLOAD_VIDEO, 2947, 1, 2, 3) == 2947 * 35 * 3
    assert S.duration("x", 1470, 1, 2, 3) == 1470 * 35 * 2
    assert S.duration("x", 981, 1, 2, 3) == 981 * 35 * 1
    assert S.drop_last(["A1", "A2", "A3", "offload"]) == ["A1", "A2", "A3"]
    assert S.drop_last(["A3", "A1"]) == ["A1"]
    assert S.drop_last(["A1"]) == []


# ---------------------------------------------------------------- the driver, with fake workers
needs_frames = pytest.mark.skipif(not os.path.isdir(FRAMES), reason="needs the HELD frame folders")


def drive(tmp_path, gpus, what="all", t0=None, ram=512, **env):
    box, out = tmp_path / "box", tmp_path / "out"
    box.mkdir(exist_ok=True)
    if t0 is not None:
        (box / "g1_t0").write_text(f"{t0}\n")
    e = dict(os.environ, G1_BOX=str(box), G1_OUT=str(out), G1_SAM3PY=sys.executable, G1_GPUS=str(gpus),
             G1_RAM_GB=str(ram), G1_CORES="64", G1_WORKER=os.path.join(TESTS, "fake_sam3_worker.py"), **env)
    p = subprocess.run([sys.executable, DRIVER, what], env=e, capture_output=True, text=True, timeout=1200)
    return p, box, out


def events(box):
    return [json.loads(line) for line in open(box / "schedule_events.jsonl")]


@needs_frames
@pytest.mark.parametrize("gpus", [8, 4])
def test_driver_all_pass_longest_first(tmp_path, gpus):
    p, box, _ = drive(tmp_path, gpus)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-2000:]
    cfg = json.load(open(box / "g1_config.json"))
    assert cfg["A1"] and cfg["A2"] == 3 and cfg["A3"] and cfg["offload"] and cfg["gpus"] == gpus
    res = json.load(open(box / "checks.json"))
    assert sorted(r["gpu"] for r in res["c0"]["runs"]) == list(range(gpus))          # check 0 on every GPU
    assert [res[c]["gpus"] for c in ("c1", "c3", "c4")] == [[0], [2], [3]] and res["c2"]["2"]["gpus"] == [1]
    ev = events(box)
    assert [e["video"] for e in ev[:3]] == LONG3 and [e["gpu"] for e in ev[:3]] == [0, 1, 2]
    assert all(e["alone"] and e["gpu_frames_running"] == [e["frames"]] and not e["a1"] for e in ev[:3])
    assert ev[0]["offload"] and not any(e["offload"] for e in ev[1:])
    plan = json.load(open(box / "plan.json"))
    for grp in ("first20", "rest"):
        assert plan[grp] == sorted(plan[grp], key=lambda v: (-next(e["frames"] for e in ev if e["video"] == v), v))
        es = [e for e in sorted(ev, key=lambda e: e["seq"]) if e["group"] == grp]
        for i, e in enumerate(es):                                                   # out of order only
            if any(x["frames"] > e["frames"] for x in es[i + 1:]):                  # when longer ones wait
                assert e["waiting_longer"] > 0
    rows = [json.loads(line) for line in open(box / "sam3_manifest.jsonl")]
    t_rest = min(e["t"] for e in ev if e["group"] == "rest")                         # the rest waits for
    assert t_rest >= max(r["t_end"] for r in rows if r["group"] == "first20")       # the first 20 and the projection
    assert all(e["gpu_predicted_gib"] < 21 for e in ev if not e["alone"])
    assert all(e["a1_total_gb"] <= e["a1_limit_gb"] for e in ev)
    proj = json.load(open(box / "projection.json"))
    assert not proj["stop"] and proj["gpus"] == gpus and LONG3[0] in proj["running"]
    assert json.load(open(box / "sam3_done.json"))["videos"] == 160


@needs_frames
def test_driver_a1_ram_limit_drops_a1_not_the_video(tmp_path):
    p, box, _ = drive(tmp_path, 8, ram=64 + 120)                       # A1 limit 120 GB
    assert p.returncode == 0, p.stdout[-2000:]
    ev = events(box)
    assert all(e["a1_total_gb"] <= 120 for e in ev)
    assert any(not e["a1"] and e["frames"] <= S.LONG for e in ev)
    assert any(e["a1"] for e in ev)


@needs_frames
def test_driver_check0_fails_on_one_gpu(tmp_path):
    p, box, _ = drive(tmp_path, 8, "checks", FAKE_DIFF="c0_default/g5_")
    assert p.returncode == 3
    assert json.load(open(box / "STOP_check0.json"))["failed_gpus"] == [5]


@needs_frames
def test_driver_projection_above_48_stops(tmp_path):
    p, box, _ = drive(tmp_path, 4, FAKE_RATE="0.5")
    assert p.returncode == 6, p.stdout[-2000:]
    proj = json.load(open(box / "STOP_projection.json"))
    assert proj["projected_gpu_hours"] > 48 and proj["stop"]
    assert not any(e["group"] == "rest" for e in events(box))


@needs_frames
def test_driver_hard_stop_at_48(tmp_path):
    p, box, _ = drive(tmp_path, 8, "checks", t0=time.time() - 48 / 8 * 3600)
    assert p.returncode == 4 and json.load(open(box / "STOP_cap.json"))["billed_gpu_hours"] >= 48


@needs_frames
def test_driver_video_failure_stops(tmp_path):
    p, box, _ = drive(tmp_path, 4, FAKE_FAIL="ILSVRC2015_train_00125006")
    assert p.returncode == 5 and json.load(open(box / "STOP_video.json"))["video"] == "ILSVRC2015_train_00125006"


@needs_frames
def test_driver_a3_mismatch_reports_size_and_continues(tmp_path):
    p, box, _ = drive(tmp_path, 4, "checks", FAKE_DIFF="c3_A3")
    assert p.returncode == 0, p.stdout[-2000:]
    res = json.load(open(box / "checks.json"))
    assert not res["c3"]["passed"]
    assert res["c3"]["runs"][0]["difference_size"]["max_abs_delta_out_probs"] == pytest.approx(1e-6, rel=1e-3)
    cfg = json.load(open(box / "g1_config.json"))
    assert not cfg["A3"] and cfg["check5_options"] == ["A1", "A2", "offload"]


@needs_frames
def test_driver_check5_drops_offload_then_stops_before_offload_video(tmp_path):
    p, box, _ = drive(tmp_path, 4, "all", FAKE_DIFF="c5_A1+A2+A3+offload")
    assert p.returncode == 7, p.stdout[-2000:]
    res = json.load(open(box / "checks.json"))
    assert [c["options"] for c in res["c5"]] == [["A1", "A2", "A3", "offload"], ["A1", "A2", "A3"]]
    rows = [json.loads(line) for line in open(box / "sam3_manifest.jsonl")]
    assert len(rows) == 159 and S.OFFLOAD_VIDEO not in {r["video"] for r in rows}


# ---------------------------------------------------------------- the worker's offload switch
def test_worker_adds_offload_only_to_start_session_while_on():
    import sam3_worker as W

    class P:
        def __init__(self):
            self.seen, self.model = [], "m"

        def handle_request(self, r):
            self.seen.append(r)
            return {}
    p = P()
    w = W._Offload(p)
    w.handle_request(dict(type="start_session", resource_path="x"))
    w.on = True
    w.handle_request(dict(type="start_session", resource_path="x"))
    w.handle_request(dict(type="add_prompt", text="dog"))
    assert p.seen == [dict(type="start_session", resource_path="x"),
                      dict(type="start_session", resource_path="x", offload_video_to_cpu=True),
                      dict(type="add_prompt", text="dog")]
    assert w.model == "m"


# ---------------------------------------------------------------- the EOV-MMP side on several GPUs
sys.path.insert(0, os.path.join(RPT, "g1", "box"))
import eov_parallel as EP  # noqa: E402


def test_eov_split_longest_first_balanced():
    frames = {"a": 100, "b": 90, "c": 50, "d": 40, "e": 10}
    assert EP.split(list(frames), frames, 2) == [["a", "d", "e"], ["b", "c"]]
    assert sorted(sum(EP.split(list(frames), frames, 8), [])) == sorted(frames)


def run_eov(tmp_path, what, **env):
    e = dict(os.environ, G1_OUT=str(tmp_path / "out"), G1_BOX=str(tmp_path / "box"), G1_GPUS="8", G1_CORES="64",
             G1_EOVPY=sys.executable, G1_EOV=os.path.join(TESTS, "fake_eov_held.py"), **env)
    return subprocess.run([sys.executable, os.path.join(RPT, "g1", "box", "eov_parallel.py"), what], env=e,
                          capture_output=True, text=True, timeout=300)


def test_eov_preflight_identical_across_gpus_ignores_timing(tmp_path):
    p = run_eov(tmp_path, "preflight")
    assert p.returncode == 0, p.stdout + p.stderr
    assert all(json.load(open(tmp_path / "box" / "eov_preflight.json"))["identical"].values())


def test_eov_preflight_gpu_dependence_stops(tmp_path):
    p = run_eov(tmp_path, "preflight", FAKE_EOV_GPU_DIFF="1")
    assert p.returncode == 8 and (tmp_path / "box" / "STOP_eov_gpu_dependence.json").exists()


@needs_frames
def test_eov_full_runs_every_video_once_over_8_gpus(tmp_path):
    (tmp_path / "box").mkdir()
    (tmp_path / "box" / "sam3_done.json").write_text("{}")
    p = run_eov(tmp_path, "full")
    assert p.returncode == 0, p.stdout + p.stderr
    rows = [json.loads(line) for f in sorted((tmp_path / "out" / "eov").glob("manifest_*.jsonl")) for line in open(f)]
    assert len(rows) == 160 and len({r["video"] for r in rows}) == 160 and len({r["gpu"] for r in rows}) == 8


# ---------------------------------------------------------------- stop 1, decision D1: an exclude list
@needs_frames
def test_driver_exclude_list_runs_the_other_159(tmp_path):
    box = tmp_path / "box"
    p, box, _ = drive(tmp_path, 8, "checks")
    assert p.returncode == 0, p.stdout[-2000:]
    e = dict(os.environ, G1_BOX=str(box), G1_OUT=str(tmp_path / "out"), G1_SAM3PY=sys.executable, G1_GPUS="8",
             G1_RAM_GB="512", G1_CORES="64", G1_WORKER=os.path.join(TESTS, "fake_sam3_worker.py"))
    p = subprocess.run([sys.executable, DRIVER, "videos", "--exclude", S.OFFLOAD_VIDEO], env=e,
                       capture_output=True, text=True, timeout=1200)
    assert p.returncode == 0, p.stdout[-2000:]
    done = json.load(open(box / "sam3_done.json"))
    assert done["videos"] == 159 and done["excluded"] == [S.OFFLOAD_VIDEO] and S.OFFLOAD_VIDEO not in done["files"]
    ev = events(box)
    assert S.OFFLOAD_VIDEO not in {x["video"] for x in ev} and [x["video"] for x in ev[:2]] == LONG3[1:]
    assert sum(x["group"] == "first20" for x in ev) == 20
    assert json.load(open(box / "plan.json"))["excluded"] == [S.OFFLOAD_VIDEO]


@needs_frames
def test_eov_full_skips_sensitivity_only_for_the_excluded(tmp_path):
    (tmp_path / "box").mkdir()
    (tmp_path / "box" / "sam3_done.json").write_text(json.dumps(dict(excluded=[S.OFFLOAD_VIDEO])))
    p = run_eov(tmp_path, "full")
    assert p.returncode == 0, p.stdout + p.stderr
    rows = [json.loads(line) for f in sorted((tmp_path / "out" / "eov").glob("manifest_*.jsonl")) for line in open(f)]
    assert len(rows) == 160 and [r["video"] for r in rows if not r["sensitivity"]] == [S.OFFLOAD_VIDEO]
