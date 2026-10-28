"""RPT G1 schedule and driver tests (A29), CPU only: the projection, the scheduling rule, check 5's
drop order, and the machine driver end to end with a fake SAM 3 worker (no GPU, no SAM 3).

Run from studies/:  python -m pytest rpt/tests/test_g1_schedule.py
"""
import gzip
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


# ---------------------------------------------------------------- pure functions
def test_projection_rates_per_video_kind():
    frames = {"a": 100, "b": 1080, S.OFFLOAD_VIDEO: 2947}
    p = S.projection(2.0, ["a", "b", S.OFFLOAD_VIDEO], frames, 0.05, 0.1, 0.2)
    want = 2.0 + (100 * 35 * 0.05 + 1080 * 35 * 0.1 + 2947 * 35 * 0.2) / 3600
    assert p["projected_hours"] == pytest.approx(want) and p["stop"] == (want > 35)


def test_projection_stop_strictly_above_35():
    assert not S.projection(35.0, [], {}, 1, 1, 1)["stop"]
    assert S.projection(35.0001, [], {}, 1, 1, 1)["stop"]


def test_job_options():
    cfg = dict(A1=True, A2=2, A3=True, offload=True)
    assert S.job_options("x", 981, cfg) == dict(reuse_backbone=True, offload_video_to_cpu=False, alone=False)
    assert S.job_options("x", 1080, cfg) == dict(reuse_backbone=False, offload_video_to_cpu=False, alone=True)
    assert S.job_options(S.OFFLOAD_VIDEO, 2947, cfg)["offload_video_to_cpu"]
    assert not S.job_options("x", 500, dict(cfg, A1=False))["reuse_backbone"]


def job(frames, a1=True, alone=False):
    return dict(frames=frames, reuse_backbone=a1, alone=alone)


def test_gpu_sum_under_21():
    assert S.predicted_peak_gib(130) == pytest.approx(7.3, abs=0.05)
    assert S.predicted_peak_gib(981) == pytest.approx(12.5, abs=0.1)
    assert S.can_start(job(130), [job(130)], 3)
    assert not S.can_start(job(130), [job(130), job(130)], 3)          # 21.9 GiB
    assert S.can_start(job(30), [job(30), job(30)], 3)                 # 3 x 6.7 = 20.1 GiB
    assert not S.can_start(job(981), [job(500)], 3)                     # 12.5 + 9.6 > 21


def test_a1_ram_cap_and_alone():
    assert not S.can_start(job(800), [job(700)], 2) or S.a1_ram_gb(800) + S.a1_ram_gb(700) < 150
    assert S.a1_ram_gb(800) + S.a1_ram_gb(700) > 150
    assert not S.can_start(job(1080, False, True), [job(30)], 2)
    assert S.can_start(job(1080, False, True), [], 2)
    assert not S.can_start(job(30), [job(1080, False, True)], 2)
    assert not S.can_start(job(30), [job(30)], 1)                       # no free worker


def test_order_rest_and_drop_last():
    fr = {"b": 10, "a": 20, "z": 1080, "y": 1470, S.OFFLOAD_VIDEO: 2947}
    assert S.order_rest(list(fr), fr) == ["a", "b", "y", "z", S.OFFLOAD_VIDEO]
    assert S.drop_last(["A1", "A2", "A3", "offload"]) == ["A1", "A2", "A3"]
    assert S.drop_last(["A3", "A1"]) == ["A1"]
    assert S.drop_last(["A1"]) == []


# ---------------------------------------------------------------- the driver, with a fake worker
needs_frames = pytest.mark.skipif(not os.path.isdir(FRAMES), reason="needs the HELD frame folders")


def drive(tmp_path, what="all", t0=None, **env):
    box, out = tmp_path / "box", tmp_path / "out"
    box.mkdir(exist_ok=True)
    if t0 is not None:
        (box / "g1_t0").write_text(f"{t0}\n")
    e = dict(os.environ, G1_BOX=str(box), G1_OUT=str(out), G1_SAM3PY=sys.executable,
             G1_WORKER=os.path.join(TESTS, "fake_sam3_worker.py"), **env)
    p = subprocess.run([sys.executable, DRIVER, what], env=e, capture_output=True, text=True, timeout=900)
    return p, box, out


@needs_frames
def test_driver_all_pass(tmp_path):
    p, box, out = drive(tmp_path)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    cfg = json.load(open(box / "g1_config.json"))
    assert cfg["A1"] and cfg["A2"] == 3 and cfg["A3"] and cfg["offload"]
    assert cfg["check5_options"] == ["A1", "A2", "A3", "offload"]
    done = json.load(open(box / "sam3_done.json"))
    assert done["videos"] == 160
    rows = [json.loads(l) for l in open(box / "sam3_manifest.jsonl")]
    assert sorted(r["video"] for r in rows if r["phase"] == "first20") == sorted(done["files"])[:20]
    by = {r["video"]: r for r in rows}
    assert by[S.OFFLOAD_VIDEO]["options"]["offload_video_to_cpu"] and rows[-1]["video"] == S.OFFLOAD_VIDEO
    assert not by["ILSVRC2015_train_00237001"]["options"]["reuse_backbone"]
    proj = json.load(open(box / "projection.json"))
    assert not proj["stop"] and proj["n_remaining"] == 140


@needs_frames
def test_driver_check0_mismatch_stops(tmp_path):
    p, box, _ = drive(tmp_path, "checks", FAKE_DIFF="c0_default")
    assert p.returncode == 3 and (box / "STOP_check0.json").exists()


@needs_frames
def test_driver_a3_mismatch_reports_size_and_continues(tmp_path):
    p, box, _ = drive(tmp_path, "checks", FAKE_DIFF="c3_A3")
    assert p.returncode == 0, p.stdout[-2000:]
    res = json.load(open(box / "checks.json"))
    assert not res["c3"]["passed"]
    assert res["c3"]["runs"][0]["difference_size"]["max_abs_delta_out_probs"] == pytest.approx(1e-6, rel=1e-3)
    cfg = json.load(open(box / "g1_config.json"))
    assert not cfg["A3"] and cfg["check5_options"] == ["A1", "A2", "offload"]


@needs_frames
def test_driver_check5_drops_last_added_then_stops_before_offload_video(tmp_path):
    p, box, _ = drive(tmp_path, "all", FAKE_DIFF="c5_A1+A2+A3+offload")
    assert p.returncode == 7, p.stdout[-2000:]
    res = json.load(open(box / "checks.json"))
    assert [c["options"] for c in res["c5"]] == [["A1", "A2", "A3", "offload"], ["A1", "A2", "A3"]]
    rows = [json.loads(l) for l in open(box / "sam3_manifest.jsonl")]
    assert len(rows) == 159 and S.OFFLOAD_VIDEO not in {r["video"] for r in rows}


@needs_frames
def test_driver_cap_stops(tmp_path):
    p, box, _ = drive(tmp_path, "checks", t0=time.time() - 35 * 3600)
    assert p.returncode == 4 and (box / "STOP_cap.json").exists()


@needs_frames
def test_driver_failure_in_first20_stops(tmp_path):
    p, box, _ = drive(tmp_path, "all", FAKE_FAIL="ILSVRC2015_train_00025029")
    assert p.returncode == 5 and json.load(open(box / "STOP_video.json"))["phase"] == "first20"


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
