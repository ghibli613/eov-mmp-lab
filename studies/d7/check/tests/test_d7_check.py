"""The D7 check's code (A34) on the laptop: the driver end to end with the fake worker, the class-free
adapter path, the equal-budget cut, the video order, the guard's decision and the binding rule.
No SAM 3, no GPU; the only real files read are G0's committed output for ...00010024 (HELD-A) and two
HELD-A frame folders' file counts."""
import gzip
import hashlib
import importlib
import json
import os
import subprocess
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = os.path.dirname(HERE)
D7 = os.path.dirname(CHECK)
REPO = os.path.dirname(os.path.dirname(D7))
sys.path[:0] = [CHECK, os.path.join(REPO, "studies", "rpt")]
sys.modules["held_b_guard"] = types.SimpleNamespace(install=lambda: None)   # no audit hook in the test process
SV = importlib.import_module("select_videos")
AN = importlib.import_module("d7_analyse")
del sys.modules["held_b_guard"]
import classfree as CF  # noqa: E402

VIDEOS = ["ILSVRC2015_train_00010024", "ILSVRC2015_train_00024005"]     # HELD-A, with frames on the laptop


def driver(tmp_path, stage, **env_extra):
    lst = tmp_path / "video_list.json"
    if not lst.exists():
        lst.write_text(json.dumps(dict(final=VIDEOS)))
        (tmp_path / "video_list.json.sha256").write_text(hashlib.sha256(lst.read_bytes()).hexdigest() + "  video_list.json\n")
    env = dict(os.environ, G1_BOX=str(tmp_path / "box"), G1_OUT=str(tmp_path / "out"), G1_SAM3PY=sys.executable,
               G1_WORKER=os.path.join(HERE, "fake_d7_worker.py"), D7_LIST=str(lst), **env_extra)
    return subprocess.run([sys.executable, os.path.join(CHECK, "d7_run.py"), stage], cwd=REPO, env=env,
                          capture_output=True, text=True)


def manifest(tmp_path):
    p = tmp_path / "out" / "manifest.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def test_driver_check0_identical_then_arms(tmp_path):
    for s in ("check0", "ref35", "arms"):
        r = driver(tmp_path, s)
        assert r.returncode == 0, r.stdout + r.stderr
    c0 = json.load(open(tmp_path / "box" / "check0.json"))
    assert c0["identical"] and c0["compare"]["prompts_compared"] == 35
    rows = manifest(tmp_path)
    assert sorted((r["stage"], r["video"]) for r in rows) == sorted(
        [("check0", VIDEOS[0])] + [(a, v) for a in ("armA", "armB") for v in VIDEOS])
    b = [r for r in rows if r["stage"] == "armB"][0]
    assert [p["name"] for p in b["passes"]] == ["person", "animal", "vehicle", "object"]
    assert [p["name"] for p in [r for r in rows if r["stage"] == "armA"][0]["passes"]] == ["object"]
    r = driver(tmp_path, "arms")                                     # resumes: nothing is rerun
    assert r.returncode == 0 and len(manifest(tmp_path)) == len(rows)


def test_driver_check0_not_identical_runs_the_reference(tmp_path):
    for s in ("check0", "ref35"):
        r = driver(tmp_path, s, FAKE_DIFF="check0")
        assert r.returncode == 0, r.stdout + r.stderr
    c0 = json.load(open(tmp_path / "box" / "check0.json"))
    assert not c0["identical"] and c0["difference_size"]["max_abs_delta_out_probs"] > 0
    assert sorted(r["video"] for r in manifest(tmp_path) if r["stage"] == "ref35") == sorted(VIDEOS)


def test_driver_out_of_memory_stops_without_skipping(tmp_path):
    assert driver(tmp_path, "check0").returncode == 0
    r = driver(tmp_path, "arms", FAKE_OOM=VIDEOS[1])
    assert r.returncode == 5
    stop = json.load(open(tmp_path / "box" / "STOP_armA.json"))
    assert stop["video"] == VIDEOS[1] and "out of memory" in stop["result"]["error"]
    assert not any(r["stage"] == "armB" for r in manifest(tmp_path))


def rle(box, h=60, w=80):
    from pycocotools import mask as mask_util
    m = np.zeros((h, w), dtype=np.uint8, order="F")
    m[box[1]:box[3] + 1, box[0]:box[2] + 1] = 1
    return mask_util.encode(m)["counts"].decode("ascii")


def masklet(oid, box, frames, prob=0.9):
    return dict(obj_id=oid, frames=[dict(f=f, box=list(box), box_sam3=list(box), prob=prob, rle=rle(box), started=f == frames[0],
                                         matched=True, det=[], trk=None) for f in frames])


def test_classfree_merges_duplicates_across_prompts_and_tiers():
    s3 = dict(width=80, height=60, prompts=[
        {"class": "person", "masklets": [masklet(1, (10, 10, 30, 40), range(0, 20))]},
        {"class": "object", "masklets": [masklet(1, (10, 10, 30, 40), range(0, 20), prob=0.8),       # duplicate: dropped
                                         masklet(2, (50, 5, 70, 25), range(5, 15), prob=0.6),
                                         masklet(3, (50, 40, 60, 50), range(5, 15), prob=0.3)]}])  # candidate tier
    out = CF.tracks_classfree(s3)
    assert len(out["duplicates"]) == 1
    assert [(t["tid"], t["category"]) for t in out["repo"]] == [(0, "person"), (1, "object")]
    assert out["repo"][1]["begin_fid"] == 5 and out["repo"][1]["end_fid"] == 15


def test_top_k_keeps_the_highest_scores():
    repo = [dict(tid=i, score=s, category="x", begin_fid=0, end_fid=1, trajectory={0: [0, 0, 1, 1]})
            for i, s in enumerate([0.6, 0.9, 0.7, 0.9])]
    got = CF.top_k(repo, 2)
    assert [t["score"] for t in got] == [0.9, 0.9] and [t["tid"] for t in got] == [0, 1]
    assert CF.top_k(repo, 10) == [dict(t, tid=i) for i, t in enumerate(repo)]


def test_video_order_and_guard_decision():
    held_a = [f"ILSVRC2015_train_{i:08d}" for i in range(80)]
    order, with_files = SV.ordered(held_a, lambda v: not v.endswith("7"))
    assert [str(x) for x in np.random.default_rng(20261103).permutation(sorted(held_a))] == order
    assert with_files == [v for v in order if not v.endswith("7")]
    assert SV.decide(5.0)[0] == "list20" and SV.decide(4.99)[0] == "need40"
    assert SV.decide(4.0, 5.0)[0] == "list40" and SV.decide(4.0, 4.9)[0] == "stop" and SV.decide(-1.0, -0.5)[0] == "stop"


def test_binding_rule():
    r = AN.rule(dict(armA=4.0, armB=6.0, ref35=10.0))
    assert r["better_arm"] == "armB" and r["continue_d7"]
    r = AN.rule(dict(armA=4.9, armB=2.0, ref35=10.0))
    assert r["better_arm"] == "armA" and not r["continue_d7"]
    assert AN.rule(dict(armA=3.0, armB=3.0, ref35=6.0))["continue_d7"]
