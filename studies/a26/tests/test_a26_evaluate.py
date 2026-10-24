"""evaluate.py's pure parts on synthetic relations: agreement with the unmodified evaluator, the
bootstrap, the reading's boundaries; and the gate helpers in tracks.py and run.py."""
import json

import numpy as np
import pytest

from a26.common import CONFIGS, FRAMES_GT, FRAMES_WHOLE, LARGE, MODERATE
from a26.evaluate import aggregate, boot_metrics, draws, filter_split, per_video, reading
from a26.run import projection
from a26.tracks import compare_predictions, compare_tracks


def _splits():
    from utils import paths
    return json.load(open(paths.OBJ_SPLIT_INFO)), json.load(open(paths.PRED_SPLIT_INFO))


def _traj(x, n):
    return [[x + i, 10.0, x + i + 40.0, 60.0] for i in range(n)]


def _synthetic(seed=0, n_videos=12):
    """GT and predictions over real class names: base and novel predicates, base and novel objects."""
    obj, pred = _splits()
    objs = sorted(c for c in obj["cls2split"] if c != "__background__")
    base_p = sorted(c for c, s in pred["cls2split"].items() if s == "base")
    novel_p = sorted(c for c, s in pred["cls2split"].items() if s == "novel")
    rng = np.random.default_rng(seed)
    gt, preds = {}, {}
    for v in range(n_videos):
        vid = f"vid{v:02d}"
        g, p = [], []
        for k in range(int(rng.integers(1, 6))):
            trip = [objs[int(rng.integers(len(objs)))], (novel_p if k % 2 else base_p)[int(rng.integers(5))],
                    objs[int(rng.integers(len(objs)))]]
            b, n = int(rng.integers(0, 50)), int(rng.integers(10, 40))
            g.append(dict(triplet=trip, duration=[b, b + n], sub_traj=_traj(5.0 * k, n), obj_traj=_traj(100.0 + k, n)))
            for j in range(int(rng.integers(0, 4))):        # some hits, some shifted misses, some wrong triplets
                shift = int(rng.integers(0, 3)) * 15
                t = trip if j < 2 else [trip[0], base_p[0], trip[2]]
                p.append(dict(triplet=t, score=float(rng.uniform(0.01, 1)), duration=[b + shift, b + shift + n],
                              sub_traj=_traj(5.0 * k, n), obj_traj=_traj(100.0 + k, n)))
        if v == 3:
            g = [r for r in g if r["triplet"][1] in base_p] or g      # a video with no novel GT
        if v == 5:
            p = []                                                    # a video with no prediction
        gt[vid] = g
        preds[vid] = p
    return gt, preds


@pytest.mark.parametrize("split", ["novel", "all"])
def test_per_video_aggregate_equals_the_official_evaluator(tmp_path, split):
    from inference.video_relation_detection_openvoc import eval_relation_detection_openvoc
    gt, preds = _synthetic()
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(gt))
    m, rec = eval_relation_detection_openvoc(target_split_pred=split, prediction_results=json.loads(json.dumps(preds)),
                                             gt_json=str(path), rt_hit_infos=True)
    obj, pred = _splits()
    g, p = filter_split(gt, preds, split, obj, pred)
    pv = per_video(g, p)
    agg = aggregate([pv[v] for v in pv])
    assert 0 < m < 1 and len(pv) >= 5
    assert abs(agg["mAP"] - 100 * m) < 1e-9
    assert abs(agg["R@50"] - 100 * rec[50]) < 1e-4 and abs(agg["R@100"] - 100 * rec[100]) < 1e-4
    assert all(len(rels) > 0 for rels in g.values())
    if split == "novel":
        assert "vid03" not in pv                      # the evaluator drops a video with no kept GT


def test_bootstrap_equals_the_evaluator_on_resamples():
    from inference.video_relation_detection_openvoc import evaluate_v2
    gt, preds = _synthetic(seed=1)
    obj, pred = _splits()
    g, p = filter_split(gt, preds, "all", obj, pred)
    pv = per_video(g, p)
    order = sorted(pv)
    stats = np.asarray([pv[v] for v in order])
    idx = draws(len(order), n_boot=5)
    bm = boot_metrics(stats, idx)
    for b in range(5):
        keys = [order[j] for j in idx[b]]
        gb = {f"{v}#{k}": g[v] for k, v in enumerate(keys)}
        pb = {f"{v}#{k}": [dict(r) for r in p.get(v, [])] for k, v in enumerate(keys)}
        m, rec, _, _ = evaluate_v2(gb, pb, viou_threshold=0.5)
        assert abs(100 * m - bm["mAP"][b]) < 1e-9
        assert abs(100 * rec[50] - bm["R@50"][b]) < 1e-4 and abs(100 * rec[100] - bm["R@100"][b]) < 1e-4


def test_draws_are_fixed_and_shared():
    a, b = draws(37), draws(37)
    assert a.shape == (1000, 37) and np.array_equal(a, b)
    assert a.min() >= 0 and a.max() <= 36
    assert not np.array_equal(draws(37, seed=1), a)


@pytest.mark.parametrize("delta,ci,size,direction", [
    (-1.2, (-2.0, -0.5), "large", "lower under the clean protocol"),
    (-LARGE, (-1.5, -0.2), "large", "lower under the clean protocol"),
    (-0.99, (-1.5, -0.2), "moderate", "lower under the clean protocol"),
    (-MODERATE, (-0.6, -0.01), "moderate", "lower under the clean protocol"),
    (-0.29, (-0.5, -0.1), "small", "not read"),
    (-1.5, (-3.0, 0.1), "small", "not read"),                  # the CI includes 0
    (-0.5, (-1.0, 0.0), "small", "not read"),                  # touching 0 is including it
    (1.3, (0.4, 2.2), "large", "higher under the clean protocol"),
    (0.4, (0.1, 0.8), "moderate", "higher under the clean protocol"),
])
def test_reading(delta, ci, size, direction):
    r = reading(delta, *ci)
    assert r["size"] == size and r["direction"] == direction


def test_compare_helpers():
    t = [dict(tid=0, category="dog", score=0.5, begin_fid=0, end_fid=2, boxes=[[1.0, 2, 3, 4], [1.0, 2, 3, 5]])]
    assert compare_tracks(t, json.loads(json.dumps(t))) == []
    u = json.loads(json.dumps(t))
    u[0]["boxes"][1][3] = 6.0
    assert compare_tracks(u, t) == ["tid 0: 1 boxes differ"]
    nan = [dict(t[0], score=float("nan"))]
    assert compare_tracks(nan, json.loads(json.dumps(nan))) == []
    p = [dict(triplet=["dog", "chase", "cat"], score=0.25, duration=[0, 2], sub_traj=[[1, 2, 3, 4]] * 2,
              obj_traj=[[5, 6, 7, 8]] * 2)]
    q = json.loads(json.dumps(p))
    q[0]["score"] += 5e-7
    assert compare_predictions(q, p, 1e-6) == []
    q[0]["score"] += 1e-5
    assert len(compare_predictions(q, p, 1e-6)) == 1
    assert compare_predictions([], p, 1e-6) == ["instance count 0 vs reference 1"]


def test_projection():
    rows = [dict(video_len=100, encode_seconds=10.0,
                 configs={n: dict(seconds=(20.0 if r == "gt" else 30.0), frames=(80 if r == "gt" else 100))
                          for n, r, _ in CONFIGS}),
            dict(video_len=100, encode_seconds=10.0,
                 configs={n: dict(seconds=(20.0 if r == "gt" else 30.0), frames=(80 if r == "gt" else 100))
                          for n, r, _ in CONFIGS})]
    p = projection(rows)
    want = (0.1 * FRAMES_WHOLE + 2 * 0.25 * FRAMES_GT + 2 * 0.3 * FRAMES_WHOLE) / 3600
    assert abs(p["hours"] - round(want, 2)) < 0.006
    assert p["cost_usd"][0] < p["cost_usd"][1]


def test_evaluation_path_on_synthetic_runs(tmp_path):
    """load_split -> score_split -> resample_check -> bootstrap_deltas -> reading, on files in
    run.py's format: R0u = R0 exactly (Δ 0, CI [0, 0]); R0w loses every other prediction."""
    import gzip
    from a26.common import CONFIG_NAMES, SPLITS
    from a26.evaluate import bootstrap_deltas, load_split, resample_check, score_split
    gt, preds = _synthetic(seed=2, n_videos=15)
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(gt))
    variants = {"R0": preds, "R0u": preds, "R0w": {v: p[::2] for v, p in preds.items()},
                "R0uw": {v: p[1::2] for v, p in preds.items()}}
    for name, pr in variants.items():
        (tmp_path / name).mkdir()
        for v, p in pr.items():
            rec = dict(video=v, config=name, predictions={"all": p, "novel": p}, tracks=[])
            with gzip.open(tmp_path / name / f"{v}.json.gz", "wt") as f:
                json.dump(rec, f)
    obj, pred = _splits()
    videos = sorted(preds)
    official, stats, order = {}, {s: {} for s in SPLITS}, {}
    for name in CONFIG_NAMES:
        official[name] = {}
        for s in SPLITS:
            pr = load_split(str(tmp_path), name, videos, s)
            off, pv, g, p = score_split(pr, gt, str(gt_path), s, obj, pred)
            official[name][s] = off
            order.setdefault(s, sorted(pv))
            assert sorted(pv) == order[s]
            stats[s][name] = np.asarray([pv[v] for v in order[s]])
            if name == "R0":
                resample_check(g, p, order[s], stats[s][name], n=3)
    d = bootstrap_deltas(stats, official, order)
    assert set(d) == {"R0u-R0", "R0w-R0", "R0uw-R0"} and set(d["R0w-R0"]) == set(SPLITS)
    assert d["R0u-R0"]["novel"]["mAP"] == dict(delta=0.0, ci=[0.0, 0.0])
    w = d["R0w-R0"]["all"]["R@100"]
    assert w["delta"] < 0 and w["ci"][0] <= w["delta"] <= w["ci"][1] + 1e-9
    r = reading(d["R0uw-R0"]["novel"]["mAP"]["delta"], *d["R0uw-R0"]["novel"]["mAP"]["ci"])
    assert r["size"] in ("large", "moderate", "small")
    with gzip.open(tmp_path / "R0u" / "vid00.json.gz", "wt") as f:      # a file under the wrong name
        json.dump(dict(video="vid00", config="R0", predictions={}), f)
    with pytest.raises(SystemExit):
        load_split(str(tmp_path), "R0u", ["vid00"], "all")
