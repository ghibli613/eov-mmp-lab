"""A31 §1's procedure on synthetic flags (no HELD data)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import make_split as S  # noqa: E402


def synthetic(n=160, seed=1):
    r = np.random.default_rng(seed)
    held = [f"ILSVRC2015_train_{i:08d}" for i in range(n - 18)] + [f"ILSVRC2015_val_{i:08d}" for i in range(18)]
    novel = {v: bool(r.random() < 0.4) for v in held}
    frames = {v: int(r.integers(30, 3000)) for v in held}
    return held, novel, frames


def test_exactly_80_80_and_disjoint():
    held, novel, frames = synthetic()
    s = S.split(held, novel, frames)
    assert len(s["held_a"]) == len(s["held_b"]) == 80
    assert set(s["held_a"]) | set(s["held_b"]) == set(held) and not set(s["held_a"]) & set(s["held_b"])


def test_alternation_continues_across_strata():
    held, novel, frames = synthetic()
    s = S.split(held, novel, frames)
    assert [s["half"][v] for v in s["order"]] == ["A", "B"] * 80
    assert any(st["count"] % 2 for st in s["strata"])          # an odd stratum exists, so a restart would differ


def test_strict_median_and_stratum_order():
    held, novel, frames = synthetic()
    s = S.split(held, novel, frames)
    med = float(np.median([frames[v] for v in held]))
    assert s["median_frames"] == med
    assert all(s["long"][v] == (frames[v] > med) for v in held)
    assert [(st["novel"], st["long"]) for st in s["strata"]] == S.STRATA
    for st in s["strata"]:
        assert st["members_sorted"] == sorted(st["members_sorted"]) and sorted(st["permuted"]) == st["members_sorted"]


def test_deterministic_and_seed_matters():
    held, novel, frames = synthetic()
    assert S.split(held, novel, frames)["held_a"] == S.split(held, novel, frames)["held_a"]
    assert S.split(held, novel, frames)["held_a"] != S.split(held, novel, frames, seed=1)["held_a"]


def test_one_generator_in_stratum_order():
    """Reproduce the permutations by hand with one generator."""
    held, novel, frames = synthetic()
    s = S.split(held, novel, frames)
    rng = np.random.default_rng(S.SEED)
    for st in s["strata"]:
        assert [str(x) for x in rng.permutation(st["members_sorted"])] == st["permuted"]
