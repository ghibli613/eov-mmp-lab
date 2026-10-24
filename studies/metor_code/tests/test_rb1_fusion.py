"""RB1 (A23): `--uniform_fusion`. Off = the released per-class weights exactly,
quirks included; on = 0.3/0.7 for all 35 classes."""
from __future__ import annotations

import random
import re
from pathlib import Path

from models.fusion import fuse_scores

ROOT = Path(__file__).resolve().parent.parent


def _released(similarity, scores):
    """The two loops exactly as they were in End2End_Model.forward at 3be8147."""
    for nb in [1,3,8,12,14,23]:
        scores[nb] = similarity[nb]*0.6 + scores[nb]*0.4
    for nb in [0,2,4,5,6,7,9,10,11,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,32,33,34]:
        scores[nb] = similarity[nb]*0.3 + scores[nb]*0.7
    return scores


def _vec(seed):
    r = random.Random(seed)
    return [r.random() for _ in range(35)], [r.random() for _ in range(35)]


def test_flag_off_is_bit_identical_to_the_released_loops():
    for seed in range(20):
        sim, sc = _vec(seed)
        assert fuse_scores(sim, list(sc), uniform=False) == _released(sim, list(sc))


def test_released_quirks_are_preserved():
    sim, sc = _vec(0)
    out = fuse_scores(sim, list(sc))
    assert out[31] == sc[31]                                    # turtle: in neither list, unfused
    once = sim[23] * 0.6 + sc[23] * 0.4
    assert out[23] == sim[23] * 0.3 + once * 0.7                # red_panda: fused twice, in order
    assert out[1] == sim[1] * 0.6 + sc[1] * 0.4                 # antelope: 0.6/0.4
    assert out[0] == sim[0] * 0.3 + sc[0] * 0.7


def test_uniform_is_0_3_0_7_once_for_every_class_including_turtle_and_red_panda():
    sim, sc = _vec(1)
    out = fuse_scores(sim, list(sc), uniform=True)
    for c in range(35):
        assert out[c] == sim[c] * 0.3 + sc[c] * 0.7, c


def test_the_two_settings_differ_exactly_where_the_released_code_is_special():
    sim, sc = _vec(2)
    a = fuse_scores(sim, list(sc), uniform=False)
    b = fuse_scores(sim, list(sc), uniform=True)
    differ = {c for c in range(35) if a[c] != b[c]}
    assert differ == {1, 3, 8, 12, 14, 23, 31}


def test_in_place_mutation_as_released():
    sim, sc = _vec(3)
    ref = list(sc)
    out = fuse_scores(sim, ref)
    assert out is ref


def test_end2end_model_no_longer_hard_codes_the_weights():
    """If the literal loops were still in End2End_Model the flag would be inert."""
    src = (ROOT / "models" / "end2end_model.py").read_text()
    assert "similarity[nb]*0.6" not in src and "similarity[nb]*0.3" not in src
    assert re.search(r"fuse_scores\(similarity, scores,\s*uniform=getattr\(self\.args, 'uniform_fusion', False\)\)", src)
