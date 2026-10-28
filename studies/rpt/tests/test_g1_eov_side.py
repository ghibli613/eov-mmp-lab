"""RPT G1, the EOV-MMP side's helpers (A29 items 4 and 7), CPU only: the 3 missing videos' GT, the
box-sensitivity inputs through the repo's gen_feats_test (zero features, no modelC), the batch
encoder. No checkpoint, no GPU.

Run from studies/:  python -m pytest rpt/tests/test_g1_eov_side.py
"""
import hashlib
import json
import os
import sys

import pytest
import torch

RPT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(RPT))
for p in (os.path.join(RPT, "g1"), RPT, REPO, os.path.join(REPO, "pilot_analysis", "stage1c")):
    if p not in sys.path:
        sys.path.insert(0, p)

import eov_held as E  # noqa: E402
import metrics as M  # noqa: E402

FRAMES = os.path.join(REPO, "data", "vidvrd", "frames")
V = "ILSVRC2015_train_00010024"
needs_data = pytest.mark.skipif(not os.path.isdir(os.path.join(FRAMES, V)), reason="needs HELD frames")


def test_missing3_file_matches_its_pin():
    p = os.path.join(RPT, "g1", "held_gt_missing3.json")
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == E.PINS["studies/rpt/g1/held_gt_missing3.json"]
    d = json.load(open(p))
    assert sorted(d) == E.NO_ECC and sum(map(len, d.values())) == 25


def test_to_repo_dense_trajectory():
    t = [dict(tid=0, category="dog", score=0.7, begin_fid=5, end_fid=7, boxes=[[1, 2, 3, 4], [2, 3, 4, 5]])]
    r = E.to_repo(t)[0]
    assert r["trajectory"] == {5: [1, 2, 3, 4], 6: [2, 3, 4, 5]} and r["score"] == 0.7


@needs_data
def test_batch_encoder_answers_from_the_batch():
    n = len(os.listdir(os.path.join(FRAMES, V)))
    gp, pp = torch.arange(n * 2.).reshape(n, 2), torch.arange(n * 3.).reshape(n, 1, 3)
    enc = E._batch_encoder(FRAMES, V, gp, pp)
    files = sorted(os.listdir(os.path.join(FRAMES, V)))
    g, p = enc(os.path.join(FRAMES, V, files[7]))
    assert torch.equal(g, gp[7:8]) and torch.equal(p, pp[7:8])


@needs_data
def test_sensitivity_inputs_run_through_gen_feats_test():
    """A HELD relation's GT pair, built as item 7 builds it, gives gen_feats_test's (0, 1) pair with
    one clip per 30 frames (the repo's clip grid), and the cut SAM 3 pair likewise."""
    from relation import FrameFeatures, relation_head
    rels = E.held_relations()[V]
    rel = next(r for r in rels if r["duration"][1] - r["duration"][0] >= 30)
    tr = M.tracks_from_list([dict(tid=0, category=rel["triplet"][0], score=0.9, begin_fid=rel["duration"][0],
                                  end_fid=rel["duration"][1], boxes=rel["sub_traj"]),
                             dict(tid=1, category=rel["triplet"][2], score=0.8, begin_fid=rel["duration"][0],
                                  end_fid=rel["duration"][1], boxes=rel["obj_traj"])])
    inp = M.sensitivity_inputs(rel, dict(sam3=(0, 1), eov=(1, 0)), tr, tr)
    enc = lambda path: (torch.zeros(1, 768), torch.zeros(1, 576, 768))
    stub = lambda m, it: (dict(logits=[0] * it["clip_feat"].shape[1], int_logit=[], windows=[], slot_inputs=None), 0.0)
    for src in ("gt", "sam3"):
        pairs = relation_head(V, E.to_repo(inp[src]), "train", FrameFeatures(V, FRAMES, enc), None, run_pair=stub)
        p = [q for q in pairs if q["sub"] == 0 and q["obj"] == 1]
        n = rel["duration"][1] - rel["duration"][0]
        assert len(p) == 1 and len(p[0]["logits"]) == n // 30 + (1 if n % 30 >= 10 else 0)
        assert p[0]["sbj_cls"] == rel["triplet"][0] and p[0]["obj_cls"] == rel["triplet"][2]
