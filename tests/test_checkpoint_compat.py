"""The released checkpoints must keep loading with zero key mismatches.

This repository was restructured after those checkpoints were trained, and every
later refactor risks silently renaming a parameter. `load_state_dict(strict=False)`
would not complain -- it would just leave that tensor randomly initialised and
quietly cost several mAP. The counts below were recorded in
docs/10_Known-issues.md when the author supplied the files.

Skipped when the checkpoints are not present, so the suite still runs on a
machine that only has the code.
"""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="model construction is hard-coded to CUDA")


def _args():
    import sys
    argv = sys.argv
    sys.argv = ["pytest"]
    try:
        from utils.parser_func import parse_args
        return parse_args()
    finally:
        sys.argv = argv


def _load(path):
    if not path or not os.path.exists(path):
        pytest.skip(f"checkpoint not present: {path}")
    return torch.load(path, weights_only=False)["state_dict"]


def test_relation_checkpoint_loads_cleanly():
    from models.relation_classifier import Model
    args = _args()
    state = _load(args.relation_ckpt)
    model = Model(args).cuda()
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert len(state) == 196
    assert (missing, unexpected) == ([], [])


def test_object_classifier_checkpoint_loads_cleanly():
    from models.object_classifier import Classifier
    args = _args()
    state = _load(args.obj_classifier_ckpt)
    model = Classifier(args).cuda()
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert len(state) == 155
    assert (missing, unexpected) == ([], [])
