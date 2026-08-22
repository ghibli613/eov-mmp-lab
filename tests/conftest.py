"""Shared fixtures.

The tests are deliberately runnable on a small card: nothing here builds the
detector or loads a checkpoint. What they protect is the wiring -- imports,
config integrity, the pure functions in the inference path, and the feature-
stream gating the ablations depend on.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def args():
    """The default argument namespace, as cli/train.py would build it."""
    argv = sys.argv
    sys.argv = ["pytest"]
    try:
        from utils.parser_func import parse_args
        return parse_args()
    finally:
        sys.argv = argv


@pytest.fixture(scope="session")
def cuda():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    return torch.device("cuda")
