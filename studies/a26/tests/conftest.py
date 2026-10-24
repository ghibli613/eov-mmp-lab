"""A26 tests: CPU only, stubs for the detector, classifier and relation head; no CLIP model.

METOR's root goes first on sys.path, so `models`, `utils`, `inference` are METOR's -- the code the
machine runs. Run on their own, from studies/:  python -m pytest a26/tests
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STUDIES = os.path.dirname(os.path.dirname(HERE))      # METOR/studies, which holds `a26`
METOR = os.path.dirname(STUDIES)
for p in (STUDIES, METOR):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

import pytest  # noqa: E402


def pytest_collection_modifyitems(session, config, items):
    outside = [i for i in items if not str(i.path).startswith(HERE)]
    if outside:
        raise pytest.UsageError("a26/tests put METOR's root first on sys.path; run them on their own "
                                "(from studies/: python -m pytest a26/tests)")


@pytest.fixture
def cpu_cuda(monkeypatch):
    """The repo's .cuda() calls on the CPU, copying as a GPU transfer would (see d2/tests/conftest.py:
    deep_sort_app.gather_sequence_info edits boxes in place on a shared view otherwise)."""
    import torch
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self.clone())
    monkeypatch.setattr(torch.Tensor, "cpu", lambda self, *a, **k: self.clone())
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self, *a, **k: self)


@pytest.fixture
def args():
    """A26's run arguments (common.RUN_ARGV), fresh per test: the fusion switch lives on them."""
    from a26.common import RUN_ARGV
    argv = sys.argv
    sys.argv = ["pytest", *RUN_ARGV]
    try:
        from utils.parser_func import parse_args
        return parse_args()
    finally:
        sys.argv = argv
