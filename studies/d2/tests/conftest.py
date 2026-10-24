"""D2 tests: CPU only, stubs for the detector, classifier and relation head; no CLIP model.

The pipeline's code root (default: the EOV-MMP working repo, where the baseline was made;
override with D2_CODE_ROOT) goes first on sys.path, so `models`, `utils`, `inference` are its.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STUDIES = os.path.dirname(os.path.dirname(HERE))      # METOR/studies, which holds `d2`
sys.path.insert(0, STUDIES)
from d2.common import use_code_root  # noqa: E402

use_code_root()

import pytest  # noqa: E402


def pytest_collection_modifyitems(session, config, items):
    """The code root above would also shadow METOR's own `models` for METOR's tests; refuse a
    mixed session rather than run them against the wrong package."""
    outside = [i for i in items if not str(i.path).startswith(HERE)]
    if outside:
        raise pytest.UsageError("d2/tests put the EOV code root first on sys.path; run them on their own "
                                "(from studies/: python -m pytest d2/tests) and METOR's separately (from the repo root: "
                                "python -m pytest tests)")


@pytest.fixture
def cpu_cuda(monkeypatch):
    """Run the repo's .cuda() calls on the CPU. On a GPU, .cuda() and .cpu() copy; on the CPU
    they would return the same tensor, and deep_sort_app.gather_sequence_info would then edit the
    detections' boxes in place (box[2] -= box[0] on a shared numpy view). Copying keeps the CPU
    run faithful to the GPU one."""
    import torch
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self.clone())
    monkeypatch.setattr(torch.Tensor, "cpu", lambda self, *a, **k: self.clone())
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self, *a, **k: self)


@pytest.fixture(scope="session")
def args():
    argv = sys.argv
    sys.argv = ["pytest"]
    try:
        from utils.parser_func import parse_args
        return parse_args()
    finally:
        sys.argv = argv


@pytest.fixture(scope="session")
def repo_fns():
    from d2.stages import repo
    return repo()
