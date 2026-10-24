"""Where things are after the move into ov-vidvrd-lab (A26m). Not hashed by the chain.

The study files were copied byte for byte and are not edited; this file is the only place that
knows the new layout.

    ov-vidvrd-lab/
        (root)               the authors' EOV-MMP code at fdeeeb4, data/, output/, pilot_analysis/
        studies/             METOR's studies/ (prereg, rb1, d2, design_review, a26, rpt)
        studies/metor_code/  METOR's repo root without studies/: its code, tests and docs

The study code still takes "METOR's root" to be the folder above studies/. That is now
ov-vidvrd-lab's root, which holds the EOV-MMP code. `use_metor_code()` gives the role back to
METOR's code:
    - the top-level packages METOR has (most of them share a name with EOV-MMP's) are always
      imported from studies/metor_code, whatever the order of sys.path;
    - a26.common.METOR_ROOT is set to studies/metor_code.
Without it, nothing changes: the D2 code keeps the EOV-MMP code root, as A25 fixes.

    cd ov-vidvrd-lab/studies
    bash migration/setup_links.sh                      # once: metor_code/data, metor_code/output
    python -m pytest -p paths a26/tests                # A26's tests, on METOR's code
    python -m pytest d2/tests                          # D2's tests, on the EOV-MMP code root
    (cd metor_code && python -m pytest tests)          # METOR's own suite (RB1's tests)
    python -m paths -m a26.estimate                    # a study module, on METOR's code
    python -m paths rb1/scripts/fallback_probe.py      # a study script, on METOR's code
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import os
import runpy
import sys

STUDIES = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(STUDIES)                       # ov-vidvrd-lab
THESIS = os.path.dirname(REPO)
EOV_ROOT = REPO                                       # a26.common.EOV_ROOT and d2.common.CODE_ROOT agree
METOR_CODE = os.path.join(STUDIES, "metor_code")

# METOR's top-level packages (every folder of its root with an __init__.py).
METOR_PACKAGES = ("cli", "data_loading", "inference", "metor", "models", "ops", "third_party", "utils", "vlm")


class _MetorCodeFirst(importlib.abc.MetaPathFinder):
    """Resolve METOR's top-level packages in studies/metor_code; leave every other import alone."""

    def find_spec(self, name, path=None, target=None):
        if path is None and name in METOR_PACKAGES:
            return importlib.machinery.PathFinder.find_spec(name, [METOR_CODE])
        return None


def use_metor_code() -> None:
    if not any(type(f).__name__ == "_MetorCodeFirst" for f in sys.meta_path):
        sys.meta_path.insert(0, _MetorCodeFirst())
    if STUDIES not in sys.path:
        sys.path.insert(0, STUDIES)
    importlib.import_module("a26.common").METOR_ROOT = METOR_CODE


def pytest_configure(config):
    """As a pytest plugin (`-p paths`)."""
    use_metor_code()


def _main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit(__doc__)
    use_metor_code()
    if argv[0] == "-m":
        sys.argv = argv[1:]
        runpy.run_module(argv[1], run_name="__main__", alter_sys=True)
    else:
        sys.argv = argv
        runpy.run_path(argv[0], run_name="__main__")


if __name__ == "__main__":
    _main(sys.argv[1:])
