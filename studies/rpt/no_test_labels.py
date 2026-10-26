"""Test labels are never opened in RPT's G0 (A28). Every G0 runner calls install() first: from then on
an attempt, by any code in the process, to open a test annotation, a test GT file, or session 3's
test caches raises PermissionError. Stdlib only, so both environments can use it."""
import os
import sys

BLOCKED = (
    f"{os.sep}anno{os.sep}test{os.sep}",            # data/vidvrd/anno/test/<video>.json
    "test_relation_gt",                              # data/vidvrd/data/test_relation_gt.json
    "test_object_trajectories",                      # ..._gt.json, ..._meta.json
    f"{os.sep}cache_s3{os.sep}test{os.sep}",         # session 3's PredCls test cache
    f"{os.sep}cache_s3_sgdet{os.sep}",               # session 3's SGDet test cache
)


def blocked(path) -> bool:
    try:
        p = os.path.abspath(os.fsdecode(path))
    except TypeError:
        return False                                 # a file descriptor, not a path
    return any(b in p for b in BLOCKED)


def _hook(event, args):
    if event == "open" and args and blocked(args[0]):
        raise PermissionError(f"A28: test labels are never opened in G0 ({os.fsdecode(args[0])})")


def install():
    """Irreversible for the process, as audit hooks are."""
    if not getattr(sys, "_rpt_no_test_labels", False):
        sys.addaudithook(_hook)
        sys._rpt_no_test_labels = True
