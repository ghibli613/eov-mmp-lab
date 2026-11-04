"""The HELD-B guard (A31 §5). After install(), opening a sealed path in this process raises PermissionError.

Sealed:
- any path that contains a HELD-B video ID (ILSVRC2015_train_… or ILSVRC2015_val_…), except under the
  repo's frame and video folders (frames are inputs, not labels): annotation files, G0's and G1's SAM 3
  outputs, EOV-MMP's outputs, any other per-video derivative;
- every aggregate of A31 §5(a) and §5(b), by file or by directory prefix.
install() also installs the test-label guard. Audit hooks cannot be removed, so the seal lasts for the
life of the process. Stdlib only, so both environments can use it.

    import sys; sys.path.insert(0, "studies/d7"); import held_b_guard; held_b_guard.install()
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SPLIT = os.path.join(HERE, "held_split.json")
SPLIT_SHA = "02415b25cfcbd398337c76331a5e4b54306bd8ec9c7579220503a0a87078f18a"
ID = re.compile(r"ILSVRC2015_(?:train|val)_\d{8}")

AGGREGATE_FILES = [
    # §5(a): extracted, then sealed
    "data/vidvrd/data/train_object_trajectories_gt.json",
    "pilot_analysis/stage1c/gt_tracks_train.json",
    "pilot_analysis/stage1c/gt_tracks_train_optionA.json",
    "studies/rpt/g1/held_gt_missing3.json",
    "studies/rpt/g0/held_gt_smoke3.json",
    "studies/rpt/g1/results_futility/futility.json",
    "studies/rpt/g1/results_descriptive/descriptive.json",
    "studies/postg1/results_screen/sam3_screen.json",
    # §5(b): sealed without an extract
    "studies/rpt/g1/results_futility/box_style_values_157.json",
    "studies/postg1/logs/sam3_screen_run.log",
]
AGGREGATE_GLOBS = ["studies/rpt/g1/eov_out/manifest_g*.jsonl", "studies/rpt/g1/eov_out/run_info_g*.json"]
AGGREGATE_DIRS = [
    "studies/rpt/g1/viz_unmatched",                 # §5(a): HELD-A images copied to the extracts
    "pilot_analysis/stage1c/task2_results",
    "studies/rpt/g0/logs", "studies/rpt/g0/box_out", "studies/rpt/g0/dryrun_laptop",
    "studies/rpt/g1/logs", "studies/rpt/g1/box_out",  # g1/logs holds sam3_manifest.jsonl (§5(a))
]
PILOT_DATA_GLOBS = ["pilot_analysis/*/data", "pilot_analysis/*/data_small"]   # held_gt, held_pairs, slots, ...


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def held_b(split_path=SPLIT, expected=SPLIT_SHA) -> frozenset:
    got = _sha256(split_path)
    if got != expected:
        raise RuntimeError(f"A31: {split_path} hashes to {got}, not the pinned {expected}")
    ids = json.load(open(split_path))["held_b"]
    if len(ids) != 80:
        raise RuntimeError("A31: HELD-B must have 80 videos")
    return frozenset(ids)


def sealed_paths(repo=REPO):
    """-> (files, directory prefixes), absolute."""
    files = {os.path.join(repo, p) for p in AGGREGATE_FILES}
    for g in AGGREGATE_GLOBS:
        files |= set(glob.glob(os.path.join(repo, g)))
    dirs = {os.path.join(repo, d) for d in AGGREGATE_DIRS}
    for g in PILOT_DATA_GLOBS:
        dirs |= {p for p in glob.glob(os.path.join(repo, g)) if os.path.isdir(p)}
    return frozenset(files), tuple(sorted(d.rstrip(os.sep) + os.sep for d in dirs))


def exempt_roots(repo=REPO):
    roots = [os.path.join(repo, "data", "vidvrd", "frames"), os.path.join(repo, "data", "vidvrd", "videos")]
    try:
        sys.path.insert(0, repo)
        from utils import paths
        roots += [paths.FRAME_DIR, paths.VIDEO_DIR]
    except Exception:
        pass
    finally:
        if sys.path and sys.path[0] == repo:
            sys.path.pop(0)
    return tuple(sorted({os.path.abspath(r).rstrip(os.sep) + os.sep for r in roots}))


def make_check(b_ids, files, dirs, exempt):
    def blocked(path) -> str | None:
        try:
            p = os.path.abspath(os.fsdecode(path))
        except TypeError:
            return None                               # a file descriptor, not a path
        if p in files or p.startswith(dirs):
            return "a sealed aggregate"
        if set(ID.findall(p)) & b_ids and not p.startswith(exempt):
            return "a HELD-B per-video file"
        return None
    return blocked


def install():
    """Irreversible for the process, as audit hooks are. Also installs the test-label guard."""
    sys.path.insert(0, os.path.join(REPO, "studies", "rpt"))
    import no_test_labels
    no_test_labels.install()
    if getattr(sys, "_d7_held_b_guard", False):
        return
    files, dirs = sealed_paths()
    blocked = make_check(held_b(), files, dirs, exempt_roots())

    def hook(event, args):
        if event == "open" and args:
            why = blocked(args[0])
            if why:
                raise PermissionError(f"A31: HELD-B is sealed ({why}: {os.fsdecode(args[0])})")
    sys.addaudithook(hook)
    sys._d7_held_b_guard = True
