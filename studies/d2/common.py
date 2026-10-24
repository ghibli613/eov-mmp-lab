"""Constants, paths and logging shared by the D2 scripts. Every number here is fixed by A25."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
METOR_ROOT = os.path.dirname(os.path.dirname(HERE))   # METOR/studies/d2 -> METOR (A25b)
THESIS = os.path.dirname(METOR_ROOT)

# The code that made the baseline tracks: the EOV-MMP working repo at fdeeeb4 (A25).
CODE_ROOT = os.environ.get("D2_CODE_ROOT", os.path.join(THESIS, "ov-vidvrd-lab"))
EOV_COMMIT = "fdeeeb4d20145abde48d2f64f9e39f85dcd1fca6"

DATA = os.path.join(HERE, "data")
LOGS = os.path.join(HERE, "logs")
CODE_PINS = os.path.join(LOGS, "code_pins.json")   # written for A25a

# The test-path front end, relative to the code root; pinned by A25a and checked on the GPU machine.
FRONT_END_FILES = (
    "models/end2end_model.py", "models/gen_labels.py", "models/tracking/deep_sort_app.py",
    "models/tracking/deep_sort/tracker.py", "models/tracking/deep_sort/track.py",
    "models/tracking/deep_sort/kalman_filter.py", "models/tracking/deep_sort/linear_assignment.py",
    "models/tracking/deep_sort/nn_matching.py", "models/tracking/deep_sort/iou_matching.py",
    "models/tracking/deep_sort/detection.py", "models/tracking/application_util/preprocessing.py",
    "models/tracking/aflink/AppFreeLink.py", "models/tracking/aflink/dataset.py",
    "models/tracking/aflink/model.py", "models/methods/detectors/ov_prompt/post_process.py",
    "models/methods/detectors/ov_prompt/model.py", "models/object_classifier.py",
    "utils/parser_func.py", "utils/paths.py", "data_loading/dataset.py", "cli/common.py",
    "inference/video_relation_detection_openvoc.py")

# ---- A25's numbers
THR = 0.5                    # vIoU threshold for found / missed / oracle pass
CUT = 0.35                   # the pipeline's score cut (post_process.py:132, strict >)
REC_FLOOR = 0.01             # recorded queries: cut score >= this (plus every selected query)
FLOOR = 0.05                 # O1a's primary floor
FLOORS = (0.01, 0.05, 0.1)   # Table 4
FRAG_SHARE = 0.10            # O2-frag: >= 2 tracks each holding >= 10% of the GT extent's frames
UNTRACKED_SHARE = 0.5        # O2-other: "untracked" if < 50% of good frames are assigned
N_BOOT, BOOT_SEED = 1000, 0  # as D1's boot_share
SCORE_LEVELS = (0.1, 0.2, 0.3)  # A25c, Table 6 (descriptive): O1b oracle boxes' cut scores
EMPTY = [0, 0, -1, -1]       # no box: adds 0 to volume and overlap in the repo's +1-convention viou

NOVEL_OBJ = frozenset({"antelope", "bear", "cattle", "fox", "giant_panda", "horse", "lizard",
                       "motorcycle", "red_panda", "watercraft"})

# ---- inputs pinned by A25 (paths relative to THESIS)
PINS = {
    "ov-vidvrd-lab/pilot_analysis/time_check/data/tc_missed.json":
        "2e4c53b94597142df4500ab08ed3302e05f25f6b06a19c2c883f6f7a22373df2",
    "ov-vidvrd-lab/pilot_analysis/time_check/data/tc_tables.json":
        "c3cb77f9752e9315a2e3a33f365fed79cd04c76f7cfa2ee448f3ce715926bf80",
    "ov-vidvrd-lab/pilot_analysis/time_check/tc_run.py":
        "fae863dd65ee0c09a2cabfdab7a5f59df2e81832e8113c7c0383f7a5238d5996",
    "session3_box_copy/SHA256SUMS_session3.txt":
        "2cc2d0710f6a92d68ae27ecf74973a9dd122dc1594b9788cff4b0c560567954b",
    "ov-vidvrd-lab/data/vidvrd/data/test_relation_gt.json":
        "b7a8219fa1b00e6d10c8bc5f37a1df7482b316dd401a1c870678817c4a921aeb",
    "ov-vidvrd-lab/data/vidvrd/data/test_object_trajectories_gt.json":
        "94df4451d06f2c3ee7270b2d452d574b466f9d5c79ebb0704f0af277dd15699a",
    "ov-vidvrd-lab/data/vidvrd/data/openvoc_obj_class_spilt_info.json":
        "9ac86fc36f2c7679f76bc19039057b40263796ea9e8536b6294bfb157b186dec",
    "ov-vidvrd-lab/data/vidvrd/data/openvoc_pred_class_spilt_info.json":
        "e6de075e119eb539b134490b1f3d047abf0f340a2d3f467cb7ba8b14c0d77043",
    "ov-vidvrd-lab/data/vidvrd/data/VidVRD_ECC_test.json":
        "6e7842bee9887c85164b673becc1e61040bfa31bc33b921202e4b2d9649faa2b",
    "ov-vidvrd-lab/output/ckpt/baseline_fbce_vidvrd_bs1_lr1e-05_dim512_none_rel_mot_clip_bbox_end2end_base-001.pth":
        "d306952a51e9ca94c201ecf580f75a99221ae658dc187ae0ff38ff8b7dfc0e70",
    "ov-vidvrd-lab/output/ckpt/AFLink_epoch20.pth":
        "b35cbeddd3acc48fece820bd640640e6bfb1f5fbf570aa79af26c6a38958daa4",
    "METOR/studies/d2/logs/anno_test_sha256.txt":
        "c98b1b84a323b0852575bd6488cb0fb8a15ecfe09fc1c11a571ee266ee4d2a85",
}
CACHE_SUMS = os.path.join(THESIS, "session3_box_copy", "SHA256SUMS_session3.txt")
CACHE_SGDET = os.path.join(THESIS, "session3_box_copy", "cache_s3_sgdet")
TC_MISSED = os.path.join(THESIS, "ov-vidvrd-lab/pilot_analysis/time_check/data/tc_missed.json")
META = os.path.join(CODE_ROOT, "data", "vidvrd", "data")
ANNO_TEST = os.path.join(CODE_ROOT, "data", "vidvrd", "anno", "test")


def use_code_root(root: str | None = None) -> str:
    """Put the pipeline's code root first on sys.path (before METOR's own `models`)."""
    root = os.path.abspath(root or CODE_ROOT)
    if sys.path[:1] != [root]:
        sys.path.insert(0, root)
    return root


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_or_none(path: str) -> str | None:
    return sha256(path) if os.path.isfile(path) else None


def verify_pins(pins: dict = PINS, base: str = THESIS) -> dict:
    """Every pinned input must hash to its A25 value; raises on the first mismatch (stop rule)."""
    out = {}
    for rel, want in pins.items():
        got = sha256(os.path.join(base, rel))
        if got != want:
            raise SystemExit(f"!! STOP: {rel} has SHA256 {got}, A25 pins {want}")
        out[rel] = got
    return out


def git_head(root: str) -> dict:
    try:
        head = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", root, "status", "--porcelain"], text=True).splitlines()
        return dict(head=head, dirty=dirty)
    except Exception as e:  # noqa: BLE001 -- a machine copy may have no .git
        return dict(error=str(e))


def env_info() -> dict:
    import importlib
    out = dict(python=sys.version.split()[0], platform=platform.platform(), host=platform.node(),
               executable=sys.executable, omp_num_threads=os.environ.get("OMP_NUM_THREADS"))
    for m in ("numpy", "torch", "scipy", "sklearn", "cv2"):
        try:
            out[m] = importlib.import_module(m).__version__
        except Exception:  # noqa: BLE001
            pass
    try:
        import torch
        out["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
            out["cudnn"] = torch.backends.cudnn.version()
            out["tf32"] = dict(cudnn=torch.backends.cudnn.allow_tf32, matmul=torch.backends.cuda.matmul.allow_tf32)
    except Exception:  # noqa: BLE001
        pass
    return out


def code_hashes() -> dict:
    files = sorted(f for f in os.listdir(HERE) if f.endswith(".py"))
    return {f"d2/{f}": sha256(os.path.join(HERE, f)) for f in files}


def write_log(name: str, config: dict, inputs: list, outputs: list, extra: dict | None = None) -> str:
    os.makedirs(LOGS, exist_ok=True)
    rec = dict(step=name, time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), command=" ".join(sys.argv),
               config=config, env=env_info(), code=code_hashes(),
               repos=dict(metor=git_head(METOR_ROOT), code_root=dict(path=CODE_ROOT, **git_head(CODE_ROOT))),
               inputs={p: sha256(p) for p in inputs}, outputs={p: sha256(p) for p in outputs})
    if extra:
        rec.update(extra)
    path = os.path.join(LOGS, f"{name}.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    return path
