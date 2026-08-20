"""Every filesystem path the project uses, in one place.

The imported EOV-MMP code hard-coded paths to its authors' machine
(`/data/wyq/jixf/jixf/VidVRD-baseline/dataset/vidvrd`) and to a `../output`
directory relative to wherever the script happened to be run from. Both break
the moment anyone else runs the code, and a wrong path that nothing reads stays
invisible until something reads it.

Everything is derived from the repository root, so the project is portable, and
each entry can be overridden with an environment variable for a machine with the
data on a different disk:

    VIDVRD_DATA_ROOT=/mnt/big/vidvrd python -m cli.train ...

Layout:

    data/vidvrd/
        videos/                 1000 .mp4, as downloaded
        frames/<video>/%06d.jpg decoded frames, 1-INDEXED (tools/extract_frames.py)
                                trajectory frame 0 is 000001.jpg -- gen_labels.py
        anno/{train,test}/      per-video annotation JSON (the only copy --
                                the GT builder reads this one too)
        data/                   trajectories, class splits, relation GT
    output/
        ckpt/                   checkpoints, in and out
        log/                    training logs
"""
from __future__ import annotations

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# --------------------------------------------------------------------- data
DATA_ROOT = _env("VIDVRD_DATA_ROOT", os.path.join(REPO_ROOT, "data", "vidvrd"))

VIDEO_DIR = os.path.join(DATA_ROOT, "videos")
FRAME_DIR = _env("VIDVRD_FRAME_DIR", os.path.join(DATA_ROOT, "frames"))
ANNO_DIR = os.path.join(DATA_ROOT, "anno")
ANNO_TRAIN_DIR = os.path.join(ANNO_DIR, "train")
ANNO_TEST_DIR = os.path.join(ANNO_DIR, "test")

#: trajectories, class splits and relation ground truth (shipped, not derived)
META_DIR = os.path.join(DATA_ROOT, "data")

TRAIN_TRAJ = os.path.join(META_DIR, "train_object_trajectories_gt.json")
TEST_TRAJ_GT = os.path.join(META_DIR, "test_object_trajectories_gt.json")
TEST_TRAJ_DET = os.path.join(META_DIR, "test_object_trajectories_meta.json")
TEST_RELATION_GT = os.path.join(META_DIR, "test_relation_gt.json")
OBJ_SPLIT_INFO = os.path.join(META_DIR, "openvoc_obj_class_spilt_info.json")
PRED_SPLIT_INFO = os.path.join(META_DIR, "openvoc_pred_class_spilt_info.json")

#: per-category CLIP image-embedding bank (tools/build_clip_object_bank.py).
#: A RECONSTRUCTION -- the authors never published theirs. See docs/PORT_STATUS.md.
CLIP_FEAT_BANK = os.path.join(META_DIR, "clip_L14_feat_vidvrd.pkl")

#: evaluation ground truth, derived from the annotations by
#: third_party/vidvrd_ii_helper/prepare_gts_for_eval.py
GT_JSON_DIR = os.path.join(REPO_ROOT, "data", "gt_jsons")

#: per-video camera-motion (ECC) matrices used by deep_sort's camera_update.
#: EOV shipped neither file. Absent -> no camera compensation; see
#: models/end2end_model.py.
ECC_TRAIN = os.path.join(META_DIR, "VidVRD_ECC_train.json")
ECC_TEST = os.path.join(META_DIR, "VidVRD_ECC_test.json")

# ------------------------------------------------------------------- output
OUTPUT_ROOT = _env("VIDVRD_OUTPUT_ROOT", os.path.join(REPO_ROOT, "output"))
CKPT_DIR = os.path.join(OUTPUT_ROOT, "ckpt")
LOG_DIR = os.path.join(OUTPUT_ROOT, "log")

#: default AFLink tracker weights; override with --path_AFLink
AFLINK_CKPT = os.path.join(CKPT_DIR, "AFLink_epoch20.pth")


def ensure_output_dirs() -> None:
    for d in (CKPT_DIR, LOG_DIR):
        os.makedirs(d, exist_ok=True)


def describe() -> str:
    """One line per path with whether it exists — for start-up diagnostics."""
    rows = []
    for name in ("VIDEO_DIR", "FRAME_DIR", "ANNO_TRAIN_DIR", "ANNO_TEST_DIR",
                 "META_DIR", "CLIP_FEAT_BANK", "GT_JSON_DIR", "CKPT_DIR", "LOG_DIR"):
        p = globals()[name]
        rows.append(f"  {name:16s} {'OK ' if os.path.exists(p) else 'MISSING'}  {p}")
    return "\n".join(rows)


if __name__ == "__main__":
    print(describe())
