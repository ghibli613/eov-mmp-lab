#!/usr/bin/env python
"""One command to fetch and prepare every input this project needs.

    python tools/prepare_data.py              # do everything that is not done yet
    python tools/prepare_data.py --check      # report only, change nothing
    python tools/prepare_data.py --steps frames,bank

Steps, in dependency order:

  1. videos   1000 .mp4 from HuggingFace `shangxd/imagenet-vidvrd`      4.2 GB
  2. anno     800 train + 200 test annotation JSON, same dataset
  3. meta     trajectories, class splits, vocabularies, prior -- MMP repo
  4. frames   decode videos to 1-indexed JPEGs                          ~42 GB
  5. gt       evaluation ground truth, derived from the annotations
  6. bank     per-category CLIP image-embedding bank for the detector

Every step is skipped if its output already looks complete, so re-running after
an interruption is cheap and safe. Nothing here is synthetic: videos,
annotations, trajectories and class splits are downloaded, and the frames, GT
and CLIP bank are computed from them.

NOT covered: the four pretrained checkpoints. They are not publicly downloadable
-- see README section 4 and docs/PORT_STATUS.md section 5.1.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from utils import paths  # noqa: E402

HF_DATASET = "shangxd/imagenet-vidvrd"
MMP_RAW = ("https://raw.githubusercontent.com/wangyongqi558/"
           "MMP_OV_VidVRD/main/dataset/vidvrd/data")
MMP_FILES = [
    # trajectories -- EOV trains and evaluates on MMP's GT tracklets
    "train_object_trajectories_gt.json",
    "test_object_trajectories_gt.json",
    "test_object_trajectories_meta.json",
    "test_relation_gt.json",
    # open-vocabulary base/novel splits
    "openvoc_obj_class_spilt_info.json",
    "openvoc_pred_class_spilt_info.json",
    # vocabularies and the predicate prior, read by data_loading/dataset.py
    "id2object.json",
    "object2id.json",
    "id2predicate.json",
    "predicate2id.json",
    "prior.pkl",
]

GT_FILES = ["VidVRDtest_gts.json", "VidVRDtest_segment_gts.json"]
BANK = os.path.join(paths.META_DIR, "clip_L14_feat_vidvrd.pkl")
GT_DIR = os.path.join(REPO, "data", "gt_jsons")


# --------------------------------------------------------------- state checks
def n_videos() -> int:
    d = paths.VIDEO_DIR
    return len([f for f in os.listdir(d) if f.endswith(".mp4")]) if os.path.isdir(d) else 0


def n_anno() -> tuple[int, int]:
    def c(p):
        return len([f for f in os.listdir(p) if f.endswith(".json")]) if os.path.isdir(p) else 0
    return c(paths.ANNO_TRAIN_DIR), c(paths.ANNO_TEST_DIR)


def n_meta() -> int:
    return sum(os.path.exists(os.path.join(paths.META_DIR, f)) for f in MMP_FILES)


def n_frame_dirs() -> int:
    d = paths.FRAME_DIR
    return len(os.listdir(d)) if os.path.isdir(d) else 0


def n_gt() -> int:
    return sum(os.path.exists(os.path.join(GT_DIR, f)) for f in GT_FILES)


def status() -> dict:
    tr, te = n_anno()
    return {
        "videos": (n_videos(), 1000),
        "anno":   (tr + te, 1000),
        "meta":   (n_meta(), len(MMP_FILES)),
        "frames": (n_frame_dirs(), 1000),
        "gt":     (n_gt(), len(GT_FILES)),
        "bank":   (int(os.path.exists(BANK)), 1),
    }


def report():
    print("  step      have / need")
    for k, (have, need) in status().items():
        mark = "OK " if have >= need else "-- "
        print(f"  {mark}{k:8s} {have:5d} / {need}")


# ------------------------------------------------------------------- the steps
def _fully_extracted(zip_path: str, out_dir: str) -> bool:
    """True when every .mp4 in the zip is on disk at exactly the right size."""
    with zipfile.ZipFile(zip_path) as z:
        for i in z.infolist():
            if not i.filename.endswith(".mp4"):
                continue
            dst = os.path.join(out_dir, os.path.basename(i.filename))
            if not os.path.exists(dst) or os.path.getsize(dst) != i.file_size:
                return False
    return True


def step_videos():
    from huggingface_hub import hf_hub_download
    os.makedirs(paths.VIDEO_DIR, exist_ok=True)
    for part in ("vidvrd-videos-part1.zip", "vidvrd-videos-part2.zip"):
        print(f"    downloading {part} ...")
        p = hf_hub_download(HF_DATASET, part, repo_type="dataset", local_dir=paths.VIDEO_DIR)
        with zipfile.ZipFile(p) as z:
            for m in (n for n in z.namelist() if n.endswith(".mp4")):
                dst = os.path.join(paths.VIDEO_DIR, os.path.basename(m))
                if os.path.exists(dst):
                    continue
                with z.open(m) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out)
        if _fully_extracted(p, paths.VIDEO_DIR):
            os.remove(p)          # 4.2 GB of bytes we would otherwise hold twice
    print(f"    {n_videos()} videos")


def step_anno():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(HF_DATASET, "vidvrd-annotations.zip", repo_type="dataset")
    # ONE copy, under anno/. Both the training dataset and the GT builder read it
    # through utils.paths -- see step_gt.
    with zipfile.ZipFile(p) as z:
        for n in (x for x in z.namelist() if x.endswith(".json")):
            split, fn = n.split("/")[-2], n.split("/")[-1]
            out_dir = os.path.join(paths.ANNO_DIR, split)
            os.makedirs(out_dir, exist_ok=True)
            with z.open(n) as src, open(os.path.join(out_dir, fn), "wb") as out:
                shutil.copyfileobj(src, out)
    tr, te = n_anno()
    print(f"    {tr} train + {te} test annotations")


def step_meta():
    """Trajectories and class splits, from MMP -- EOV expects these exact names."""
    import urllib.request
    os.makedirs(paths.META_DIR, exist_ok=True)
    for f in MMP_FILES:
        dst = os.path.join(paths.META_DIR, f)
        if os.path.exists(dst):
            continue
        print(f"    {f} ...")
        urllib.request.urlretrieve(f"{MMP_RAW}/{f}", dst)
    print(f"    {n_meta()}/{len(MMP_FILES)} files")


def step_frames():
    subprocess.check_call([sys.executable, os.path.join(REPO, "tools", "extract_frames.py")])


def step_gt():
    os.makedirs(GT_DIR, exist_ok=True)
    helper = os.path.join(REPO, "third_party", "vidvrd_ii_helper")
    sys.path.insert(0, helper)
    from prepare_gts_for_eval import prepare_gts_for_vidvrd  # noqa: E402
    # NOTE: their __main__ writes video-level data under the segment-level
    # filename; call the function directly with the right flag for each.
    prepare_gts_for_vidvrd(os.path.join(GT_DIR, "VidVRDtest_gts.json"), segment_gt=False)
    prepare_gts_for_vidvrd(os.path.join(GT_DIR, "VidVRDtest_segment_gts.json"), segment_gt=True)


def step_bank():
    subprocess.check_call([sys.executable,
                           os.path.join(REPO, "tools", "build_clip_object_bank.py"),
                           "--per-class", "50"])


STEPS = [
    ("videos", step_videos, lambda s: s["videos"][0] >= 1000),
    ("anno",   step_anno,   lambda s: s["anno"][0] >= 1000),
    ("meta",   step_meta,   lambda s: s["meta"][0] >= len(MMP_FILES)),
    ("frames", step_frames, lambda s: s["frames"][0] >= 1000),
    ("gt",     step_gt,     lambda s: s["gt"][0] >= len(GT_FILES)),
    ("bank",   step_bank,   lambda s: s["bank"][0] >= 1),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report state and exit")
    ap.add_argument("--steps", default=None,
                    help="comma-separated subset, e.g. frames,bank")
    ap.add_argument("--force", action="store_true", help="run steps even if they look done")
    args = ap.parse_args()

    if args.check:
        report()
        return 0

    wanted = args.steps.split(",") if args.steps else [n for n, _, _ in STEPS]
    print("Before:")
    report()
    print()

    for name, fn, done in STEPS:
        if name not in wanted:
            continue
        if done(status()) and not args.force:
            print(f"  [skip] {name} — already complete")
            continue
        print(f"  [run ] {name}")
        fn()

    print("\nAfter:")
    report()
    missing = [k for k, (h, n) in status().items() if h < n]
    if missing:
        print(f"\n  INCOMPLETE: {', '.join(missing)}")
        return 1
    print("\n  All data ready. Remaining blocker: the four pretrained checkpoints "
          "(README section 4).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
