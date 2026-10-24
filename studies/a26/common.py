"""Constants and paths shared by the A26 scripts. Every number here is fixed by A26."""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
METOR_ROOT = os.path.dirname(os.path.dirname(HERE))   # METOR/studies/a26 -> METOR
THESIS = os.path.dirname(METOR_ROOT)
EOV_ROOT = os.path.join(THESIS, "ov-vidvrd-lab")       # the data and weights live here on the laptop
EOV_COMMIT = "fdeeeb4d20145abde48d2f64f9e39f85dcd1fca6"

DATA = os.path.join(HERE, "data")
LOGS = os.path.join(HERE, "logs")

# ---- the four configurations: (name, frame range, uniform fusion). Run in this order per video.
CONFIGS = (("R0", "gt", False),       # the released test path (A23's R0)
           ("R0u", "gt", True),       # A23's R0u: uniform 0.3 / 0.7 fusion
           ("R0w", "whole", False),   # the whole video instead of the GT-derived range
           ("R0uw", "whole", True))   # both
CONFIG_NAMES = tuple(c[0] for c in CONFIGS)
SPLITS = ("novel", "all")
METRICS = ("mAP", "R@50", "R@100")

# ---- the run (as session 3 and D2 ran the released test path)
SEED = 3407
CKPT = "output/ckpt/baseline_fbce_vidvrd_bs1_lr1e-05_dim512_none_rel_mot_clip_bbox_end2end_base-001.pth"
AFLINK = "output/ckpt/AFLink_epoch20.pth"
RUN_ARGV = ("--frame_stride", "1", "--arch", "metor-compat", "--n_iter", "0")
SCORE_TOL = 1e-6        # the 2-video gate: R0's instance scores against session 3's

# ---- the reading (A26, fixed before any number exists)
N_BOOT, BOOT_SEED = 1000, 0
CI = (2.5, 97.5)
LARGE, MODERATE = 1.0, 0.3            # SGDet novel mAP points, |Δuw|

# ---- session 3's R0 (the pairing check), from cache_s3_sgdet/task6_sgdet.json
S3_SUMS = os.path.join(THESIS, "session3_box_copy", "SHA256SUMS_session3.txt")
S3_SUMS_SHA = "2cc2d0710f6a92d68ae27ecf74973a9dd122dc1594b9788cff4b0c560567954b"
S3_DIR = os.path.join(THESIS, "session3_box_copy", "cache_s3_sgdet")
S3_R0 = {"novel": 15.785916254291479, "all": 27.184818048508298}
REFERENCE = os.path.join(DATA, "r0_reference.json")    # session 3's predictions for the gate videos
R0_TOL = 0.01          # R0 on the laptop: |mAP - session 3's| at most this (points); the reported precision

# ---- the cost estimate's inputs (measured; see estimate.py)
FRAMES_WHOLE = 52104     # every frame of the 200 test videos (session 3's manifest)
FRAMES_GT = 46035        # frames inside the GT-derived ranges (D2 report, Table 7)
STOP_HOURS = 18.0        # the 2-video projection above this: stop and ask
