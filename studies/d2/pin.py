#!/usr/bin/env python3
"""Write d2/logs/code_pins.json for A25a: the d2 code and tests, the EOV front end, the baseline
tracks and the GPU run's inputs. `record_run.py` and `analyse.py` refuse to run on anything else.

    python -m d2.pin
"""
from __future__ import annotations

import json
import os

from d2.common import (CODE_PINS, CODE_ROOT, EOV_COMMIT, FRONT_END_FILES, HERE, PINS, code_hashes, git_head,
                       sha256)


def main():
    gh = git_head(CODE_ROOT)
    dirty = [line[3:] for line in gh.get("dirty", [])]            # "XY path"; only pilot_analysis/ may differ (A25)
    if gh.get("head") != EOV_COMMIT or any(not p.startswith("pilot_analysis/") for p in dirty):
        raise SystemExit(f"!! STOP: the code root is not clean at {EOV_COMMIT}: {gh}")
    tests = os.path.join(HERE, "tests")
    out = dict(
        eov_commit=EOV_COMMIT,
        d2_code=code_hashes(),
        d2_tests={f"d2/tests/{f}": sha256(os.path.join(tests, f)) for f in sorted(os.listdir(tests)) if f.endswith(".py")},
        d2_box={f"d2/box/{f}": sha256(os.path.join(HERE, "box", f)) for f in sorted(os.listdir(os.path.join(HERE, "box")))
                if os.path.isfile(os.path.join(HERE, "box", f))},                     # A25c: the machine scripts
        front_end_eov={f: sha256(os.path.join(CODE_ROOT, f)) for f in FRONT_END_FILES},
        baseline_tracks_json=sha256(os.path.join(HERE, "data", "baseline_tracks.json")),
        inputs=dict(end2end_ckpt=PINS["ov-vidvrd-lab/output/ckpt/baseline_fbce_vidvrd_bs1_lr1e-05_dim512_none_rel_mot_clip_bbox_end2end_base-001.pth"],
                    aflink=PINS["ov-vidvrd-lab/output/ckpt/AFLink_epoch20.pth"],
                    ecc_test=PINS["ov-vidvrd-lab/data/vidvrd/data/VidVRD_ECC_test.json"],
                    obj_split=PINS["ov-vidvrd-lab/data/vidvrd/data/openvoc_obj_class_spilt_info.json"]))
    with open(CODE_PINS, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"{CODE_PINS}: {sha256(CODE_PINS)}")


if __name__ == "__main__":
    main()
