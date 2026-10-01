#!/usr/bin/env python3
"""Camera-motion (ECC) matrices for videos missing from VidVRD_ECC_{split}.json.

    ~/miniconda3/envs/repro-next/bin/python pilot_analysis/stage1c/ecc_supplement.py test

FRAMES ONLY, no labels. The generator of the shipped ECC files is not in any
local tree. This reconstructs it (StrongSORT-style ECC: MOTION_EUCLIDEAN, frames
downscaled 0.1x, eps 1e-5, 100 iterations, warp previous -> current, stored 3x3
under the 1-based key of the current frame). Checked against the shipped train
file: rotation terms agree to ~1e-4, translations to 0.05-0.4 px.
Output: pilot_analysis/stage1c/VidVRD_ECC_{split}_supplement.json (the repo's file is untouched).
"""
import sys
sys.dont_write_bytecode = True
import json, os, cv2, numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
split = sys.argv[1] if len(sys.argv) > 1 else "test"
shipped = json.load(open(os.path.join(ROOT, "data", "vidvrd", "data", f"VidVRD_ECC_{split}.json")))
vids = sorted(f[:-5] for f in os.listdir(os.path.join(ROOT, "data", "vidvrd", "anno", split)))
missing = [v for v in vids if v not in shipped]


def ecc(src, dst, scale=0.1, eps=1e-5, max_iter=100):
    s = cv2.resize(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    d = cv2.resize(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY), (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    w = np.eye(2, 3, dtype=np.float32)
    try:
        _, w = cv2.findTransformECC(s, d, w, cv2.MOTION_EUCLIDEAN,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps), None, 1)
    except cv2.error:
        return None
    w[0, 2] /= scale; w[1, 2] /= scale
    return np.vstack([w, [0, 0, 1]]).tolist()


out, failed = {}, {}
for v in missing:
    fdir = os.path.join(ROOT, "data", "vidvrd", "frames", v)
    frames = sorted(os.listdir(fdir))
    out[v], failed[v] = {}, 0
    prev = cv2.imread(os.path.join(fdir, frames[0]))
    for k in range(2, len(frames) + 1):             # 1-based key of the current frame
        cur = cv2.imread(os.path.join(fdir, frames[k - 1]))
        m = ecc(prev, cur)
        if m is None:
            failed[v] += 1                           # absent key -> the tracker treats it as no compensation
        else:
            out[v][str(k)] = m
        prev = cur
json.dump(out, open(os.path.join(os.path.dirname(__file__), f"VidVRD_ECC_{split}_supplement.json"), "w"))
print({v: (len(out[v]), f"failed {failed[v]}") for v in out})
