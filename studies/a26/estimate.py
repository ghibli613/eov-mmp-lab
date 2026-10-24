#!/usr/bin/env python3
"""Laptop: A26's GPU time and cost estimate, and what the machine needs. Prints only.

    cd METOR/studies && python -m a26.estimate

Inputs, all measured on an RTX 4090 with the released test path:
  - session 3's SGDet manifest (verified against SHA256SUMS_session3.txt): every test video's frame
    count, and the forward time per video (measured after the loader, so without encoding);
  - D2's report (d2_report.md, Step 2 table): forward time 10,873 s over 46,035 GT-range frames,
    a projected 3.5 h for the whole pass with encoding.
The run encodes each video once and runs four forwards (common.CONFIGS); a whole-video forward is
scaled by frames. modelC's second (novel) call is not separable in either measurement: +10% margin.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from a26.common import CONFIGS, FRAMES_GT, FRAMES_WHOLE, S3_DIR, S3_SUMS, S3_SUMS_SHA, STOP_HOURS  # noqa: E402
from a26.run import PRICE, sha256  # noqa: E402

D2_FORWARD_S = 10873        # d2_report.md:75, the forward over the GT ranges, recorders on
D2_PASS_H = 3.5             # d2_report.md:74, the whole pass, encoding included
MARGIN = 1.10
SETUP_H = 1.0               # upload, setup, token fetch (D2: about 30 min), the gate, the copy-off


def main():
    if sha256(S3_SUMS) != S3_SUMS_SHA:
        raise SystemExit("!! STOP: SHA256SUMS_session3.txt differs from its pin")
    sums = dict(reversed(l.strip().split(None, 1)) for l in open(S3_SUMS))
    man_path = os.path.join(S3_DIR, "manifest.jsonl")
    if sha256(man_path) != sums["cache_s3_sgdet/manifest.jsonl"]:
        raise SystemExit("!! STOP: session 3's SGDet manifest differs from SHA256SUMS_session3.txt")
    man = [json.loads(l) for l in open(man_path)]
    frames = sum(r["frames"] for r in man)
    s3_forward_h = sum(r["seconds"] for r in man) / 3600
    if len(man) != 200 or frames != FRAMES_WHOLE:
        raise SystemExit(f"!! STOP: {len(man)} videos, {frames} frames")

    fwd_gt = max(D2_FORWARD_S / 3600, s3_forward_h)          # one GT-range forward pass, hours
    encode = D2_PASS_H - D2_FORWARD_S / 3600                  # encoding + I/O, once per video
    fwd_whole = fwd_gt * FRAMES_WHOLE / FRAMES_GT
    per = {n: fwd_gt if r == "gt" else fwd_whole for n, r, _ in CONFIGS}
    gpu = (encode + sum(per.values())) * MARGIN
    lo, hi = gpu * 0.85, gpu * 1.15
    machine = (lo + SETUP_H, hi + SETUP_H)

    print("A26 GPU estimate (RTX 4090; measured inputs, nothing run)")
    print(f"  test videos {len(man)}, frames {frames:,} (whole), {FRAMES_GT:,} (GT ranges, {FRAMES_WHOLE / FRAMES_GT:.3f}x)")
    print(f"  one GT-range forward pass: {fwd_gt:.2f} h (D2 {D2_FORWARD_S / 3600:.2f} h; session 3 {s3_forward_h:.2f} h)")
    print(f"  encoding, once per video: {encode:.2f} h")
    for n, h in per.items():
        print(f"  forward {n:5s} {h:.2f} h")
    print(f"  GPU total, x{MARGIN} for the second modelC call and the writes: {gpu:.1f} h (range {lo:.1f}-{hi:.1f} h)")
    print(f"  machine time with setup, fetch, the 2-video gate and the copy-off: {machine[0]:.0f}-{machine[1]:.0f} h")
    print(f"  cost at ${PRICE[0]:.2f}/h (measured): ${machine[0] * PRICE[0]:.1f}-{machine[1] * PRICE[0]:.1f}; "
          f"at ${PRICE[1]:.2f}/h: ${machine[0] * PRICE[1]:.1f}-{machine[1] * PRICE[1]:.1f}")
    print(f"  stop and ask if the 2-video gate projects more than {STOP_HOURS:.0f} h")
    print("""
Machine
  GPU    one RTX 4090, 24 GB (as session 3 and D2: R0 must reproduce their tracks bit for bit)
  RAM    >= 64 GB (setup stops below 32 GB; one 645-frame video already exceeded 12.7 GB in the pilot)
  disk   >= 60 GB free (D2's machine: 60 GB was enough; A26 adds about 1-2 GB of predictions)
  image  a PyTorch -devel image with nvcc (the deformable-attention operator is built on the machine)

Copied from the laptop (upload.sh; no push)
  ~/ov-vidvrd     the EOV-MMP working repo as `git archive fdeeeb4` (the fetch tools and the CUDA operator)
  ~/METOR         METOR as `git archive` of the A26 commit (the code that runs, and studies/), without
                  studies/d2/data/rec
  data            the class splits, id maps, prior.pkl, train trajectories; and the test annotations the
                  pipeline reads: test_relation_gt.json, test_object_trajectories_gt.json, anno/test/*.json
                  (deleted from the machine right after the run, and logged)

Fetched on the machine
  Hugging Face, with the author's read-only token (fetch_with_token.sh; unset and removed before the run):
                  the eval weights (end-to-end checkpoint, AFLink, CLIP bank, both ECC files; 2.92 GB),
                  the 200 test videos' frames (shards), SHARDS.json and MANIFEST.json
  public          CLIP ViT-L/14@336px from OpenAI's URL (its SHA256 is checked); pip packages""")


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
