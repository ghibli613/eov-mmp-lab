# Stage 1, session 3 — Task 2 report: the 5-video check passed

**Date:** 2026-10-01. **Status:** stopped for approval before Task 3, the full
cache.

## Setup

- **Box:** Vast.ai RTX 4090 (24 GB), 251 GB RAM, 96 cores, 60 GB disk, PyTorch
  template (torch 2.11.0+cu128), `OMP_NUM_THREADS=8`.
- **Code:** repo at `5745120`, plus the fix `b3edbfb`.
- **Videos:** 3 train + 2 test, picked deterministically by frame-count
  percentile:

| split | video | frames | pairs | slots | time | cache file |
|---|---|---:|---:|---:|---:|---:|
| train | ILSVRC2015_train_00290008 | 92 | 6 | 10 | 13.7 s (incl. warm-up) | 1.2 MB |
| train | ILSVRC2015_train_00533001 | 187 | 6 | 6 | 6.5 s | 1.1 MB |
| train | ILSVRC2015_train_00770001 | 407 | 8 | 10 | 13.9 s | 1.6 MB |
| test | ILSVRC2015_train_00194007 | 178 | 2 | 12 | 16.9 s | 1.6 MB |
| test | ILSVRC2015_train_00190003 | 377 | 2 | 16 | 13.1 s | 2.4 MB |

Frame encoding accounts for almost all the time per video.

## Verification

| check | result |
|---|---|
| model weights | 196 of 196 loaded from the end-to-end checkpoint; 0 missing, 0 unexpected |
| cached logits vs the model's own scores | **exact**: max residual 0.0 on every call |
| fidelity on the pilot's detected pairs (`segments_raw_all.json`) | **passes**: scores match to within 7.7e-7, and the top-20 predicate set is identical in **30 of 30** segments |
| shapes vs `interface_audit.md` B4 | all match (per slot, per pair, per frame, once) |
| frames decoded on the box vs the private bundle | **byte-identical**: 555 of 555 frames, max pixel difference 0 |

The pilot's dump stores only the top 20 of 132 scores per segment, so those 20
are what the fidelity check compares. Because box-decoded frames are
byte-identical to the bundle, the train frames decoded on the box come from the
same pipeline as the test frames.

**One bug surfaced on the first real run.** Storing each pair's results raised
a key clash (`obj`), because the track ID and the object embedding used the same
field name. It's fixed in `b3edbfb`, and all five videos above ran on the fixed
code.

## Projection for Task 3 (all 1,000 videos)

| | value |
|---|---|
| frame encoding | 0.048 s/frame (0.041 without warm-up) × 296,204 frames |
| everything else | 0.17 s/pair × 10,318 pairs |
| **GPU time** | **≈ 4–4.5 h** |
| **cache size** | **≈ 1.75 GB** |
| **cost** | 4.5 h × the box's hourly price; ≈ $1.50–3 at $0.30–0.60/h |
| disk | fits in 60 GB: frames are decoded or fetched, cached, then deleted, one video at a time |

Frames are encoded one at a time, exactly as `Dataset_new` does, which is what
keeps the fidelity exact. Batching would be faster but changes fp16 numerics, so
it wasn't done.

## Decisions needed

1. **Approve Task 3?** If yes, the streaming runner gets written first (about
   30 minutes) and dry-run locally.
2. **Push** the local commits (`b3edbfb`, `f8644c9`, plus the runner) before
   Task 3 starts, so the box runs exactly what's in git.
3. **The box is idle and billing.** If the decision will take a while, stop the
   instance (don't destroy it). Stop/start keeps everything, including the setup
   and the weights, and halts the GPU charge.

Raw records: `pilot_analysis/stage1c/task2_results/`.
