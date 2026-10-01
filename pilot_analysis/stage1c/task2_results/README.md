# Session 3, Task 2: the 5-video check on the GPU box

**Box:** Vast.ai RTX 4090 (24 GB), 251 GB RAM, 96 cores, 60 GB disk, PyTorch
template (torch 2.11.0+cu128), `OMP_NUM_THREADS=8`. The repo checkout was at
`5745120`, plus the one-line fix `b3edbfb`, which was copied to the box before
the rerun. Raw records are in this directory.

## Videos

Picked deterministically by `session3_tasks.py select-videos`:

| split | video | frames | tracks | pairs | slots | time | encode | file | residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | ILSVRC2015_train_00290008 | 92 | 3 | 6 | 10 | 13.7 s | 12.9 s | 1.2 MB | 0.0 |
| train | ILSVRC2015_train_00533001 | 187 | 6 | 6 | 6 | 6.5 s | 5.9 s | 1.1 MB | 0.0 |
| train | ILSVRC2015_train_00770001 | 407 | 5 | 8 | 10 | 13.9 s | 13.0 s | 1.6 MB | 0.0 |
| test | ILSVRC2015_train_00194007 | 178 | 2 | 2 | 12 | 16.9 s | 16.3 s | 1.6 MB | 0.0 |
| test | ILSVRC2015_train_00190003 | 377 | 2 | 2 | 16 | 13.1 s | 11.9 s | 2.4 MB | 0.0 |

The first video's time includes model warm-up.

## Verification

- **`modelC` weights:** 196 loaded from the end-to-end checkpoint, 0 missing, 0
  unexpected.
- **Hook fidelity:** the logits rebuilt from the hooked tensors, run through the
  model's own methods, reproduce its returned scores with a max residual of
  **0.0** on every call.
- **Dump fidelity, on detected trajectories:** recomputing `modelC` on the pairs
  in the pilot's `segments_raw_all.json` reproduces the dumped scores to within
  **7.7e-7** (float rounding). The top-20 predicate set is identical in **30 of
  30** segments (2 + 28). The dump holds only the top 20 of 132 scores per
  segment, so that is what could be compared.
- **Shapes match audit B4.** Per slot: logits 132 (fp32), interactiveness logit,
  `visual_pre_embeddings` 3072, spatial-decoder tokens 4×768, and slot inputs
  clip 4×768 / bbox 4×24 / rel 42 / mot 42. Per pair: conditioning mean 3072,
  meta_net bias 768, subject and object embeddings 768. Per frame: CLIP-L 768;
  RoI per (track, frame) and union per (pair, frame), 768 each. Once: `ctx`
  16×768, `token_suffix` 132×60×768, templates 132×768.
- **Frame provenance:** frames decoded on the box are **byte-identical** to the
  private bundle's: 555 of 555 frames, max pixel difference 0. Train frames
  decoded on the box therefore match the pipeline that produced the test frames.

## Projection for Task 3 (all 1,000 videos)

| | value |
|---|---|
| frame encoding | 0.048 s/frame (0.041 without warm-up) × 296,204 frames |
| everything else | 0.17 s/pair × 10,318 pairs |
| **GPU time** | **≈ 3.9–4.5 h** |
| **cache size** | **≈ 1.75 GB** |
| disk | fits in 60 GB if frames are streamed: decode or fetch, cache, then delete, one video at a time |

Frame encoding dominates. Frames are encoded one at a time, exactly as
`Dataset_new` does, which is what keeps the fidelity exact. Batching would be
faster but would change fp16 numerics, so it wasn't done.
