# Stage 1, session 3 — status: Task 1 done, Task 2 blocked on a GPU host

Everything here ran on the local machine: CPU, plus a few MB of GPU for a smoke
test. It cost $0 and rented nothing. The repo is untouched. The frozen artefacts
still match `stage1b/preregistration.md`: all six SHA256s, and the amended file's
own hash.

| file | purpose |
|---|---|
| `gt_tracks.py` | Task 1: GT tracks (train = option B, test = repo densifier) + ECC check → `gt_tracks_{train,test}.json`, `gt_tracks_report.json` |
| `optionB.py` | option-B train tracks (`runs_option_b`) and the re-score of the frozen axes on that grid → `optionB_rescore.{md,json}` |
| `ecc_supplement.py` | ECC for videos missing from the shipped files, frames only → `VidVRD_ECC_test_supplement.json` |
| `cache_wrapper.py` | Tasks 2/3/6: B6 cache via forward hooks, resumable, SHA256 manifest; dump-fidelity mode |

## Task 1 — whole GT tracks

Tracks are built with the repo's `add_initial_frames` + `interpolate_and_adjust_frames`,
as briefed, with int frame keys. From test annotations the script reads boxes and
categories only: `relation_instances` is deleted from each parsed file before
anything else touches it.

| split | annotated tracks | tracks kept | pairs | slots | pairs > 40 slots |
|---|---:|---:|---:|---:|---:|
| test | 587 | 583 | 1,392 | 8,508 | 0 |
| train | 2,430 | **1,617** | **3,728** | **5,714** | 0 |

Test matches pilot §B.9 exactly.

**⚠ Train does not survive the repo's densifier.** `interpolate_and_adjust_frames`
was written for detector output. It **rejects any track with < 65% frame
coverage**, and where a gap exceeds 15% of the span it keeps only the longer
side. Train and test annotations differ in kind:
- **train:** 33.5% of tracks are below 65% coverage, with 605 gaps over 100 frames.
  Objects seem to be boxed mainly where relations are labelled.
- **test:** 0.7% of tracks are below 65% coverage.

So on train the densifier drops 813 whole tracks and 8,835 more boxes. The three
options, counted on the same grid:

| train tracks built by | tracks | pairs | slots | cost |
|---|---:|---:|---:|---|
| A. repo densifier (as briefed) | 1,617 | 3,728 | 5,714 | loses 33% of tracks; train and test get identical code |
| B. split at gaps > 30 frames, densify each segment | 4,044 | 8,926 | 13,558 | keeps every annotated box; invents none across gaps longer than one slot |
| C. whole span, interpolate every gap | 2,430 | 6,326 | 23,222 | invents boxes across gaps of 100+ frames; 10 pairs > 40 slots |

**Decided: option B** (preregistration amendment A5, 2026-10-01).
`gt_tracks_train.json` is now option B: 4,044 tracks, 8,926 pairs, 13,558 slots
(12,365 labelled), and 0 pairs over 40 slots. Every one of the 189,345 annotated
boxes is kept unchanged, and no interpolated stretch is longer than 30 frames.
Option A is kept as `gt_tracks_train_optionA.json`, for the record. HELD was
re-scored on the option-B grid with the frozen thresholds, and no measured axis
got worse by more than 5 points (`optionB_rescore.md`). Test is unaffected.

**Pre-roll.** As briefed, the repo path adds two frames before each track start,
so 447 of 583 test tracks (and 1,106 train tracks) begin at frame −2. The
wrapper's per-frame RoI features skip frames below 0. `gen_feats_test` only
reads mid-frames (≥ 13), so it is unaffected. PredCls output extents may start
at −2, which costs up to 2 frames of vIoU on the first slot of those pairs.

## ECC for test

- `VidVRD_ECC_test.json` covers **199 of 200** test videos, with the train key
  convention (keys 2 … frame_count). **Missing: `ILSVRC2015_train_00250013`**
  (a test video, despite its name). 17 train videos also lack ECC, as in
  session 1.
- On test boxes, key `t+1` is again the best of the ECC readings: median
  residual 1.995 vs 2.095 (key `t`) and 3.148 (inverse). No compensation gives
  1.872. As on train, that's expected when the camera follows the subject.
- **Generator.** The script behind the shipped files isn't in this repo or any
  sibling tree. I reconstructed it as StrongSORT-style ECC: Euclidean motion,
  frames downscaled 0.1×, ε 1e-5, 100 iterations, previous → current. Checked on
  train frames against the shipped matrices, rotations agree to ~1e-4 and
  translations to 0.05–0.4 px (for shifts up to 18 px). That's the same method up
  to decoder noise.
- **The missing test video** is computed from frames only in
  `VidVRD_ECC_test_supplement.json`: 36 of 37 matrices. One step didn't
  converge and is left absent, which the tracker treats as no compensation.

## Task 2 — wrapper written, plumbing verified; the 5-video run needs a GPU host

**This machine can't run it.** The GTX 1650 has 4 GB, of which 1.66 GB is already
in use, and there's 7 GB of system RAM. CLIP-L + TagCLIP + `modelC` don't fit in
the remaining ~2.4 GB, before counting the fp32 CLIP copies `modelC` loads at
init. WSL would spill into system RAM and thrash, which is what the pilot
measured (> 31 min on one video).

**`cache_wrapper.py`:**
- Builds only `modelC`, loading its weights from the end-to-end checkpoint and
  recording missing and unexpected keys, plus the repo's two frame encoders.
- Uses `Dataset_new`'s frame loop at stride 1, but discards the detector's
  `patch_` tensor, so RAM stays at ~0.9 MB per frame.
- Feeds GT tracks through the repo's `gen_feats_test`.
- Captures the B6 set with four forward hooks.
- Splits pairs over 40 slots into consecutive ≤ 40-slot windows and records
  them. Skips finished videos using the SHA256 manifest.
- **Checks fidelity on every call:** it rebuilds `sigmoid(logits) × sigmoid(int)`
  from the hooked tensors using the model's own methods, and records the
  residual against the model's returned scores.
- **Dump-fidelity mode:** it recomputes `modelC` on the detected pairs in
  `segments_raw_all.json` and compares against the dumped top-20 scores. The dump
  contains everything this needs: pair boxes, categories, segment ranges, and
  the top-20 predicates with scores. The only limit is that it holds the top 20
  of 132 scores per segment, so fidelity is checked on those 20.

**Verified here, with random tensors of the real shape:**
- the pair order matches `gen_feats_test` item for item, including categories
  and durations
- slot inputs are 4×768 / 4×24 / 42 / 42, as audit B4 says
- RoI pooling skips the pre-roll frames
- an 85-slot pair splits into windows 0–40, 40–80 and 80–85

**Not yet verified**, because it needs the GPU: encoders, `modelC` loading,
hooks, and both fidelity checks.

**Deviation:** logits and interactiveness logits are stored in fp32, not fp16.
They get ranked in post-processing, fp16 could reorder near-ties, and it costs
about 0.5 KB per slot.

**Estimate for the host run:**
- *Throughput.* The pilot measured 37–60 s per video for the full pipeline on a
  Vast.ai RTX 4090 with `OMP_NUM_THREADS=8`. The GT path skips the detector and
  tracker, but adds per-frame RoI pooling and one text-tower pass per pair.
- *Time.* ≈ 8–17 h for Task 3 (1,000 videos) and ≈ 2–3 h for Task 6.
- *Price.* The pilot recorded none. At an **assumed** $0.30–0.60/h for a 4090,
  that's ≈ $3–12 in total.
- *Size.* ≈ 3 GB (audit B5), plus fp32 logits.
- The 5-video run gives the real per-video time.
