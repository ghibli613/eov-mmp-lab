# Step 0 report — the agreed plan, A26 and its code, ready for the GPU (not run)

**For:** the plan's author. **Status:** stopped before any GPU use, as instructed. The next step
is the thesis author's approval of the spend.

**Done:**
1. **`studies/design_review/agreed_plan.md`.** It merges reviews 1–3, your acceptances and the
   final decisions:
   - steps, costs, decision points, risks and the fallback framing;
   - the outside check set: one VidOR train part, MD5-verified, videos with ≥ 2 annotated
     objects; ImageNet-VID dropped;
   - VidOR val untouched; a VidOR evaluation deferred, pending the author's decision on
     VidOR-trained models;
   - the realistic target, about +1 novel mAP on R0;
   - both bars, 16.74 (defensible) and 17.95 (prudent);
   - RB1's order: R2 → R1 → R3.
2. **A26 is written and appended:** the protocol audit, with its reading fixed.
   - The pre-registration is now `9485e1a8f72971db328b7c8a71a1c3ff96c301b3701a42ab8853b11e8559c494`
     (444/444). The first 70,428 bytes are unchanged. The append is logged in
     `studies/a26/logs/prereg_log.txt`.
3. **The whole-video wrapper, the runner, the evaluator and the machine scripts** are in
   `studies/a26/`. **30 CPU tests with stubs all pass.**
4. **The GPU estimate, the machine needs and the fetch list** are printed by `python -m a26.estimate`
   (§3).

**Not done:**
- **No GPU** was used.
- **No test-label file** was opened.
  - The laptop read session 3's prediction files and manifest, the shipped ECC file, and D2's
    report.
  - D2's recordings were not touched (A25d).
- **Commits are local only:** `486a962` (the code and plan), then a second commit with A26, this
  report and the index. Nothing was pushed.

---

## 1. Two changes from the brief, for you to check

1. **R0 is re-run, not reused.** This adds about 3 GPU-h (about $1–2).
   - The four configurations then come from one code path, on one machine, in one run.
   - Every video carries an identity gate: R0's tracks must equal session 3's, bit for bit.
   - Session 3's dumps were made by a reconstruction script (`eval_from_cache.predict`), not by
     the repo's chain. Pairing a re-run R0 with R0u, R0w and R0uw is cleaner than pairing
     session 3's files with them.
   - Step 0's cost in the plan is now **≈ 15 GPU-h, $5–11**, against the earlier ≈ 12 GPU-h,
     $4–8.
2. **The reading is symmetric in sign.** The earlier draft only had thresholds for Δuw < 0.
   - The whole video could also *raise* the score: the detector gets more frames, and the 15%
     rule's base changes.
   - So the words use |Δuw|: "large" for ≥ 1.0, "moderate" for ≥ 0.3, each needing the CI to
     exclude 0. The direction is stated.
   - The fallback framing moves up only if the clean protocol scores **lower**.
   - `agreed_plan.md` §5 matches.

## 2. A26 in brief

**The configurations.** All four use the released checkpoint and METOR's code unmodified
(`--arch metor-compat --n_iter 0`, seed 3407):
- **R0:** the GT-derived range, per-class fusion (as released);
- **R0u:** uniform 0.3/0.7 fusion (A23's R0u);
- **R0w:** the whole video, `begin_fid = 0`, `end_fid = video_len`. This also moves the 15%
  rule's base `min_l`;
- **R0uw:** both.

**Why METOR's code and not EOV `fdeeeb4`.** A23 defines R0u by METOR's `--uniform_fusion`. METOR's
test path is unchanged since `7e87c9b` (A23a). Gate 1 below proves on the GPU that it reproduces
EOV.

**The run.**
- Each video is encoded once, then gets four forwards.
- Both predicate splits come from one forward: modelC is called twice on the same inputs, as the
  pilot's fused path (B.21). The pilot's novel-branch predictions already equal session 3's
  (scores within 1.1e-7).
- Predictions go through the repo's `process_pred` → `association` → `format_`.

**Gates. Any failure stops the run, and the author is asked.**
1. On every video, R0's tracks must equal session 3's (D2's ground rule 3).
2. **The 2-video gate:**
   - R0's predictions, both splits, must equal session 3's (scores within 1e-6);
   - the projection of the full run must be at most **18 h**.
3. **On the laptop:**
   - all 800 files must match the manifest;
   - R0's mAP must equal session 3's (15.786 / 27.185) within 0.01.

**Evaluation (laptop, once).**
- The evaluator is unmodified, as `_predict_split` calls it.
- **The paired bootstrap:**
  - 1,000 resamples, `default_rng(0)` per split, the same draws for every configuration;
  - computed from per-video statistics taken from the evaluator's own functions, asserted
    against the official numbers;
  - on the first two resamples, cross-checked by running the evaluator itself.

**The reading.** Only one quantity is read: **Δuw, novel mAP**.
- "large" → the clean protocol becomes a primary protocol of the final session;
- "moderate" → both protocols are reported;
- "small" → the published comparison stands, with disclosure.

Δu, Δw, the `all` split and the recalls are descriptive, with no verdict words.

**Recorded in A26:**
- **R0u's evaluation moves** from RB1's session to here, with the reason: it is released-checkpoint
  only. RB1's session reuses it. A23's rule still binds R1–R3.
- **The count:** the seventh test evaluation by pre-registration order, the sixth actually
  performed. No method is evaluated.
- **A26's outputs are bound like D2's recordings:** never used for design or tuning.
- **Test annotations on the machine:** 202 files.
  - `Dataset_new` reads `test_relation_gt.json` and `test_object_trajectories_gt.json`.
  - `gen_feats_test` reads `anno/test/<video>.json`, for each video's height and width
    (`models/gen_labels.py:672-675`).
  - Only the pipeline reads them; no D2 code runs on the machine. They are deleted after the
    run, and the deletion is logged.
- **A small fact found on the way:** the shipped ECC file covers frames 2 to the last frame of
  199 test videos, so the whole video, not only the GT range. It has no entry for one video,
  which runs without camera compensation in every configuration, as released.

## 3. The estimate (`python -m a26.estimate`)

| | hours |
|---|---|
| encoding, once (from D2's 3.5 h pass − 3.02 h forward) | 0.48 |
| R0, R0u: GT range (D2's measured 10,873 s forward) | 3.02 each |
| R0w, R0uw: whole video (× 52,104 / 46,035 frames) | 3.42 each |
| **GPU total** (+10% for modelC's second call and the writes) | **14.7 (12.5–16.9)** |
| **machine**, with setup, the token fetch, the gate and the copy-off | **13–18** |

**Cost: $4.7–6.3 at the measured $0.35/h; $8.1–10.7 at $0.60/h.** The run stops and asks if the
gate projects more than 18 h.

**The machine:**
- one **RTX 4090**, as session 3 and D2, because gate 1 needs bit-identical tracks;
- **≥ 64 GB RAM** (`setup.sh` refuses less);
- **≥ 60 GB disk**;
- a PyTorch **-devel** image (`nvcc` builds the deformable-attention operator).

**What it gets:**
- **Copied from the laptop** (`box/upload.sh`; no push):
  - the EOV repo as `git archive fdeeeb4`, for the fetch tools and the CUDA operator;
  - METOR as `git archive HEAD`, without `studies/d2/data/rec`;
  - the class splits, id maps, `prior.pkl` and the train trajectories;
  - the 202 test annotation files.

  Each file's hash is checked on the machine.
- **From Hugging Face, with the author's token** (`box/fetch_with_token.sh`, run by the author;
  the token rules as in D2):
  - the eval weights: end-to-end checkpoint, AFLink, CLIP bank, both ECC files (2.92 GB);
  - the 200 test videos' frames.
- **From the public URL:** CLIP ViT-L/14@336px, with its SHA256 checked.

## 4. The tests (`studies/a26/logs/pytest_a26.txt`, 30 passed)

- **R0 through the A26 runner equals the repo's own `cli/common.py::_predict_split`,** run once
  per split with modelC unwrapped. This is on the CPU stub pipeline: the real post-processor,
  deep_sort, AFLink and format code, with a stub detector, classifier and relation head. Tracks
  and predictions both match.
- **Each configuration** sees the right frames and the right fusion flag. The flag is restored,
  even on error.
- **The order of configurations** does not change any output.
- **The whole video moves the 15% rule:** a 7-frame track is kept at min_l 40 and dropped at
  min_l 60.
- **`whole_video` on the real `Dataset_new.__getitem__`,** with stub encoders and synthetic
  frames.
- **The evaluator:**
  - the per-video aggregates equal the unmodified evaluator, on synthetic relations with real
    class names;
  - the bootstrap equals the evaluator on resamples;
  - the full evaluation path works on files in the runner's format;
  - the reading's boundaries are right, including a CI touching 0.

## 5. After approval

1. Rent the machine above and send the SSH line.
2. On the laptop: `bash studies/a26/box/upload.sh root@HOST PORT`.
3. `setup.sh`.
4. **The author runs `fetch_with_token.sh`.** I report when the token is gone.
5. `run.sh check`, then `run.sh gate`. I report the gate and the projection.
6. `run.sh full`.
7. `cleanup.sh`, which deletes the test annotations and logs it.
8. `copy_back.sh`. I report when the copy is verified, and the author destroys the machine.
9. `python -m a26.evaluate` on the laptop, once.
10. `studies/a26/a26_report.md`, then stop.

**The decision needed now:** approve about **$5–11** (13–18 machine-hours, with a stop at an 18 h
projection), on a 4090 with ≥ 64 GB RAM and ≥ 60 GB disk.
