# D2 — Steps 0 and 1 report (stopped at the Step 2 gate)

> **Test-label error analysis, disclosed; no method evaluated.** No test label has been
> opened yet, and no GPU has been used.

**Status.** Steps 0 and 1 are complete. A25, A25a and A25b are appended, hashed, set
to 444 and committed locally. Nothing is pushed. The batch is **stopped at the Step 2
gate**:
- no saved intermediate outputs exist, so the recording pass needs a GPU;
- it needs your approval of the estimate in §6, plus three decisions (§7).

**Location.** At the author's request, `prereg/`, `rb1/` and `d2/` now live under
`METOR/studies/` (A25b, §10). **Every `d2/…`, `rb1/…` and `prereg/…` path below means
`studies/d2/…`, `studies/rb1/…` and `studies/prereg/…`.**

Date: 2026-10-10 · METOR repo (`ghibli613/metor`), local `~/Documents/thesis/source/METOR` ·
EOV working repo `~/Documents/thesis/source/ov-vidvrd-lab` (`fdeeeb4`, read only)

---

## 1. Pre-registration chain

| item | value |
|---|---|
| ov-pairtime prereg (A24–A24i, closed), final SHA256 | `981bf067da43c41df8759a260c9bde61014b411809e13bf9838f62f6906b5ce5`: equals line 1 of its `.sha256`, file 444, not modified |
| METOR prereg before A25 (after A23a) | `bdfe25e62b20d3d82c743c848197d3448aff9d40fe6e98efdac3fd6134af268b` (verified) |
| after **A25** (definitions, reading rule) | `2b54bf65fb43e3150ba4942bc8f14eea76e0ff3953b33568f35299b507de69ea` · A25 text alone `0124eba3…cbd906` |
| after **A25a** (code pins) | `f18440cb7737ecb1c562963648ffeae66cb4580799cc510330dabfce714ee175` · A25a text alone `91891c78…cdb5a8` |
| after **A25b** (the folder move, §10) — current | `58211c14debd881df31f670ff6722f7ae575558d39863803e40c8b83723d6d6a` · A25b text alone `ad00798d…34e55` |
| commits (local, not pushed) | A25 alone, before any D2 code existed: `3a54d6e944bc1bf772cca479556dcccd59b30092` · code + A25a: `b4d53a062c90980aca1421c1cb645f9d33bd7f96` · move + A25b: `45c3c3a637ca0294ad1e7178346a1b8ab12eecaa` |

How each append was done:
- the prior hash was verified, and the earlier bytes were confirmed unchanged
  afterwards;
- `preregistration.sha256` got a new first line and an `after-…` line;
- both files were set back to 444;
- the before/after hashes are logged in `d2/logs/prereg_log.txt`, with backups next
  to it.

A25 states three things:
- the chain continued in ov-pairtime;
- D2 is the chain's second disclosed test-label analysis, after D1. It is not a
  test evaluation, and the order of test evaluations recorded in A24i is unchanged;
- O1 (A22) stays cancelled. D2 does not revive its gates.

## 2. Inputs, verified (ground rule 4): all match

| input | SHA256 | checked against |
|---|---|---|
| D1 `time_check/data/tc_missed.json` (buckets) | `2e4c53b9…3df2` | D1 report §6, `logs/tc_analyse.json` |
| D1 `tc_tables.json`, `tc_run.py` | `c3cb77f9…bf80`, `fae863dd…d5996` | D1 report §6 |
| `session3_box_copy/SHA256SUMS_session3.txt` | `2cc2d071…954b` | EOV prereg pin, D1 logs |
| baseline tracks and outputs: the 204 files of `cache_s3_sgdet/` | each equal to its line | `sha256sum -c` |
| `test_relation_gt.json` / `test_object_trajectories_gt.json` (bytes only) | `b7a8219f…1aeb` / `94df4451…699a` | `rb1/logs/data_meta_sha256.txt` |
| class split, ECC test matrices | `9ac86fc3…6dec`, `6e7842be…a2b` | same |
| end-to-end checkpoint, `AFLink_epoch20.pth` | `d306952a…0e70`, `b35cbedd…a4` | `rb1/logs/checkpoint_sha256.txt` |
| the 200 `anno/test/*.json` (bytes only; first pinned in A25) | listed in `d2/logs/anno_test_sha256.txt` (`c98b1b84…2a85`) | — |
| EOV pilot prereg (frozen at A22) | `442623b8…11f0` | A23 |

The brief's "201 of 484 involve only base objects" agrees with A22 §1d: U1 = 262, of
which 61 involve a novel object. These counts are reused, not recomputed.

## 3. Step 0 — the front end, stage by stage

The code read is EOV `fdeeeb4`, the commit session 3 ran from to make the baseline.
Its tracked code has not changed since. METOR's copy is the same on the test path,
apart from the fusion call (bit-identical with RB1's flags off). The full map, with
METOR line numbers too, is in `d2/stage_map.md`.

| # | stage | code fact | where (EOV) |
|---|---|---|---|
| 0 | frames | every frame in [begin_fid, end_fid), **the min/max of the video's GT object trajectory extents**, not the video's length | `data_loading/dataset.py:120-126` |
| 1 | detector | 300 queries; sigmoid probabilities over 35 classes; no NMS | `parser_func.py:226`; `post_process.py:191` |
| 2 | **cut** | keep a query iff its **max class probability is > 0.35**. If none passes, keep the frame's single best query. Then the top 300 (query, class) entries. | `post_process.py:194-195`, `:221-233`, `:264-282` |
| 3 | **dedupe** | `filter` → `unique_boxes`: a box is dropped if **any later box** (lower score) has IoU > 0.8 with it, so the lower-scored duplicate survives | `gen_labels.py:145-162`; `end2end_model.py:211-221` |
| 3b | clip, fuse | boxes clipped to the image; the 35 scores fused with modelB. Tracking does not use the scores. | `end2end_model.py:224-265` |
| 4 | **deep_sort** | settings below the table | `deep_sort_app.py:125-209`; `parser_func.py:302-373`; `tracker.py:38`; `kalman_filter.py:148-149` |
| 5 | **AFLink** | `thrT` (0, 30), `thrS` 75, `thrP` 0.05; then one row per (frame, id) | `end2end_model.py:283-293`; `AppFreeLink.py:73-109` |
| 6 | scores to rows | each row takes the scores of the same-frame detection with the highest IoU > 0.5 | `end2end_model.py:295-313` |
| 7 | **filter, in order** | the five rules below the table | `gen_labels.py:363-371`, `:381`, `:385-413`, `:416-428`, `:626`; `end2end_model.py:319` |
| 8 | naming | argmax of the mean 35-vector over scored rows; a track with no scored row gets class 0 and a NaN score | `gen_labels.py:603-622` |

**deep_sort (stage 4).**
- **Its own filters do nothing:**
  - every detection has **confidence 1** (`:55`), so `min_confidence` 0.2 removes
    nothing;
  - the minimum height is 0;
  - NMS runs at overlap 1.0, and overlap ≤ 1 by construction.
- **Matching:**
  - vanilla matching, cost 0.98 · cosine + 0.02 · Mahalanobis, χ² gate;
  - max cosine distance 0.35, budget 100, EMA 0.9;
  - ECC camera compensation; IoU stage 0.7.
- **NSA with confidence 1:** the measurement noise is 0, so **an updated box
  equals its detection's box**.
- **Lifecycle:** `max_age` 30, `n_init` 3. Only confirmed tracks with
  time_since_update ≤ 1 are output, so a track's first 2 frames never are.

**Filter (stage 7), in order:**
1. 2 frames are added before the first row, copying its box (this makes up for
   `n_init`);
2. **coverage:** the rows, the 2 included, must be ≥ 65% of the span;
3. **gap trim:** repeatedly cut at gaps > 15% of the span, keeping the side with more
   rows;
4. interpolation and rounding;
5. **length ≥ 10 frames** and **range > 15% of (end_fid − begin_fid)**.

**Where the code differs from the brief (and from A22 §1a):**
1. **The length rule is ≥ 10 frames, not ≥ 12.** `len(track) >= 12` also counts the
   dict's `category` and `score` keys.
   - A test on the repo's own `format_trajectories_test` confirms it: 8 rows plus 2
     added frames survive, 7 plus 2 do not.
   - The baseline's shortest final track has exactly 10 frames.
2. **Two steps are missing from the brief's five:** the 0.8-IoU **dedupe** between
   the cut and the tracker, and the **gap trim** inside the filter. Each gets its own
   label.
3. **The range in the 15% rule is the GT-extent range** of stage 0, not the video's
   length.
4. **The cut and the dedupe are the only detection filters before tracking.**
   deep_sort's own filters are no-ops.

**Saved intermediate outputs: none usable.**
- Session 3's SGDet cache stores only `model.traj` (the final, filtered, named
  tracks), pair logits and features.
- No raw detections, cut output, tracker output or filter decisions exist anywhere
  on the laptop. A22's O1 dump was never run.
- **So Step 2 needs a GPU.**

## 4. What A25 fixes (summary, for review before money is spent)

**Missed side.** A U1 relation's subject or object whose best **final** track has
vIoU < 0.5. This is D1's `best_viou`: the repo's `viou`, with the track cut to the
GT relation's extent.

**Detection sets per frame** (boxes clipped as in the pipeline):
- **raw(s):** cut score ≥ s, **plus** the cut's selected queries, so the fallback
  query never counts as "not detected";
- **cut:** the queries the cut selected;
- **tracker input:** the detections deep_sort receives, after the dedupe.

The cut score is the detector's max class probability, the score the 0.35 cut uses.

**Oracle link.**
- Per GT frame, take the detection with the highest IoU (repo `iou`; class ignored;
  ties go to the lowest query index).
- A frame where the best IoU is 0, or the set is empty, has **no box**. It is
  encoded as `[0, 0, −1, −1]`, which adds exactly 0 to the repo's +1-convention,
  volumetric `viou`.

**Pre-filter tracks.** The rows after AFLink and score assignment, grouped by id,
with missing frames inside linearly interpolated (as the pipeline would do). No
added frames and no rounding.

**Stages, per missed side; the first that applies:**

| label | condition |
|---|---|
| **O1a** | oracle over raw(0.05) < 0.5 |
| **O1b** | oracle over cut < 0.5 |
| **O2** | no pre-filter track ≥ 0.5. Sub-labels: **dedupe** (oracle over tracker input < 0.5) → **frag** (the good frames go to ≥ 2 tracks, each holding ≥ 10% of the GT extent) → **other**, split descriptively into untracked (< 50% of good frames assigned) / tracked-but-off |
| **O3** | some pre-filter track ≥ 0.5. The best one is replayed through the repo's own filter functions, and its first failing rule named: **F-cov** (`:381`), **F-len** (`:626`, a prior trim flagged), **F-range** (`:319`), or **F-reshape** (kept, but the final version is < 0.5; split into gap trim / other) |
| **O4** | the replayed track is kept with final vIoU ≥ 0.5. Impossible for a missed side if ground rule 3 holds; the report explains any case. |

**Relation label:** the most upstream stage among its missed sides. **S0** = no
missed side: both sides are found, but by the same track, so D1's s ≠ o pair fails.
S0 is reported, stays in the denominator, and is in no reading group.

**Class split:** a side by its GT class; a relation is "base-only" or "involves
novel".

**Funnel (Table 5):** every GT object track in `anno/test`, over its whole extent.
Cumulative stages: F1 oracle raw(0.05) → F2 cut → F3 tracker input → F4 pre-filter
track → F5 final track. Base vs novel.

**CIs:** D1's `boot_share`, unchanged: 1,000 resamples of whole videos, seed 0.

**Reading.** By relation, pooled. The groups are **link** = O2 + O3, **rescore** =
O1b and **detector** = O1a. The largest by count names the module, as in the brief.
The report gives:
- the largest and second-largest groups, with CIs;
- the paired bootstrap CI of their difference, and "not separated" if it includes
  0;
- in a tie in count, every tied group.

**Stop rules:**
- an input hash mismatch;
- a ground-rule-3 difference on any video;
- a replay mismatch;
- a failed post-processor check during the run;
- a U1 join failure;
- missing or corrupted recordings.

## 5. Step 1 — recorders, classifier, analysis, tests

**Code, all in `d2/`; no pipeline file of either repo edited:**
- **`recorders.py`** adds a read-only forward hook on the post-processor and wraps
  `run` (deep_sort), `AFLink.link`, `format_trajectories_test` and
  `interpolate_and_adjust_frames`. Each wrapper copies first, then returns the
  original's own result.
  - Per frame it saves every query with cut score ≥ 0.01, plus the selected ones.
  - During the run it checks that the selected set and boxes equal the
    post-processor's output.
- **`record_run.py`** is the GPU runner, set up exactly as session 3's
  `sgdet_cache.py`: seed 3407, frame stride 1, the same checkpoint, AFLink weights
  and ECC.
  - After every video it checks ground rule 3 against
    `d2/data/baseline_tracks.json` (658 tracks extracted from the verified cache)
    and stops on any difference.
  - It is resumable.
  - Before loading anything, it refuses to start unless the 22 EOV front-end files,
    the d2 code and the inputs match `d2/logs/code_pins.json`.
- **`stages.py` and `analyse.py`** hold the classifier, the tables and the reading.
  `analyse.py` enforces the order: pins, cache, annotations (as bytes), code,
  baseline, every recording, ground rule 3 and the replay check on every video.
  **Only then** does it open the test labels.

**Tests (CPU, stubs, no CLIP):**
- **Totals:** **23 passed** on the EOV code root, and 23 on METOR's copy. METOR's
  own suite still gives **169 passed**. `record_run.py --check_only` verifies every
  pin.
- **Recorders on vs off** (off/on/off) through the repo's real `OVPostProcess`,
  dedupe, deep_sort, AFLink (real weights) and filter: identical tracks, and the
  originals restored.
  - A mutant that shifts every tracker input by 1 px changes the tracks, so the
    check has power.
  - Shifting one frame-0 box changes nothing: that frame is in deep_sort's
    tentative phase, and with NSA every later update snaps to its detection.
- **The classifier:** in a synthetic video, every relation lands in the stage it was
  built for at all three floors (O1a, O1b, O2-dedupe, O2-frag, F-cov, F-range,
  F-len, F-reshape-trim, S0). Also checked: the funnel, the join, ties, and that
  the labels are not opened before verification.
- **Faithful reuse:** D1's `boot_share`, taken from `tc_run.py` by AST, equals D2's.
  The vectorised IoU equals the repo's `iou` bit for bit.

**Differences from A25, recorded in A25a (old / new / reason):**
1. **The U1 join.**
   - *A25:* an ambiguous key stops the batch.
   - *New:* within a key, rows are matched to GT entries by D1's recomputed
     `best_pair_min_viou`. Rows equal in key **and** value take the equally many
     matching GT entries in file order. Unequal counts still stop the batch.
     **Every** row's recomputed value must equal D1's.
   - *Reason:* two same-class, same-predicate, same-span relations whose objects
     are both untracked give identical D1 rows. Stopping there would protect
     nothing.
2. **Added stop rules:**
   - front-end and code pins checked on the machine and in the analysis;
   - the cache and annotation files re-hashed at analysis time;
   - no repeated (frame, id) rows;
   - `pruning_threshold` must be 0.35.
3. **Timing rule.** The 2-video check runs `ILSVRC2015_val_00046001` and
   `ILSVRC2015_train_00047020` (41.6 s and 13.0 s in session 3). If it takes more
   than 25% longer, the run stops for a new estimate.
4. **`modelC.tgt_split = "all"`,** as session 3 set it. This has no effect on the
   tracks.
5. **CPU tests copy on `.cuda()` and `.cpu()`.** On a CPU, deep_sort would
   otherwise edit boxes in place through a shared numpy view. This affects only the
   tests.

## 6. Step 2 — time and cost estimate (approval needed)

| | |
|---|---|
| Measured basis | Session 3 ran the **same forward** on these 200 test videos in **209.4 min** on a Vast.ai RTX 4090 (`OMP_NUM_THREADS=8`). The per-video sum was 175 min: median 33.5 s, max 398 s. |
| Added by D2 | Read-only recorders: per frame, a sigmoid/max over 300 × 35 and a small GPU → CPU copy; per video, an npz write. Estimated **≤ 10%**. The 2-video check measures it, with the 25% stop rule above. |
| GPU time | **≈ 4 h** (3.6–4.5 h), including model load and the 2-video check |
| Cost | **≈ $1.20–2.70** at an assumed $0.30–0.60/h. No hourly price has ever been recorded in the chain, so this is 4 h × the RB1 machine's actual rate. |
| Disk / copy-off | < 2 GB of recordings, verified against the manifest's SHA256 on the laptop |
| Setup | Shared with RB1: the EOV repo at `fdeeeb4`, the checkpoints, 7.5 GB of test frames, metadata. D2 runs **before RB1's own runs**, with A24h's token rules: no HF token during the run, and a token-string scan before and after. |

**Sequence:**
1. `--check_only`;
2. the 2 videos (ground rule 3, timing);
3. all 200;
4. copy off and verify;
5. Steps 3–4 on the laptop;
6. `d2/d2_report.md`, then a hard stop.

## 7. Decisions needed

1. **Approve the Step 2 estimate** (≈ 4 h on the RB1 machine, before RB1's runs).
2. **Test annotations on the rented machine.** The released test path reads them:
   - `test_object_trajectories_gt.json` sets the frame range;
   - `Dataset_new` loads `test_relation_gt.json`;
   - `gen_feats_test` reads `anno/test`.

   Session 3 ran the same way, and A25 discloses it. No D2 code reads them there,
   and all label analysis runs on the laptop. **This differs from M1's A24h rule**
   ("test labels never go to the rented machine"). Accept it as disclosed, or say
   if you want something else. Any substitute input files would have to pass
   ground rule 3 and be pre-registered first.
3. **Getting `d2/` onto the machine.** The commits are local. Either push METOR to
   GitHub, or copy `d2/` across. Either way, `d2/data/baseline_tracks.json` (3.3 MB,
   git-ignored like every `data/` folder) must be copied; the runner checks it
   against its pin.

## 8. Surprises and caveats so far

- **The dedupe keeps the lower-scored duplicate.** `unique_boxes` drops the earlier
  (higher-scored) of two boxes overlapping by more than 0.8. This doesn't change
  the cut but changes which box is tracked; O2-dedupe will measure it.
- **The processed frames and the 15% rule are GT-derived** (stage 0). This is a
  property of the released test path, already noted by A22 for the frames, and
  disclosed in A25.
- **The order of stages can be non-monotone.** A track can exceed the oracle by
  interpolating across frames with no detection. A side can therefore be labelled
  O1a or O1b even though some pre-filter track reaches 0.5. A25 keeps the brief's
  order, and the report will count such cases.
- **Housekeeping.** The study folders now live under `studies/` (§10). Commits are
  local; nothing is pushed.

## 9. Files (all under `studies/`)

| file | SHA256 |
|---|---|
| `d2/logs/code_pins.json`, current (re-pinned by A25b; pins every code, test, front-end and input file) | `9e47a91d7710405125da31764770df3f34b9637377c1764151f0a456575b847e` |
| `d2/logs/code_pins.A25a.json` (A25a's pins, kept unchanged) | `2e9cfd9291cf21096ece0eacaa13c27765e356ffbd2303603afd1c24c94d9d54` |
| `d2/data/baseline_tracks.json` | `3d9c5c86b9a80681cb1aa6643bab94f1b6bdea4448ecdb6151797ca63793df56` |
| `d2/stage_map.md` | `730a057121aa9dbb083666f5621e9ba0cf412a2cb6d69f21713324757949ce34` |
| A25a test logs: d2 on the EOV root / on METOR's copy / METOR's suite / `--check_only` | `ad419e3c…1202` / `6e6f6f25…3791` / `a192f0a0…0c5f` / `41067811…e996` |
| A25b test logs, the same four in the new layout | `0c7cc4f6…65976` / `3fbdb7a8…1f205` / `6999cd90…25b42` / `1be79f91…abfb5d` |

## 10. Folder move (A25b)

The author asked to gather the pre-registration and the batch folders into one place.
`prereg/`, `rb1/` and `d2/` moved from the METOR root into **`studies/`**, as git
renames (commit `45c3c3a`).

- **Nothing in the record changed.** All 51 files were byte-identical across the
  move, so the prereg's hash chain is intact.
- **Paths cited before the move now need the `studies/` prefix.** That includes the
  closed ov-pairtime header's absolute path to `METOR/prereg/preregistration.md`.
  A25b's path map resolves every such citation, and so does the index at
  `studies/README.md`. The frozen files themselves are untouched.
- **Path-only edits the move forced,** each listed in A25b with its old and new
  hash:
  - `d2/common.py`: the repo root is now two levels up;
  - the d2 test conftest and a comment;
  - RB1's two fallback scripts (their repo root and log path);
  - the commands in `rb1/README.md` and `d2/README.md`.

  None of RB1's 14 pinned code files was touched.
- **d2 was re-pinned, and every suite reran in the new layout with the same
  results:** d2 23/23 on both code roots, METOR 169 passed, and `--check_only`
  verifies every pin. The `d2` commands now run from `studies/`.
