# Pre-registration — batch RB1 (clean EOV-MMP baseline; METOR's iterative-enhancement module)

This file **continues the pilot pre-registration** at
`/home/totoro/Documents/thesis/source/ov-vidvrd-lab/pilot_analysis/stage1b/preregistration.md`
(the EOV-MMP working repo), **frozen at amendment A22**, SHA256
`442623b82cf5392c480e889ab160d5e0606300acc9d588d338e0a806036211f0`. That file is
not modified by RB1. Amendment numbering continues from it: the first entry here
is A23.

Hash convention (as in the pilot file): `preregistration.sha256` lists the
current SHA256 of this file first, then one `after-…` line per earlier state,
each giving the hash the file had **before** the named amendment was appended.
Both files are made read-only (444) after every append. Each append stops at the
first failed step and logs the SHA256 before and after in `rb1/logs/prereg_log.txt`.

## Amendments

#### A23 (2026-10-07): batch RB1 — a clean EOV-MMP baseline, and what METOR's iterative-enhancement module adds on it; the fifth test evaluation (disclosed)

**This is baseline construction and a module measurement. It is not the thesis
method.**

**A22's GPU run (O1 step 2) is cancelled. It is not run.** Nothing of O1 beyond
its step 1 (A22; `pilot_analysis/objects/o1_step1_inventory.md` in the EOV repo) will be produced under A22.
Its gates G-O1 and G-O2 are not evaluated. If O1 is wanted again it needs a new
amendment.

**What exists before A23, and what was done to write it.**
- No model was built or run. No GPU was used. The machine is a 4 GB GPU / 7 GB
  RAM laptop; it can do code reading, CPU tests with a stub CLIP, and hashing.
- Code was read in the METOR reconstruction repo (`ghibli613/metor`, local path
  `~/Documents/thesis/source/METOR`, base commit
  `3be814763e2d9acffa26dc5ff20fdde249bce3fe`, working tree clean).
- SHA256 of the checkpoints, the CLIP weights and the metadata files were
  computed (files read as bytes only; **that includes the three
  `test_*.json` files under `data/vidvrd/data/`, hashed and not opened**).
- Counts in A22 §1c/§1b are reused, not recomputed.
- The pilot preregistration in the EOV repo was verified, read-only, at
  `442623b82cf5392c480e889ab160d5e0606300acc9d588d338e0a806036211f0` (the value
  after A22; logged in `rb1/logs/prereg_log.txt`). **It is not modified.** This
  file continues it; RB1's amendments live here.

**Why.** O1's step 1 found that the vendored EOV-MMP step-4 code feeds novel
boxes with their labels to the detector, and fuses detector and classifier
scores with per-class weights. The six classes with the 0.6 / 0.4 weights
(antelope 1, bear 3, cattle 8, fox 12, giant_panda 14, red_panda 23) are
**all novel categories**. The released numbers are therefore not a clean
open-vocabulary baseline, and a delta measured against them would be
ambiguous. RB1 builds the clean baseline and measures METOR's module on it.

---

##### The training-target audit (read from code at the base commit)

Step 4 is `cli/train.py` → `End2End_Model.forward` (training branch). Every
place it builds or consumes a training target:

| # | where | what the released code does | novel involved? |
|---|---|---|---|
| T1 | `models/gen_labels.py:260-278` `gen_box_label`, called at `end2end_model.py:69` | boxes and class ids for **every** trajectory fragment of `train_object_trajectories_gt.json` (`cls2id`, no split filter). 1,244 of 4,834 fragments are novel (A22 §1b). | **yes** |
| T2 | `models/methods/detectors/ov_prompt/model.py:309-321` (`forward_train`) | the detector's text queries and exemplar image queries are the **labels present in the frame** (`uniq_labels`), so a frame's novel labels become queries. `all_train_token` is False, so `seen_list` is not used. | **yes** (follows T1) |
| T3 | `end2end_model.py:70-72` `len(labels) > 0` gate | a frame with no targets is skipped for the detector and contributes no detections to tracking. | frames with only novel objects are *not* skipped in the released code |
| T4 | `gen_labels.py:238-257` `gen_frame_pre_label` → `target['pred_label']` → `set_criterion.py:404-416` | frame-level predicate multi-hot, **base predicates only**, but taken from every relation instance in the frame, **whatever the categories of its subject and object**. | predicates: no. a base predicate whose subject or object is a novel object: **yes** |
| T5 | `gen_labels.py:496-583` `gen_feats` (relation targets) | the track pair is matched to GT objects of **every** category (`gen_hit_tid` over all of `vid_gt_trajs`); the predicates come from `gen_label` over **all** of `relation_instances`, written into a 132-wide `pre_label`. Object labels `sbj_label`/`obj_label` are the **detector's predicted track class**, not GT. | **yes**: GT novel objects can be hit; novel predicates are in `pre_label` |
| T6 | `models/relation_classifier.py:528-545` (`Model.forward`, training) | with `src_split = base`: `pre_labels` = base columns only; object labels pass through `convert_oid_on_base`, which maps novel ids to −1 (ignored). **But `int_labels` = `sum(pre_label over all 132 columns) > 0`** — a pair whose only relations are novel predicates is a positive for the interactiveness loss. | **yes**: a novel-predicate leak into `int_loss` |
| T7 | object classifier (`modelB`) | **not called** in the training forward; no loss, no gradient (AdamW skips parameters with no grad). | no |
| T8 | upstream of step 4: the stage 1–3 checkpoints | **not inspectable** (A22 §1b: no step-1/2/3 training code in the repo). Whether they saw novel data is not known here. | unknown |

Two points the audit settles about the released run:
- The leak T6 exists whatever the detector did; it is a property of the loss.
- T3 means that, in the released code, frames whose objects are all novel were
  trained on and tracked. The clean setting changes that (below).

---

##### What "clean" means here, and where it is enforced

Flag `--exclude_novel_train` (default off; off = the code exactly as released).
When on:

| filter | applies at | effect |
|---|---|---|
| F1 | `Dataset_new.__getitem__`, train, `gt_traj` | fragments whose category is novel are removed. **`begin_fid`/`end_fid` are computed before the filter**, so the processed frame range is the released one. Frames are kept. |
| F2 | same place, `gt_rel` | relation instances removed if the predicate is novel, **or** the subject or the object tid has a novel category (category from the video's `subject/objects`). |
| F3 | `gen_feats`, `vid_anno` | the same two removals: novel tids dropped from `vid_gt_trajs`, and `relation_instances` filtered as in F2. A track over a novel object gets no GT hit and so no relation labels: it is a negative pair. Novel objects become background. |
| F4 | consequences, not code | T1/T2: novel labels never reach the detector's targets or text queries. T4/T5/T6: `pre_label` has no novel column set, so `int_labels` carries no novel-predicate signal. |

**Choice, fixed here: frames whose objects are all novel.** After F1 such a
frame has no targets, and the existing `len(labels) > 0` gate (T3) skips it for
the detector and for tracking, exactly as the released code treats a frame with
no objects. The alternative, running the detector on an empty target set,
crashes in the text-query step and would be new behaviour. The number of frames
skipped this way is logged. Novel objects in frames that also contain base
objects are background, as intended.

**Asserted and logged during training** (a run aborts on a non-zero count):
- novel category ids in detector targets (`target['labels']`): 0;
- novel category ids among the detector's selected text queries: 0;
- novel-predicate columns set in any `pre_label`: 0;
- novel-category GT tids in the matched hits used for labels: 0;
- novel-predicate or novel-object relation instances reaching `gen_label`: 0.

Logged, per epoch: fragments removed (of 4,834 and of the novel 1,244),
relation instances removed and why (novel predicate / novel subject or object),
frames skipped as all-novel, and the counts above.

**What this does not cover, and is stated in every report:** T8. The stage 1–3
checkpoints may have seen novel data, so **R2 and R3 are clean for step 4 only**.

---

##### Fusion

Flag `--uniform_fusion` (default off). When on, at `end2end_model.py:261-264`
every one of the 35 classes uses `0.3 · modelB + 0.7 · detector`. This is the
code's own setting for most classes, fixed a priori. **Nothing is tuned and nothing
is searched.** Off: the released code, including turtle (31) unfused and
red_panda (23) fused twice.

---

##### Configurations (fixed)

| id | starts from | training data | fusion | IEM | trained here |
|---|---|---|---|---|---|
| R0 | released end-to-end checkpoint | — | per-class (as released) | off | no. Re-evaluated in Step 4. Session 3 measured SGDet novel 15.79 / all 27.18 |
| R0u | the same checkpoint | — | **uniform 0.3 / 0.7** | off | no. evaluation only |
| R1 | stage 1–3 checkpoints | as released (novel included) | per-class | off (`--n_iter 0`) | yes. reproduces the released step 4 |
| R2 | stage 1–3 checkpoints | **novel removed** (F1–F3) | **uniform** | off (`--n_iter 0`) | yes. **the clean baseline** |
| R3 | stage 1–3 checkpoints | **novel removed** (F1–F3) | **uniform** | **on** (`--n_iter 2`) | yes. **clean baseline + METOR's iterative enhancement module** |

**Correction fixed here: what R3 contains.** `--arch metor-compat` attaches
**only the iterative enhancement module (IEM)**: `metor/compat.py`,
`attach_metor_modules` → `METORFeatEmbedding`. **Contextual refinement encoding
(CRE) is not wired into compat.** R3 = R2 + IEM. It measures the IEM, **not CRE
and not METOR as published**. The report and every sentence about R3 say so. The
CRE `pool`/`inject` modes (doc 32 §6) are not used by this path.

**`--n_iter` defaults to 2 in compat** (`compat.py:82`,
`getattr(args, "n_iter", 2)`). R1 and R2 therefore pass `--n_iter 0`
**explicitly**. At startup each run logs and asserts:
- R1, R2: `featEmbedding.iem is None`;
- R3: `featEmbedding.iem` exists with 2 layers.

IEM hyperparameters in R3 are the repo's, unchanged: `alpha` 0.9, `n_heads` 8,
`dropout` 0.1, `st_max_len` 512, `residual_mapping` False, `n_iter` 2, hidden 768.
R0 and R0u use `--n_iter 0` (the released checkpoint has no IEM keys).

**R2 and R3 differ only in `--n_iter`.** Same stage 1–3 checkpoints, same
filters, same fusion, same seed, same data order, same epochs, same learning
rate, same optimizer and scheduler.

---

##### Training protocol (fixed)

- **Fixed 5 epochs** (EOV-MMP's step-4 budget): `--max_epoch 5`, `--lr 1e-5`,
  `--batch_size 1`, AdamW (weight decay 0.01, as in `cli/train.py`),
  `MultiStepLR` milestones [15, 20, 25] (never reached), seed 3407.
  The released checkpoint's own config says epoch 6; that gap is noted, not
  chased. **The last epoch (epoch 5) is the final model.**
- **No model selection on test.** The per-epoch test evaluation in
  `cli/train.py` is disabled by a flag (`--no_epoch_eval`, default **off** so
  the released behaviour is unchanged; RB1 runs pass it **on**). Test is touched
  only in Step 4.
- **Identical training for R2 and R3.**
  - The training `DataLoader` gets its own `torch.Generator` seeded 3407; any
    other shuffling gets its own seeded generator. The video order of R1, R2 and
    R3 is logged and **asserted identical** on the 5-video run.
  - The global RNGs are re-seeded to 3407 **after the model is built and before
    the first step**, so building the IEM (R3) does not change anything the
    other runs draw afterwards from the CPU stream, including the detector's
    query sampling (`torch.randperm`, `model.py:308-321`).
  - **Known limit, stated:** the IEM has dropout 0.1, which draws from the CUDA
    stream; R3's CUDA stream therefore differs from R2's after the first IEM
    forward. That is inherent in having the module and cannot be removed without
    changing the module. One seed per configuration.
  - `set_trainable` must leave the IEM parameters trainable in R3. Trainable
    parameter counts are logged per configuration; **R3 − R2 must equal the
    IEM's parameter count**, or the run aborts.
- **Checkpointing and resume.**
  - A checkpoint is written at the end of **every** epoch: model, optimizer,
    scheduler, epoch, and the RNG states (Python, NumPy, torch CPU, torch CUDA,
    and the loader generator).
  - `--resume <ckpt>` restores all of it and continues exactly.
  - On the small run: 2 epochs straight vs 1 epoch + resume + 1 epoch. Reported:
    bit-identical or within float tolerance (and which).
  - After every epoch the checkpoint and logs are copied off the machine to
    persistent storage. A run must survive losing the machine.
- **Failure rule.** If more than 1% of the 800 training videos fail in a run,
  stop and report.

---

##### The test evaluation (once, together, at the end)

**This is the fifth test evaluation.** The earlier four are disclosed in this
file and the session reports.

- **Models:** R0, R0u, R1, R2, R3 (R0 re-confirmed). One session, after all
  training is finished. Nothing is evaluated on test before that.
- **Settings:** SGDet, predicate splits `novel` and `all`, through the unmodified
  repo evaluator (`cli/common.py::_predict_split` →
  `eval_relation_detection_openvoc`). PredCls only if that path supports it with
  no new code in this repo; otherwise not reported. **The evaluator is not
  edited.**
- **Metrics:** mAP, Recall@50, Recall@100, for `novel` and `all`. **mAP_o is not
  reported**: it is not produced by this evaluation path, and adding it needs
  new code. (If the upstream object-evaluation script runs on the dumped
  trajectories unchanged, it may be added and will be labelled as an addition.)
- **Per-video predictions** are dumped, for each configuration and split, by a
  thin script outside the evaluator (`rb1/scripts/`), which calls
  the same `process_pred` → `association` → `format_` → evaluator chain as
  `_predict_split` and writes the formatted per-video predictions. The headline
  numbers must equal what `_predict_split` returns; the script asserts this.
- **Paired video-level bootstrap**, 1,000 resamples, seed 0, 95% percentile
  interval. One resample draws test videos with replacement, **the same draw for
  every configuration**; each configuration's mAP / R@50 / R@100 is recomputed
  with the unmodified evaluator on the resampled videos (duplicates get distinct
  keys, and the GT file is resampled the same way). Differences:

  | difference | what it measures |
  |---|---|
  | R0u − R0 | the per-class fusion weights |
  | R1 − R0 | does our retraining reproduce the released step 4 |
  | R2 − R1 | removing novel data and the per-class weights together |
  | **R3 − R2** | **what the IEM adds on a clean baseline (primary)** |

**Reading (fixed):**
- **R3 − R2** (mAP, SGDet): **"helps"** if it is positive on **both** `novel` and
  `all`, **and** the `novel` CI excludes 0. **"Inconclusive"** if it is positive
  but the `novel` CI includes 0. **Otherwise "no gain".** The words are about the
  IEM only.
- **R1 vs R0:** if R1's `novel` mAP is more than 1.0 point from R0's, our
  retraining does **not** reproduce the release. The report says so, and R2 and
  R3 are comparable **only to R1**, not to R0.
- R0u − R0 and R2 − R1 carry no verdict words; they are reported with their CIs.
- No other contrast is read. Anything else in the report is descriptive.

---

##### Pins at A23

| artefact | SHA256 |
|---|---|
| METOR base commit | `3be814763e2d9acffa26dc5ff20fdde249bce3fe` (clean tree) |
| `output/ckpt/checkpoint_vidvrd0059_new_1e-5.pth` (`--detector_ckpt`, stage 1) | `93c4457cd4a3e5b2db821753409bc3dc3dc0cd9d2d1bac3646f6efd82e8966da` |
| `output/ckpt/vidvrd_backboneViT-L_14@336px_lr0.01vision-guided.pth` (`--obj_classifier_ckpt`, stage 2) | `3525a173e26f6534f52a275a8d3e22a114a392765b5a1abda80aac5db5eff098` |
| `output/ckpt/baseline_fbce_vidvrd_bs1_lr0.0001_drop0.5_dim512_none_rel_mot_clip_bbox_stage2_new_L14_e2e.pth` (`--relation_ckpt`, stage 3) | `d9839ed789134e9acaad7b85106bf93186dc34b0f05b87a2faa9ee32cbbd2f24` |
| `output/ckpt/baseline_fbce_vidvrd_bs1_lr1e-05_dim512_none_rel_mot_clip_bbox_end2end_base-001.pth` (the released end-to-end checkpoint; R0, R0u) | `d306952a51e9ca94c201ecf580f75a99221ae658dc187ae0ff38ff8b7dfc0e70` |
| `output/ckpt/baseline_fbce_vidvrd_bs1_lr1e-05_dim512_none_rel_mot_clip_bbox_train42-003.pth` (present, **not used** by RB1) | `adff1b04af31260c3e50e83c6115ead231b4c855305f49b74c4546b968c60f38` |
| `output/ckpt/AFLink_epoch20.pth` | `b35cbeddd3acc48fece820bd640640e6bfb1f5fbf570aa79af26c6a38958daa4` |
| CLIP ViT-L/14@336px weights (`~/.cache/clip/ViT-L-14-336px.pt`; equals the hash OpenAI embeds in its download URL) | `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02` |
| metadata files (`data/vidvrd/data/*`) | listed in `rb1/logs/data_meta_sha256.txt` |
| checkpoint hash list | `rb1/logs/checkpoint_sha256.txt` |

The hashes of every **changed or new code file** in the METOR repo, the commit
that holds them, and the hashes of the Step-4 evaluation scripts are **not
known yet**. They are pinned by a dated pin addendum, **A23a**, appended **after
the Step-1 code and tests exist and before any GPU run**. It also pins the
rented machine's checkpoint hashes (they must equal the table above). **No
model is built or run on any GPU before A23a exists.** Any change to a design
choice above is a new amendment, written before the result it affects.

**Order.**
1. **Step 0** this amendment.
2. **Step 1, CPU:** the flags (`--exclude_novel_train`, `--uniform_fusion`,
   `--no_epoch_eval`, `--resume`, the seeded loader), the per-epoch checkpoint,
   tests that pin both behaviours (flag off = released; flag on = filters and
   zero counts), the full existing test suite, and the check that with all flags
   off `--n_iter 0` is still bit-identical to EOV. Then A23a.
3. **Step 2, rented GPU, 5 training videos per configuration** (and 5 test
   videos only as a plumbing check; **no number from them is reported**), then a
   measured runtime and cost estimate, then **stop and ask.**
4. **Step 3:** full training of R1, R2, R3.
5. **Step 4:** the single test session. Then the report
   `rb1/rb1_report.md` and a hard stop.

#### A23a (2026-10-07): pin addendum to A23 — Step-1 code, tests, and what differs from A23

Written **after the Step-1 code and tests exist and before any GPU run.** No
model has been built, trained or evaluated on a GPU. No test annotation was
opened (the only test-side files touched are the three `test_*.json` metadata
files, hashed as bytes in A23). **Nothing in A23's design choices is changed**;
the differences below are about how it was implemented, and each states the old
choice, the new one and the reason.

**Pins.**

| artefact | value |
|---|---|
| METOR code commit | `7e87c9bb8c45cd4699469546c3c4310db20b87b2` (parent `3be814763e2d9acffa26dc5ff20fdde249bce3fe`, the A23 base). Working tree clean at this commit apart from `prereg/` and `rb1/`. |
| changed or new files, SHA256 | listed in `rb1/logs/changed_files_sha256.txt` (14 files: `cli/common.py`, `cli/train.py`, `cli/train_state.py`, `data_loading/feature_dataset.py`, `docs/23_Configuration-and-flags.md`, `models/end2end_model.py`, `models/fusion.py`, `models/gen_labels.py`, `models/novel_filter.py`, `utils/parser_func.py`, and four `tests/test_rb1_*.py`) |
| `models/novel_filter.py` | `fe8ba0a351a6f5ba216abd7dbde752160e7cbd9d512de72e2a7e5d65cbed50bb` |
| `models/fusion.py` | `e43221cffe05818f34f20512bd1859e773914538ec19d1d5916871f9e1365f4b` |
| `cli/train_state.py` | `d909fd280d7fe489efc18d39e377098b0aa0c3aa73923abe45065ec8c869ff83` |
| `cli/train.py` | `83d71acff81672225214df4e0ba5beb5e0a019d9d6d748b8dc55917bf31002d3` |
| `models/end2end_model.py` | `e454a7cb9ea92c0fb736a6307632c7048e6920b136d13785e4d5673df838b9ef` |
| `models/gen_labels.py` | `319550e0fdaa049f4f19b46f0a63ae62b58b7d6f450fa48fdd1996a62d8b8624` |
| `data_loading/feature_dataset.py` | `9e2f5d2221cbac6223d79a57e8708e6fbffd6af926a5b37db6fbb1997022119b` |
| `utils/parser_func.py` | `024a2e05bff9ede5f9ed0eb002bf92ecd2fd5e72a4c509f77eda27f287bd21c2` |
| `cli/common.py` | `079223905d398a98ff346c38af034c93b4643c1f0d2b5eaab92bd1c390c556c6` |
| test log at the pinned commit | `rb1/logs/pytest_step1_at_7e87c9b.txt`, SHA256 `edff868f7e5691606c1dd124793abf3b5638f17032e6818b54747082956677e2` |
| test log at the base commit (clean worktree) | `rb1/logs/pytest_baseline_3be8147_clean.txt`, SHA256 `1b90f3d8818d6b9a41507a102e364b738d9052b8a6702760cf776a4945249800` |

**Test results.** Base commit, clean worktree: **137 passed**. Pinned commit:
**169 passed** (the 137 plus 32 new in `tests/test_rb1_*.py`), no failures, no
skips. `tests/test_compat_equivalence.py` (4 tests) passes: **with every RB1 flag
off, `--n_iter 0` is still bit-identical to the EOV baseline, and wrapping still
preserves every released-checkpoint parameter name and shape.** The new tests
pin both settings of each flag:
- fusion off = a verbatim copy of the released loops, bit for bit, including
  red_panda fused twice and turtle unfused; on = 0.3/0.7 once for all 35;
- `--exclude_novel_train` on a real training video: no novel fragment, relation
  or predicate survives; `gen_feats` gives the same pairs with and without the
  flag, novel predicates are in `pre_label` without it and **zero** with it, and
  the interactiveness target (any predicate set) has strictly fewer positives;
  flag off, the dataset item equals the annotation file;
- exact resume: 2 epochs straight equals 1 epoch + resume + 1 epoch, **bit
  identical** weights and identical video order, with a different seed given to
  the resuming process; a mutant that skips restoring the loader generator makes
  the test fail (checked once by hand, not a test file);
- video order is the same whatever random numbers were drawn before (R2 vs R3);
- startup aborts when `--expect_n_iter` disagrees with the model, when the IEM is
  frozen, or when its names match a frozen pattern; R3's trainable count minus
  R2's equals the IEM's parameter count;
- every RB1 flag defaults to the released behaviour.

**What the tests do not cover (needs the GPU, Step 2).** The training forward
through the detector, `criterionA`, `postprocessorA`, deep_sort and AFLink with
the flags. The pieces around them were run on CPU on real annotations (target
builders, `gen_feats`, filters, loader, loop, checkpointing); the detector end
was not run, because the machine cannot hold the model.

**Differences from A23, each with old / new / reason.**

1. **R1 is not literally "the released code".**
   - *A23:* R1 reproduces the released step 4 "as released".
   - *Found:* the released target builders **cannot run on the files in this
     repo**. `gen_frame_pre_label` indexes `predicate[0]` and raised
     `KeyError: 's'`; `gen_box_label` indexes `category[0]` and `box[0][0]` and
     raised `TypeError`. The annotation and trajectory files hold strings and flat
     `[x1,y1,x2,y2]`, the code assumes one-element lists and nested coordinates.
     Run on a real training video before any edit. Step-4 training as vendored
     therefore never ran here (the repo's docs say nothing has been trained).
   - *New:* both functions accept either form. **This is not behind a flag**:
     the old behaviour is a crash. R1 is "the released logic with a format fix
     that changes nothing else"; the report states it. The semantic reading (the
     same categories, boxes and predicates) is unambiguous and is pinned by tests.
   - *Consequence for the reading:* the R1 − R0 "does retraining reproduce the
     release" rule is unchanged, but a failure there may now also come from the
     authors' data pipeline differing from this one; the report says so if it
     happens.
2. **`--resume`.** *A23:* "add `--resume <ckpt>`". *Found:* `--resume` already
   existed as a bare `store_true` that nothing read. *New:* it takes an optional
   path; bare or absent means no resume, as before. Nothing depended on it.
3. **Flag names A23 did not give.** A23 fixed the behaviours; the names are
   `--epoch_ckpt_dir`, `--mirror_dir`, `--loader_seed`, `--expect_n_iter`,
   `--small_train_videos`, besides `--exclude_novel_train`, `--uniform_fusion`,
   `--no_epoch_eval` and `--resume`. All default to the released behaviour.
   `--small_train_videos N` takes N videos evenly spaced over the sorted training
   list, first and last included (5 → positions 0, ¼, ½, ¾, last).
4. **Re-seeding.** *A23:* global RNGs are re-seeded after the model is built.
   *New:* done when `--loader_seed` ≥ 0, which every RB1 run passes; with the
   flag off the released stream is untouched. No effect on RB1 runs.
5. **The asserted counts.** *A23:* five counts asserted. *New:* four are asserted
   exactly as listed (novel ids in detector targets; novel-predicate entries in
   `pre_label`; novel GT tids in label matching; relation instances reaching
   `gen_label`, which follows from the filter and is covered by the `pre_label`
   check and the tid check); "novel ids among the detector's text queries" is
   asserted through the targets' labels, because the queries are exactly the
   labels present in the frame (T2). A violation aborts the run. **Added, logged
   only:** `track_labels_novel_id_ignored_by_loss`. A track with no scored
   detection falls back to class id 1 (antelope, a novel category) in
   `format_trajectories`; the relation loss maps novel ids to −1 and ignores
   them (`convert_oid_on_base`). That is released behaviour, present in R1 and
   R2 alike, and no novel label is used; it is counted so the report can say so.
6. **No change to `models/relation_classifier.py`.** A23's T6 (the
   interactiveness target built from all 132 predicate columns) is handled at the
   source: with `--exclude_novel_train` no novel column is ever set in `pre_label`,
   so `int_labels` carries no novel-predicate signal. Verified by test on a real
   video (above). The loss code itself is the released one.
7. **The Step-4 evaluation scripts (dump of per-video predictions, the paired
   bootstrap) do not exist yet.** *A23* said A23a would pin them. *New:* they are
   written after training and before any test evaluation, tested on synthetic
   predictions only, and pinned by a further amendment (**A23b**) written before
   the test session. They read no test data until then.
8. **Git identity.** The commit was made with `-c user.name=kexauso613
   -c user.email=kexauso613@gmail.com` (the author of the base commit); no git
   configuration was changed.

**Not changed from A23:** the configurations R0, R0u, R1, R2, R3; what R3
contains (the iterative enhancement module only, not CRE); fusion; the filters
F1–F3 and the all-novel-frame choice; the training protocol (5 epochs, last
epoch, seed 3407); the evaluation settings; the bootstrap; the reading rules; the
pins of A23 (checkpoints, CLIP weights, metadata).

**Order.** Step 2 next: the rented machine (setup, hash verification, CUDA
operators, the test suite), then 5 training videos per configuration, then a
measured runtime and cost estimate, then **stop and ask**. No GPU run and no
spending before the estimates are approved.

#### A25 (2026-10-10): batch D2 — at which front-end stage are D1's U1 object tracks lost? A test-label error analysis (disclosed); no method evaluated

**This is a disclosed test-label error analysis, like D1 (A20). It is not a test
evaluation of any method.** It evaluates no method, and it fixes no method,
threshold or parameter of any future model. Its reading will inform which
object-side module is built next, and the thesis discloses that. Anything built
after it is designed and tuned on training videos only, under its own amendment.

**The chain since A23a.** The chain continued in the ov-pairtime repo
(`~/Documents/thesis/source/ov-pairtime/prereg/preregistration.md`, batch M1,
amendments A24–A24i). M1 is closed (A24i, 2026-10-09: G-M1 FAIL). That file was
verified on 2026-10-10. It is read-only (444), and its final SHA256 is
`981bf067da43c41df8759a260c9bde61014b411809e13bf9838f62f6906b5ce5`, the first line
of its `preregistration.sha256`. D2 does not modify it. **This file's SHA256 before
A25 is `bdfe25e62b20d3d82c743c848197d3448aff9d40fe6e98efdac3fd6134af268b`** (after
A23a), the same value the ov-pairtime header cites as its base. Numbering
continues after A24i, so this entry is A25.
- A24i records that M1's test session was chronologically the fifth test
  evaluation in the chain, and that RB1's (A23) has not run. D2 changes neither.
- D2 is the chain's second disclosed test-label analysis, after D1 (A20).
- **Relation to A22.** A23 cancelled O1's GPU step. D2 asks a different question
  (where, on test videos, D1's U1 tracks are lost). It does not revive O1 or its
  gates G-O1 and G-O2, so O1 stays cancelled. D2 reuses A22 §1a's code reading
  where it agrees, and corrects it in the places listed below.

**What exists before A25, and what was done to write it.**
- No model was built or run, and no GPU was used. The machine is the 4 GB GPU /
  7 GB RAM laptop.
- **Code read:** the EOV-MMP working repo (`~/Documents/thesis/source/ov-vidvrd-lab`)
  at `fdeeeb4d20145abde48d2f64f9e39f85dcd1fca6`. Session 3 ran from this commit to
  make the baseline SGDet tracks, and its tracked code has not changed since; only
  `pilot_analysis/` differs. METOR's copy (`5ff9fcb59ff1834d3baeac200ed5be8fa46eda23`)
  is the same on the test path, apart from the fusion call (bit-identical with
  RB1's flags off, A23a).
- **No test label was opened in D2.** `test_relation_gt.json`,
  `test_object_trajectories_gt.json`, the 200 `anno/test/*.json` and D1's
  `tc_missed.json` were hashed as bytes only. Counts quoted from D1 and A22 are
  reused, not recomputed.

**Inputs, verified on 2026-10-10.** A mismatch stops the batch.

| input | SHA256 | checked against |
|---|---|---|
| D1 `time_check/data/tc_missed.json` (bucket file) | `2e4c53b94597142df4500ab08ed3302e05f25f6b06a19c2c883f6f7a22373df2` | D1 report §6; `time_check/logs/tc_analyse.json` |
| D1 `time_check/data/tc_tables.json` | `c3cb77f9752e9315a2e3a33f365fed79cd04c76f7cfa2ee448f3ce715926bf80` | same |
| D1 `time_check/tc_run.py` (its `best_viou` and `boot_share` are reused) | `fae863dd65ee0c09a2cabfdab7a5f59df2e81832e8113c7c0383f7a5238d5996` | D1 report §6 |
| `session3_box_copy/SHA256SUMS_session3.txt` | `2cc2d0710f6a92d68ae27ecf74973a9dd122dc1594b9788cff4b0c560567954b` | the EOV prereg pin; D1's logs |
| baseline tracks and outputs: the 204 files of `cache_s3_sgdet/` (200 `test/*.pt`, `manifest.jsonl`, `sgdet_novel.json`, `sgdet_all.json`, `task6_sgdet.json`) | each equal to its line in `SHA256SUMS_session3.txt` | `sha256sum -c` |
| `data/vidvrd/data/test_relation_gt.json` | `b7a8219fa1b00e6d10c8bc5f37a1df7482b316dd401a1c870678817c4a921aeb` | `rb1/logs/data_meta_sha256.txt`; D1's log |
| `data/vidvrd/data/test_object_trajectories_gt.json` | `94df4451d06f2c3ee7270b2d452d574b466f9d5c79ebb0704f0af277dd15699a` | `rb1/logs/data_meta_sha256.txt` |
| `data/vidvrd/data/openvoc_obj_class_spilt_info.json` | `9ac86fc36f2c7679f76bc19039057b40263796ea9e8536b6294bfb157b186dec` | same |
| `data/vidvrd/data/VidVRD_ECC_test.json` | `6e7842bee9887c85164b673becc1e61040bfa31bc33b921202e4b2d9649faa2b` | same |
| `output/ckpt/…end2end_base-001.pth` | `d306952a51e9ca94c201ecf580f75a99221ae658dc187ae0ff38ff8b7dfc0e70` | `rb1/logs/checkpoint_sha256.txt` |
| `output/ckpt/AFLink_epoch20.pth` | `b35cbeddd3acc48fece820bd640640e6bfb1f5fbf570aa79af26c6a38958daa4` | same |
| the 200 `data/vidvrd/anno/test/*.json` (first pinned here) | listed in `d2/logs/anno_test_sha256.txt`, SHA256 `c98b1b84a323b0852575bd6488cb0fb8a15ecfe09fc1c11a571ee266ee4d2a85` | — |
| EOV pilot pre-registration, frozen at A22 | `442623b82cf5392c480e889ab160d5e0606300acc9d588d338e0a806036211f0` | A23 |

**The front end, as the code shows it.** The full map, with line references in
both repos, is `d2/stage_map.md`. In order:
0. Frames in [begin_fid, end_fid), the min/max of the video's **GT object
   trajectory extents** (`dataset.py:120-126`).
1. Detector: 300 queries, sigmoid class probabilities over 35 classes, no NMS.
2. **Cut:** keep a query iff its max class probability is > 0.35, else the frame's
   single best query; then the top 300 (query, class) entries
   (`post_process.py:194-195`, `:221-233`, `:264-282`).
3. **Dedupe:** a box is dropped if a later box in that order has IoU > 0.8
   (`gen_labels.py:145-162`). Boxes are clipped and the scores are fused with
   modelB; tracking does not use the scores.
4. **deep_sort:** every detection has confidence 1, so its confidence, height
   and NMS (1.0) filters remove nothing (`deep_sort_app.py:55`, `:171-178`).
   Settings: vanilla matching, 0.98 · cosine + 0.02 · Mahalanobis, max cosine
   0.35, budget 100, EMA 0.9, NSA (so an updated box is its detection's box),
   ECC, IoU stage 0.7, `max_age` 30, `n_init` 3. Output: confirmed tracks with
   time_since_update ≤ 1.
5. **AFLink:** `thrT` (0, 30), `thrS` 75, `thrP` 0.05 (`end2end_model.py:283-293`).
6. Each row takes the scores of the same-frame detection with the highest IoU
   > 0.5 (`:295-313`).
7. **Filter, in order:** 2 frames added before the first row (`gen_labels.py:363-371`);
   **coverage** ≥ 65% of the span (`:381`); **gap trim**, cutting at gaps > 15%
   of the span and keeping the larger side (`:385-413`); interpolation and
   rounding (`:416-428`); **length ≥ 10 frames** (`:626`, `len(track) ≥ 12`
   counting the `category` and `score` keys); **range** > 15% of
   (end_fid − begin_fid) (`end2end_model.py:319`).
8. **Naming:** argmax of the mean 35-vector over scored rows (`gen_labels.py:603-622`).

**Corrections to A22 §1a and to the batch brief.**
- The length rule is ≥ 10 frames, not ≥ 12.
- Two steps are not among the brief's five stages: the dedupe (3) and the gap
  trim (7). D2 gives each its own label.
- The range in the 15% rule is the GT-extent range of item 0, not the video's
  length.

**Saved intermediate outputs: none usable.** Session 3's SGDet cache keeps only
`model.traj` (the final, filtered, named tracks), pair logits and features
(`stage1c/sgdet_cache.py`). No raw detections, cut output, tracker output or
filter decisions were saved. So D2 needs a GPU recording pass (Step 2).

---

##### Recording (Step 2)

Code lives in `d2/` in this repo. **No pipeline file of either repo is edited.**
Every hook is read-only:
- **A forward hook on `postprocessorA["bbox"]`.** Per frame it saves every query
  with cut score ≥ 0.01 (box in the post-processor's pixel coordinates, cut score,
  argmax class, query index) and the queries the cut selected. In the run it also
  checks that the selected set and boxes equal the post-processor's own output.
- **Wrappers that copy before calling the original:**
  - `models.end2end_model.run`: the detections deep_sort receives, and deep_sort's
    output rows;
  - `AFLink.link`: its input and output rows;
  - `models.end2end_model.format_trajectories_test`: its input rows `temp`, with
    the assigned scores;
  - `models.gen_labels.interpolate_and_adjust_frames`: input and output frames
    per call.
- **The final `model.traj[video]`.**

**How it runs.**
- The run matches session 3's `sgdet_cache.py`: seed 3407, `--frame_stride 1`,
  the checkpoint and AFLink weights above, and the shipped `VidVRD_ECC_test.json`.
  The code root is the EOV repo at `fdeeeb4`. The full forward runs unchanged,
  modelC included.
- **Ground rule 3.** For every video, the final tracks must equal the baseline
  cache's tracks exactly: the same ids, classes, begin/end, every box and the
  score. Any difference stops the run and is reported. The check runs per video
  during the run, and again on the laptop.
- **2 videos first** (the first and last of the test list), then all 200. This
  happens **only after a time and cost estimate is approved**.
- **Test annotations on the machine.** The released test path reads test
  annotation files: `Dataset_new('val')` reads `test_relation_gt.json` and
  `test_object_trajectories_gt.json`, and `gen_feats_test` reads `anno/test/*.json`,
  as in session 3. They are on the machine for that reason only, and no D2 code
  reads them there. All test-label analysis runs on the laptop.
- **Setup and token rules** are those of the RB1 machine, following A24h. No
  Hugging Face token is present during the run. Token-shaped strings are scanned
  for (counts only) before and after. Everything is copied off and verified by
  SHA256.

---

##### Definitions (fixed)

**Sources**
- **U1 relations.** D1's rows in `tc_missed.json` with pass `novel`, task `sgdet`
  and bucket `U1` (262 expected; A22 §1d).
  - Each row is joined to `test_relation_gt.json` by (video, subject class,
    predicate, object class, duration), among the GT that D1's `keep` filter
    admits.
  - If a key matches several GT entries, the one chosen is the entry whose D1
    `best_pair_min_viou`, recomputed with D1's code on the baseline tracks,
    equals the row's (|Δ| < 1e-9).
  - Any row left unmatched or ambiguous stops the batch.
- **Final tracks:** the baseline cache tracks. By ground rule 3 these are also the
  recorded ones.
- **Pre-filter tracks:**
  - the rows of `temp` (after AFLink and score assignment, before the filter),
    grouped by track id;
  - boxes x1y1x2y2 as in `temp`;
  - missing frames inside filled by linear interpolation per coordinate
    (`np.interp`, as at `gen_labels.py:416-419`);
  - no rounding and no added frames;
  - extent [first row frame, last row frame + 1).
- **vIoU** is the repo's `viou` (`inference/video_relation_detection_openvoc.py:73`),
  called as D1 calls it. The track is cut to its overlap [lo, hi) with the GT
  extent [gb, ge), and the score is `viou(track[lo:hi], [lo, hi], gt_traj, [gb, ge])`;
  it is 0 if they don't overlap.
- **The cut score** of a query is its max over the 35 classes of the detector's
  sigmoid probability (`post_process.py:194`), the score the 0.35 cut uses.
- **Detection sets per frame.** All boxes are clipped exactly as at
  `end2end_model.py:224-232`.
  - **raw(s):** queries with cut score ≥ s, plus the queries the cut selected. So
    the cut's fallback query is never counted as "not detected".
  - **cut:** the queries the cut selected (> 0.35, or the fallback).
  - **tracker input:** the detections passed to deep_sort, after the dedupe.
- **Oracle link over a set.**
  - For each frame f in [gb, ge), take the detection with the highest IoU to the
    GT box. IoU is the repo's `iou` (`:48`, +1 convention), the class is ignored,
    and ties go to the lowest query index, or the first in order.
  - If that IoU is 0, or the set is empty, the frame has no box. It is encoded as
    [0, 0, −1, −1], which adds 0 to both volume and overlap in the repo's `viou`.
  - The oracle vIoU is the repo's `viou` of that frame-by-frame track against the
    GT over [gb, ge).

**Per side** (each U1 relation's subject and object, separately)
- **Missed side:** its best final track has vIoU < 0.5 (D1's `best_viou`).
- Each missed side gets **the first of these stages that applies, tested in this
  order:**
  - **O1a, not detected:** oracle over raw(0.05) < 0.5.
  - **O1b, below the cut:** oracle over cut < 0.5.
  - **O2, tracking:** no pre-filter track reaches vIoU ≥ 0.5. Its sub-labels, in
    order:
    - **O2-dedupe:** oracle over tracker input < 0.5.
    - **O2-frag, fragmented.** "Good frames" are the GT frames where the
      tracker-input oracle's box has IoU ≥ 0.5 with the GT box. Each good frame
      goes to the pre-filter track whose raw row box on that frame has the
      highest IoU with the oracle box, if that IoU is ≥ 0.5. The side is
      fragmented iff ≥ 2 distinct tracks each get ≥ 10% of the GT extent's
      frames.
    - **O2-other:** the rest. Reported descriptively within it: "untracked" (< 50%
      of the good frames assigned to any track) vs "tracked but off".
  - **O3, filter:** some pre-filter track reaches ≥ 0.5. The one with the highest
    pre-filter vIoU (ties: lowest id) is replayed through the repo's own
    functions, and its first failing rule is named:
    - **F-cov:** coverage < 65% (`:381`);
    - **F-len:** < 10 frames after the gap trim (`:626`); a trim before it is
      flagged;
    - **F-range:** ≤ 15% of the range (`end2end_model.py:319`);
    - **F-reshape:** it survives all three, but its final track's vIoU is < 0.5.
      Split into **gap trim** (the trim at `:385-413` shortened it) and **other
      reshaping** (the added frames, interpolation, rounding).
  - **O4, kept:** the replayed track survives and its final vIoU is ≥ 0.5. If
    ground rule 3 holds, this cannot happen to a missed side; the report explains
    any case.
- **Replay check.** For every video, the repo's `format_trajectories_test` on the
  recorded `temp`, followed by the 15% rule, must reproduce the recorded final
  tracks exactly. Otherwise the batch stops.

**Per relation**
- **Relation label:** the most upstream stage among its missed sides, in the order
  O1a < O1b < O2 < O3 < O4. Within O2: dedupe < frag < other. Within O3: the rule
  order above.
- **S0, no missed side.** Both sides have a final track ≥ 0.5, yet D1's best pair
  (s ≠ o) failed. Such relations are reported, kept in the denominator, and in
  none of the reading's groups.
- **Class split.** A side is base or novel by its GT category
  (`openvoc_obj_class_spilt_info.json`). A relation is "base-only" if both sides
  are base, and "involves novel" otherwise.

**Funnel** (descriptive, Table 5)
- **Tracks:** every GT object track in the 200 test annotation files
  (`anno/test/*.json`), one per tid, over the frames where it has a box. Its
  extent runs from the first to the last of those frames + 1, and a missing frame
  inside is empty for the GT too.
- **Stages,** cumulative, each at vIoU ≥ 0.5 over the whole track:
  - F1: oracle over raw(0.05);
  - F2: oracle over cut;
  - F3: oracle over tracker input;
  - F4: some pre-filter track;
  - F5: some final track.
- **Reported:** the share of all GT tracks at each stage, base vs novel.

**CIs:** 95% percentile intervals from 1,000 resamples of whole videos, seed 0, as
in D1's `boot_share`.

---

##### Tables (Step 3)

All over the U1 relations, SGDet, novel split:
1. Count, share and CI per stage, by relation and by side.
2. The same, split base / novel.
3. O3 per rule (by side).
4. O1a at floors 0.01, 0.05 and 0.1, by relation and by side. Only raw(s)'s floor
   changes.
5. The funnel.

---

##### Reading (Step 4)

By relation, pooled over the U1 relations.
- **Groups:**
  - **link** = O2 + O3;
  - **rescore** = O1b;
  - **detector** = O1a.

  S0 and O4 stay in the denominator but in no group.
- **The largest group by count names the object-side module:**
  - link → relation-aware track linking and keeping;
  - rescore → relation-aware rescoring of detections before the cut;
  - detector → a detector limit that relations can't recover, so the
    clean-detector step moves up.
- **The report states** the largest and second-largest groups, with shares and
  CIs, and the paired bootstrap 95% CI of the difference in share (largest −
  second).
  - If that CI includes 0, the reading line adds "not separated from <second>".
  - A tie in count names both groups.
- The reading names a direction for the next design. It does not fix that
  module's design, thresholds or parameters.

---

##### Stop rules and order

**Stop rules.** D2 stops and reports, fixing nothing, at the first of:
- an input hash mismatch;
- a ground-rule-3 difference on any video;
- a replay mismatch;
- a failed post-processor check during the run;
- a U1 join failure;
- recorded data missing, or failing its hash after the copy.

**Order.**
1. **Step 1, laptop, $0.** Write in `d2/`: the recorders, the stage classifier and
   the Step-3 analysis code. Test them on stubs and synthetic cases:
   - recorders on vs off give identical outputs;
   - the classifier's labels are right;
   - the length rule holds on the repo function.

   **A25a pins the code and the test logs before any GPU run.**
2. **Step 2, GPU.** A time and cost estimate, then **stop and ask**. On approval,
   on the RB1 machine and before RB1's own runs:
   - 2 videos, with the ground-rule-3 check;
   - then all 200;
   - copy everything off and verify it.
3. **Steps 3–4, laptop.**
   - Verify every input and recording.
   - **Only then open the test labels** (`test_relation_gt.json`,
     `anno/test/*.json`, `tc_missed.json`), in `d2/analyse.py`.
   - The tables, then the reading.
4. **`d2/d2_report.md`**, then a **hard stop**. Nothing is built in D2.

#### A25a (2026-10-10): pin addendum to A25 — Step-1 code, tests, and what differs from A25

Written **after the Step-1 code and tests exist, and before any GPU run.** Since A25:
- no model has run on a GPU;
- no test label has been opened. The test-side files were hashed as bytes, and the
  baseline's detected tracks were read from session 3's verified cache. Those are
  detector outputs, not labels.

**Nothing in A25's definitions, thresholds or reading rule is changed.** The
differences below are implementation details. Each states the old choice, the new
one and the reason.

**Pins.** A25 was committed before any D2 code existed, as METOR commit
`3a54d6e944bc1bf772cca479556dcccd59b30092`. The code below is committed together
with this amendment.

| artefact | SHA256 |
|---|---|
| `d2/logs/code_pins.json`: the 8 code files, the 4 test files, the 22 EOV front-end files at `fdeeeb4`, the baseline tracks and 4 run inputs. `record_run.py` and `analyse.py` refuse to run if any of these differ. | `2e9cfd9291cf21096ece0eacaa13c27765e356ffbd2303603afd1c24c94d9d54` |
| `d2/recorders.py` | `89dc36508b74a2fe4542ebfd4bf1ee8a7337332310bb3851a9227c6edcc9cccc` |
| `d2/record_run.py` | `61cc5d0a22c0110187e779825471014c53fe1dbe9abc58cacb3f21a91a86b4c5` |
| `d2/stages.py` | `feab43dd9f4faf68ee886b6594dc792b9b8ea27327fa000aa3f6a64251d36ecf` |
| `d2/analyse.py` | `fe6a7a676175bf20f992ec8a1e79a6fa4ec38969cdfb822098dd8badbef825db` |
| `d2/baseline_tracks.py` | `22ce431e566dd5405f2876a982eef32742bdccd583bf3d16011f27c3d1244362` |
| `d2/common.py` | `0a6ea4dacc4a71e791cf2adb3e1dc60d18542b7fada801369f36f153286a429e` |
| `d2/pin.py` | `b519054b0d82b586de26bc80f7a814c64a87400d63a4413b19f3af953dad3c89` |
| `d2/data/baseline_tracks.json` (200 videos, 658 tracks, from the 200 verified `cache_s3_sgdet/test/*.pt`) | `3d9c5c86b9a80681cb1aa6643bab94f1b6bdea4448ecdb6151797ca63793df56` |
| `d2/stage_map.md` (Step 0) | `730a057121aa9dbb083666f5621e9ba0cf412a2cb6d69f21713324757949ce34` |
| test log, d2 on the EOV code root | `d2/logs/pytest_d2_eov_root.txt`, `ad419e3c517abf4b7f971a049794a0e19b1991273e7ecbdda18c0027eb6b1202` |
| test log, d2 on METOR's vendored front end | `d2/logs/pytest_d2_metor_root.txt`, `6e6f6f25f896ed43e2874d90fdddd858428b492d5a58d4040d5a6417bdc53791` |
| test log, METOR's own suite with `d2/` present | `d2/logs/pytest_metor_tests_with_d2.txt`, `a192f0a0ad1370a5f3de02b241efb423fb9d1eac6f004f8b1aeab6526f8d0c5f` |
| `record_run.py --check_only` on the laptop | `d2/logs/record_run_check_only_laptop.txt`, `41067811e6fa7ef0ac5ecdbfda0710a05f6bd14ca1381e390d06747ed170e996` |

**Test results.**
- d2: **23 passed** on the EOV code root and **23 passed** on METOR's vendored
  copy.
- METOR's own suite: **169 passed**, the same as at A23a.
- `--check_only` verifies every pin and exits 0. The laptop goes no further,
  because building the dataset loads the real CLIP model (ground rule 5).

The tests run the repo's own test branch of `End2End_Model.forward` on the CPU,
with the real `OVPostProcess`, dedupe, deep_sort, AFLink (real weights) and filter.
The detector and classifier are stubs, and the relation head is not reached. They
show:
- **Ground rule 3 on stubs.** With the recorders on and off (off, on, off), the
  tracks are identical. Afterwards, the originals are restored and no hook is
  left.
  - A mutant that shifts every tracker input box by 1 px changes the tracks, so
    the comparison is not vacuous.
  - Shifting a single frame-0 box changes nothing: that frame is in deep_sort's
    tentative phase, and with NSA at confidence 1 every later update snaps to its
    own detection.
  - The recorded data is consistent with the pipeline. The cut's fallback is
    recorded, the duplicate removed by the dedupe is visible, and the repo's
    filter replayed on the recorded rows reproduces the tracks.
- **The length rule is ≥ 10 frames,** run on the repo's own
  `format_trajectories_test`: 8 rows plus the 2 added frames survive, 7 plus 2
  do not. The baseline itself contains 10-frame tracks (its shortest final track
  has 10 frames).
- **The stage classifier.** In a synthetic video, every relation lands in the
  stage it was built for, at all three floors: O1a, O1b, O2-dedupe, O2-frag, F-cov,
  F-range, F-len, F-reshape-trim and S0. The funnel, the join, ties in the reading
  and the separation statement were also checked.
- **The order is enforced.** `analyse.py` stops at verification (no recordings)
  without calling `open_labels`. On a code root other than the pinned EOV front
  end, it stops earlier, at the code pins.
- **Faithful reuse.** D1's `boot_share`, taken from `tc_run.py`'s source by AST,
  equals D2's. The vectorised IoU equals the repo's `iou` bit for bit. An empty
  frame adds 0 to the repo's `viou`.

**Differences from A25, each with old / new / reason.**
1. **The U1 join when rows are indistinguishable.**
   - *A25:* a key with several GT entries is resolved by D1's recomputed
     `best_pair_min_viou`; a row left unmatched or ambiguous stops the batch.
   - *New:* within a key, rows and GT entries are matched by value
     (|Δ| < 1e-9).
     - Rows equal in key and value carry nothing else D2 uses. They take the
       equally many matching GT entries in file order, so the set of GT entries
       analysed is still determined.
     - Unequal counts stop the batch.
     - **Every** row's recomputed value must equal D1's, not only on ambiguous
       keys.
   - *Reason:* two GT relations with the same classes, predicate and span, whose
     objects are both untracked, give identical D1 rows. Stopping there would
     protect nothing.
2. **Added checks, each a stop rule:**
   - the code root's 22 front-end files and d2's code must equal
     `code_pins.json`, both on the machine and in the analysis;
   - the analysis re-hashes the 204 cache files and the 200 annotation files;
   - `temp` may not repeat a (frame, id) row;
   - the recorder aborts if the post-processor's `pruning_threshold` is not 0.35.
3. **A timing rule for Step 2.** The 2-video check runs the first and last
   videos of the test list. By session 3's log these are
   `ILSVRC2015_val_00046001` and `ILSVRC2015_train_00047020`, at 41.6 s and 13.0 s.
   If the check takes more than 25% longer than session 3 on the same videos, the
   run stops for a new estimate before the 200.
4. **`modelC.tgt_split = "all"`,** as session 3's `sgdet_cache.py` set it. modelC
   runs after the tracks are final, so the tracks are unaffected.
5. **CPU tests copy on `.cuda()` and `.cpu()`.** On a CPU,
   `deep_sort_app.gather_sequence_info` would edit the detections' boxes in place
   through a shared numpy view. On a GPU, `.cpu()` copies. This affects only the
   tests.
6. **Details A25 left implicit, now fixed in code:**
   - O2-other's "untracked" includes the case of no good frame.
   - The replay's "trimmed" flag means the interpolated frame range differs from
     [first row − 2, last row].
   - Table 3 also counts F-len after a trim.
   - The report's checks are descriptive: missed sides in O4; S0 relations; sides
     labelled O1a/O1b although some pre-filter track reaches 0.5; replayed-kept
     tracks absent from the final tracks.
   - A tie for the largest group names every tied group.
7. **Recording format.**
   - Per video: `test/<video>.npz` (arrays) and `test/<video>.json` (meta, the
     filter's output, the final tracks).
   - A manifest holds each file's SHA256.
   - The run is resumable.
8. **Code root.** METOR's vendored copy passes the same 23 tests. The GPU run uses
   the EOV repo at `fdeeeb4`, which made the baseline, with its 22 front-end files
   equal to the pins.

**Not changed from A25:** the inputs and their pins; the recording plan and ground
rule 3; every definition and threshold (0.5, 0.35, floors 0.01 / 0.05 / 0.1, 10%,
50%); the stage order; the tables; the bootstrap; the reading rule; the stop rules;
the order of steps.

**Order.** Step 2 next: a time and cost estimate, then **stop and ask**. No GPU run
and no spending happen before the estimate is approved.

#### A25b (2026-10-10): the study folders move under `studies/`; path fixes and a d2 re-pin; nothing else changes

Written after the move and before any GPU run. No test label has been opened, and no
model has run.

**What moved.** At the author's request, `prereg/`, `rb1/` and `d2/` moved from the
METOR repo root into `studies/`, as a git rename.
- All 51 files, git-ignored ones included, were hashed before and after the move.
  Every one is **byte-identical**.
- This file therefore keeps its hash chain. It now lives at
  `studies/prereg/preregistration.md`.
- An index is at `studies/README.md`.

**Path map.** It resolves every path cited before A25b.

| cited as | now |
|---|---|
| `prereg/…` (A23 onward, `docs/23_Configuration-and-flags.md`, `cli/train.py`) | `studies/prereg/…` |
| `/home/totoro/Documents/thesis/source/METOR/prereg/preregistration.md` (the closed ov-pairtime header) | `/home/totoro/Documents/thesis/source/METOR/studies/prereg/preregistration.md` |
| `rb1/…` (A23, A23a, RB1's reports and logs) | `studies/rb1/…` |
| `d2/…` (A25, A25a, D2's files and logs) | `studies/d2/…` |

Some of these citations sit in files this move did not edit:
- the ov-pairtime pre-registration, which is closed and is **not modified**;
- the hash-pinned `docs/23_Configuration-and-flags.md`;
- `cli/train.py`, a comment only;
- the historical logs.

Their citations are resolved by this map.

**Files edited because of the move, old → new SHA256.** Only paths changed: no
logic, definition or threshold.

| file | change | old | new |
|---|---|---|---|
| `d2/common.py` | the repo root is two levels up; the pinned annotation list's path | `0a6ea4da…a429e` | `eb9a8c7ab81382c2baacc377357157251c1d04bee8618fcca4d8968fa6e7cd38` |
| `d2/record_run.py` | a comment | `61cc5d0a…6b4c5` | `a13200b13a9b823aba58010bed5ab7cb78734e24e5a81e225aeb569cf09606eb` |
| `d2/tests/conftest.py` | the folder holding `d2`; a message | `6fea3029…e556d` | `91dbb0e1481335cd59c60e93f61384ca2601d664a93b871fc361ffe683dae38c` |
| `d2/README.md` | commands run from `studies/` | `0a2a7131…bbf14` | `6a998fbcf93ef12fe63230f92651f255f54a73180ef534f977ba52f4bc4cb0a8` |
| `rb1/scripts/fallback_probe.py` | the repo root is three levels up; usage line | `3e8edce5…fc60a` | `feb1dd82a678ca178061d03e90bb4ea1077a20620f14bedbeaa67a6aa7e415dd` |
| `rb1/scripts/fallback_count_gpu.py` | log path and usage line | `a049f52a…b8866` | `94983c6231298957a5c991fcf54633a3d57bb7cb348b9311fbdbf34e9ba6b7c5` |
| `rb1/README.md` | `--epoch_ckpt_dir` and `--resume` under `studies/rb1/runs/` | `cc0f5174…3424c` | `e2dea8be7042b9053c9b583158dc78a8db649b8ff940afad028e6f8d234a2627` |

The RB1 scripts were pinned by no hash, only by commit `5ff9fcb`. None of RB1's
14 pinned code files (A23a) is touched.

**d2 re-pin.**
- `d2/logs/code_pins.json` is now
  `9e47a91d7710405125da31764770df3f34b9637377c1764151f0a456575b847e`.
- A25a's version is kept unchanged as `d2/logs/code_pins.A25a.json`
  (`2e9cfd9291cf21096ece0eacaa13c27765e356ffbd2303603afd1c24c94d9d54`). A25a's
  test logs are also kept unchanged.
- **Unchanged since A25a:** the EOV front-end pins, the baseline tracks
  (`3d9c5c86…df56`), the inputs, and every other d2 file.

**Tests in the new layout.** The new logs sit next to A25a's.

| run | result | log, SHA256 |
|---|---|---|
| d2 on the EOV code root | **23 passed** | `d2/logs/pytest_d2_eov_root_A25b.txt`, `0c7cc4f6f7bb47805f386c8dad2eb677b7230c66e34500cedf16657d72465976` |
| d2 on METOR's vendored front end | **23 passed** | `d2/logs/pytest_d2_metor_root_A25b.txt`, `3fbdb7a85845ebd77868d80f29e6b42ada80e91c707c5d4cdd7d4fd48421f205` |
| METOR's own suite | **169 passed** | `d2/logs/pytest_metor_tests_A25b.txt`, `6999cd9067abe74ba5ceae2ba082d4d3e90662990d9ac0b629b5a321bdb25b42` |
| `record_run.py --check_only` | every pin verified | `d2/logs/record_run_check_only_laptop_A25b.txt`, `1be79f91feb5fd3c06c41f15092cfcbb74f6dd1887113886e63f6aa026abfb5d` |

`fallback_probe.py` now resolves the repo root to METOR, and imports METOR's
`models`.

**Not changed:** A25's definitions, thresholds, reading rule, stop rules and order;
the inputs and their pins; RB1's design (A23, A23a). The `d2` commands now run from
`studies/`.

**Order.** As in A25a: Step 2 next, with a time and cost estimate, then **stop and
ask**.

#### A25c (2026-10-10): Step 2 approved on its own machine; a descriptive O1b score table; the test annotations on the machine; the frame-range finding; re-pin

Written **before any GPU run.** No test label has been opened, and no model has run.

**Approval.** The plan's author reviewed Steps 0–1 (`studies/d2/step1_report.md`)
and approved Step 2 as estimated, with the changes below. **A25's definitions,
stages, thresholds, reading rule and stop rules are unchanged.**

**1. A descriptive addition: Table 6, the cut scores of the O1b oracle's boxes.**
- **Which boxes.** For every side labelled O1b at the primary floor 0.05: in each
  GT frame, the box the raw(0.05) oracle chose (A25's oracle, unchanged), and its
  cut score.
- **Per side:**
  - the median cut score over the frames that have an oracle box;
  - the share of the GT extent's frames whose oracle box scores ≥ 0.1, ≥ 0.2 and
    ≥ 0.3 (a frame with no oracle box counts as below);
  - the share whose box the cut selected.
- **Reported:** across O1b sides (median and IQR), and pooled over their GT
  frames; for all sides, base and novel.
- **Descriptive only.** No stage, threshold or reading depends on it.
- **The author's reason:** to show whether "below the cut" means just below,
  where relations could lift the box, or far below, close to not detected.
- **Code.** Only the analysis changes; the recordings already hold every score.
  - `stages.oracle_choice` now returns the index the oracle picks. `oracle_track`
    is rebuilt on it and makes the same choices; the existing tests pass
    unchanged.
  - Added: `stages.oracle_scores`, `DetSets.raw_scores`,
    `analyse.o1b_score_table`, and `common.SCORE_LEVELS = (0.1, 0.2, 0.3)`.

**2. Step 2 runs on its own machine, not the RB1 machine.** RB1 still needs its
PredCls amendment and its training frames.
- **Machine:** one RTX 4090 with ≥ 32 GB RAM and ≥ 60 GB disk, on the Vast.ai
  PyTorch template, rented by the author. `OMP_NUM_THREADS=8`.
- **Sequence:**
  1. `--check_only`;
  2. the 2 videos (ground rule 3, and A25a's 25% timing rule);
  3. all 200;
  4. the cleanup (item 3);
  5. copy off and verify.

  **Stop and ask if the run is projected to exceed 6 h.**
- **Inputs:**
  - **code:** `git archive fdeeeb4` from the laptop;
  - **from the laptop:** the 11 small metadata files and the 200 test annotations
    (`box/upload_files.txt`);
  - **from the private Hugging Face weights repo:** the end-to-end checkpoint,
    AFLink, the CLIP feature bank and the ECC files;
  - **from the private frames bundle:** the 200 test videos' frames. This is
    session 3's bundle, with each shard SHA256-checked;
  - **from OpenAI's public URL:** CLIP, whose own download checks the SHA256 in
    its URL.

  Every file is checked against the laptop's SHA256: `box/expected_upload.sha256`
  (519 files) and `box/expected_fetch.sha256` (7 files, including the bundle's
  `SHARDS.json` and `MANIFEST.json`). The 14 metadata files equal RB1's recorded
  hashes, and the checkpoints equal A25's pins.
- **`studies/` is copied with rsync, not pushed.** It holds the pre-registration
  and the leak findings.
- **The token** follows the author's rules: a new fine-grained, read-only token,
  with read access to the frames dataset and the weights repo.
  - The author runs `box/fetch_with_token.sh` interactively. It reads the token,
    hidden, into the `HF_TOKEN` of that process only.
  - The token is never written to any file, `HF_HOME` included.
  - Token-shaped strings are scanned for, counts only, before and after.
  - The token is unset, and any token file removed, before the script ends (also
    on failure). This happens after the frames are fetched and before the
    recording run starts.
  - The author is told when the token is gone, and deletes it on Hugging Face.
- **After the verified copy-off,** the author destroys the machine.

**3. Test annotations on the machine: accepted and disclosed.**
- **Why they must be there.** The released SGDet test path reads them:
  - `Dataset_new('val')` reads `test_relation_gt.json` and
    `test_object_trajectories_gt.json`; the latter sets the frame range (item 4);
  - `gen_feats_test` reads `anno/test/*.json`.

  Ground rule 3 requires session 3's inputs.
- **Only the pipeline reads them** on the machine: no D2 code, and no manual
  inspection.
- **They are deleted right after the run.** `box/cleanup.sh` logs each file's
  SHA256, deletes it, and checks it is gone. The log is copied off.
- **This differs from M1's A24h rule** ("test labels never go to the rented
  machine"). M1's models read no test annotation; here the released test path
  itself needs them, and ground rule 3 needs the same inputs as session 3.

**4. A further finding for the report: the released test path takes its frame
range from the GT.**
- **Quoted:** `data_loading/dataset.py:120-126` (begin_fid and end_fid are the
  min and max of the video's GT object trajectories) and
  `models/end2end_model.py:314-323` (the 15% rule over `end_fid − begin_fid`),
  EOV `fdeeeb4`.
- **Measured from metadata only:** the recordings' own `video_len`, `begin_fid`
  and `end_fid`, the values the pipeline computed. No label file is read for it.
- **Table 7, descriptive:**
  - per video, the share of its length outside [begin_fid, end_fid): median, p75,
    and the maximum with its video;
  - the number of videos with any frame outside;
  - the total frames outside, over all frames.

**Pins.**

| artefact | SHA256 |
|---|---|
| `d2/logs/code_pins.json` (now also pins the 12 files of `d2/box/`) | `95816845ec3a367ed890796f5018421253a6edf0e78b63a891852c2420b09028` |
| A25b's pins, kept as `d2/logs/code_pins.A25b.json` | `9e47a91d7710405125da31764770df3f34b9637377c1764151f0a456575b847e` |
| `d2/stages.py` | `593f9809a0202b736afee27d304f7f9878071063135028217d477f8683f18255` |
| `d2/analyse.py` | `d466262f8a73f8c7f2b989f9ecc658fc300353331643eccf80b812244a38b130` |
| `d2/common.py` | `77240af3b292f9885d7ac4493e004c7555a879a57f8f351a9671a4d0ab382b5f` |
| `d2/pin.py` | `70d627db838b03c2f06192b4e6f59261d203f89e53b8ed2fc70889605489ef94` |
| `d2/tests/test_d2_analyse.py` | `59e7607fc7be7cbaf61a86948a40f2775a2ceaf1b19a6e712ad34578d203d46a` |
| `d2/box/fetch_with_token.sh` | `7a42626b1e9d990a72cd982d994d316d998bafa046528d640223c56f45d332bc` |
| `d2/box/cleanup.sh` | `de8942883ee5c25bb21a43b6b46a5ed1219f6b8e7d127e9b31af8f0fa794c620` |
| `d2/box/expected_upload.sha256` / `expected_fetch.sha256` | `63e273d75fb3eaf06a2ad8b8dbecf41c2fbd7d3d2de9082abd60380aee3f4716` / `8d589f6f7322f5a95ac0c896fb50fc31edcaabc35a64179502fa0f83c42c903d` |

**Unchanged since A25b:** the recording code (`recorders.py`, `record_run.py`,
`baseline_tracks.py`), the 22 EOV front-end pins, the baseline tracks and the
inputs.

**Tests.**

| run | result | log, SHA256 |
|---|---|---|
| d2 on the EOV code root | **25 passed** (2 new, plus Table 6 checks in the synthetic video) | `d2/logs/pytest_d2_eov_root_A25c.txt`, `f663d201369274682b41ba61cbf30fd4233cf519df7880f659e14dc2300c48fb` |
| d2 on METOR's vendored front end | **25 passed** | `d2/logs/pytest_d2_metor_root_A25c.txt`, `db589613124842e69cbf9bf4fa44caaf055c3fe14672b779b0af9a6f675399bd` |
| METOR's own suite | **169 passed** | `d2/logs/pytest_metor_tests_A25c.txt`, `4af146d09f27b71fa63135bab57fd2f632f88096d08ec66c15b7dfd0961f7cc0` |
| `record_run.py --check_only` | every pin verified | `d2/logs/record_run_check_only_laptop_A25c.txt`, `792a5e37f886b6a79bbbb468b35a986b6f36ee250c9114d457743950f1686f8a` |

**Order.**
1. The machine setup;
2. Step 2;
3. the cleanup;
4. copy off and verify;
5. Steps 3–4 on the laptop. The test labels are opened only then.
6. `studies/d2/d2_report.md`, then a **hard stop**.

#### A25d (2026-10-10): D2 closed

The closing entry, written after the plan's author reviewed and accepted the D2 report.

**D2 is complete, and the reading is as reported.** The report is
`studies/d2/d2_report.md` (SHA256
`ba235a5cf35515a13f2292d86826ac2de24570ef2e3c1678bf8506ed9d8685a4`, commit `73fc5fd`).
- **The reading (A25's rule, by relation, pooled, over the 262 U1 relations):**
  - the largest group is **rescore (O1b), 96 = 36.6%** (95% CI 23.4–50.6%), which
    names relation-aware rescoring of detections before the cut;
  - the second is **detector (O1a), 93 = 35.5%** (18.9–52.3%), **not separated**
    from it (paired 95% CI of the difference: −26.4 to +29.9 points);
  - link (O2 + O3) is 67 = 25.6% (15.3–39.4%).
- **As reported, descriptively:** the order of the top two is not robust. It moves
  with the score floor (Table 4) and with a post-hoc relabelling (report §6).
  O1a + O1b is 72.1% at every floor.
- **Step 2:**
  - all 200 test videos were recorded, each identical to the baseline (ground
    rule 3);
  - the copy-off was verified;
  - the 203 test annotation files were deleted from the machine after the run,
    with their SHA256 logged; all equal the laptop's pinned copies;
  - every token scan found 0.

  The author is destroying the machine after the verified copy-off.
- **Disclosed in the report:**
  - the test annotations on the rented machine (A25c);
  - an abandoned two-process attempt (about 18 min; nothing written);
  - `cleanup.sh`'s early exit after the deletion, with the remaining checks run by
    hand and logged;
  - the finding that the released SGDet test path takes its frame range and its 15%
    rule from the GT object trajectories: 11.6% of test frames are never detected.

**Binding from here on.**
- **The test recordings are never used for designing or tuning anything.** This
  covers `studies/d2/data/rec/` (manifest
  `37d26c1c59bb1f3befafde0c4dab5d532a401987933c2748be80245b2adc28c5`, full listing
  `studies/d2/logs/rec_sha256.txt`
  `1328b7c9076cf2a20952293d5aa94a04be3e464574d16cff1ccf8861d5def662`) and every
  output derived from them: per-relation stages, oracle scores, the funnel. They are
  never used for any method, threshold, parameter, rule or selection. They may be
  cited only as this disclosed analysis.

**Record.**

| artefact | SHA256 |
|---|---|
| `studies/d2/data/d2_tables.json` | `c957eeab76fe45c96b81ba86b82e5a3472e2fd8338a0da10d96d7780fcdb770e` |
| `studies/d2/data/d2_relations.json` | `90897c6f5077265169d4a252593b68a09722a7c215d63a5d289e61d080a31a72` |
| `studies/d2/data/d2_funnel.json` | `5b170d82c3652377c514e777d27ef5182a223f75176efb26d889734043e95778` |
| `studies/d2/logs/d2_analyse.json` | `246c0b39cc0bab0f8222795dcaa02c1500965ac3ebffff0ea69772ade21d67dc` |

The analysis outputs are committed in `4f6d059`. The recordings are committed
separately, right after this amendment.

D2 is closed. The next work, a clean detector, is under its own amendment.

#### A25e (2026-10-11): a descriptive headroom count for the design review (Q7); no decision rule, no tuning

Written **before the count is computed.** It is requested by the plan's author's
design review (`studies/design_review/`, Q7). **This is descriptive only:** it sets
no decision rule, tunes nothing, and evaluates no method. Like D1 and D2, it will
inform the choice of the next direction, and the thesis discloses that.

**The question.** Of D1's 262 SGDet U1 relations (novel pass: no detected track
pair matches the GT pair), how many does the released baseline match in **PredCls**,
where it has GT tracks and GT classes? That is an upper bound on what fixing tracks
alone could recover with the released relation head.

**The single input** is D1's `tc_missed.json` (SHA256
`2e4c53b94597142df4500ab08ed3302e05f25f6b06a19c2c883f6f7a22373df2`, checked first).
It holds D1's SGDet rows (including the 262 U1) and D1's 361 PredCls-missed rows,
for the same 605 novel-split GT relations.
- **No label file is opened.**
- **No D2 output is used.** A25d bars D2's recordings and everything derived from
  them from design use.

**Definitions.**
- A U1 relation is **matched in PredCls** if D1 did not list it among the PredCls
  misses.
- Relations are identified by (video, subject class, predicate, object class,
  duration), because D1's rows carry no track ids.
- **Exact** for a key that has one known entry (a single SGDet row) and at most one
  PredCls-missed row.
- **Bounded** for keys with several entries. The lower bound assumes the PredCls
  misses fall on the U1 rows first; the upper bound assumes they fall first on the
  key's other known rows. Both, and the exact count, assume that none of the 121
  SGDet hits (not listed by D1) shares a key with a U1 relation.
- **Reported, descriptively:** matched and missed counts; the exact keys split into
  base-only / involves-novel and long / short; D1's PredCls bucket (U3, A1, N1, C1)
  for the U1 relations PredCls also misses; and SGDet novel recall now (121 / 605)
  against its value if every PredCls-matched U1 relation were recovered.

**Code.** `studies/design_review/q7_headroom.py`, SHA256
`e099e2eb180c7d90b07fa64bf146561390534c9b4b4be294fa0cfca3ed01479d`, committed
together with this amendment. Output: `studies/design_review/q7_headroom.json`.

#### A26 (2026-10-11): the protocol audit — R0u, R0 over the whole video, and both; a test evaluation of the released checkpoint only (disclosed); no method evaluated

Written **before any number of it exists.** Nothing has been run, and no GPU has been used.
The GPU run happens only after the thesis author approves the time and cost estimate below.
This is step 0 of the agreed plan (`studies/design_review/agreed_plan.md`, SHA256
`a4d4dc00f64c846c490fa6c930c3907f9011ae374de29fbc8f4d6dc82cd7a3e8`), which merges design
reviews 1–3 and the plan author's decisions.

**The question.** The released SGDet test path makes two choices at test time:
- **The frame range.** It takes each video's range, and the base of its 15% track-length rule,
  from the **GT object trajectories** (`data_loading/feature_dataset.py:123-138`;
  `models/end2end_model.py:205-210`, `:322-327`). D2 found that 11.6% of test frames are never
  detected (A25d).
- **The fusion weights.** Per-class weights favour the classifier on exactly six novel classes
  (RB1 leak list L7).

How much do these two choices move the published SGDet numbers? The answer sets how the
thesis compares itself with the published bar (agreed plan §5).

##### The four configurations

All four use the released end-to-end checkpoint, run as A23 runs R0 and R0u. Everything else is
the same in all four:
- METOR's code, unmodified, at commit `486a962`. Its test path is unchanged since `7e87c9b`
  (A23a).
- `--arch metor-compat --n_iter 0`; the run asserts that `featEmbedding.iem is None`.
- Seed 3407, `--frame_stride 1`, `AFLink_epoch20.pth`, and the shipped `VidVRD_ECC_test.json`.
  That file has entries for frames 2 to the last frame of 199 test videos, the whole video
  and not only the GT range, and no entry for one (`ILSVRC2015_train_00250013`). This is checked
  against session 3's frame counts, and is the same in all four configurations.

| config | frame range | fusion |
|---|---|---|
| **R0** | as released: [`begin_fid`, `end_fid`) from the GT trajectories | as released (per-class) |
| **R0u** | as released | **uniform** 0.3 · classifier + 0.7 · detector for all 35 classes (`--uniform_fusion`, `models/fusion.py`) |
| **R0w** | **the whole video**: `begin_fid = 0`, `end_fid = video_len` (the number of frame files) | as released |
| **R0uw** | the whole video | uniform |

- **The whole video changes two things:** the frames the detector sees, and the base of the 15%
  rule (`min_l = end_fid − begin_fid`). Nothing else in the test path reads the range.
- **The wrapper** is `studies/a26/wrappers.py::whole_video`. It works on the loader's batch,
  outside the pipeline. It refuses a GT range that is not inside the video.
- **R0 is re-run.** It is the identity check, and the pairing partner. It has been evaluated
  before (session 3, the pilot) and is not new information.

##### How it runs (`studies/a26/run.py`)

**Per video:**
- `Dataset_new('val')` encodes the frames once.
- `End2End_Model` then runs four times, in the order R0, R0u, R0w, R0uw. `args.uniform_fusion`
  is set per configuration and restored afterwards.
- One forward gives both predicate splits:
  - modelC is called in `all`, then in `novel`, on the same inputs (`wrappers.BothSplits`; the
    pilot's fused path, validated exact in PILOT-STATUS B.21);
  - the pilot's novel-branch predictions already matched session 3's to the same triplets,
    durations and boxes, with scores within 1.1e-7 (`pilot_analysis/time_check/time_check_report.md:32`).
- Predictions go through the repo's own `process_pred` → `association` → `format_`, exactly as
  `cli/common.py::_predict_split` runs them.
- The outputs are dumped per video and configuration, with the final tracks, and hashed into a
  manifest.

**Gates on the machine.** Any failure stops the run; nothing is read; the author is asked.
1. **On every video,** R0's final tracks must equal session 3's
   (`studies/d2/data/baseline_tracks.json`, extracted from session 3's cache before any D2
   recording). The rule is D2's ground rule 3, exactly.
2. **The 2-video gate** runs `ILSVRC2015_val_00046001` and `ILSVRC2015_train_00047020`, as in
   D2.
   - R0's predictions for both splits must equal session 3's (`studies/a26/data/r0_reference.json`):
     the same order, triplets, durations and trajectories, with scores within 1e-6.
   - The timed projection of the full run must be **at most 18 h**. Otherwise, stop and ask.

**On the laptop,** before anything is read:
- every file must match the manifest, for 200 videos × 4 configurations;
- R0's mAP must equal session 3's (15.786 novel, 27.185 all) within **0.01 points**;
- the videos whose R0 predictions differ from session 3's at all are listed.

##### Evaluation (laptop, once; `studies/a26/evaluate.py`)

- **The evaluator** is unmodified: `eval_relation_detection_openvoc(target_split_pred=split,
  rt_hit_infos=True)`, as `_predict_split` calls it.
- **What is reported:** SGDet `novel` and `all`; mAP, R@50 and R@100, for each configuration.
- **Per-video statistics** (AP, TP within the top 50 and top 100, number of GT) come from the
  evaluator's own `eval_detection_scores_v2` and `voc_ap`. They must aggregate to the official
  numbers: mAP within 1e-9, recall within 1e-4.
- **The paired video bootstrap:**
  - 1,000 resamples per split, `numpy.random.default_rng(0)`, over the split's evaluated videos
    in sorted order;
  - **the same draws for every configuration**;
  - 95% percentile intervals of each configuration minus R0;
  - on the first two resamples, the evaluator itself must reproduce the aggregates, with
    duplicate videos as distinct keys and the GT resampled the same way. This is the procedure
    A23 describes.
- **Point estimates** are the official numbers' differences.

##### The reading (fixed here)

**One quantity is read: Δuw = R0uw − R0, SGDet novel mAP.** It carries one of three words:
- **"large"**: |Δuw| ≥ 1.0 point, and the 95% CI excludes 0;
- **"moderate"**: 0.3 ≤ |Δuw| < 1.0, and the CI excludes 0;
- **"small"**: otherwise. A CI that touches 0 counts as including it.

For "large" and "moderate", the direction is stated: "lower under the clean protocol" if Δuw < 0,
"higher" if Δuw > 0.

**What the word decides** (agreed plan §5):
- **large:** the clean protocol (whole video, uniform fusion) becomes a primary protocol of the
  final test session, next to the published one. If the clean protocol scores lower, the
  fallback framing (agreed plan §7) moves up.
- **moderate:** both protocols are reported throughout.
- **small:** the published-protocol comparison stands, with this audit disclosed.

**Descriptive only, reported with CIs and no verdict words:**
- Δu = R0u − R0 (A23's "R0u − R0" row);
- Δw = R0w − R0;
- every `all`-split difference, and every recall.

No other contrast is read.

##### What changes in A23, and other records

| | before | now | why |
|---|---|---|---|
| R0u's test evaluation | in RB1's single test session, after all training ("Nothing is evaluated on test before that") | **here, before RB1's training** | It uses only the released checkpoint, so it informs nothing RB1 trains. The agreed plan needs the protocol effect now. RB1's session does not re-run R0u: its "R0u − R0" row is A26's, with the same checkpoint and code. A23's rule still binds R1, R2 and R3. |

- **The count.** This is the **seventh** test evaluation in pre-registration order (A23's RB1 is
  the fifth, and has not been run; A24's M1 is the sixth). It is the **sixth actually performed**,
  after the four earlier ones and M1. D2 (A25) was a test-label error analysis, not an
  evaluation. **It evaluates no method:** all four configurations are the released checkpoint.
- **Bound like D2's recordings (A25d).** A26's predictions, tracks and per-video statistics are
  never used to design or tune anything. They are used only for the reading above, and are
  cited as reported.
- **The test annotations on the machine.** The pipeline reads 202 files:
  - `test_relation_gt.json` and `test_object_trajectories_gt.json`, through `Dataset_new`;
  - the 200 files `anno/test/<video>.json`, through `gen_feats_test`, which reads each video's
    height and width (`models/gen_labels.py:672-675`, `:26-30`).

  Only the pipeline reads them on the machine. No D2 code runs there: `a26/tracks.py` holds
  copies of D2's two comparison functions. The files are deleted right after the run, with
  their SHA256 logged (`box/cleanup.sh`).
- **The token.** As in A25c: used only as `HF_TOKEN` in the author's interactive fetch; never
  written to a file; token scans before and after; unset, and any token file deleted, before
  the run starts.
- **Copying.** The METOR code is copied as `git archive` of the commit, without
  `studies/d2/data/rec`. Nothing is pushed.

##### Estimate and machine (`python -m a26.estimate`)

**The inputs are measured:**
- D2's forward time, 10,873 s over 46,035 GT-range frames;
- D2's 3.5 h whole pass, so encoding is about 0.48 h;
- session 3's 52,104 frames over the whole videos.

**The estimate:**
- forward passes: 3.02 h each for R0 and R0u; 3.42 h each for R0w and R0uw;
- GPU total: **14.7 h** (12.5–16.9), including +10% for modelC's second call;
- machine time with setup, the token fetch, the gate and the copy-off: **13–18 h**;
- cost: **$4.7–6.3** at the measured $0.35/h, **$8.1–10.7** at $0.60/h.

**The machine:**
- one **RTX 4090**, as session 3 and D2 (gate 1 needs bit-identical tracks);
- **≥ 64 GB** RAM, **≥ 60 GB** disk;
- a PyTorch -devel image with `nvcc`.

**What the machine gets:**
- **Copied from the laptop:**
  - the EOV repo as `git archive fdeeeb4`, for the fetch tools and the CUDA operator;
  - METOR as above;
  - the class splits, id maps, `prior.pkl`, the train trajectories, and the 202 test
    annotation files.
- **From Hugging Face, with the token:** the eval weights (2.92 GB) and the 200 test videos'
  frames.
- **From the public URL:** CLIP ViT-L/14@336px, with its SHA256 checked.

##### Pins

| artefact | SHA256 |
|---|---|
| METOR commit holding the A26 code | `486a962dc483f2e4789e22279ebc288ee3935668` |
| `studies/a26/logs/code_pins.json` (157 METOR code files, 9 a26 files, 7 inputs; `run.py` checks all of them on the machine) | `699508ff79f56d3678a7e95e369e4d12138fe911c69e4212b9c860be5c424d96` |
| `studies/a26/run.py` | `7437921c1eac7765323ae1d6e40df5e15c79d5e06e86515f03db32631ce77b3b` |
| `studies/a26/wrappers.py` | `55264c35f698636baeced91fb4fad78ad1322bc1b936905e46f9eee2c4524e83` |
| `studies/a26/evaluate.py` | `4b77ec161b89e41f635538b17bb5869c01b8aeb173f736ec934ce78055f3d188` |
| `studies/a26/common.py` | `30124d9cb184662acac1a461141bc824c046266627cd545dfbd487c15b40923c` |
| `studies/a26/tracks.py` | `67724d8c7c1f9c4f2a0c43ee03f845e44c4017e04f5602f33ba02423d806ef03` |
| `studies/a26/data/r0_reference.json` (from session 3's `sgdet_{all,novel}.json`, checked against `SHA256SUMS_session3.txt` `2cc2d0710f6a92d68ae27ecf74973a9dd122dc1594b9788cff4b0c560567954b`) | `d288d4c29e46266d8d737204dd4a5f1914966ede60f2af315c38e35942106270` |
| `studies/d2/data/baseline_tracks.json` | `3d9c5c86b9a80681cb1aa6643bab94f1b6bdea4448ecdb6151797ca63793df56` |
| the end-to-end checkpoint, as A23 | `d306952a51e9ca94c201ecf580f75a99221ae658dc187ae0ff38ff8b7dfc0e70` |
| `AFLink_epoch20.pth`, as A23 | `b35cbeddd3acc48fece820bd640640e6bfb1f5fbf570aa79af26c6a38958daa4` |
| `VidVRD_ECC_test.json` | `6e7842bee9887c85164b673becc1e61040bfa31bc33b921202e4b2d9649faa2b` |
| `test_relation_gt.json` (laptop evaluation; A25's pin) | `b7a8219fa1b00e6d10c8bc5f37a1df7482b316dd401a1c870678817c4a921aeb` |
| `studies/a26/logs/pytest_a26.txt` (30 CPU tests with stubs, all passed) | `3552d93ca87938e8380b1a54dc0d770aaec69f2250335260af20b85e9f4f7fb7` |

**What the tests pin:**
- R0 through the A26 runner equals the repo's own `_predict_split`, run once per split, on the
  CPU stub pipeline: tracks and predictions both;
- each configuration's frame range and fusion switch;
- that the whole video moves the 15% rule;
- that the order of configurations does not matter;
- the evaluator agreement, the bootstrap against the evaluator on resamples, and the reading's
  boundaries.

**Order.**
1. This amendment.
2. **The author approves the spend.**
3. **On the machine:** upload, setup, the token fetch, `check`, then the 2-video `gate`. Stop on
   any gate failure, or a projection over 18 h. Then the full run, the cleanup (test
   annotations deleted and logged), and the copy-off. The author destroys the machine.
4. **On the laptop:** `evaluate.py`, once. Then the report `studies/a26/a26_report.md`, and stop.

Any change to the above is a new amendment, written before the result it affects.

#### A26m (2026-10-24): repository migration, no content change

Written after the move. No model was trained or evaluated, no GPU was used, and no test
annotation was opened. Nothing in A23–A26 changes.

**What moved.** At the author's request, the METOR repo's `studies/` and its code moved into
the EOV-MMP working repo, `~/Documents/thesis/source/ov-vidvrd-lab`. That is now the one working
repo. METOR becomes an archive.
- 775 files were copied byte for byte, and **no copied file is edited**:
  - the 764 files METOR's git tracks, at HEAD
    `16e5744217d00d47be55f04a75146e646533a58a`;
  - the 11 git-ignored D2 machine logs, `d2/logs/box/*.log`. `d2/d2_report.md` hashes nine of
    them.
- Each file has the same SHA256 at its new path.
- 677 of these files are hashed by this chain, directly or through a file it hashes. The
  closure, recomputed in the new place, reaches the same 677.
- This file keeps its hash chain. It now lives at
  `ov-vidvrd-lab/studies/prereg/preregistration.md`.
- **Backups:** git bundles of both repos, all refs, verified, in `~/Documents/thesis/archive/`.
  Every commit this chain cites is in one of them.

**Path map.** It resolves every path cited before A26m.

| cited as | now |
|---|---|
| `METOR/studies/…`, `studies/…` | `ov-vidvrd-lab/studies/…` |
| any other METOR path (`cli/…`, `models/…`, `docs/…`, `tests/…`, …) | `ov-vidvrd-lab/studies/metor_code/…` |
| "METOR's root" in the study code (the folder above `studies/`) | `studies/metor_code`, set by `studies/paths.py` |
| `ov-vidvrd-lab/…` (EOV-MMP code, pilot, data, checkpoints) | unchanged |

**New files.**
- `studies/paths.py` is the only file that knows the new layout. MIGRATION.md §4 gives the
  commands.
- **The machine flows are not re-pointed** (`a26/box/`, `d2/box/`). A26's GPU step needs a pin
  addendum that re-points them before it runs.

**Checks, from the new place** (MIGRATION.md §6).
- **Chain.** The pilot file recomputes to `442623b8…11f0`, and this file, through A26, to
  `9485e1a8…c494`. Every snapshot, hash list and link matches, including ov-pairtime's
  `981bf067…5ce5`.
- **Pins.** Every structured pin matches at its new path:
  - A23a's 14 changed files;
  - D2's code pins and its 404 test recordings;
  - A26's 157 METOR files and 9 a26 files.
- **Tests.** The counts equal the last logged ones:

  | suite | result |
  |---|---|
  | A26 | 30 passed |
  | METOR's suite (RB1) | 169 passed |
  | D2 | 25 passed, and 25 passed on METOR's code |

**Record.** `studies/MIGRATION.md`, SHA256
`4c4df2878ae3f3f7e0608f74dc61481eb2fb763e475a53f508d10a3e0a91daf0`. It holds:
- the date and METOR's HEAD;
- the bundles and their SHA256;
- every moved file (old path → new path, SHA256);
- what was left out and why;
- the checks and the test results.

It also records that the pilot file's A6–A22 were committed late, on 2026-10-24
(ov-vidvrd-lab `0f97a26`), and why their content predates A23.

**Not changed:** any definition, threshold, reading rule, stop rule, pin or order of A23–A26.
A26 stays stopped before any GPU use.
