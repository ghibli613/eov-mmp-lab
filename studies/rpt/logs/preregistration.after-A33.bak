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

#### A28 (2026-10-26): RPT G0 — SAM 3 set up, the five code checks fixed as adapter rules, a reproduction of session 3 and a 3-video smoke test on HELD; no gate result, no test data

Written before any SAM 3 output on a VidVRD video, any adapter run, the reproduction or the smoke
test exists. No model has been trained or evaluated, no test annotation has been opened, and the
only GPU use so far is a memory probe of SAM 3 with random weights and no data
(`studies/rpt/g0/logs/laptop_memory_probe.txt`). A27 stays reserved for the tracklet-keeping
design (`studies/design_review/agreed_plan.md`); this is the next free label.

**What G0 is.** The first gate of the RPT proposal (`studies/rpt/rpt-proposal.md`, version 5,
SHA256 `1f9f648262c460ccd181e7e60a79ada9b67d5293f7ef8cff2ef746958959cf83`): a frozen SAM 3 replaces
EOV-MMP's detector, tracker and track filter, and its tracks enter the released relation head
unchanged, through `gen_feats_test` + `modelC`, the path session 3 used for GT tracks. G0 sets SAM 3
up, settles the proposal's five code checks, builds the adapter and runs a small smoke test. **It
measures no gate result and makes no design choice.** Where the proposal could not be followed as
written, G0 stopped and reported (`studies/rpt/g0/interim_report_1.md`, SHA256
`786c2c148d38da76b30ca16fa3dd2fa90a48143a64e28953df6f6b1f9557f01e`); the rules below that settle
those points are the plan author's answers to that report. G0 ends with `studies/rpt/g0_report.md`,
a draft G1 amendment (not hashed) and a stop.

##### The chain checker (G0 step 3)

`studies/migration/verify_after_A26m.py` skipped its ov-pairtime check without a word once the
ov-pairtime folder was gone. Only that check changed: the file is now read from the archive bundle
`~/Documents/thesis/archive/ov-pairtime-2026-10-24-a820a0e.bundle` (commit `a820a0e`, through a
throwaway bare repository) and must hash to `981bf067…5ce5`. A missing bundle, one git cannot read,
or a different hash is a FAIL with a non-zero exit; the check is never skipped.

| | SHA256 |
|---|---|
| `studies/migration/verify_after_A26m.py` (fixed) | `3af183caad77d61ea4aca7e7fa04d2da74c354b0d4335797ede98409d0eda1e7` |
| `studies/rpt/tests/test_chain_checker.py` (7 tests: present and matching, synthetic and real; missing; altered; two unreadable bundles; `main()` exits 1 without the bundle) | `8fc440011d2d60750e17b4ee7a3d830ce95791251ac1749882a3b394f33ea13b` |
| `studies/rpt/g0/logs/verify_after_A26m_fixed.txt` (the real chain: 66 checks, all passed, head `ac77e243…a591`) | `26154ba93572ae25bdc36ed502a13902a962d5c95d427dd54efca61838378677` |

##### SAM 3, frozen

- **Code:** `github.com/facebookresearch/sam3` at `f1647e8d0c1d280991f4d1b7d974c2704537fe66`
  (2026-10-23), cloned outside this repo to `~/Documents/thesis/source/sam3`. Never edited.
- **Checkpoint:** `facebook/sam3`, file `sam3.pt`, Hugging Face revision
  `3c879f39826c281e95690f02c7821c4de09afae7`, 3,450,062,241 bytes; its Git LFS pointer is the git
  blob `5b7c2eaba7c1e2a81f26bc7476bfe80e43e979e4` (135 bytes), and `config.json` is the blob
  `3e600e765b0f1b4c52680aed9ad242c9a90f7ed1`. The repository is gated and the author's access is
  pending; without access Hugging Face exposes no SHA256 (the file's HEAD request, its raw LFS
  pointer and `paths-info` all answer 401; the tree API masks the LFS oid). **The SHA256 is pinned
  in a pin addendum, A28a, written after the download and before any SAM 3 code touches data.**
  A28a also checks that the LFS pointer rebuilt from that SHA256 and the size hashes to the blob
  above, so the file is the one this revision named. It is loaded only from a local path after its
  SHA256 is checked; SAM 3's own download code is never reached.
- **Model:** `build_sam3_video_predictor()` (`sam3/model_builder.py:818`) with every default: SAM 3,
  the November 2025 release. **Not SAM 3.1**, which the repo added on 2026-03-27 (`facebook/sam3.1`;
  `build_sam3_predictor()` defaults to it); the proposal names SAM 3.
- **Licence:** the SAM License (last updated 2025-11-19; `LICENSE` SHA256
  `4dea99bfaa016e21bc860d73f344236bd1e5c4977d1a9a8fd32f822b500ae1be`, unchanged since the initial
  commit; the Hugging Face repo's `LICENSE` is the same text without its final newline). §1.a grants
  use; §1.b.ii requires that a publication acknowledge the use of SAM Materials. Outputs are not SAM
  Materials; no clause restricts them.
- **Environment:** its own conda env `sam3` (conda-forge Python 3.12.15; the default Anaconda
  channels were not used, since they need a Terms-of-Service acceptance that is the author's to
  give), torch 2.10.0+cu128 as SAM 3's README installs it, `pip install -e ".[notebooks]"`, psutil,
  and setuptools 78.1.0 (`model_builder.py:8` imports `pkg_resources`, which setuptools 84 no longer
  ships). `pip freeze`: `studies/rpt/g0/logs/env/pip_freeze_sam3_laptop.txt`, SHA256
  `7831e667dcbe4decdd0d970c25338efd3e3c62606f31bade73d01d165db7fe7e`. The two sides talk only through
  files.

##### The five code checks, and the rules they fix (G0 step 4)

1. **Gap filling.** The released track code (`models/gen_labels.py`, called by
   `format_trajectories`, `:446-451`, and `format_trajectories_test`, `:605-610`):
   - adds two frames before the first box, copies of it (`add_initial_frames`, `:363-371`);
   - drops a track with boxes on < 65% of its span (`:380-382`), and, while a gap is longer than 15%
     of the span, drops the shorter side of the largest gap (`:385-413`);
   - fills every remaining internal gap by **linear interpolation** of each coordinate between the
     boxes on either side (`np.interp` over the present frames, `:416-419`), then rounds every box of
     the span to integers and clamps it at 0 (`round_and_positive`, `:422-423`, `:426-428`);
   - adds nothing after the last box (`end_fid` = last frame + 1, `:480`, `:635`); keeps a track
     only with ≥ 12 frames, pre-roll included (`:471`, `:626`).

   **Rule.** Internal gaps of a SAM 3 ID are filled by `:416-423` exactly: `np.interp` over the
   frames where SAM 3 output the ID, then the repo's own `round_and_positive` on every frame of
   [first, last]. The coverage rule, the gap trim and the length filter are rule 8, which RPT
   replaces. **No pre-roll** (settled: interim report 1, D4).

2. **Score field.**
   - SAM 3 thresholds the **detection probability** `sigmoid(pred_logits)`
     (`sam3/model/sam3_video_base.py:602`): a detection is kept if it is > 0.5
     (`score_threshold_detection`, `:608`, with mask NMS at the same threshold,
     `sam3_image.py:846-861`), and a masklet is started only by a kept detection ≥ 0.7
     (`new_det_thresh`) that matches no existing masklet (`:210-213`). The video post-processing
     (`_postprocess_output`, `sam3_video_inference.py:432-524`) thresholds no score.
   - **Presence × object: yes, already.** With the presence token (on by default:
     `has_presence_token=True` → `supervise_joint_box_scores`, `model_builder.py:680`, `:739`),
     `pred_logits` = logit(σ(class logit) × σ(presence logit)), clamped to ±10
     (`sam3_image.py:345-353`).
   - **The output field** is `out_probs` (`sam3_video_inference.py:450-452`, `:519`) =
     `obj_id_to_score`, set once, to the probability of the detection that started the masklet
     (`sam3_video_base.py:1059-1064`; the code's own comment: "first frame detection score",
     `sam3_video_inference.py:409`), and never updated (`:1079-1080` only marks removals). SAM 3's
     own multi-prompt video evaluation scores each masklet with this one value, on every frame
     (`sam3_video_inference.py:946-950`; `sam3_video_base.py:2049-2075`, `per_frame_scores = scores`).
   - **Detected / propagated / absent:** not in the output, but identifiable from SAM 3's own
     per-frame association: a masklet is *detected* on frame t if it was started by a detection on
     t, or a kept detection was matched to it on t (mask IoU ≥ `assoc_iou_thresh` 0.1); this is the
     definition SAM 3's masklet confirmation uses (`sam3_video_base.py:1968-1970`, `:2011-2017`). It
     is read through a read-only wrapper around `run_tracker_update_planning_phase`; SAM 3's code is
     not edited and its outputs do not change.
   - Per-frame detection probabilities exist inside SAM 3, but are not attached to masklets; when
     several kept detections match one masklet on a frame (allowed: `im_mask`, `:251`, feeds
     `det_to_matched_trk_obj_ids`, `:150`), SAM 3 defines no single value.

   **Rule (settled: interim report 1, D2).** The per-frame value is the masklet's `out_probs` in that
   frame's output; the track score is its mean over detected frames, which equals the probability of
   the masklet's starting detection (≥ 0.7). Also recorded per frame, **not used**: the
   probabilities of the matched detections (with a count of frames with more than one) and SAM 3's
   tracker score. A masklet with no detected frame has no score; it is reported and not paired.

3. **Clip grid.** Clips are not on a video grid. For each ordered pair, clips start at the pair's
   overlap start, `begin_fid = max` of the two tracks' starts (`gen_labels.py:657`), in steps of 30
   (`:13`, `:693`); a tail ≥ 10 frames is the last clip and a tail < 10 is cut (`:661-666`,
   `:694-697`). A clip's features are read at its middle frame, `begin_fid + len(clip)//2`
   (`:81-82`, `:106`, `:117`). Middle frames are shared only by pairs whose overlap starts agree
   modulo 30, always by (s, o) and (o, s). On HELD with session 3's GT tracks
   (`studies/rpt/g0/check3_middle_frames.py`, SHA256 `6843dbff…cede9`; output
   `logs/check3_middle_frames.json`, `44911fb2…f9cd`): 628 of 47,038 frames are read (1.3%, 4.0
   clips per frame read); a cache of only those frames is 2.5 MB per video at the median (max 17.7
   MB) against 145 MB (max 2.49 GB) for every frame, at fp16.
   **Rule.** Features are computed on demand for exactly the frames `gen_feats_test` requests, and
   memoized per (video, frame); never for every frame.

4. **Window joining.** Session 3's `run_pair` (`pilot_analysis/stage1c/cache_wrapper.py:166-180`)
   splits a pair's clip sequence into consecutive windows of ≤ 40 (`:46`, `:171`), runs `modelC`
   once per window (`:173-175`) and concatenates the per-clip outputs in order (`:176`, `:179`);
   `eval_from_cache.predict` then runs `process_pred` → `association` on each pair's whole joined
   sequence (`eval_from_cache.py:72-77`). So per-clip predictions are joined **before** the greedy
   merge. No pair in session 3 had more than 40 clips, so the join ran only in its random-tensor
   check; the released SGDet path does not window at all (`models/end2end_model.py:326-337`).
   **Rule.** `run_pair` is called, not reimplemented; its joined output goes to `process_pred` →
   `association` → `format_`.

5. **Confident cut: 0.5.** SAM 3 keeps a detection if its probability is > 0.5: the video model's
   `score_threshold_detection` ("prob threshold for detection outputs -- only keep detections above
   this threshold", `sam3_video_base.py:270-272`, `:608`; 0.5 in `model_builder.py:747`), the image
   API's `confidence_threshold=0.5` (`sam3_image_processor.py:17`, `:199`; SAM 3's example notebooks
   use `Sam3Processor(model, confidence_threshold=0.5)`), and the predictor API's default
   `output_prob_thresh` 0.5 (`sam3_base_predictor.py:76-79`; passed only to SAM 3.1 models).
   `new_det_thresh` = 0.7 (`model_builder.py:750`) starts a masklet; it does not keep a detection.
   **Consequence:** at SAM 3's defaults every masklet starts from a detection ≥ 0.7, so every
   masklet is confident and **the candidate tier [0.1, 0.5) is empty by construction.**
   **Rule (settled: interim report 1, D1).** No second SAM 3 pass and no SAM 3 setting is changed.
   In the same default pass, SAM 3's raw detections with probability in [0.1, 0.5) are recorded
   per frame, read-only: box, probability, prompt. They are read from the detector's
   `forward_grounding` output as it returns, before NMS lowers the logits of every detection ≤ 0.5
   in place (`sam3_image.py:846-861`), through the same kind of wrapper as the status; the frame
   comes from a wrapper on `_build_multigpu_buffer_next_chunk` (`sam3_image.py:822`), with no GPU
   sync. They are pre-NMS, as SAM 3 never runs NMS on them. The smoke test reports their counts;
   G1 will report how many GT objects missed by confident tracks have one. The candidate tier and
   Verify are decided in the G2 amendment.

##### The adapter (`studies/rpt/`; G0 step 7)

SAM 3 side (its own environment):
- **Prompts:** the 35 names of `data/vidvrd/data/object2id.json` (SHA256 `350c554c…2b37`), in id
  order, `_` replaced by a space; nothing else, no synonym. Each masklet carries its prompt's name.
- **Protocol:** SAM 3's own multi-prompt video protocol (`Sam3VideoInference.forward`,
  `sam3_video_inference.py:928-952`), through the predictor API: one session per video on its JPEG
  folder; per prompt, `add_prompt` at frame 0, then `propagate_in_video` forward from frame 0 over
  every frame, then `reset_session`. Default settings throughout (thresholds 0.5 / 0.7, hotstart
  15 / 8 / 8, image size 1008, frames held on the GPU, bf16 autocast as the predictor sets it, no
  `torch.compile`). `torch.manual_seed(0)`; SAM 3 inference draws no random numbers.
- **Written per video:** `studies/rpt/g0/sam3_out/<video>.json.gz`: per prompt and masklet, per
  output frame the RLE mask, the tight box of the returned mask, SAM 3's own box, `out_probs`, the
  frame status and the recorded detection and tracker scores; per prompt its time and its raw
  [0.1, 0.5) detections; per video the peak GPU memory.

EOV-MMP side (unchanged repo code):
- **Identity:** one (prompt, SAM 3 ID) is one track; a new ID is a new track; nothing is split.
- **Boxes:** on every frame where SAM 3 returns a non-empty mask for the ID, the tight box of that
  mask, `[x1, y1, x2, y2]` in pixels with `torchvision.ops.masks_to_boxes`' convention (the function
  SAM 3 uses for its own box). SAM 3's own box (from the mask before its non-overlap step,
  `sam3_video_inference.py:492-515`) is recorded beside it. A frame with an empty returned mask is
  not an output frame.
- **Gaps:** rule 1. Frames before the first or after the last output frame are not part of the
  track.
- **Status per frame:** *detected* or *propagated* (output frames, rule 2) or *filled* (gap frames).
- **Score and tiers:** rule 2; confident if score ≥ 0.5 (rule 5), candidate if 0.1 ≤ score < 0.5,
  dropped if < 0.1. With no loop in G0, only confident tracks are paired (proposal §3.1: candidates
  are for the loop).
- **Duplicates across prompts (settled: interim report 1, D3):** two masklets from different
  prompts whose per-frame mask IoU, averaged over the frames where both have an output mask, is
  > 0.7 are duplicates. The higher-scoring masklet is kept unchanged (frames, boxes, score, class)
  and the other is removed, greedily in descending score (ties: prompt id, then SAM 3 ID); each kept
  track is still exactly one SAM 3 ID. G0 reports both masklets of every duplicate and the frames the
  removed one had outside the shared frames.
- **Pairs:** `gen_feats_test` (`gen_labels.py:642`) on the confident tracks, split `train` (the HELD
  annotation is read for width, height and the frame count only); its own overlap ≥ 10 and tail
  rules, in its own order.
- **Windows:** rule 4.
- **Features:** CLIP-L and TagCLIP-L through the repo's own encoders, `_shared_encoder` and
  `_transform_resize(336, 336)` (`data_loading/dataset.py`), exactly as session 3's
  `encode_frames` per frame (`cache_wrapper.py:78-98`): batch 1, fp16 on CUDA, frame `f` = the f-th
  file in sorted order; whole video, the GT range of `dataset.py:120-135` bypassed; on demand and
  memoized (rule 3).
- **Gap exposure:** a clip whose middle frame is a *filled* frame of its subject or object is
  flagged and counted.
- **No test labels:** every G0 runner installs an audit hook that raises on opening any path under
  `anno/test`, any `test_relation_gt*` or `test_object_trajectories*` file, or anything of session
  3's test caches.

##### The reproduction of session 3 (G0 step 8)

- **Videos:** the first 5 HELD IDs in sorted order (`pilot_analysis/stage1/train_split.json`,
  SHA256 `095dd89c…1f1a`): `ILSVRC2015_train_00008005`, `…00010024`, `…00024005`, `…00025029`,
  `…00027003`.
- **Input:** their GT tracks from `gt_tracks.load_tracks("train")` (option B,
  `gt_tracks_train.json`, SHA256 `87b40fff…636d`), through the adapter's track check and feature
  wrapper into `gen_feats_test` + `modelC` (session 3's `build_modelC`, from the end-to-end
  checkpoint `d306952a…0e70`, `tgt_split all`, `--frame_stride 1`). No SAM 3 step applies to GT
  tracks.
- **Compared with** `cache_s3/train/<video>.pt` (each matching `SHA256SUMS_session3.txt`, SHA256
  `2cc2d071…954b`, as checked on 2026-10-26), pair by pair in `gen_feats_test` order.
- **Bar:** max |logit − cached logit| ≤ 1e-4 over every pair, clip and predicate column.
  Reported, not bars: the interactiveness-logit residual, the slot-input residuals, and whether
  each requested frame's `global_proj` equals the cached one (frame provenance).
- **Where:** session 3's stack: an RTX 4090, torch 2.11.0+cu128, `OMP_NUM_THREADS=8`. The frames are
  the laptop's JPEGs, which today's cv2 5.0.0 re-decodes from the laptop's mp4 files byte for byte
  (all 1,300 frames of the 5 videos), as session 3 decoded on its box. A run on the laptop's GTX
  1650 (sm_75), made first to test the code path, is labelled a dry run; its residual is reported
  and does not count.
- **On a fail:** find where the paths diverge (frames, features, windows, ordering) and report.
  Nothing is adjusted to make it pass.

##### The smoke test (G0 step 9)

- **Videos:** the first 3 of the 5: `…00008005` (130 frames), `…00010024` (64), `…00024005` (981).
- **Pipeline:** SAM 3 (35 prompts, every frame of the whole video) → adapter → `gen_feats_test` +
  `modelC` → `process_pred` (top 20 per clip, no prior) → `association` → `format_` (top 200) →
  `eval_relation_detection_openvoc(target_split_pred="base")` with `gt_json` = the HELD GT of these
  3 videos from `pilot_analysis/stage2/data/held_gt.json` (SHA256 `78d55c69…4dee`; built from the
  training annotations, base relations only, so the novel-predicate instances of the training
  videos stay unused). No T1 rules, no loop.
- **Reported per video:** frames; SAM 3 time per prompt and in total; confident, candidate and
  dropped by class; the raw [0.1, 0.5) detections, by class; SAM 3's own removals; duplicates;
  ordered pairs and clips; middle frames inside filled gaps; frames with several matched detections;
  peak GPU memory. Then the G1 cost extrapolated to the 160 HELD videos (47,038 frames) against G1's
  runtime bar, and 18 images (6 evenly spaced frames per video, GT and SAM 3 boxes) in
  `studies/rpt/g0/viz/`.
- **The evaluator's numbers are labelled "smoke test, in-sample, not a result"**: EOV-MMP was trained
  on HELD. They are not read, compared or used for anything.

##### A runtime measurement, to inform G1 only (not a design change)

SAM 3 recomputes the image backbone for every frame of every prompt: its `forward_image`
(`vl_combiner.py:81`) depends on the image alone and runs on a `backbone_out` rebuilt per frame
(`sam3_video_base.py:580-590`; `sam3_image.py:136-166`). On smoke video 1 only, after the default
runs and the relation side:
1. **A determinism control:** the default run again, on its first 3 prompts.
2. **A reuse run:** all 35 prompts, with a read-only wrapper on `forward_image` that keeps the
   output SAM 3 computed for a frame (a CPU copy; position encodings bitwise equal to an earlier
   frame's are kept once) and gives later prompts a GPU copy with the same tensor sharing. On
   random weights the output is 212.6 MiB per frame in bf16, about half of it position encodings.
3. **Reported:** whether masks, boxes, scores, statuses and the raw detections are bitwise
   identical to the default run (`studies/rpt/g0/sam3_compare.py`), the differences if not, and
   the speedup. If they are not identical, the reuse is not used. The control separates a change
   from GPU non-determinism. G0 uses the default run for everything else.
4. **Budget:** this runs last and only if fewer than 95 minutes have passed since the machine's
   setup began.

##### The laptop check (G0 step 6)

SAM 3 at its default settings on `…00010024` (the shortest smoke video, 64 frames) with one prompt,
"bicycle" (the class of its GT track 0), on the GTX 1650 (4 GB), after A28a. Reported: whether it
runs, peak GPU memory, time per frame. Nothing is lowered or changed to make it fit. It is stopped
after 30 minutes of wall time if unfinished, and reported as such.

##### The rented machine (approved by the author)

An RTX 4090 (session 3's GPU), ≥ 32 GB RAM, a Vast.ai PyTorch image with torch 2.11.0+cu128 for the
EOV-MMP side, SAM 3 in its own virtual environment as above, `OMP_NUM_THREADS=8`,
`--frame_stride 1`. Approved under 2 GPU-hours. The run stops and the author is asked if the
measured SAM 3 time on the first smoke video projects the 3 videos beyond 1.25 GPU-hours (with
about 35 minutes of set-up, fetch, reproduction and copy-back, the machine stays under 2 hours).
To cover steps 8 and 9 with one machine, G0 waits up to 2 days for SAM 3 access; if access takes
longer, the plan author is asked about a machine for step 8 alone. The token, as in A25c and A26:
a read-only Hugging Face token, held only as `HF_TOKEN` in the author's interactive fetch (with a
throwaway `HF_HOME`), never written to a file, log or report, unset after use, with token scans
before and after. No test file is copied to the machine. Nothing is pushed.

##### G0 pass bars

1. The reproduction: max absolute logit residual ≤ 1e-4 against `cache_s3`, on all 5 videos.
2. All adapter unit tests pass (`studies/rpt/tests/`).
3. The 3 smoke videos run end to end, SAM 3 to the evaluator, without an error.

A pass allows the G1 amendment to be written; it is not evidence for the method.

##### OPEN (not decided here)

1. The candidate tier and the Verify action: the G2 amendment (rule 5).
2. The runtime fallback's *k*, if G1's runtime bar cannot be met (proposal §3.1): the G1 amendment.

**Order.**
1. This amendment; the checker is run; a commit.
2. Laptop: the adapter's unit tests; the dry run of step 8 on the GTX 1650.
3. SAM 3 access; the author's fetch on the laptop; A28a pins the SHA256; the laptop check.
4. The machine: steps 8 and 9, then the measurement; copy-back; the author destroys it.
5. Laptop: the comparisons, the evaluation, the images, `studies/rpt/g0_report.md`, and
   `studies/rpt/g1_amendment_draft.md` (not hashed). Stop.

The adapter, its tests and the machine scripts exist as this is appended; none of them has run.
Any change to the above is a new amendment, written before the result it affects.

#### A28a (2026-10-26): pin addendum to A28 — SAM 3's checkpoint SHA256

Written after the author's access to `facebook/sam3` was granted and the checkpoint was fetched,
before any SAM 3 code has touched data. The author ran `studies/rpt/g0/fetch_sam3_ckpt.sh`
(token typed hidden, held only in `HF_TOKEN` with a throwaway `HF_HOME`); the file is at
`~/Documents/thesis/source/sam3_ckpt/`, outside the repo, and holds no token-shaped string.

| | pinned |
|---|---|
| `sam3.pt`, `facebook/sam3` revision `3c879f39826c281e95690f02c7821c4de09afae7` | SHA256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`, 3,450,062,241 bytes |
| `config.json`, same revision | SHA256 `4616385e4b21f2e5e22c875b65679185cbccfa95de42542b9166f7dc3d57160f` |

**Checked against A28's pins:** the size equals A28's; the Git LFS pointer rebuilt from this SHA256
and size (135 bytes) hashes to the git blob `5b7c2eaba7c1e2a81f26bc7476bfe80e43e979e4` that A28
pinned, and `config.json` hashes to its pinned blob `3e600e765b0f1b4c52680aed9ad242c9a90f7ed1`. So
the file is the one the pinned revision named.

**Use:** every SAM 3 run (the laptop check, steps 9, the measurement) checks this SHA256 before
loading; the machine's own fetch is checked against it too.

Nothing else in A28 changes.

#### A29 (2026-10-28): RPT G1 — SAM 3's tracks against EOV-MMP's on the 160 HELD videos

Label: A29 is the next free one (A27 is reserved; A28 and A28a are G0's). It is appended to the
chain after A28a, whose head is `0d62e8380a8b8a9e2361717fc210a351fe4f3aaf4d2119a44c946315d753947b`.

**When this was written, and what had been seen.** It was written after G0
(`studies/rpt/g0_report.md`), before any G1 run. It applies the plan author's answers to
`studies/rpt/g0/runtime_proposal.md` and the review `studies/rpt/rpt-review-g1.md`.

- **G0's 3 smoke videos are in G1.** `ILSVRC2015_train_00008005`, `…00010024` and `…00024005` are 3
  of the 160 HELD videos and of G1's first 20. The definitions below were written after seeing
  SAM 3's output on them. Every bar is judged on all 160, as pre-registered.
- **The other 157 HELD videos:** 45,863 frames (median 175, max 2,947, 3 over 1,000); 465 GT objects
  (346 base, 119 novel) of 34 classes; 4,297 base GT relations in `held_gt.json` (154 of them).

**What G1 reads.** G1 measures the tracker. No relation score on HELD is read, except the secondary
box-sensitivity hit rates, which compare box sources on the same in-sample model. EOV-MMP was
trained on HELD, so its relation scores there are in-sample.

##### Changed from the proposal (before any G1 data; reasons measured or documented)

**1. The runtime bar becomes a cost stop.**
- **Superseded text.** Proposal §6, G1 bars: *"**Runtime.** At most 1 GPU-hour per 20 videos;
  otherwise the pre-registered every-*k*-frames fallback."* Proposal §3.1: *"**Runtime fallback
  (pre-registered).** If G1's runtime bar fails, text detection runs every *k* frames, with SAM 3's
  tracker propagating in between. *k* is fixed in the amendment and never tuned on results."*
- **Reason.** G0 §9 measured SAM 3 at its defaults at 0.094 s per frame per prompt on an RTX 4090.
  That projects 43 GPU-hours for the 160 HELD videos, 5.4× the bar of 8 GPU-hours.
- **Now.** Runtime is **not a pass/fail bar of G1**. It is only a cost stop: a cap of
  **35 GPU-hours**, defined under "The cost cap" below. The every-*k* fallback and options B1–B5 of
  the runtime proposal are not used. SAM 3 runs at full fidelity: every frame, all 35 prompts,
  default settings. Only the implementation options A1–A3 and `offload_video_to_cpu` may be used,
  each only if bitwise identical to G0's output.
- **Binding consequence for the test session.** It runs SAM 3 at exactly the settings G1 validates:
  full fidelity, plus only the options that passed bitwise here. Otherwise G1 would not have
  validated the configuration that is tested.

**2. The novel-object recall bar becomes report-only.**
- **Superseded text.** Proposal §6: *"**Recall.** GT track recall at vIoU 0.5 at least 10 points
  above EOV-MMP's, overall and on novel object classes (when ≥ 30 instances)."* Only the novel part
  changes.
- **Reason.** On HELD, EOV-MMP's detector is not only in-sample. Its step-4 training targets included
  the boxes, class ids and text queries of every training trajectory, novel classes included: leaks
  T1 and T2 of RB1's training-target audit (`studies/prereg/preregistration.md:60-61`, A23). Its
  novel-object recall on HELD is therefore inflated twice, and a 10-point bar against it could stop
  RPT for a reason that has nothing to do with the test set.
- **Now.** Novel-object recall is reported in full, per class too, with no threshold. The overall
  recall bar, which includes the novel GT objects (235 of the 753 GT runs; 124 of the 472 tids), and
  the pair-recall bar stay binding.
- The review marked this the author's decision. **The plan author confirmed it explicitly before any
  G1 data.**

##### The bars

EOV-MMP's tracks are defined in "Definitions", item 1. The GT unit is the GT run (item 3), and
"matched" means matched under item 2's rule, class-aware unless stated. The same rule serves overall
recall, novel-object recall, unmatched tracks, box style and item 8, for SAM 3 and EOV-MMP alike.

| bar | quantity measured | threshold | status |
|---|---|---|---|
| **Overall recall** | fraction of the 753 HELD GT runs (item 3) matched by a SAM 3 confident track (item 2), minus the same fraction for EOV-MMP's tracks (R0w) | **≥ 10 points** | **binding** |
| **Pair recall** | fraction of HELD base GT relations (item 4) that D1's U1/U2 code recalls with SAM 3's confident tracks, minus the same for EOV-MMP's (R0w) | **≥ 10 points** | **binding** |
| **False-track burden** | mean over the 160 videos of unmatched confident tracks per video (item 5), SAM 3 ÷ EOV-MMP (R0w) | **≤ 1.5×** | **binding** |
| **Box style** | median over matched SAM 3 track and GT run pairs (item 2) of the mean per-frame box IoU over shared annotated frames (item 6) | **≥ 0.70** | **binding** |
| **Box sensitivity** | top-5 hit rate of the GT predicate (item 7) with SAM 3 boxes, minus the same with EOV-MMP's boxes | **≥ −5 points** | **secondary**: a fail does not stop the line; it moves v2 before the test (proposal §7) |
| Novel-object recall | overall recall restricted to the 235 novel GT runs (124 tids); per class too | none | **report-only** (change 2) |
| Recall per tid | a tid is recalled if any of its runs is recalled (item 3); 472 / 348 / 124 tids (all / base / novel), beside the run counts | none | report-only |
| Pair recall, scoreable pairs | pair recall with the best pair taken only over track pairs that overlap ≥ 10 frames (item 4); SAM 3 and EOV-MMP | none | report-only |
| Recall, confident plus candidates | as overall recall, with candidate-tier tracks added | none | **report-only**; identical to confident-only by construction (erratum) |
| Class-agnostic counts | every class-aware count above, class-agnostic (item 2) | none | report-only |
| Gap exposure | clips whose middle frame is a *filled* frame of its subject or object | none | report-only |
| Track counts by *N* | tracks per video, binned by confident-track count *N*: 0, 1, 2, 3–4, 5–8, ≥ 9; SAM 3 and EOV-MMP | none | report-only |
| Raw detections for missed objects | item 8 | none | report-only; input to the G2 amendment |

A fail of a binding bar stops the line (proposal §6, fallback 2). A pass only allows G2.

##### Definitions

1. **EOV-MMP's tracks** (formerly OPEN 3).
   - **What:** EOV-MMP's **final** tracks, after its own filters, i.e. the tracks its relation
     stage sees. That is the `trajectories` list after `format_trajectories_test` and the 15% rule,
     as passed to `gen_feats_test` and kept in `self.traj` (`models/end2end_model.py:314-325`).
   - **Primary: R0w,** the whole video (A26's `studies/a26/wrappers.py::whole_video`), matched to
     SAM 3's whole-video search. **R0, the GT-derived range, is reported beside it.**
   - **Leak note:** EOV-MMP's detector was trained on HELD (T1/T2), so its HELD recall is inflated.
     That makes the binding bars conservative for SAM 3.
   - **Camera compensation** (the draft's second-round OPEN 2, settled by the plan author):
     - **The problem.** The released test path gives the tracker the **test** ECC file
       (`end2end_model.py:277`). No HELD video is in it, and a video absent from the file is tracked
       without camera compensation (`tracker.py:55-57`, `end2end_model.py:19-40`).
     - **The fix.** The model's `test_ecc` attribute is set to the contents of the train ECC file,
       `data/vidvrd/data/VidVRD_ECC_train.json` (SHA256
       `202e7ffed90667c4250bf8a0490f20b22b4c7ec620f0cdabb58e0354795d4646`), with no code edited.
       EOV-MMP's HELD tracks then get the same compensation step its test tracks get.
     - **Three HELD videos are absent from that file:** `ILSVRC2015_train_00265063`,
       `ILSVRC2015_train_00679000` and `ILSVRC2015_val_00123004`. They run without compensation, as
       the released code does for missing videos.
     - The front end is fed HELD frames and training-annotation ranges only, never
       `Dataset_new('val')`, which reads test files, under the test-label guard.
2. **Matching a track to a GT run** (formerly OPEN 4; the rule set with the second-round OPEN 1).
   The same rule applies to SAM 3's and EOV-MMP's tracks.
   - **Cut-vIoU.** A track is compared with a run after being **cut to that run's frames**, by D1's
     `cut_viou` convention (`studies/d2/stages.py:147-159`). For a run over [gb, ge) and a track
     over [begin, end), the track is cut to [max(gb, begin), min(ge, end)). The cut-vIoU is the
     official `viou` (`inference/video_relation_detection_openvoc.py:73-114`) of the cut boxes
     against the run over [gb, ge), or 0 if the cut is empty. A match needs cut-vIoU ≥ 0.5.
   - **Assignment.**
     - Candidate (track, run) pairs with cut-vIoU ≥ 0.5 are taken greedily in descending cut-vIoU
       (ties: lower track id first, then lower run index).
     - A pair is accepted if its run is not yet matched, and its track is either not yet matched or
       matched only to runs of the same tid.
     - So **each run is matched by at most one track**, and **one track may match several runs of
       the same tid, but runs of only one tid.**
   - **Class-aware** for the bars: the track's class name equals the run's.
   - **Class-agnostic:** the same rule, ignoring names. Reported beside every class-aware count.
   - **Reason for the cut.** Without it, an unbroken SAM 3 track that bridges a gap of more than 30
     frames would fail vIoU against a single run just for being longer. That would penalise SAM 3 for
     keeping identity through occlusion, while EOV-MMP's tracks, which break at such gaps, would not
     be penalised.
3. **The GT unit: session 3's option-B run** (the draft's second-round OPEN 1, settled by the plan
   author). The unit for overall recall, novel-object recall, unmatched tracks, box style and item 8.
   - **Source:** `pilot_analysis/stage1c/gt_tracks_train.json` (SHA256
     `87b40fff514178f77a57d9e142bbbb9363eef34f6b4c22d8eaa88d4c7612636d`, pinned in A28). Each tid is
     split where more than 30 consecutive frames lack a box, and shorter gaps are interpolated.
   - **On HELD:** 753 runs (518 base, 235 novel) from 472 tids. 149 tids have more than one run, at
     most 8.
   - **Why not whole tids.** The official `viou` needs a dense trajectory over a contiguous span. Many
     training tids have unannotated frames inside their span (session 3 found 33.5% of train tracks
     under 65% coverage).
   - **Reported per run:** the binding overall-recall bar, box style and unmatched tracks.
   - **Reported per tid, report-only:** a tid is recalled if any of its runs is recalled. The tid
     counts (472 / 348 / 124) are reported beside the run counts (753 / 518 / 235).
4. **Pair recall** (formerly OPEN 5): D1's U1/U2 code, unchanged.
   - **D1** (`pilot_analysis/time_check/tc_run.py:232-250`). For a GT relation over [gb, ge), each
     track is cut to [max(gb, begin), min(ge, end)), and its vIoU is the official `viou` of the cut
     boxes against the GT subject (or object) trajectory over [gb, ge); 0 if the cut is empty. The
     pair score is the maximum over ordered track pairs s ≠ o of min(subject vIoU, object vIoU).
     Below 0.5 is **U1**. Otherwise, if the best pair's names differ from the GT triplet's, it is
     **U2**.
   - **The importable copy, pinned in A25a:** `studies/d2/stages.py:147-159` (`cut_viou`,
     `best_viou`, "D1's convention") and `studies/d2/analyse.py:142-148` (`best_pair_min`, "D1's U1
     quantity").
   - **A relation is recalled** (binding) if it is neither U1 nor U2. Class-agnostic recall (not U1)
     is reported beside it. As in D1, the pair rule (overlap ≥ 10 frames) is not part of the bar.
   - **Report-only, scoreable pairs:** the same recall, with the maximum over ordered track pairs
     taken only over pairs whose two tracks overlap ≥ 10 frames, as the pair rule computes it
     (`models/gen_labels.py:659`). Only such pairs are ever scored by the relation head. It is
     reported for SAM 3 and EOV-MMP, so the share of recalled relations that can raise the final
     score is visible.
   - **The relations:** HELD's base GT relations, `pilot_analysis/stage2/data/held_gt.json` (SHA256
     `78d55c69…4dee`, base only; the novel-predicate instances stay unused). It covers 157 of the 160
     videos, because stage 2 dropped the 3 without ECC: `…00265063`, `…00679000`,
     `ILSVRC2015_val_00123004`. For those 3, the same construction (`pilot_analysis/stage2/
     step1_parts.py:40-53`, base relations only) is applied to their training annotations, and
     the result is hashed before any G1 run.
5. **Unmatched confident track** (formerly OPEN 7): a confident track that matches no GT run under
   item 2's rule (class-aware). Real but unannotated objects, such as the distant person in G0,
   count for both trackers, so the bar stays relative. **The class-agnostic unmatched count is reported
   beside it**: a well-placed but misnamed track is unmatched in the bar, and the split shows how
   often.
6. **Box style** (formerly OPEN 8).
   - **Per pair:** for each matched SAM 3 track and GT run pair (item 2), the mean per-frame `iou`
     (`video_relation_detection_openvoc.py:48-70`) over the run's annotated frames (not its
     interpolated ones) where the track has a box.
   - **The bar** is the median over pairs.
   - GT and SAM 3 boxes are drawn on the middle annotated frame of each of the first 20 videos, for a
     visual check.
7. **Box sensitivity** (formerly OPEN 8; this reads modelC scores on HELD).
   - **Which relations:** HELD base GT relations that both SAM 3's and EOV-MMP's tracks recall
     (item 4).
   - **How:** per relation, the pair's clips are built by `gen_feats_test` over the relation's span
     three times: GT boxes, SAM 3's recalled pair cut to the span, and EOV-MMP's likewise. modelC
     runs each (`tgt_split all`, session 3's `run_pair`).
   - **Hit:** a clip is a hit if the GT predicate is among the top 5 **base** predicates of its
     score, `sigmoid(logit) × sigmoid(interactiveness)`. The hit rate is hits ÷ clips per box source.
8. **Raw [0.1, 0.5) detections for missed objects** (formerly OPEN 10; D1's raw recording).
   - **Which runs:** each GT run that no SAM 3 confident track matches (item 2, class-aware).
   - **Quantity:** the fraction of the run's annotated frames that have a raw detection of the **same
     class** with box IoU ≥ 0.5 against its GT box.
   - **Reported:** the distribution of that fraction (deciles, and the share of runs at 0,
     > 0, ≥ 0.25, ≥ 0.5, ≥ 0.75), base and novel separately, not only "at least one frame".
     G0 found raw detections dense (about 26 per frame, pre-NMS).
   - **Use:** report-only. It decides, in the G2 amendment only, between dropping Verify and
     redefining it on raw detections (review §4).

##### The cost cap

- **GPU-hours** are billed machine hours, wall clock × GPUs rented, not summed process time. With
  A2's concurrent processes, billed time is what counts.
- **The cap covers SAM 3:** billed hours from the start of the first-hour checks to the moment the
  last of the 160 videos' SAM 3 files is written. Set-up, the token fetch, EOV-MMP's HELD tracks and
  copy-back count in the budget, not in the cap.
- **Hard stop:** if the actual SAM 3 hours reach **35** before all 160 videos are done, stop and ask
  the author.
- **Projection after the first 20 videos** (frame-weighted, not ×8 of the first 20): hours spent so
  far + Σ over the 140 remaining videos of frames × 35 prompts × *r*(video). *r*(video) is the billed
  seconds per frame per prompt measured for that video's settings:
  - **videos of ≤ 1,000 frames:** the rate over the first 20, all of which are ≤ 1,000 frames (3,371
    frames, max 981), with the options that passed;
  - **the 2 videos of 1,001–2,300 frames** (`…00896000`, 1,080; `…00237001`, 1,470), which run alone
    without A1: the default single-process rate from check 0;
  - **`…00280000`** (2,947 frames), alone, with `offload_video_to_cpu`: the rate from its check.
  - If the projection exceeds **35**, stop and ask the author.
  - Also stop and ask if any of the first 20 videos fails to run (formerly OPEN 9). Every bar is
    judged on all 160, never on the first 20 alone.

##### G1's first machine hour: the checks

All on smoke video 1, `ILSVRC2015_train_00008005`. Each run is compared with
`studies/rpt/g0/sam3_compare.py` against G0's default output
`studies/rpt/g0/sam3_out/ILSVRC2015_train_00008005.json.gz` (SHA256 `aed5945d…05db4`): every prompt,
masklet, per-frame field and raw detection. **Pass means bitwise identical.**

| order | run | pass | not identical |
|---|---|---|---|
| 0 | the default run, as G0 | the machine reproduces G0 | **stop and ask**: nothing else can be judged |
| 1 | A1, backbone reuse | used for videos ≤ 1,000 frames | not used; the difference recorded |
| 2 | A2, 2 then 3 processes at once; each process's output compared; throughput measured | the largest passing count is used, within the scheduling rule | one process at a time; the difference recorded |
| 3 | A3, `torch.compile` (SAM 3's `compile=True`) | used | not used; the size of the difference reported: max \|Δ\| of `out_probs` and box coordinates, mask IoU, and masklets added or lost |
| 4 | `offload_video_to_cpu=True` | used for `…00280000` | `…00280000` is not run on this GPU; **stop before it and ask the author**. The fallback is a 48 GB GPU for that video, at defaults. |

- Each option is checked on its own against G0, in this order.
- An option that is not identical is never used; G1 continues without it, except at checks 0 and 4
  as stated.
- **Check 5, the passing options together** (second-round OPEN 3, settled by the plan author).
  - After checks 1–4, one more run on smoke video 1 with all passing options together, bitwise
    against G0.
  - On a mismatch, drop the option added last (order 1–4) and repeat. Stop when a combination is
    identical, or when none is left; then single-process defaults are used.
- The first 20 videos then run with the combination that passed check 5. Every attempt of every
  check goes in the G1 report, with its outcome, time and GPU memory.

##### Scheduling (identical output; fixed now, not tuned)

- **A1** keeps about 106 MB of CPU RAM per frame, so it is used only for videos of ≤ 1,000 frames.
  Concurrent A1 caches stay below about 150 GB in total.
- **GPU memory.** A process's predicted peak is 6.5 GiB + 6.3 MiB × frames (G0 §9: 7.3 GiB at 130
  frames, 12.8 GiB at 981). Concurrent processes are scheduled so that the predicted peaks sum to
  **under 21 GiB**.
- **Long videos run alone.** Videos over 1,000 frames run alone, without A1. Only `…00280000`, whose
  predicted peak (about 24.6 GiB) exceeds 21 GiB, uses `offload_video_to_cpu`.

##### The machine

- **One RTX 4090,** pinned to G0's stack so that check 0 does not fail on stack drift:
  - NVIDIA driver 580.142; the Vast.ai PyTorch template's `/venv/main` with Python 3.12.15 and torch
    2.11.0+cu128 (CUDA 12.8) for the EOV-MMP side;
  - SAM 3's own venv with torch 2.10.0+cu128;
  - both package sets equal to G0's `pip freeze`: `studies/rpt/g0/logs/box/pip_freeze_main.txt`
    (SHA256 `ed46e4da15a01b48c6c53cbd14fe9a5710a473e51a20b1a225c22063e6250613`) and
    `pip_freeze_sam3env.txt` (`42206283b1e237296b22d7df22d797860f5cfb2501f2bfb79d04dc491ffe58b1`).
  - **Image digest and cuDNN** (second-round OPEN 4, settled by the plan author): G0's logs did not
    record them, so they cannot be pinned. G1 rents the same Vast.ai PyTorch template as G0 and records the image digest and the
    cuDNN version at setup. Check 0, the default run reproducing G0 bitwise, decides whether the
    stack matches; if it fails, stop and ask.
- **≥ 192 GB RAM,** for A1's caches.
- If no such machine is available at about the budget's price, ask the plan author before renting
  anything else.
- No test file on the machine; the token as in A28. Nothing is pushed.

##### Budget (approved by the plan author)

- **Up to about $15 in total,** at about $0.35/h:
  - at most **35 GPU-hours** for SAM 3 (the cap);
  - about **3 GPU-hours** for EOV-MMP's HELD tracks, R0w and R0;
  - plus set-up and copy-back.

##### G1's SAM 3 outputs are kept

Every HELD video's SAM 3 file (the `sam3_runner.py` output) is kept, hashed into a manifest and
committed, or force-added if git-ignored, for reuse in G2. G2 then pays only for the loop's extra
prompts, not another full pass.

##### Values G0 found (fixed; A28, A28a, G0 report)

- **SAM 3:**
  - code `f1647e8d0c1d280991f4d1b7d974c2704537fe66`;
  - `sam3.pt` `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` (`facebook/sam3`
    revision `3c879f39…`, not SAM 3.1);
  - `build_sam3_video_predictor()`, every default; SAM 3's own multi-prompt protocol; 35 prompts, `_`
    replaced by a space.
- **Tiers:** confident cut **0.5** (SAM 3's `score_threshold_detection`); candidate floor **0.1**.
- **Track score:** the mean of `out_probs` over detected frames, which equals the starting
  detection's probability.
- **Erratum to A28.** A28's rule 2 says SAM 3's *"masklet is started only by a kept detection ≥ 0.7
  (`new_det_thresh`) that matches no existing masklet"* and that the track score *"equals the
  probability of the masklet's starting detection (≥ 0.7)"*. Rule 5 says *"at SAM 3's defaults every
  masklet starts from a detection ≥ 0.7"*.
  - **Corrected.** A detection is kept only if its probability is strictly above 0.5:
    `pos_pred_idx = torch.where(pred_probs > self.score_threshold_detection)`
    (`sam3/model/sam3_video_base.py:608`; NMS uses the same `>`, `sam3/perflib/nms.py:42`). The
    ≥ 0.7 test (`:211`) applies only when masklets already exist. On a frame with none, such as
    frame 0 of every prompt, every kept detection starts a masklet (`:1522-1535`). G0 observed
    starting scores of 0.594 and 0.668.
  - **The consequence.** A track's score equals its starting detection's probability, and masklets
    start only from detections above 0.5. So **the candidate tier [0.1, 0.5) is empty by
    construction at SAM 3's defaults**, on every video, not only on G0's 3.
  - No rule changes.
- **Gaps and duplicates:**
  - one SAM 3 ID is one track;
  - internal gaps are filled by `gen_labels.py:416-423`; no pre-roll;
  - cross-prompt duplicates at mean mask IoU > 0.7 keep the higher-scoring masklet unchanged.
- **Relation side and recording:**
  - pairs, windows and features as A28, the adapter as committed (`studies/rpt/adapter.py`,
    `relation.py`, `sam3_runner.py` at `b249c1d`);
  - raw [0.1, 0.5) detections are recorded read-only.
- **Measured on a 4090:**
  - **Speed:** 0.094 s per frame per prompt at defaults (0.09 when a prompt finds nothing, 0.12–0.18
    when it does).
  - **Memory:** 6.5 GiB + about 6.3 MiB per frame.
  - **Backbone reuse:** bitwise identical, 1.36× faster.
  - **Determinism:** a rerun is bitwise identical.
- **HELD:** 160 videos, 47,038 frames; 472 GT objects (tids), 348 base and 124 novel. The novel
  ones by class: antelope 29, cattle 23, horse 20, bear 13, watercraft 11, giant panda 10,
  motorcycle 7, fox 7, lizard 2, red panda 2. As option-B runs (item 3): 753 runs, 518 base and 235
  novel (antelope 54, bear 39, cattle 35, watercraft 23, horse 23, giant panda 20, red panda 16,
  fox 14, motorcycle 7, lizard 4). Base GT relations: 4,365 in `held_gt.json` (157 videos), plus
  those of the 3 videos built under item 4.

##### Videos

The 160 HELD videos of `pilot_analysis/stage1/train_split.json` (SHA256 `095dd89c…1f1a`; seed
20260928), sorted. All 160 are used. A bare number below is `ILSVRC2015_train_` + that number (142
videos); the last 18 are ImageNet-VID `val` videos that belong to VidVRD's training split, and are
written in full. The first 20 IDs are G1's "first 20".

```
00008005 00010024 00024005 00025029 00027003 00032000 00040000 00040005
00040030 00050006 00051015 00051016 00052001 00055002 00058003 00060001
00060005 00065002 00065003 00069006 00071013 00071014 00071019 00073037
00074010 00077006 00078000 00097003 00112000 00116002 00119013 00119025
00125006 00125007 00125013 00125014 00135004 00139004 00146005 00149004
00154002 00159010 00161016 00165000 00171001 00171004 00172007 00175001
00181011 00185000 00209003 00209005 00215000 00215006 00215010 00218002
00225001 00227012 00228000 00234001 00234007 00235006 00235007 00237001
00246002 00253025 00265004 00265042 00265063 00265066 00265068 00265074
00270000 00272005 00272034 00280000 00284001 00290008 00290019 00293004
00293005 00300008 00302021 00303008 00308017 00309001 00311002 00339001
00339003 00349000 00353000 00354000 00356001 00371013 00374000 00398004
00407001 00415007 00415008 00435001 00467000 00476002 00477000 00480000
00494000 00532000 00533003 00572000 00588000 00596000 00632001 00636001
00667002 00679000 00683000 00691000 00721000 00750000 00756000 00759000
00760001 00766000 00772004 00780000 00796000 00873000 00882000 00896000
00897005 00912002 00917000 00971000 00978000 00987000 00988001 00989000
01014000 01020000 01025001 01034001 01079000 01108001 ILSVRC2015_val_00007022 ILSVRC2015_val_00008001
ILSVRC2015_val_00020001 ILSVRC2015_val_00026003 ILSVRC2015_val_00027000 ILSVRC2015_val_00028003 ILSVRC2015_val_00035000 ILSVRC2015_val_00035008 ILSVRC2015_val_00035013 ILSVRC2015_val_00037004
ILSVRC2015_val_00046000 ILSVRC2015_val_00078000 ILSVRC2015_val_00082000 ILSVRC2015_val_00103000 ILSVRC2015_val_00123004 ILSVRC2015_val_00135000 ILSVRC2015_val_00150003 ILSVRC2015_val_00169000
```

##### Seeds

- SAM 3: `torch.manual_seed(0)`; its inference draws no random numbers, and G0's rerun was bitwise
  identical.
- The HELD split: seed 20260928 (fixed, stage 1).
- Bootstrap CIs, descriptive only, never a bar: `numpy.random.default_rng(0)`, 1,000 resamples over
  the 160 videos, the same draws for SAM 3 and EOV-MMP.
- No other randomness.

##### OPEN items

None. The second-round OPEN 1–4 of the draft are settled above: the GT unit and matching rule
(Definitions, items 2 and 3), camera compensation (item 1), the combined-options check (check 5)
and the stack record (the machine).

**Not changed by this amendment:** anything in A28 and A28a other than the erratum, and the two
changes listed under "Changed from the proposal". Any change to the above is a new amendment,
written before the result it affects.

#### A29a (2026-10-28): G1 on one multi-GPU machine — the machine, the driver pin, parallel checks, longest-first scheduling, a 48 GPU-hour cap

Written after A29 (chain head `b0f1ec509d3b77d03c58db8f98e63f6e3df480fa34c4e7dbb2447225c8552fd0`) and
before any G1 run. No G1 data exists. It changes how G1 runs, not what G1 measures: every bar,
definition and pass rule of A29 stands.

**Reasons.**
- No RTX 4090 offered on Vast.ai on 2026-10-28 has G0's driver 580.142 (the offers had 565, 580.82,
  580.126, 580.159, 580.173, 580.178, 590, 595, 610 and 615). So A29's driver pin cannot be met.
- The author chose one machine with several RTX 4090s, to cut G1's wall clock from about 1.5 days on
  one GPU to about 4-6 hours, at a moderately higher cost.
- On several GPUs, the long-video tail and the checks leave GPUs idle but billed. So A29's cap of 35
  GPU-hours, set for one GPU, no longer fits.

##### The machine (replaces A29's "One RTX 4090" and "≥ 192 GB RAM")

- **8 RTX 4090s on one machine, with ≥ 256 GB RAM;** if none is available, **4 RTX 4090s with ≥ 192 GB
  RAM**. Never another GPU model (no RTX 5090, no other generation): check 0 must reproduce G0 bitwise.
- **Both:** disk ≥ 200 GB, CUDA ≥ 12.8, a verified host, reliability ≥ 0.98, download ≥ 200 Mbps.
- **The image:** Vast.ai's own PyTorch template, as G0's (G0: `/venv/main`, Ubuntu 24.04.5, Python
  3.12.15, torch 2.11.0+cu128). G0 did not record the template's tag or the image digest. The
  author copies G1's digest from the instance page; setup records the OS, the driver and both cuDNN
  versions.
- **The driver pin is relaxed.** Any driver is accepted and recorded. Check 0 decides whether the
  stack matches, as A29 already says for the image and cuDNN. Python 3.12.15, torch, CUDA 12.8 and
  both package sets stay pinned to G0's `pip freeze` (setup stops on any difference).

##### The checks (adds to A29's "first machine hour")

- **Check 0 on every GPU, at once.** Each GPU reruns smoke video 1 at defaults. Every GPU must
  reproduce G0 bitwise; if any GPU does not, stop and ask.
- **Checks 1-4 at once, each on its own GPU, each against G0:** A1 on GPU 0; A2 on GPU 1 (2
  processes, then 3 if 2 passed); A3 on GPU 2; offload on GPU 3.
- **Then check 5** on GPU 0, as A29 says (all passing options together; on a mismatch, drop the option
  added last and repeat). It runs with the OpenMP thread count the videos will use, so that count is
  covered by a bitwise check too.
- **Threads:** each SAM 3 process gets `OMP_NUM_THREADS` = CPU cores ÷ concurrent processes, at most 8.

##### Scheduling (replaces A29's order; A29's per-GPU rules stand)

- **Longest first.**
  - `…00280000` (2,947 frames), with offload if it passed, `…00237001` (1,470) and `…00896000`
    (1,080) start at once, each alone on its own GPU, without A1. A GPU running one of them alone
    has one SAM 3 process; when the video ends, the GPU gets A2's other processes.
  - The first 20 videos (A29's) run, longest first, on the other GPUs, and on any GPU that frees up.
  - When the first 20 are done, the projection is made; then the other 137 run, longest first.
- **Placement.** A video starts on the GPU, among those it fits on, with the fewest running
  processes (ties: the lowest index). If the longest queued video fits on no GPU, the next-longest
  one that fits starts, so no GPU waits while a shorter video could run.
- **Per GPU, A29's rules:** the predicted peaks (6.5 GiB + 6.3 MiB × frames) of the processes on one
  GPU sum to under 21 GiB, and A2's process count is used only if A2 passed.
- **RAM, machine-wide** (replaces A29's "about 150 GB"). The A1 caches (106 MB per frame) of all
  running processes together stay at most the instance's RAM − 64 GB. The RAM is measured at the
  start: `/proc/meminfo`, or the container's cgroup limit if lower. A video whose A1 cache would
  exceed the limit runs without A1, with the same output.
- **Without offload:** `…00280000` is not started; the other 159 run; then stop and ask (A29).
- **Any video that fails to run stops G1** (confirmed by the plan author).

##### The cost cap (replaces A29's 35 GPU-hours)

- **The SAM 3 phase is capped at 48 billed GPU-hours** = billed wall clock × GPUs rented, from the
  start of check 0 to the moment the last of the 160 SAM 3 files is written. That is about 6 hours
  of wall clock on 8 GPUs, or 12 on 4. Set-up, the tests, the preflight, the token fetch, the
  EOV-MMP side and copy-back count in the budget, not in the cap.
- **Hard stop:** at 48 billed GPU-hours, stop and ask.
- **Projection after the first 20:** the remaining schedule is simulated with the scheduler's own
  placement rule, including the videos still running.
  - Each video's duration is frames × 35 × *r*, frame-weighted as in A29, with *r* the measured
    seconds per frame per prompt of one SAM 3 process:
    - for videos ≤ 1,000 frames, over the first 20 (summed process time ÷ summed frames × 35);
    - for `…00237001` and `…00896000`, check 0's (its slowest GPU);
    - for `…00280000`, check 4's.
  - A running video's elapsed time is subtracted.
  - Projected billed GPU-hours = GPUs × (wall clock so far + simulated remaining wall clock).
  - If above 48, stop and ask.

##### The EOV-MMP side (adds to A29's Definitions items 1 and 7)

- R0w, R0 and box sensitivity are split by video across the GPUs, one `eov_held.py` process per
  GPU, after all 160 SAM 3 files exist. Videos are assigned longest first to the GPU with the fewest
  frames so far.
- **A video's result must not depend on its GPU.** Before check 0, the 2 preflight videos
  (`…00010024`, `…00008005`, with G0's SAM 3 files) run on GPU 0 and on GPU 1. Their outputs must be
  identical, apart from the timing fields; otherwise, stop and ask.

##### Before check 0, on the machine

The RPT unit tests run on the machine's CPU, then the EOV-MMP preflight. Both are set-up (budget,
not cap).

##### Budget (replaces A29's "up to about $15")

- **Up to $35 in total,** including set-up, the EOV-MMP side and copy-back.
- **Price limits:** 8× RTX 4090 at most $0.50 per GPU-hour ($4.00/h for the machine); 4× at most
  $0.55 per GPU-hour ($2.20/h). Above these, ask the plan author.

**Implementation:** `studies/rpt/g1/schedule.py`, `studies/rpt/g1/box/g1_driver.py`,
`studies/rpt/g1/box/eov_parallel.py` and `studies/rpt/g1/box/setup.sh`. The commit that runs is
recorded on the machine (`~/rpt_g1_box/commits.txt`) and in the G1 report.

**Not changed by this amendment:** everything else in A29 (the bars, the definitions, the checks'
pass rules and order, offload only for `…00280000`, A1 only for videos ≤ 1,000 frames, the stop
conditions, the seeds, keeping and hashing every SAM 3 output). Any change to the above is a new
amendment, written before the result it affects.

#### A29b (2026-10-28): G1, stop 1 — `…00280000` on one Ada-generation 48 GB GPU (A29's fallback)

Written after G1's stop 1 and the plan author's decisions on it, before `…00280000` has any G1
output. When written, 11 of the first 20 videos' SAM 3 files existed and the checks' outcomes were
known (`studies/rpt/g1/g1_stop_report_1.md`). No bar quantity has been computed.

**The stop and its cause.**
- On the 8× RTX 4090 machine of A29a, `ILSVRC2015_train_00280000` (2,947 frames) ran alone with
  `offload_video_to_cpu`, the configuration check 5 had passed. It failed with CUDA out of memory
  (22.79 GiB allocated, on a card with 23.54 GiB); its first prompt had already peaked at 22.19 GiB.
- Offload moves only the video frames to the CPU; SAM 3's per-frame state, which grows with the
  video's length, stays on the GPU. Check 4 had passed on the 130-frame smoke video.
- A29a's rule "any video that fails to run stops G1" fired.

**The decisions** (the plan author, after the stop).
- **D1.** The other 159 videos run on the 8× machine, with `…00280000` excluded by an explicit,
  logged list. The 11 finished videos are kept. On the 8× machine, the EOV-MMP side computes:
  - R0w and R0 tracks for all 160 videos (they do not need SAM 3's files);
  - box sensitivity for the 159 videos with SAM 3 files.

  The 8× machine's EOV-MMP preflight outputs (the 2 smoke videos) are kept for the comparison below.
- **D3.** The time spent waiting for these decisions counts toward A29a's 48 GPU-hour cap, as
  defined. No change.
- **D2.** `…00280000` takes A29's pre-registered fallback ("a 48 GB GPU for that video, at defaults"),
  restricted as below. It runs as a separate rental, after the 8× machine is destroyed.

##### The 48 GB machine

- **GPU:** one 48 GB GPU of the Ada generation, the RTX 4090's: RTX 6000 Ada, L40S or L40. Not Ampere
  (A6000, A40) or any other generation.
- **Software:** the same stack as G1's 8× machine: Python 3.12.15, both package sets equal to G0's
  `pip freeze`, EOV-MMP's operator built with nvcc 12.8 as there; the driver is recorded.
- **Check 0 is required:** smoke video 1 at defaults, bitwise against G0's output (A29's check 0).
  **If check 0 fails, stop and ask.**
- **`…00280000`:** one process, SAM 3's defaults, no offload, no A1, no A2, no A3.
- **Box sensitivity for `…00280000`** runs there only if that machine's EOV-MMP preflight on the 2
  smoke videos is bitwise identical (apart from the timing fields) to the 8× machine's. Otherwise,
  box sensitivity is judged on 159 videos, and this is disclosed.
- **The R0w and R0 tracks used for every bar** are the 8× machine's, for all 160 videos.
- **Cost:** within A29a's total budget of $35. Above it, ask the plan author.

##### What G1 then reports

Every A29 quantity on all 160 videos, with SAM 3's file for `…00280000` from the 48 GB machine.
Box sensitivity is on 160 or 159 videos, as above. Which machine produced each SAM 3 file is
reported.

**Not changed by this amendment:** everything else in A29 and A29a. Any change to the above is a
new amendment, written before the result it affects.

#### A29c (2026-10-28): G1, stop 2 — A2 off, a measured A1 RAM rule, and out-of-memory failures moved to the 48 GB run

Written after G1's stop 2 and the plan author's decisions on it (`studies/rpt/g1/g1_stop_report_2.md`),
before the rest of D1 runs.
- **Already seen when written:** the 20 videos of the first 20 have SAM 3 files on the 8× machine,
  and A29a's projection after them was made: 22.2 projected billed GPU-hours, below 48.
- **Not computed:** no bar quantity.

**Stop 2 and its cause.**
- **The failure.** At 16:30:52 UTC, on the 8× machine, the kernel killed the SAM 3 process running
  `ILSVRC2015_val_00027000`: the memory cgroup recorded `oom_kill`. System RAM was exhausted. A29a's
  rule "any video that fails to run stops G1" fired.
- **The cause.** A29a's A1 rule (all A1 caches ≤ RAM − 64 GB, at 106 MB per frame) was too loose:
  - the caches measure **112 MB per frame** (all 20 finished videos: 111.6–112.3);
  - each of the 16 SAM 3 processes (A2: 2 per GPU) needs about 4 GB of its own.
- **Lost:** the videos in progress, including `…00237001` and `…00896000`. They run again from the
  start.

**The decisions** (the plan author), for the rest of D1 on the 8× machine:

1. **A2 off.** One SAM 3 process per GPU: 8 processes.
   - Reason: A2 measured only 1.04× in check 2, and it doubles the per-process RAM.
   - With one process per GPU and A1, each process runs exactly check 1's configuration, which was
     bitwise identical to G0 at the same OpenMP thread count (8).
2. **The A1 RAM rule uses the measured 112 MB per frame.**
   - **Total:** all A1 caches together ≤ instance RAM − 64 GB − 8 GB × the number of SAM 3 processes.
     Here: 518.9 − 64 − 8 × 8 = 390.9 GB.
   - **Live guard:** an A1 start also needs the available memory minus its own cache to stay ≥ 64 GB
     at that moment. The available memory is the lower of `/proc/meminfo`'s MemAvailable and the
     container's memory-cgroup limit minus its usage, with the inactive file cache counted as free.
   - A video failing either test runs without A1, with the same output.
3. **Out-of-memory failures no longer stop G1.**
   - **What counts:** a video that fails only from GPU or system out-of-memory, i.e. a CUDA
     out-of-memory error, or a kernel `oom_kill` of its process, seen as a rise of the cgroup's
     `oom_kill` count when the process dies.
   - **What happens to it:**
     - it is logged, and its partial output discarded;
     - it is **added to the 48 GB run of A29b**, under the same rules as `…00280000` (one process,
       defaults, no offload, A1, A2 or A3; check 0 required there);
     - a killed process is replaced on its GPU.
   - G1 continues with the other videos.
   - **Still stop and ask:**
     - on any other failure;
     - if more than 3 videos fail from out-of-memory in total (`…00280000`, the first, is not counted
       among these 3);
     - at the 48 GPU-hour hard stop.

**The projection is not repeated;** the hard stop at 48 GPU-hours protects the cap.

**After the SAM 3 phase** (as before):
- the EOV-MMP side: R0w and R0 for all 160 videos; box sensitivity for the videos with SAM 3 files
  on the 8× machine;
- then copy-back.

The videos moved to the 48 GB run are reported. Their box sensitivity follows A29b's rule for
`…00280000`.

**Not changed by this amendment:** everything else in A29, A29a and A29b. Any change to the above is a
new amendment, written before the result it affects.

#### A29d (2026-10-29): G1 — an offload erratum, and a futility rule before the 48 GB run

Written after the 8× machine's copy-back (`studies/rpt/g1/g1_8x_done_report.md`) and the plan
author's decisions on it, before the 48 GB run.
- **Already seen when written:** 157 of the 160 HELD videos have SAM 3 files; all 160 have EOV-MMP
  files (R0w and R0; box sensitivity for the 157). All are copied back and hashed (commit
  `6ea77c7`). Moved to the 48 GB run under A29c: `ILSVRC2015_train_00225001` and
  `ILSVRC2015_train_00971000`.
- **Not computed:** no bar quantity, on any set of videos. The only counts taken so far are from the
  ground truth: the 3 videos without SAM 3 files hold 39 of the 753 GT runs and 156 of the 4,390 HELD
  base GT relations.

##### 1. Erratum: SAM 3's video offload is a no-op

- **The finding.** SAM 3's `offload_video_to_cpu` has no effect in this code.
  - The request reaches SAM 3, and its frame loader keeps the frames on the CPU.
  - But the video model then copies every frame to the GPU regardless: `_construct_initial_input_batch`
    builds the input batch from all frames and moves it to the GPU with `copy_data_to_device`
    (`sam3/model/sam3_video_inference.py:145-153`, SAM 3 commit `f1647e8d`).
  - The evidence: check 4's peak allocated GPU memory with offload equals check 0's without it,
    7.316072 GiB.
- **So:**
  - The cause given in stop report 1 and in A29b ("Offload moves only the video frames to the CPU;
    SAM 3's per-frame state, which grows with the video's length, stays on the GPU") is wrong.
    `…00280000` failed with its frames on the GPU.
  - Check 4's "pass" means only that offload is a no-op. **Offload is not a validated option. It is
    not carried into G2 or the test session.**
- **No rule and no result changes.** `…00280000` was the only video that asked for offload, and it
  produced no file on the 8× machine.
- **A memory observation, for the test-session amendment.**
  - `…00225001` (836 frames) and `…00971000` (521 frames) ran out of GPU memory at 21.46 and
    22.42 GiB allocated.
  - G0's formula (6.5 GiB + 6.3 MiB × frames) gives them 11.6 and 9.7 GiB.
  - GPU memory depends on the number of objects SAM 3 tracks, not only on the frame count.
  - No rule in G1 changes.

##### 2. The futility rule, before the 48 GB run

- **First,** every binding bar of A29 is computed on the 157 videos with SAM 3 files, SAM 3 against
  EOV-MMP (R0w) on the same 157 videos.
- **Then each binding bar's best possible value for SAM 3 over all 160 videos.**
  - SAM 3 takes its best case on the 3 videos without SAM 3 files (`…00280000`, `…00225001`,
    `…00971000`).
  - EOV-MMP takes its real values on all 160 videos, from the 8× machine (R0w).

  | binding bar | SAM 3's best case on the 3 videos | the bound, against A29's threshold |
  |---|---|---|
  | Overall recall | all 39 of their GT runs count as matched | (SAM 3's matched runs on the 157 + 39) ÷ 753, minus EOV-MMP's fraction of the 753 |
  | Pair recall | all 156 of their base relations count as recalled | (SAM 3's recalled relations on the 157 + 156) ÷ 4,390, minus EOV-MMP's fraction of the 4,390 |
  | False-track burden | SAM 3 has 0 unmatched confident tracks on them | (SAM 3's unmatched tracks on the 157 ÷ 160), divided by EOV-MMP's mean over the 160 |
  | Box style | one extra pair at IoU 1.0 per GT run, 39 in all, the most possible (each run is matched by at most one track) | the median over SAM 3's pairs on the 157 plus those 39 |

  Box sensitivity is secondary and not part of this rule.
- **If any binding bar fails even at its best case,** G1 fails for certain.
  - The 3 videos are not run, and nothing is rented.
  - `studies/rpt/g1_report.md` gives the verdict and this rule's computation.
- **Otherwise,** the plan author sees the outcome and decides on the 48 GB run (A29b, A29c).
  - If it runs, every bar is judged on all 160 videos as A29 says.
  - The 157-video numbers are not used for any decision.
- **Either way,** the 157-video numbers and the bounds are reported in `g1_report.md`.

**Not changed by this amendment:** everything else in A29, A29a, A29b and A29c. Any change to the
above is a new amendment, written before the result it affects.

#### A30 (2026-11-02): post-G1 Task 2 — the SAM 3 screen on HELD (the last chance for SAM 3 as a detector)

Written before the screen runs, from the agreed post-G1 plan (`studies/postg1/monday-pack.md`, Part 1,
Task 2). **Exploration on HELD, in-sample for EOV-MMP; not a test result.** The trimming oracle
(Task 1) is not hashed yet and takes the next free label.
- **Already seen when written:** G1's bars and descriptive numbers on these 157 videos
  (`studies/rpt/g1_report.md`), and the 20 inspected tracks
  (`studies/rpt/g1/viz_unmatched_observations.md`, SHA256 `b26f1976…6667`).
- **Not computed:** no rule's effect, at any setting.

##### Data

- **Videos:** the 157 HELD videos with G1 SAM 3 files.
- **SAM 3:** G1's confident tracks (`studies/rpt/g1/sam3_out/`, manifest
  `studies/rpt/g1/logs/box/sam3_manifest.jsonl`, SHA256 `9d508db4…a45d`), read by G1's adapter.
- **EOV-MMP:** G1's R0w tracks (`studies/rpt/g1/eov_out/`) on the same 157 videos.
- **Scoring:** A29's matching, pair recall (D1's best pair) and unmatched-track count, from
  `studies/rpt/g1/metrics.py` (SHA256 `359c3d9a…dbf8`), re-run on the kept tracks.
- **Code:** `studies/postg1/sam3_screen.py` (SHA256 `b20de27d…721c`), tested by
  `studies/postg1/tests/test_sam3_screen.py` (`17cbc00d…132b`). The scorer must equal `metrics.py`
  on the no-rule configuration, the full configuration and 25 configurations drawn with
  `numpy.random.default_rng(0)`, or the run stops.

##### Boxes and how rules combine

- **"Box":** the confident track's per-frame box as G1 used it: the tight box of SAM 3's mask, with
  gap frames interpolated and rounded by the repo's `round_and_positive`.
  - Coordinates are integer pixel indices.
  - Width is x2 − x1 + 1, height y2 − y1 + 1 (the official `iou`'s convention); area is
    width × height.
  - A box's centre is its midpoint, and its diagonal is √(width² + height²).
- **Every rule is judged on the video's full set of confident tracks,** before any removal. A
  configuration removes the union of the tracks its rules flag.

##### The rules (each removes whole SAM 3 confident tracks)

1. **Tiny:** remove if min(median box width, median box height) < t px; t ∈ {8, 12, 16, 24}.
2. **Edge sliver:** remove if the box touches the left or right frame edge on ≥ f of the track's
   frames **and** its median width < w px; (f, w) ∈ {0.5, 0.8} × {16, 32}.
   - "Touches" means x1 ≤ 2 or x2 ≥ W − 3, where W is the frame width in SAM 3's file and pixel
     indices run 0 to W − 1. That is, within 2 px of the edge.
3. **Fragment:** remove if, on ≥ f of the frames it shares with some larger confident SAM 3 track,
   ≥ c of its box area lies inside that track's box; (c, f) ∈ {0.8, 0.9} × {0.5, 0.8}.
   - The larger track may be of any class. "Larger" means a strictly larger median box area.
   - The inside share on a frame is the intersection area ÷ the track's own box area.
4. **Score cut (reference only, never selected or combined):** remove if track score < s;
   s ∈ {0.6, 0.7, 0.8, 0.9}. The score is G1's track score.
5. **Geometry-only partner:** keep only if some other confident SAM 3 track overlaps it in time by
   ≥ 10 frames and, on ≥ half of those frames, their box centres are within d × the larger box's
   diagonal; d ∈ {1, 2, 3}.
   - "The larger box's diagonal" is the larger of the two boxes' diagonals on that frame.
6. **modelC partner: skipped.**
   - G1 cached modelC scores only for SAM 3 pairs that recall a GT relation (A29 item 7, box
     sensitivity), not for every pair of confident tracks.
   - No features are computed.
7. **Class gate:** keep only if EOV-MMP's R0w tracks contain its class in the same video. One
   setting.

##### The grid, the criterion and the decision

- **The grid (full search):** each of rules 1, 2, 3, 5 and 7 is off or at one of its settings:
  5 × 5 × 5 × 4 × 2 = **1,000 configurations**. Rules 4 and 6 are computed alone and reported,
  never combined (rule 6 is skipped, as above).
- **Criterion ("worth validating"):** a configuration passes if, on the 157 videos against EOV-MMP
  R0w, all three hold:
  - overall recall gain ≥ +8 points;
  - pair recall gain ≥ +8 points;
  - false-track burden ≤ 1.35×.
- **The class gate is required.** A pass counts only with rule 7 on.
  - Passes without rule 7 are reported descriptively. Option 2 would then need all 35 prompts at G1's
    cost.
  - The screen's gate uses R0w tracks (score ≥ 0.35), stricter than Option 2's planned ≥ 0.1 gate, so
    the screen is conservative here.
- **The selection rule.** Among passing configurations with rule 7 on, the one that goes forward has
  the highest min(recall gain, pair-recall gain).
  - Ties go to the lower burden, then to fewer rules on, then to the earlier configuration in grid
    order.
  - No other configuration is carried forward.
- **Fragility:** if only one or two configurations pass with rule 7 on, the pass is flagged as
  fragile.
- **Decision (binding):**
  - **Pass:** the selected configuration becomes Option 2, to be validated later on a fresh FIT
    subset.
  - **Fail:** **SAM 3 closes as a detector**, with no second round of rules on HELD. SAM 3's tracker
    remains a candidate for Option 1.

##### Report

`studies/postg1/sam3_screen_report.md`, verdict first, with:
- the configurations tried, and the passes with and without rule 7;
- the trade-off between burden and the recall gains (the Pareto front);
- for each single rule at each setting: recall, pair recall, burden, matched tracks lost and
  unmatched tracks removed;
- rule 4 alone, and the reason rule 6 is skipped;
- the class gate's removals: novel against base class, matched against unmatched;
- which of the 20 inspected tracks the selected configuration removes, and which each single rule's
  settings remove.

**Not changed by this amendment:** everything in A29–A29d. Any change to the above is a new
amendment, written before the result it affects.

#### A31 (2026-11-04): PaD — the HELD split (HELD-A, HELD-B), HELD-B's gates and gate 3, the extracts and the HELD-B guard, FIT annotation counts

Written before the split is built, from `studies/postg1/d7-session-prompt.md` (v2) and the planning
chat's reply to the developer review (`studies/postg1/dev-review-reply-1.md`). Giang's decisions
recorded there: Task 1 (the trimming oracle) is not run; A30's scope is option (a), written in the
next amendment.
- **Already seen when written:** G1's results on the 157 HELD videos with SAM 3 files
  (`studies/rpt/g1_report.md`), A29d's futility computation, the descriptive analysis and its 20
  inspected tracks, and A30's screen. All of them used every HELD video, so both halves defined
  below. Also the structure of `train_object_trajectories_gt.json`: each tid is stored as contiguous
  fragments, and 905 of the 1,958 FIT tids have more than one.
- **Not computed:** the flags, the median, the split, and any count per half or on FIT.

##### Phase names and gates (fixed)

| phase | what it is | at test time |
|---|---|---|
| Phase 1, Prompt | the main method: a learned class-free prompt for SAM 3 (P1), learned keep/drop (P3), naming by modelB (P4), EOV-MMP's relation classifier with T1 (P5) | SAM 3 |
| Phase 2, Filtered Distill | EOV-MMP's detector learns from Phase 1's boxes, cleaned by P3 | EOV-MMP's detector |
| Phase 3, Split Distill | box-only distillation plus gap fill by SAM 3's tracker; EOV-MMP's own score judges junk, no P3 | EOV-MMP's detector |
| Phase 4, Improve | further ideas, only if needed | — |

Gates 0–3 belong to Phase 1, gate 4 to Phase 2, gate 5 to Phase 3.

##### 1. The split

- **Videos:** the 160 HELD videos of `pilot_analysis/stage1/train_split.json` (SHA256 `095dd89c…1f1a`).
  142 have `ILSVRC2015_train_` IDs and 18 have `ILSVRC2015_val_` IDs.
- **Flags:**
  - **novel:** the video's annotation file (`data/vidvrd/anno/train/<video>.json`) lists in
    `subject/objects` at least one object whose category is novel in
    `data/vidvrd/data/openvoc_obj_class_spilt_info.json` (`cls2split`). This reads all 160 HELD
    annotation files. It happens before the guard is installed.
  - **long:** the number of entries in the video's frame folder (`data/vidvrd/frames/<video>/`) is
    strictly above `numpy.median` of the 160 counts.
- **Strata, in this fixed order:** (novel, long), (novel, not long), (not novel, long),
  (not novel, not long).
- **Procedure:**
  - one generator, `numpy.random.default_rng(20261102)`;
  - for each stratum in that order: its video IDs sorted (Python `sorted` on the strings), then
    `rng.permutation` of that list;
  - the four permuted lists concatenated in stratum order;
  - A, B, A, B… assigned along the whole list, starting with A. The alternation does not restart per
    stratum, so the split is exactly 80/80.
- **Recorded:** the halves of the 3 videos without SAM 3 files (`…00280000`, `…00225001`, `…00971000`).
- **Output:** `studies/d7/held_split.json` (the two lists, the flags, the frame counts, the median, the
  strata, the input hashes), and its SHA256 in `studies/d7/held_split.json.sha256`.
- **Code:** `studies/d7/make_split.py`; the test-label guard is installed first.

##### 2. The roles

- **HELD-A:** prompt selection, the naming check, and P3's training (cross-validated by video inside
  HELD-A).
- **HELD-B: sealed.** It serves at most three pre-registered, tracker-level gates, with nothing ever
  tuned on it:
  - **gate 3:** the Phase 1 (Prompt) system against EOV-MMP R0w;
  - **only if gate 3 passes, gate 4:** the Phase 2 (Filtered Distill) student against R2;
  - **only if gate 3 passes, gate 5:** the Phase 3 (Split Distill) student against R2.
  - Each gate gets its own amendment before it runs. All uses are disclosed. No other use is allowed.

##### 3. Gate 3, fixed now

A later amendment may only make it stricter.
- **Tracker-level, not mAP,** since modelC is in-sample on HELD.
- **G1's matching** (A29's rules), **class-aware**, against EOV-MMP R0w on HELD-B. All three must
  hold:
  - overall recall gain ≥ +8 points;
  - pair recall gain ≥ +8 points;
  - burden ≤ 1.35×.
- Class-agnostic numbers are reported.
- **No skips:** gate 3 runs every HELD-B video, on a 48 GB GPU planned in advance.

**P3 follows option (b).** SAM 3 tracks that match a novel GT object are unlabelled like any other
unmatched track, and are never excluded. Novel GT is used only to report suppression.

##### 4. Disclosure

- **HELD-B is sealed from now on, not unseen.** These all used it: G1, A29d's futility rule, the
  descriptive analysis (including the 20 inspected tracks) and A30's screen.
- **P3's starting rule family** (tiny, edge sliver, fragment) was seen on all of HELD.

##### 5. The extracts, then the guard

**What counts as an aggregate here.** A file is an aggregate if it holds HELD-B ground truth or
HELD-B system output for more than one video. The repo was searched for every file that mentions a
HELD video ID, both ID forms, inside gzip files too.

**(a) Data aggregates: extracts, then blocked.**
- **Extracts:** a FIT-only extract where the file holds FIT videos, and a HELD-A-only extract where it
  holds HELD-A videos. Each keeps the file's structure, restricted to the subset's video keys, in
  `studies/d7/extracts/`. Its SHA256 goes in `studies/d7/extracts/SHA256SUMS`, and the source's
  SHA256 in `studies/d7/extracts/manifest.json`.
- **Ground truth, keyed by video:**
  - `data/vidvrd/data/train_object_trajectories_gt.json` (FIT, HELD-A);
  - `pilot_analysis/stage1c/gt_tracks_train.json` and `gt_tracks_train_optionA.json` (FIT, HELD-A);
  - `pilot_analysis/stage2/data/held_gt.json` and `held_pairs.json` (HELD-A);
  - `studies/rpt/g1/held_gt_missing3.json` and `studies/rpt/g0/held_gt_smoke3.json` (HELD-A);
  - the 5-video pilot subsets (HELD-A): `pilot_analysis/stage2/data_small/held_gt.json` and
    `held_pairs.json`, `pilot_analysis/stage2/data/held_gt_small5.json`,
    `pilot_analysis/boundary/data/held_gt_small.json`, `pilot_analysis/f1/data_small/held_gt_small.json`.
- **Results files:** `studies/rpt/g1/results_futility/futility.json`,
  `studies/rpt/g1/results_descriptive/descriptive.json`, `studies/postg1/results_screen/sam3_screen.json`.
  - Their HELD-A extract keeps only the per-video entries of HELD-A videos: the `files` hashes, and
    for the last two the `viz_sample` and `inspected_20` entries of HELD-A videos.
  - Their pooled numbers mix both halves, so they are left out.
- **Images:** `studies/rpt/g1/viz_unmatched/`. The HELD-A images are copied to
  `studies/d7/extracts/viz_unmatched.held_a/`.
- **Manifests later code needs for hash checks:** `studies/rpt/g1/logs/box/sam3_manifest.jsonl` and
  `studies/rpt/g1/eov_out/manifest_g*.jsonl`. Their HELD-A extract keeps the lines of HELD-A videos.

**(b) Blocked without an extract.** Each holds HELD-B data that cannot be split by video, or is a run
record that no later code reads:
- **Values without video keys:** `studies/rpt/g1/results_futility/box_style_values_157.json` (555
  values pooled over 157 videos).
- **Pilot data directories:** these hold HELD ground truth and HELD results as aligned arrays without
  video keys (`slots.npz`, `pos_meta.json`, `*.npy`, `*.pt`). All of `pilot_analysis/*/data/` and
  `pilot_analysis/*/data_small/`, and `pilot_analysis/stage1c/task2_results/`.
- **Run records:**
  - `studies/rpt/g0/logs/`, `studies/rpt/g0/box_out/`, `studies/rpt/g0/dryrun_laptop/`;
  - `studies/rpt/g1/logs/`, `studies/rpt/g1/box_out/`, `studies/rpt/g1/eov_out/run_info_g*.json`;
  - `studies/postg1/logs/sam3_screen_run.log`;
  - logs, per-run manifests, schedules and driver output with per-video lines.

**(c) Not blocked, because they hold no HELD-B labels or outputs:**
- prose and committed records: reports, `preregistration.md` and its backups, drafts;
- code and tests that name a few video IDs;
- ID lists and schedules (`train_split.json`, `gt_tracks_report.json`, `step0_selection.json`);
- frame-derived inputs: frames, videos, `VidVRD_ECC_train.json`, the class-level CLIP bank.

**The guard** (`studies/d7/held_b_guard.py`) is an audit hook like the test-label guard. It reads the
HELD-B list from `held_split.json`, after checking its SHA256. From then on, for the life of the
process, opening any of these raises `PermissionError`:
- **per-video files:** any path that contains a HELD-B video ID, of either form, except under
  `data/vidvrd/frames/` and `data/vidvrd/videos/`. This covers the annotation files, G0's and G1's
  SAM 3 outputs, EOV-MMP's outputs and any other per-video derivative;
- **every path in (a) and (b),** by file or by directory prefix.

The guard covers Python processes that install it. Every script of this session installs both
guards first. Shell tools are not covered, so no shell command reads a blocked path.

**Committed:** `held_split.json` and its hash, the extracts with `SHA256SUMS` and `manifest.json`, the
guard and its tests (`studies/d7/tests/test_held_b_guard.py`).

##### 6. FIT annotation counts (report only)

Computed from the FIT-only extract of `train_object_trajectories_gt.json`; nothing is decided by them.
- **Sampled frames:** frame indices f with f mod 30 = 0, from 0 to the frame-folder count − 1.
- **A track** is a (video, tid). Its fragments are merged. Its span runs from its first to its last
  annotated frame.
- **Reported:**
  - the share of sampled frames with no GT box: over all sampled frames, and over the sampled frames
    inside the union of the video's track spans;
  - the share of sampled (frame, track) pairs, with the frame inside the track's span, where the track
    has no box on that frame (inside an annotation gap);
  - the distribution of track coverage (annotated frames ÷ span length): quantiles, the share under
    65%, and the share with full coverage.
- **Output:** `studies/d7/fit_counts.json`, from `studies/d7/fit_counts.py`.

**Not changed by this amendment:** anything in A28–A30. Any change to the above is a new amendment,
written before the result it affects.

#### A32 (2026-11-04): A30's scope — its verdict stands for the system it tested; PaD Phase 1 is a system A30 did not test (written after A30's result)

Written after A30's result (`studies/postg1/sam3_screen_report.md`, commit `b04b714`), from Giang's
decision 2 in `studies/postg1/d7-session-prompt.md` (v2): **option (a)**. The alternative, option (b),
read A30's words literally ("SAM 3 closes as a detector") and would have ended PaD Phase 1. That
option is recorded here as not taken. **Nothing is run under this amendment, and Task 2 is not
re-run.**
- **Already seen when written:** A30's full result on the 157 HELD videos with SAM 3 files.
  - 0 of 500 gated configurations pass.
  - The class gate alone gives recall +4.90 (from +9.80), pair recall +7.42 (from +9.16) and burden
    0.97× (from 1.84×).
  - 16 configurations pass without the gate.
  - Also A31's split.

##### 1. A30's verdict stands for the system it tested

- **That system:** the 35-name SAM 3 (one pass per class name, at G1's settings), with A30's fixed
  hand rules and the class gate (rule 7: a track is kept only if EOV-MMP's R0w tracks contain its
  class in the same video).
- **Closed by A30, unchanged:** Option 2 (that system as the detector), and C6b (that system's boxes
  as a teacher for distillation). Neither is reopened by any later amendment.
- **A30's statement that SAM 3's tracker remains a candidate for Option 1** is unchanged.

##### 2. PaD Phase 1 is a system A30 did not test

| | A30's system | PaD Phase 1 |
|---|---|---|
| prompt | the 35 class names, one pass each | one learned class-free prompt, trained on FIT |
| keep/drop | fixed hand rules and the class gate | a learned keep/drop model (P3) |
| names | the prompt's class name | EOV-MMP's auxiliary classifier, modelB (P4) |
| class gate | required (rule 7) | **none** |

- **Where A30's failure came from:** the class gate. It cut the recall gain from +9.80 to +4.90
  points, below the +8 bar. Every other rule only removes tracks. The gate is defined by EOV-MMP's
  own class list in each video, and PaD has no such gate.
- **So A30's "SAM 3 closes as a detector" is read as applying to the system it tested.**
- **This reading was written after A30's result.** The thesis says so in the method chapter, not in a
  footnote, and states the order of events:
  1. A30's binding words;
  2. its fail;
  3. this scope statement;
  4. PaD's own gates (A31 and later).

##### 3. P3 and A30's rules

- **P3 is a learned model.** It is fitted on P1's own HELD-A tracks, cross-validated by video inside
  HELD-A, and judged on sealed HELD-B at gate 3 (A31). It is not a second round of A30's rules.
- **Its rule family is reused, and its thresholds are refitted.** The geometric rules (tiny, edge
  sliver, fragment) are P3's starting family. No threshold chosen on the 35-name tracks is carried
  over, and none was selected by A30, since nothing passed.
- **The family was seen on all of HELD,** HELD-B included (A31 §4).
- **The rules' effect is already reported** in `sam3_screen_report.md`. It is P3's descriptive starting
  point, and nothing new is computed for it.

##### 4. Gate 0

PaD gate 0 is recorded as **"A30, scoped"**. Its verdict closes Option 2 and C6b, and it supplies P3's
starting rule family descriptively. Nothing was run for it in this session.

**Not changed by this amendment:** A30's text, data, grid, criterion and result; anything in A31. Any
change to the above is a new amendment, written before the result it affects.

#### A33 (2026-11-04): PaD — the naming check: modelB alone names G1's 35-name SAM 3 tracks on HELD-A (laptop, $0; CLIP and modelB only, no SAM 3)

Written before the check runs, from `studies/postg1/d7-session-prompt.md` (v2), Step 3.
- **Purpose:** gates 2 and 3 are class-aware, and modelB has never named objects alone. This measures
  that before anything is rented.
- **Exploration on HELD-A, in-sample** for modelB and R0w on HELD's base objects. Not a test result.
- **Already seen when written:**
  - **sizes:** 78 HELD-A videos with SAM 3 files, 294 confident tracks, 41,077 detected-frame boxes,
    18,570 GT boxes, 20,036 frames to encode;
  - **speeds on the laptop's GTX 1650,** measured on random features and one FIT frame: modelB
    0.39 s per box, CLIP and TagCLIP together 1.35 s per frame;
  - **a plumbing dry run** on one HELD-A video with dummy models, which computes no name.
- **Not computed:** no modelB name, on any HELD box.

##### Data

- **Videos:** the 78 HELD-A videos (A31) that have G1 SAM 3 files.
- **Tracks:** G1's 35-name SAM 3 confident tracks after A29's duplicate rule. That is the track set G1
  counted: `adapter.tracks_from_sam3(...)["repo"]` (`studies/rpt/adapter.py`, SHA256 `25a1aaf1…3b94`),
  read from `studies/rpt/g1/sam3_out/`.
- **EOV-MMP R0w:** G1's per-video files (`studies/rpt/g1/eov_out/`).
- **Ground truth:** from A31's HELD-A extracts (`studies/d7/extracts/manifest.json`, SHA256
  `7466bfd4…e483`, each extract checked against it):
  - the option-B runs (`gt_tracks_train.held_a`), with boxes read from the per-video annotations;
  - the base relations (`held_gt.held_a` and `held_gt_missing3.held_a`), as G1 used them.
- **Matching and pooling:** A29's, from `studies/rpt/g1/metrics.py` (SHA256 `359c3d9a…dbf8`).

##### Naming (P4), as in the developer review §B6

- **Boxes:** for each track, every **detected** frame, meaning a frame the adapter marks as started or
  matched by a SAM 3 detection. The box is the track's box on that frame: the tight box of SAM 3's
  mask.
- **Per box, exactly as `End2End_Model`'s test loop without the fusion**
  (`models/end2end_model.py:222-264`, SHA256 `ce636cf9…c0c1`):
  - **the frame's TagCLIP `patch_proj`,** from the call `studies/rpt/relation.py` makes
    (`encode_image_tagclip(x, 336, 336, attn_mask=1)` on `_transform_resize(336, 336)`). The frame
    is the f-th file of the folder in sorted order. The CLIP global pass is skipped, because naming
    does not read it and it does not affect `patch_proj`.
  - **pooling:** the box is clipped to the image, scaled to the 24 × 24 grid and turned into a mask
    by `create_mask`. Then a masked mean and an L2 norm, with the original's number types (a float32
    box, integer sizes).
  - **modelB:** `Classifier(args)` (`models/object_classifier.py`, `5652f678…a9cf`;
    `vlm/text_encoder.py`, `a2096fe0…2597`) with EOV-MMP's parsed default arguments. Its weights
    are the 155 `modelB.*` tensors of the released end-to-end checkpoint (SHA256 `d306952a…0e70`),
    as R0w loaded them. Eval mode, softmax output, one box per call.
  - **class order:** `new_order` (`end2end_model.py:257-259`).
- **A track's 35-vector** is the mean over its detected frames, and its **name** is the argmax
  (`numpy.argmax`). **No fusion** with any decoder score.
- **Code:** `studies/d7/naming_check.py` (SHA256 `98221359…0554`), tested by
  `studies/d7/tests/test_naming_check.py` (`b6a0722c…709f`). The tests check the pooling against an
  independent coverage-weighted mean, the class order, the stop rule, and renaming through A29's
  matching.
- **The run is resumable per video.** Outputs go to `studies/d7/naming_check/videos/<video>.json.gz`,
  with hashes in `manifest.jsonl`.

##### Measures

These are class-aware, with A29's matching, on the 78 videos, for three systems:
- **(i)** G1's tracks with **the prompt names**;
- **(ii)** the same tracks with **modelB's names**;
- **(iii)** EOV-MMP R0w.

For each system:
- object recall (A29's run recall), overall and base / novel;
- pair recall;
- class-aware burden (unmatched confident tracks per video).

The class-agnostic numbers are reported too.

Also reported:
- **modelB's naming accuracy on the tracks matched class-agnostically,** base / novel (by the GT run's
  class). The prompt names' accuracy on the same matches is given beside it.
- **modelB's top-1 on HELD-A GT boxes,** base / novel. Each annotated frame of each HELD-A GT run is
  one box. As a secondary figure, the argmax of each run's mean vector, as for tracks.
- **The runtime.**

##### The stop rule (binding as stop-and-report; not a kill)

"Gain" is the class-aware overall object-recall gain over (iii), in points.
- **If (i)'s gain is ≤ +2 points:** stop and report. The check cannot discriminate.
- **Else, if (ii)'s gain is less than half of (i)'s gain:** stop. The Step 4 rental request is not
  prepared; the result is reported. Naming is then the bottleneck, and P4 is redesigned first.
- **Otherwise:** continue to Step 4.

**Disclosed:** modelB and R0w are both in-sample on HELD's base objects, so the novel split is the
honest part.

**Runtime estimate:** about 10–11 hours on the laptop. That is modelB at about 6.5 h (59,647 boxes ×
0.39 s), plus TagCLIP at about 4 h (20,036 frames).

**Report:** `studies/d7/naming_check_report.md`, verdict first, with the runtime.

**Not changed by this amendment:** anything in A28–A32. Any change to the above is a new amendment,
written before the result it affects.
