# RB1 — Step 1 report (CPU only, before any GPU run)

**Status: Steps 0 and 1 are complete. Nothing has run on a GPU. Stopped, waiting for approval of Step 2 setup.**
Baseline construction and METOR-module measurement; not the thesis method. No test evaluation has happened (the fifth is still ahead).

Date: 2026-10-07 · Repo: `ghibli613/metor`, local `~/Documents/thesis/source/METOR`

---

## 1. What was done

| Step | Result |
|---|---|
| Step 0 | A23 written and appended to `prereg/preregistration.md` in this repo. It continues the pilot pre-registration (EOV repo, frozen at A22, SHA256 `442623b82cf5392c480e889ab160d5e0606300acc9d588d338e0a806036211f0`, verified and logged). The EOV repo was **not modified**. A22's GPU run (O1 step 2) is recorded as **cancelled, not run**. |
| Step 1 | Flags, tests, audit, full test suite. Code committed locally as `7e87c9bb8c45cd4699469546c3c4310db20b87b2` (parent `3be814763e2d9acffa26dc5ff20fdde249bce3fe`). Not pushed. |
| A23a | Pin addendum appended: commit hash, SHA256 of all 14 changed files, test results, and every difference from A23 (old choice / new choice / reason). |

**Hash chain** (`prereg/preregistration.sha256`, both files now 444):

| state | SHA256 |
|---|---|
| header only | `056299d13a77e7f21fcabbcab054b9eed3ba248ab47f961064c27b4706d18f0e` |
| after A23 | `30eecc0255f974af9c639aac4ac36c61f6f251a92f7cedbf0f7384ce88717f6d` (A23 text alone `0f2c50b4…8413c`) |
| after A23a (current) | `bdfe25e62b20d3d82c743c848197d3448aff9d40fe6e98efdac3fd6134af268b` (A23a text alone `c42b8f39…3937`) |

Every append verified the prior hash and that the earlier bytes were unchanged, and logged before/after in `rb1/logs/prereg_log.txt`.

**Housekeeping, disclosed:**
- An earlier attempt to append A23 into the EOV file failed (permission denied; the file is 444). My script kept going and **overwrote the EOV `preregistration.sha256`** with a line for an amendment that was never appended. I restored it from backup immediately; it is byte-identical to before (`146021a9…`, confirmed with `cmp`). EOV `preregistration.md` was never changed.
- The five files I had written under the EOV `pilot_analysis/rb1/` were copied into this repo's `rb1/` (hash-verified, logged in `rb1/logs/move_log.txt`), then the originals and the two empty directories were removed.
- Two interim pytest logs were deleted (one was run while edits were in progress).

## 2. Tests

| run | result |
|---|---|
| base commit `3be8147`, clean worktree | 137 passed |
| pinned commit `7e87c9b` | **169 passed** (137 + 32 new), no failures, no skips |
| `tests/test_compat_equivalence.py` | 4 passed: with all flags off, `--n_iter 0` is still **bit-identical to EOV**; wrapping still preserves every released-checkpoint parameter name |

The new tests pin both settings of each flag. Highlights:
- fusion off = a verbatim copy of the released loops, bit for bit (red_panda fused twice, turtle unfused); on = 0.3/0.7 once for all 35 classes;
- on a real training video, `--exclude_novel_train` leaves no novel fragment, relation or predicate; `gen_feats` gives the same pairs with and without it; novel predicates are in `pre_label` without it and **zero** with it; the interactiveness target has strictly fewer positives;
- **resume is exact**: 2 epochs straight equals 1 + resume + 1, bit-identical weights and identical video order, even when the resuming process was given a different seed. A hand-made mutant that skips restoring the loader generator makes this test fail (so it is not vacuous);
- video order is independent of how many random numbers model construction drew (the R2-vs-R3 requirement);
- startup aborts if the IEM layer count disagrees with `--expect_n_iter`, if the IEM is frozen, or if its parameter names match a frozen pattern; R3's trainable count minus R2's equals the IEM's parameter count.

**What the tests do not cover (needs the GPU):** the training forward through the detector, `criterionA`, `postprocessorA`, deep_sort and AFLink with the flags on. Everything around it was run on CPU on real annotations (target builders, `gen_feats`, filters, loader, loop, checkpointing). More crashes like the one in §3 are possible; Step 2's 5-video run is the first real check.

## 3. Finding: the released step-4 training path could not run on this repo's data

Run on a real training video before any edit:
- `gen_frame_pre_label` indexes `predicate[0]` → `KeyError: 's'`.
- `gen_box_label` indexes `category[0]` and `box[0][0]` → `TypeError`.

The annotation and trajectory files hold strings and flat `[x1,y1,x2,y2]`; the code assumes one-element lists and nested coordinates. So step-4 training as vendored never ran here.

**Fix:** both functions now accept either form. It is **not behind a flag**, because the old behaviour is a crash. R1 is therefore "the released logic with a format fix that changes nothing else", not literally the released code. The R1 − R0 rule is unchanged, but if R1 misses R0 by more than 1 point on novel, part of the cause may be that the authors' data pipeline differed from this one. This is recorded in A23a as difference 1.

## 4. Training-target audit (every place targets are built)

| # | where | released behaviour | novel involved? |
|---|---|---|---|
| T1 | `models/gen_labels.py` `gen_box_label` (called at `end2end_model.py:69`) | box + class id for every fragment of `train_object_trajectories_gt.json`, no split filter (1,244 of 4,834 fragments are novel) | **yes** |
| T2 | `models/methods/detectors/ov_prompt/model.py:309-321` | detector text queries and exemplar queries = the labels present in the frame | **yes** (follows T1) |
| T3 | `end2end_model.py` `len(labels) > 0` gate | a frame with no targets is skipped for the detector and tracking | all-novel frames were *trained on* in the release |
| T4 | `gen_frame_pre_label` → `set_criterion.py:404-416` | frame-level predicate multi-hot, base predicates only, but from every relation whatever its subject/object categories | base predicate on a novel object: **yes** |
| T5 | `gen_feats` (relation targets) | track pairs matched to GT objects of every category; predicates from all relations into a 132-wide `pre_label`; object labels are the detector's predicted class, not GT | **yes** |
| T6 | `models/relation_classifier.py:528-545` | `pre_labels` sliced to base, **but `int_labels` is computed from all 132 columns**, so a pair with only novel predicates is an interactiveness positive | **yes — a novel-predicate leak into the loss** |
| T7 | object classifier (modelB) | not called in training; no loss, no gradient | no |
| T8 | stage 1–3 checkpoints | not inspectable | unknown |

Also: the six classes with the 0.6/0.4 fusion weights (antelope, bear, cattle, fox, giant_panda, red_panda) are **all novel categories**; turtle (31) is unfused; red_panda (23) is fused twice.

**What `--exclude_novel_train` does:**
- F1: drops novel fragments from detector targets.
- F2: drops relations whose predicate is novel or whose subject/object is a novel object.
- F3: the same on the annotation `gen_feats` reads, so a track over a novel object gets no GT hit and no labels (a negative pair).
- The processed frame range is computed **before** filtering; frames whose objects are all novel stay in the range and are skipped by the existing `len(labels) > 0` gate (count logged per epoch).
- T6 is fixed at the source (no novel column is ever set in `pre_label`); `relation_classifier.py` is unchanged.
- Aborts the run if a novel id reaches detector targets, a novel predicate entry reaches `pre_label`, or a novel GT tid reaches label matching.
- Logged only: `track_labels_novel_id_ignored_by_loss`. A track with no scored detection falls back to class id 1 (antelope, novel); the relation loss maps novel ids to −1 and ignores them. This is released behaviour, present in R1 and R2 alike; no novel label is used.

**Caveat (stated in A23):** T8. R2 and R3 are clean for step 4 only.

## 5. Flags added (all default to released behaviour)

`--exclude_novel_train`, `--uniform_fusion`, `--no_epoch_eval`, `--epoch_ckpt_dir`, `--resume <ckpt>` (was an unused bare `store_true`; now takes a path), `--mirror_dir`, `--loader_seed`, `--expect_n_iter`, `--small_train_videos N`. Documented in `docs/23_Configuration-and-flags.md`; run commands in `rb1/README.md`.

## 6. Configurations (unchanged from A23)

| id | starts from | training data | fusion | IEM |
|---|---|---|---|---|
| R0 | released e2e checkpoint | — | per-class | off (already measured 15.79 / 27.18) |
| R0u | same | — | uniform 0.3/0.7 | off |
| R1 | stage 1–3 | as released (novel included) | per-class | off, `--n_iter 0 --expect_n_iter 0` |
| R2 | stage 1–3 | novel removed | uniform | off, `--n_iter 0 --expect_n_iter 0` |
| R3 | stage 1–3 | novel removed | uniform | **IEM only**, `--n_iter 2 --expect_n_iter 2` |

R3 contains only the iterative enhancement module; CRE is not wired into compat. It measures the IEM, not CRE and not METOR as published. `--n_iter` defaults to 2 in compat, hence the explicit flag and the startup assertion. Primary contrast: R3 − R2 (paired video-level bootstrap, 1,000 resamples, seed 0).

## 7. Pins (full list in A23 and A23a)

Checkpoints (SHA256, also in `rb1/logs/checkpoint_sha256.txt`):

| file | SHA256 |
|---|---|
| detector (stage 1) `checkpoint_vidvrd0059_new_1e-5.pth` | `93c4457cd4a3e5b2db821753409bc3dc3dc0cd9d2d1bac3646f6efd82e8966da` |
| object classifier (stage 2) `vidvrd_backboneViT-L_14@336px_lr0.01vision-guided.pth` | `3525a173e26f6534f52a275a8d3e22a114a392765b5a1abda80aac5db5eff098` |
| relation (stage 3) `…stage2_new_L14_e2e.pth` | `d9839ed789134e9acaad7b85106bf93186dc34b0f05b87a2faa9ee32cbbd2f24` |
| released end-to-end `…end2end_base-001.pth` (R0, R0u) | `d306952a51e9ca94c201ecf580f75a99221ae658dc187ae0ff38ff8b7dfc0e70` |
| `AFLink_epoch20.pth` | `b35cbeddd3acc48fece820bd640640e6bfb1f5fbf570aa79af26c6a38958daa4` |
| CLIP ViT-L/14@336px (equals the hash in OpenAI's download URL) | `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02` |

`…train42-003.pth` is present but **not used**. The three `test_*.json` metadata files were hashed as bytes only, not opened.

## 8. Not done, and why

- **Step-4 evaluation scripts** (per-video prediction dump, paired bootstrap) are not written yet. Per A23a difference 7 they will be written and tested on synthetic predictions only, then pinned by **A23b** before the test session.
- **No GPU work.** This machine has a 4 GB GTX 1650 and 7 GB RAM; it cannot hold the model.
- Rough, **unmeasured** cost guide: earlier SGDet cost 0.24 s per processed frame; an epoch is 162,330 frames, i.e. ~11 GPU-hours forward-only. With backward and tracking at 1.5–2×, about 80–110 GPU-hours per configuration, 240–320 for three. At roughly $0.35–0.70/hour that is about **$85–230**. Three parallel machines cost the same and finish in a third of the time. The 5-video run replaces this with a measured figure.

## 9. What is needed before Step 2

The person relaying these messages (not Claude chat) has to provide GPU access; I can't rent a machine. Addendum D requirements: 24 GB GPU, ≥32 GB system RAM, `OMP_NUM_THREADS=8`, ≥150 GB disk, the 42 GB frames, the checkpoints above, and the CUDA ops in `ops/` built for the machine.

Decisions needed:
1. How will I get GPU access (SSH details, or a machine already rented to the spec above)?
2. Where should `--mirror_dir` point for the per-epoch copies?
3. Should the code be pushed to GitHub for the machine to clone, or copied over directly? (Neither `prereg/` nor `rb1/` is committed or pushed yet.)
4. Run R1, R2, R3 on one machine in sequence, or on three in parallel (same cost, a third of the wall time, but each must pass the D.4 test run and input hashes must match)?

**No GPU run and no spending until the Step 2 estimates are approved.**

## 10. Files

- `prereg/preregistration.md`, `prereg/preregistration.sha256`: A23, A23a
- `rb1/logs/`: checkpoint, CLIP and metadata hashes; `prereg_log.txt`, `move_log.txt`; `code_commit.txt`; `changed_files_sha256.txt`; test logs at the base and pinned commits; pre-amendment backups
- `rb1/scripts/mutation_check_resume.py`: the by-hand check that the resume test is not vacuous
- `rb1/README.md`: run commands
- code: `models/novel_filter.py`, `models/fusion.py`, `cli/train_state.py`, and edits to `cli/train.py`, `cli/common.py`, `models/end2end_model.py`, `models/gen_labels.py`, `data_loading/feature_dataset.py`, `utils/parser_func.py`; tests `tests/test_rb1_*.py`
