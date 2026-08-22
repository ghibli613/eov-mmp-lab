# Known issues and blockers

Ordered by what stops you first. Everything here was measured or reproduced, not
assumed.

Last verified: 2026-08-20 (checkpoints received and verified same day).

---

## 1. Checkpoints — RESOLVED (2026-08-20, sent by the author)

The author supplied all four components plus two **fully trained end-to-end
models**. Installed under `output/ckpt/`:

| File | What | Notes |
|---|---|---|
| `checkpoint_vidvrd0059_new_1e-5.pth` | step-1 detector, 755 tensors, epoch 59 | Drive's `-002` dedup suffix stripped |
| `..._stage2_new_L14_e2e.pth` | step-3 relation classifier, 196 tensors, epoch 19 | |
| `vidvrd_backboneViT-L_14@336px_lr0.01vision-guided.pth` | step-2 object classifier, 155 tensors | arrived as `.pth.zip` — that is torch's own zipfile format, **not** an archive to extract; renamed only |
| `AFLink_epoch20.pth` | tracker, 144 tensors | matches `paths.AFLINK_CKPT` |
| `..._end2end_base-001.pth` | **trained end-to-end**, 1106 tensors, epoch 6 | mAP 26.88 all / 15.64 novel |
| `..._train42-003.pth` | **trained end-to-end**, 1106 tensors, epoch 2 | mAP 24.33 all / 13.70 novel |

**Every one loads with 0 missing and 0 unexpected keys** against the
restructured model — verified by building the full model and calling
`load_state_dict(..., strict=False)` on each. This matters: the repo was
restructured after these were trained, and a key mismatch would have silently
loaded a half-initialised model rather than erroring.

The end-to-end checkpoints' internal layout (`modelA`/`modelB`/`modelC` =
755/155/196 tensors) matches `End2End_Model`'s constructor order, and their
stored `config` matches this repo's argument defaults.

So `--eval_only` can run today:

```bash
python -m cli.train --eval_only \
  --ckpt_path output/ckpt/baseline_fbce_vidvrd_bs1_lr1e-05_dim512_none_rel_mot_clip_bbox_end2end_base-001.pth \
  --dataset vidvrd --path_AFLink output/ckpt/AFLink_epoch20.pth
```

Note the training *code* for steps 1–3 is still unreleased (see
[01_Architecture.md §4](01_Architecture.md#4-what-is-publicly-available)) — these
checkpoints are the outputs, not the recipe. Retraining them from scratch is
still not possible from this repository.

---

## 1b. torch.load and `weights_only` — FIXED, but know about it

PyTorch 2.6 changed `torch.load`'s default from `weights_only=False` to `True`,
which refuses any checkpoint containing more than tensors. **All four EOV
checkpoints store an `argparse.Namespace` under `config`**, so every one of them
would have failed to load with an `UnpicklingError` — on a version of PyTorch
newer than the authors used.

Every live `torch.load` now passes `weights_only=False` explicitly. These files
are produced locally or come from the paper's authors, so loading them fully is
intended, and the call sites say so.

Found by constructing the whole model with the checkpoint loads stubbed out;
it is not visible until something actually loads a file.

## 2. VRAM — halved on 2026-08-22, still not enough to train here

**Measured: the two datasets alone allocated 5.32 GB**, before the detector,
relation model, or classifier existed. On a 4.3 GB card that survived only by
spilling into system RAM under WSL2, which is why a probe that should fail in
seconds took six minutes.

The cause was in `data_loading/dataset.py`:

```python
self.clip, _    = clip.load('ViT-L/14@336px', device='cuda')
self.tagclip, _ = clip_tagclip.load('ViT-L/14@336px', device='cuda')
```

Each `Dataset_new` put **two** CLIP-L models on the GPU, and `cli/train.py`
builds two datasets — four CLIP-L instances for feature extraction alone.

**Fixed:** both encoders are frozen and stateless, so they are now loaded once
per process through `_shared_encoder()` and shared between the train and val
datasets. That halves this cost, from four model copies to two.

It does **not** make training possible on a 4 GB card — the detector, relation
model and classifier are still to come, and `patch_`/`patch_proj` per frame are
large. A T4 (16 GB) or better is still required. This machine is fine for
development, for the test suite, and for constructing and checkpoint-loading the
models, which is what `tests/test_checkpoint_compat.py` exercises.

**Not** optional, despite appearances: `patch_` is the detector's input
(`models/end2end_model.py`), so CLIP encoding is the visual front-end of the
whole pipeline, not merely the relation stream's feature extractor. There is no
configuration in which the end-to-end path skips it.

---

## 3. Every frame is re-encoded, every epoch

`__getitem__` encodes **every frame of the video, every time it is sampled**,
with both CLIP and TagCLIP. Nothing is cached.

| | |
|---|---|
| train frames | 244,100 |
| test frames | 52,104 |
| ×2 encoders | **592,408 ViT-L/14@336 forwards per epoch** |
| ×20 epochs | **11.8 M forwards** |

### Caching the features does NOT help — correcting an earlier claim

An earlier version of this document recommended precomputing the features to
disk as "the single biggest improvement available". **That was wrong**, and the
arithmetic shows why. Per frame the dataset materialises:

| tensor | shape | fp16 for 296,204 frames |
|---|---|---|
| `patch_` | (576, 1024) | **349 GB** |
| `patch_proj` | (576, 768) | **262 GB** |
| `global_proj` | (768,) | 0.5 GB |
| | | **612 GB total** |

That is 15× *larger* than the 42 GB of JPEGs it would replace. The patch-token
tensors dominate, and they are exactly what TagCLIP produces. Caching only
`global_proj` is cheap (0.5 GB) but it is the smallest part of the work.

### What does help: the sampling the authors left commented out — now a flag

Upstream shipped the stride-30 sampling commented out inside `__getitem__`. As of
2026-08-22 it is implemented properly and exposed as **`--frame_stride`**:

```bash
python -m cli.evaluate --frame_stride 30 ...    # the paper's config, ~30x cheaper
python -m cli.evaluate ...                      # default 1: encode every frame
```

At stride 30 it encodes frames whose 1-indexed number is ≡ 1 (mod 30) and reuses
those features for the 29 in between — from 592,408 forwards per epoch to about
19,750.

This is not a hack: the paper's Implementation Details say *"For all
experiments, video frames are sampled every 30 frames."* The shipped code
contradicts the paper it implements.

**The default is 1, i.e. upstream's behaviour, deliberately.** Which variant the
supplied checkpoints were trained with is still unknown, so stride 30 may not
match how they were produced. This is the first thing to settle when a 16 GB
machine is available, because every subsequent number is a delta against
whichever baseline you pick. Measure both on a few videos before trusting either.

---

## 4. CLIP object bank — RESOLVED (authors' file installed)

The authors' `clip_L14_feat_vidvrd.pkl` (278 MB) is now in place, replacing the
5 MB reconstruction. They differ substantially, which is why results on the
reconstruction were never comparable:

| | reconstruction | authors' |
|---|---|---|
| exemplars | ~1,500 | **189,345** |
| per category | 15–50 | 180–26,010 |
| dtype | float32 | float16 |
| L2-normalised | yes | **no** |

They embedded every ground-truth box, unnormalised, in half precision.
`tools/build_clip_object_bank.py` is kept for reference and for datasets where
no bank exists, but it is no longer used for VidVRD.

---

## 5. Camera-motion compensation — RESOLVED

`VidVRD_ECC_train.json` (34 MB) and `VidVRD_ECC_test.json` (7.5 MB) arrived with
the checkpoints and are installed under `data/vidvrd/data/`.

Structure verified against `Tracker.camera_update`: 199 videos, **string** frame
keys (the code does `str(int(frame))`), 3×3 homographies with `‖I−M‖` well under
the 100 cutoff in `get_matrix`. The tolerant loader picks them up automatically
and the warning no longer fires.

---

## 6. Hyperparameters do not match the paper

The default `max_epoch` is 20, with `MultiStepLR(milestones=[15,20,25])`.
The paper's step 4 says **5 epochs**, and that milestone schedule belongs to
step 3. The released config is not cleanly the paper's step 4.

Full table in [01_Architecture.md §5](01_Architecture.md#5-where-the-code-and-the-paper-disagree).

---

## 7. Model selection uses the test set

`train.py` evaluates on the test set after every epoch and keeps the best-mAP
checkpoint. This is the convention across this benchmark — RePro and MMP do the
same — so following it keeps numbers comparable. **State it explicitly in any
write-up.**

---

## Resolved

| Issue | Resolution |
|---|---|
| CUDA extension would not build | Built and verified: kernel matches the PyTorch reference to 8.7e-19. Required editing vendored CUDA source — see [11_Port-status.md](11_Port-status.md) |
| `torchvision.__version__[:3]` version gate | Compared `(major, minor)` numerically |
| ~25 hard-coded paths | Routed through `utils/paths.py` |
| `--clip_feat_path` default pointed at a non-existent path | Now `paths.CLIP_FEAT_BANK`. A second declaration in `utils/arguments.py` looked like a conflict but was inert; that file has since been deleted (2026-08-22 cleanup) |
| Eval paths broken — would crash at end of epoch 1 | Fixed |
| No evaluate-only path | `--eval_only` added |
| `gpu_id=3` selected no GPU on single-GPU machines | `${gpu_id:-0}` |
| `.gitignore` used inline comments, so `output/` and `ops/build/` never matched | Rewritten |
| Undefined names in live code (`F`, `random`, `_logger`) | Fixed and exercised |
| CLIP bank written with `pickle.dump` but read with `torch.load` | Producer now uses `torch.save`; existing file converted, tensors verified identical |
| `torch.load` refusing checkpoints under torch>=2.6 | `weights_only=False` at every live call site |
