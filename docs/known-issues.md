# Known issues and blockers

Ordered by what stops you first. Everything here was measured or reproduced, not
assumed.

Last verified: 2026-08-20.

---

## 1. Checkpoints — BLOCKER for training

`cli/train.py` loads four pretrained components before training:

```
output/ckpt/checkpoint_vidvrd0059_new_1e-5.pth                       step 1: detector
output/ckpt/baseline_fbce_..._stage2_new_L14_e2e.pth                 step 3: relation classifier
output/ckpt/vidvrd_backboneViT-L_14@336px_lr0.01vision-guided.pth    step 2: object classifier
--path_AFLink <file>                                                 tracker
```

All four are unconditional `load_state_dict` — no guard, no fallback.

**The download is not the real problem.** The code that *produces* the first
three was never released — upstream's To-Do list still has "Object Detection
Training" and "Relationship Detection Training" open, and the only training
loops in their repo are step 4 and the AFLink tracker. See
[architecture.md §4](architecture.md#4-what-is-publicly-available).

Sources, in order of likely success:

1. Email the author — their README invites it: `1285441164yq@gmail.com`
2. Baidu: `https://pan.baidu.com/s/1jZbnkYexAZQcGApyBUhQjw?pwd=jfec`
3. Google Drive folder `1IH5fyfZN7on_DMsv55385scd0YrlueTO` — returns **401** to
   automated download; re-tested 2026-08-20. May work in a signed-in browser.

**Testing does not need the first three.** `--eval_only` loads one trained
end-to-end checkpoint, which supersedes them. AFLink is still required —
trajectory association is part of inference.

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

## 2. VRAM

**Measured: the two datasets alone allocate 5.32 GB**, before the detector,
relation model, or classifier exist. On a 4.3 GB card this survives only by
spilling into system RAM under WSL2, which is why a probe that should fail in
seconds took six minutes.

The cause is in [`data_loading/dataset.py:61-62`](../data_loading/dataset.py):

```python
self.clip, _    = clip.load('ViT-L/14@336px', device='cuda')
self.tagclip, _ = clip_tagclip.load('ViT-L/14@336px', device='cuda')
```

Each `Dataset_new` puts **two** CLIP-L models on the GPU, and `train.py` builds
two datasets — four CLIP-L instances for feature extraction alone, plus more
inside the models.

A T4 (16 GB) or better is required. This machine is fine for development and
debugging, not for training.

---

## 3. No feature cache

`__getitem__` re-encodes **every frame of the video, every time it is sampled**,
with both CLIP and TagCLIP. Nothing is cached anywhere.

Measured against the actual frame counts:

| | |
|---|---|
| train frames | 244,100 |
| test frames | 52,104 |
| ×2 encoders | **592,408 ViT-L/14@336 forwards per epoch** |
| ×20 epochs | **11.8 M forwards** |

The frames never change, so this is pure recomputation and it dominates the cost
of a run. Precomputing these features to disk once would be the single biggest
improvement available, and it would also remove the two CLIP models from each
dataset — largely fixing issue 2.

This is how MMP already works: it reads pre-extracted features from disk and
ships `scripts/features.py` to build them.

---

## 4. The CLIP object bank is a reconstruction

The authors never published `clip_L14_feat_vidvrd.pkl`.
[`tools/build_clip_object_bank.py`](../tools/build_clip_object_bank.py) rebuilds
it from real GT boxes, real frames and the real CLIP encoder — nothing synthetic
— but the recipe is a guess.

**Results built on it are not directly comparable to the paper.** Say so in any
write-up. Details in [data.md](data.md#datavidvrddataclip_l14_feat_vidvrdpkl--the-clip-object-bank).

---

## 5. Camera-motion compensation is disabled

`VidVRD_ECC_{train,test}.json` were never shipped and cannot be reconstructed
from anything here. `End2End_Model` used to load them unconditionally and die.

It now loads them if present, else uses `{}` and prints a warning.
`Tracker.camera_update` guards with `if video in ecc.keys()`, so an empty dict
makes compensation a no-op.

**This is a real quality loss on moving-camera videos**, not an equivalence.

---

## 6. Hyperparameters do not match the paper

`run_train.sh` uses `max_epoch=20` with `MultiStepLR(milestones=[15,20,25])`.
The paper's step 4 says **5 epochs**, and that milestone schedule belongs to
step 3. The released config is not cleanly the paper's step 4.

Full table in [architecture.md §5](architecture.md#5-where-the-code-and-the-paper-disagree).

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
| CUDA extension would not build | Built and verified: kernel matches the PyTorch reference to 8.7e-19. Required editing vendored CUDA source — see [port-status.md](port-status.md) |
| `torchvision.__version__[:3]` version gate | Compared `(major, minor)` numerically |
| ~25 hard-coded paths | Routed through `utils/paths.py` |
| `--clip_feat_path` declared twice with different values | Both resolve to `paths.CLIP_FEAT_BANK` |
| Eval paths broken — would crash at end of epoch 1 | Fixed |
| No evaluate-only path | `--eval_only` added |
| `gpu_id=3` selected no GPU on single-GPU machines | `${gpu_id:-0}` |
| `.gitignore` used inline comments, so `output/` and `ops/build/` never matched | Rewritten |
| Undefined names in live code (`F`, `random`, `_logger`) | Fixed and exercised |
| CLIP bank written with `pickle.dump` but read with `torch.load` | Producer now uses `torch.save`; existing file converted, tensors verified identical |
| `torch.load` refusing checkpoints under torch>=2.6 | `weights_only=False` at every live call site |
