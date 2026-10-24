# What is not implemented

Ordered by how much it matters. §1 dominates everything else in this document
combined.

---

## 1. The MS-COCO decoder initialisation ⚠

> "The Transformer decoder is initialized with the parameters of an object
> detector pre-trained on the MS-COCO dataset, excluding novel object
> categories." — Sec. 4.2

**Not wired up.** [`metor/model.py`](../metor/model.py) defaults to
`VanillaQueryDecoder`, a plain `nn.TransformerDecoder` starting from random
weights.

### Why this dominates

Open-vocabulary detection ability does not come from VidVRD's 800 training
videos. It comes from a detector pretrained on COCO with far more data, with CLIP
distilled into it so its region features live in CLIP's text space.

Without it, `mAP_o` = 55.09 is unreachable. And since a triplet counts only at
vIoU ≥ 0.5 on **both** trajectories, poor trajectories cap relation mAP too.
Expect any run of this repository as it stands to fall well short — for this
reason alone, before any [doc 30](30_Paper-gaps.md) entry is considered.

### What it takes

The real detector is vendored at
[`models/methods/detectors/ov_prompt/`](../models/methods/detectors/ov_prompt/)
with its CUDA kernels in [`ops/`](../ops/).

1. Build it with `--det_token_num 100` instead of 300.
2. Load COCO weights with the novel Open-VidVRD categories removed.
3. Feed it `Q̂` from Eq. 8 in place of `self.query_embed.weight` — the hook is in
   `OVDETR.forward_train`.
4. Pass it as `decoder=` to `METOR.__init__`, which exists for this.

A real project, not a config change — and note EOV-MMP never released the
*training* code for its steps 1–3, so there is no reference loop to copy.

→ [doc 12 §4](12_Transformer-decoder-and-queries.md)

---

## 2. Evaluation for `--arch metor`

`python -m cli.train --arch metor` trains. There is no matching evaluation path.

The vendored post-processing
([`inference/post_process.py`](../inference/post_process.py) →
[`third_party/vidvrd_eval_api/`](../third_party/vidvrd_eval_api/)) expects
trajectories produced by the deep_sort + AFLink tracker. METOR regresses its own
(Eq. 5), so the adapter between the two is unwritten.

`--arch metor-compat` evaluates today, because it *is* the EOV path.

What is needed: convert `(P, T, 4)` regressed trajectories plus `(P, T, n_rel)`
predicate scores into the `pred_rels` structure `process_pred` / `association` /
`format_` consume. Mostly bookkeeping — durations, per-clip splitting, and the
`sbj × obj × pre` ranking.

---

## 3. VidOR

`configs/metor_vidor.yaml` records the paper's VidOR settings so the
configuration is on record, but **the VidOR data is not in this workspace**. Only
VidVRD is present. Nothing here has been run against VidOR.

---

## 4. A per-step frame cap ⚠

**This is an open defect, not just a missing feature.**

The model encodes all of a video's key frames in one forward with gradients
retained through the frozen ViT. Measured over the real dataset at
`--frame_stride 30`:

| | key frames per video |
|---|---|
| median | 7 |
| mean | 10.3 |
| p90 | 22 |
| p99 | 47 |
| **max** | **184** |

Rough activation cost for ViT-L/14@336 backward: ~1.2 GB/frame in fp32, ~0.6 GB
in fp16. So the median video needs ~8 GB fp32, the p90 video ~26 GB, and the
184-frame video is impossible on any single GPU.

**About 10% of videos will OOM regardless of hardware.** The fix is a per-step
frame cap, or chunked ViT encoding with gradient checkpointing. Until then,
training will die partway through an epoch.

→ [doc 40](40_Hardware.md)

---

## 5. Checkpoint resume

`train_metor` saves a checkpoint every epoch but there is no `--resume`. Since
30 epochs will not finish inside a single Colab session, this is needed in
practice before any real training run.

---

## 6. Known deviations

Things implemented differently from the paper **on purpose**, and off by default.

### `--residual_mapping`

Makes `M_f` predict a delta, so Eq. 11 becomes
`Ô⁽ᵏ⁾ = αÔ⁽ᵏ⁻¹⁾ + (1−α)(M_f(R̂⁽ᵏ⁾) + Ô⁽ᵏ⁻¹⁾)`.

Starts as the exact identity and tends to train more stably, but it changes the
algebra: the subject–object difference is preserved exactly rather than scaled by
`α^k`. **Off by default**; both readings are pinned by tests.

→ [doc 13 §4](13_Iterative-enhancement.md)

### `--cre_mode pool`

An approximation of Eq. 7 that pools cached patch embeddings instead of running
the ViT with appended tokens. Not equivalent — the patches never see the context
tokens, and the interaction is not iterated through 24 layers. Default for
`--arch metor-compat` only.

→ [doc 11 §4](11_Contextual-refinement-encoding.md)

---

## 7. Priority order

If the goal is reproducing the paper's numbers:

1. §1 — the COCO decoder init. Nothing else matters until this is done.
2. §4 — the frame cap, or training cannot complete an epoch.
3. §5 — resume, or it cannot complete a run.
4. §2 — evaluation, or you cannot read the result.
5. Then [doc 30 §5](30_Paper-gaps.md) for the choices worth sweeping.
