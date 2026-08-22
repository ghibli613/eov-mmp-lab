# Extension guide — where each change hooks in

Companion to [90_Research-ideas.md](90_Research-ideas.md), which argues *what*
is worth doing. This says *where*: which file, which function, and what each
change would establish.

Ordered by evidence-per-hour, not by ambition.

---

## 1. How to extend this — the ablations, and where each one hooks in

Ordered by evidence-per-hour, not by ambition. Every one of these is a change to
the **relation side or the ranking**; the detector is effectively frozen because
the training code for steps 1–2 was never released
([01_Architecture.md §4](01_Architecture.md#4-what-is-publicly-available)).

### 1.1 The geometry-only baseline — the go/no-go

**Claim to test:** how much of EOV's mAP survives with no appearance at all.

**Hook:** already built. `--clip_feat` is now a working gate
(§3.1); `Model(args)` with `args.clip_feat = False` zeroes the visual stream and
runs. `tests/test_feature_streams.py::test_geometry_only_configuration_runs`
exercises it.

**What it does not do:** this is a *step-3* experiment — retrain the relation
classifier on ground-truth trajectories and compare. It is **not** an end-to-end
experiment, because `patch_` is the detector's input (known-issues §2) and the
end-to-end path cannot run without CLIP encoding at all. Step-3 training code is
not in this repository; MMP's is public and is the intended starting point.

If a geometry-only relation classifier lands near the full model, §5's claim is
established empirically and the taxonomy caveat evaporates.

### 1.2 Rebalancing the four-way text ensemble — free

**Claim to test:** the uniform `cat([sbj, obj, uni, learned]) / 2` in §3.3 is
arbitrary, and the learned stream is the base-overfit one.

**Hook:** `split_text_embeddings` in
[models/relation_classifier.py](../models/relation_classifier.py). Give each of
the four blocks its own scalar weight, and allow different weights for the base
and novel splits. This is the F-VLM / ViLD-ensemble trick and it needs **no
retraining** — one eval pass per setting.

**Do not tune it on the test set.** See §6.6.

### 1.3 Calibrating the object scores — free

**Claim to test:** the `sbj_scr * obj_scr * pre_scr` product in `format_`
systematically demotes novel objects.

**Hook:** `format_` in [inference/post_process.py](../inference/post_process.py),
plus the softmax in `Model.forward`. A single additive logit bias per split is
enough to test it. Sweep `--max_per_video` at the same time; it is a free
parameter nobody in the lineage reports.

### 1.4 Fixing association — free

**Hook:** `association` in the same file. Three independent changes: allow a
one-clip gap instead of `break`; pool clip scores with max or top-k instead of
mean; allow merging predicates that share a component (`walk_left`/`run_left`).

`tests/test_inference_postprocess.py` pins the current behaviour, including the
gap-splitting, so each change shows up as a deliberate diff.

### 1.5 Real temporal aggregation — the method

**Claim to test:** §5 says the model is temporally blind within a clip; this is
the fix that §5 motivates.

**Hook:** `_extract_clip_feat` in [models/gen_labels.py](../models/gen_labels.py).
Replace the single `mid_fno` sample with k sampled frames, masked-pooled per
frame using that frame's box, then aggregated (mean, or attention). The patch
features for every frame are already materialised by the dataset, so this costs
no extra encoding — only the pooling.

Interaction to watch: with `--frame_stride 30`, the "middle frame" of a clip is
already a *reused* neighbour's features, so this change and that flag are not
independent. Settle the stride question first.

### 1.6 The protocol — do this before 6.2 and 6.3

6.2, 6.3 and 6.4 are all tuning, and this benchmark's convention is to select on
the test set (known-issues §7). Tuning *more* things on the test set will make
the result indefensible.

**Hook:** carve a pseudo-novel split out of the base predicates — hold out ~15
base categories, treat them as unseen for tuning, never touch test. The split
files are `configs/VidVRD_pred_class_spilt_info_v2.json`, read through
`utils/paths.PRED_SPLIT_INFO`; `tests/test_config.py` pins the published split so
a new one has to be introduced deliberately rather than by editing in place.

### 1.7 Things deliberately left in place

- `vlm/backbones/internvideo/` (8,787 lines) is unreachable but retained: §6.5
  may want a real video encoder, and it is already vendored for this task.
- `prior.pkl` and `--use_prior` are kept but should stay off; a prior that
  generalises to unseen compositions would have to come from text space, not
  from counts.

---
