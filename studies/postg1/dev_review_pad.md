# Developer review of PaD, before implementation

2026-11-04 · Part 1 of `dev-review-pack.md` · read-only: no GPU, no downloads, no test labels, no HELD-B
file opened · code read: this repo at `c34b569`, SAM 3 at `f1647e8` (`~/Documents/thesis/source/sam3`,
unedited), RB1's vendored copy `studies/metor_code/` · one CPU probe (§A2, appendix)

SAM 3 paths are relative to the SAM 3 checkout. All other paths are relative to this repo.

## Summary

**Verdict: Phase 1 can be built, and its core (P1, the learned prompt) carries the least risk.**
- **The gradient path is verified.** A CPU probe used SAM 3's own detector classes at real width,
  with random weights, a tiny text encoder and no ViT. With every SAM 3 parameter frozen, SAM 3's
  shipped box-only loss sends non-zero gradients to both CoOp tokens and a learned text-encoder
  output. Twenty Adam steps lower the loss from 11.59 to 8.96, and SAM 3's weights stay bit-identical
  (§A2).
- **No SAM 3 edit is needed.** One wrapper on the instance's `backbone.forward_text` serves both
  training and the video predictor. It is the same pattern G1's runner uses on the detector.

**The risks lie elsewhere:**
1. **Naming (P4).** EOV-MMP has never named objects with its auxiliary classifier (modelB) alone,
   and nothing in the repo measures it. Gates 2 and 3 are class-aware, so naming errors subtract
   directly from recall (§B6).
2. **One pass is not 35× cheaper, and needs more memory, not less.** A generic or learned prompt
   puts every object into a single SAM 3 pass. Tracking cost and memory grow with the number of
   objects (§A5).
3. **VidVRD's labels will teach suppression unless they are handled.** Frames without boxes train
   the presence head toward "nothing here". Annotation gaps turn annotated objects into negatives.
   Track boxes come from masks, which P1's loss never trains (§A3, §A5).

**Part 6 cannot be run as written at Step 2.** It re-runs Task 2, which already ran under A30. A30's
binding rule was "fail → SAM 3 closes as a detector; no second round of rules on HELD". Part 6 also
narrows that rule after its result. Steps 1 and 4 need smaller fixes (§E11).

**Phase 2 is feasible with work, at about twice the code the planners estimate.**
- `pred_ious` is already trained: an IoU loss on matched queries, on in both checkpoints. It is
  not an objectness head.
- The CLIP-embedding loss also needs masking for class-free targets, which the reviews missed.
- The step-4 GPU path has never run (§C9).

**Level 2 is doubtful as specified** (§D10).

**Changes to make before starting, in order:**
1. **Part 6 Step 2:** drop the re-run. Write one amendment instead:
   - A30's verdict stands for the 35-name SAM 3 with rules (Option 2, C6b);
   - PaD's one-prompt detector is a system A30 did not test;
   - this scope statement is written after A30's result, and the thesis says so (§E11).
2. **Step 1:**
   - make the split procedure exact;
   - make the HELD-B guard cover the aggregate files, writing FIT-only and HELD-A-only extracts
     first;
   - state that HELD-B is sealed from now on, but was used by G1 and A30.
3. **Add a $0 naming check before any rental.** Rename G1's 35-name SAM 3 tracks on HELD-A with
   modelB alone, and compare class-aware recall against the prompt names (§B6).
4. **The D7 check:**
   - run it on one 48 GB GPU with "no skips";
   - write a new entry point for non-class prompts;
   - report recall at an equal track budget;
   - close the guard's edge cases;
   - fix the cost estimate and the check-0 video;
   - add a P1 full-scale smoke step on the same machine, after the arms (§E11, §E12).
5. **P1 design:**
   - preprocess frames for training exactly as the video predictor does (square 1008);
   - exclude frames with no GT box;
   - treat interpolated gap boxes as ignore regions;
   - choose the prompt on an image-level proxy, then run video inference on the final one or two;
   - define the KgCoOp anchor;
   - check that masks agree with boxes for the learned prompt (§A3–A5).

| point | mark |
|---|---|
| A1 where a learned prompt enters SAM 3 | **feasible** |
| A2 gradients reach the prompt | **feasible** (verified on CPU, tiny build) |
| A3 training code and losses, box-only | **feasible with work** |
| A4 cost of P1 training | **feasible** on a 24 GB 4090 |
| A5 learned prompt in video inference; presence | **feasible with work** |
| B6 naming with modelB (P4) | **feasible with work**; faithfulness to EOV-MMP **doubtful** |
| B7 learned keep/drop (P3) features | **feasible with work** |
| B8 relations on P4-named tracks (P5) | **feasible** |
| C9 RB1's step-4 code for Phase 2 | **feasible with work**; the never-run GPU path is the risk |
| D10 Level 2 | **doubtful** |
| E11 Part 6 | Step 2 **blocker** as written; Steps 1 and 4 **feasible with fixes** |

---

## A. Phase 1, P1: the learned class-free prompt

### A1. Where a learned prompt can enter SAM 3: **feasible**

**The text path:**

| stage | code |
|---|---|
| tokenizer, context length 32 | `sam3/model_builder.py:500-509` |
| `token_embedding`, 49,408 × 1,024 | `sam3/model/text_encoder_ve.py:199` |
| plus positional embedding; 24-layer causal transformer, width 1,024; `ln_final`; every token kept (`pool_type` "none") | `TextTransformer.forward`, `:232-254` |
| `resizer` (linear, 1,024 → 256) | `:288`, `:319` |
| padding mask = token id ≠ 0 | `:305`, inverted at `:315` |

**The output** is `language_features` (32 tokens × batch × 256) plus a mask.

**It enters the detector in four places,** all through `_encode_prompt` (`sam3_image.py:168-211`):
- the fusion encoder, where image tokens attend to the prompt (`:213-236`);
- the decoder's text cross-attention and its presence token, in every layer (`:253-283`;
  `decoder.py:320-323`);
- the dot-product scorer, on the mean-pooled prompt (`model_misc.py:752-786`; used at
  `sam3_image.py:317-321`);
- the segmentation head's prompt cross-attention (`:403-411`).

**CoOp tokens.** `TextTransformer.forward` takes token ids only; it has no `inputs_embeds` argument.
There are two ways in, both without edits:
- **Our module calls SAM 3's submodules in order:** token embedding, then placeholder rows replaced
  by learned vectors, then the positional embedding, the transformer with its causal mask,
  `ln_final` and `resizer`. That is about 15 lines mirroring `text_encoder_ve.py:236-245` and
  `:301-319`. **The probe does exactly this.**
- **A forward hook on `encoder.token_embedding`** that replaces a placeholder token's rows. It
  works too, but it is less clear.

**The output variant.** Learning `language_features` directly is the same injection point with
n × 256 parameters.
- Don't use VETextEncoder's pre-encoded tuple branch (`text_encoder_ve.py:320-326`).
  `SAM3VLBackbone` slices the batch by `len(captions)` (`vl_combiner.py:170-172`), and a tuple has
  length 3.

**Without editing SAM 3: yes.** Replace `backbone.forward_text` on the instance with a wrapper.
- For a sentinel caption, the wrapper returns the learned features; anything else goes to the
  original method.
- Training calls `self.backbone.forward_text` (`sam3_image.py:564`). The video model calls
  `self.detector.backbone.forward_text` (`sam3_video_base.py:560-562`). One wrapper covers both.
- G1's Recorder already wraps detector methods on the instance this way
  (`studies/rpt/sam3_runner.py:78-129`). No subclassing is needed.

**Recommendation:** CoOp tokens are the method; the output variant is an ablation.
- The output variant bypasses the text encoder's prior, so it drifts more easily.
- Drift is the risk for novel objects, and for masks (§A5).

### A2. Whether gradients reach the prompt: **feasible (verified on CPU)**

**The image training path has no blocker.** `Sam3Image.forward_grounding` (`sam3_image.py:441-497`)
runs under no `no_grad`. What exists is benign:

| site | effect |
|---|---|
| `decoder.py:611` | reference boxes are detached between layers, as standard in DETR; the prompt still reaches every layer through cross-attention |
| `encoder.py:306` | `no_grad` around constant reference points only |
| `sam3_image.py:55`, `:348-349` | presence detach is off by default |
| `model_misc.py:779-784` | in-place `*=` and `clamp_` on a matmul output that autograd does not save: safe |
| clamps at ±12 (scorer) and ±10 (joint score, `sam3_image.py:351-353`) | zero gradient for saturated queries only |
| `act_ckpt_utils.py:48-51` | activation checkpointing uses `use_reentrant=False`, on only in train mode |

**The blockers are all in the video predictor:**
- `@torch.inference_mode()` on 9 entry points of `sam3_video_inference.py` (`:54`, `:93`, `:252`,
  `:800`, `:839`, `:1059`, `:1477`, `:1547`, `:1677`);
- also `sam3_video_predictor.py:153`, `:176` and `sam3_base_predictor.py:54`, `:114`.

So P1 trains on frames through the image path. The video predictor is for inference only, which
also bounds Level 2 (§D10).

**The probe** (appendix; `CUDA_VISIBLE_DEVICES=` in the laptop's `sam3` env):
- **Classes:** SAM 3's own fusion encoder, decoder with presence token, dot-product scorer and
  geometry encoder, at d_model 256 with random weights.
- **Text and image:** a tiny `VETextEncoder` (width 64, 2 layers) instead of the 24-layer one; no
  ViT, with random features on a 12 × 12 map.
- **Loss:** `Sam3LossWrapper` with `Boxes` and `IABCEMdetr` (presence on), at ODinW's weights, with
  3 GT boxes.
- **Frozen:** all 33,760,053 SAM 3 parameters.

| learned part | mode | loss | ‖grad‖ | SAM 3 params with a grad |
|---|---|---|---|---|
| CoOp tokens (4 × 64) | train (DAC o2m, 5 aux layers) | 164.65 | 6.80 | 0 |
| CoOp tokens | eval (matching called by hand) | 11.59 | 1.32 | 0 |
| text-encoder output (32 × 256) | train | 168.21 | 13.65 | 0 |
| text-encoder output | eval | 11.59 | 2.47 | 0 |

- **Optimisation:** 20 Adam steps on the CoOp tokens (eval mode) lower the loss from 11.59 to 8.96.
  SAM 3's parameters stay bit-identical.
- **Not shown:** scale, memory, time and real convergence. These need the full-scale smoke step
  (§A4).

**Probe-only workarounds** (none touches SAM 3's files):
- a stub for `torchmetrics`;
- a no-op `pin_memory` (`geometry_encoders.py:649` needs CUDA);
- a PyTorch focal loss in place of the Triton kernel (`loss_fns.py:153-157`, GPU only);
- the decoder built with `resolution=None`, because `decoder.py:300-306` caches coordinates on
  "cuda".

### A3. Training code and losses, box-only: **feasible with work**

**What SAM 3 ships:**
- `sam3/train/`: a Hydra trainer, matchers and losses.
- **The ODinW config is box-only** ("loss config (no mask loss)",
  `train/configs/odinw13/odinw_text_only_train.yaml:76-106`):
  - `Boxes` with L1 5.0 and GIoU 2.0;
  - `IABCEMdetr` with presence: `loss_ce` 20, presence 20, `pos_weight` 5 (`:176-186`);
  - a one-to-many matcher.
- **The model** is built with `build_sam3_image_model(eval_mode=false, enable_segmentation=…)`
  (`:350-354`).

**Switching masks off is clean:** `enable_segmentation=False` (`model_builder.py:579`, `:622-626`)
builds no segmentation head, so no `pred_masks` and no `Masks` loss.

**What we write**, about 800–900 lines including tests:

| item | size |
|---|---|
| the prompt module (CoOp and output variants, KgCoOp term) and the `forward_text` wrapper | ~100 lines |
| the FIT frame dataset (below) | ~150 lines |
| a short training loop instead of the Hydra trainer, which is built for full fine-tuning, SLURM and parameter groups | ~200 lines |
| a backbone-feature cache (the ViT is frozen and does not depend on the prompt) | ~50 lines |
| an image-level evaluation on HELD-A frames (§A4) | ~150 lines |
| tests | ~250 lines |

**The FIT frame dataset:**
- one frame in 30;
- frames resized to a 1008 square, as the video predictor does (`model/io_utils.py:58`), not with
  the trainer's aspect-keeping resize and pad (ODinW yaml `:31-45`);
- boxes as normalised cxcywh;
- `BatchedFindTarget` built by hand (`model/data_misc.py:142-180`).

**Gotchas the probe found:**
- **Normalisation:** `Sam3LossWrapper(normalization="global")` calls `torch.distributed.all_reduce`
  (`train/loss/sam3_loss.py:74-76`). Use "local" on one GPU.
- **The matcher:** it is built only when `eval_mode=False` (`model_builder.py:325-338`), and
  matching runs only in train mode (`sam3_image.py:495-496`). In eval mode, call
  `_compute_matching` yourself, as the probe does.
- **Missing packages:** `torchmetrics`, `hydra`, `submitit` and `fvcore` are not in G1's pinned SAM 3
  environment; they are the `[train]` extra (`pyproject.toml:68-79`).
  - `IABCEMdetr` imports torchmetrics only for a logged F1 (`loss_fns.py:11`, `:509`).
  - Installing extras changes the environment that check 0 pins. Use a separate venv, or install
    torchmetrics alone and re-run check 0 after it.
- **Image model vs video detector.** They share weights (`detector.*` keys,
  `model_builder.py:545-547`) but score differently.
  - The video detector multiplies presence into `pred_logits` (`supervise_joint_box_scores=True`,
    `:739`; `sam3_image.py:345-353`). The image model does not (default False, `:54`).
  - Train as ODinW does, with separate class and presence losses on the image model. The video
    model then scores class × presence, as SAM 3 is meant to be used.

**Two problems specific to VidVRD, both to be fixed before training:**
1. **Frames with no GT box.** IABCE drops every per-query loss there (`keep_loss`,
   `loss_fns.py:423-425`), but keeps the presence loss with target 0 (`:431-440`).
   - Empty frames therefore teach only "nothing here" to the presence head, which at video time
     multiplies every detection in the frame.
   - **Fix:** exclude frames with no GT box.
2. **Gaps.** Many training tids have unannotated frames inside their span; session 3 found 33.5%
   of training tracks under 65% coverage (`studies/rpt/a29_for_review.md:320-322`). FIT is
   probably similar.
   - A sampled frame inside a gap makes an annotated object a negative.
   - **Fix:** interpolate tids across gaps and treat the interpolated boxes as ignore regions. Zero
     the negative term for queries with IoU > 0.5 to them, about 30 lines. Or skip such frames.

**A lever if option (b) suppresses too much:** `weak_loss` with `is_exhaustive=False` removes all
negatives of a frame (`loss_fns.py:447-460`). That is a softer option (b), available without new
code.

### A4. The cost of P1 training: **feasible on a 24 GB 4090**

**Frames:** FIT has about 197k frames (640 × a mean of 308; `option1_design.md:154`). At one in 30
that is about 6.6k frames, fewer once empty frames are excluded.

**Backbone, once:**
- G0's backbone-reuse measurement implies about 25 ms per backbone pass per frame on a 4090:
  423 s → 311 s saved 34 passes on each of 130 frames (`g0_report.md:200`). That is about 3 minutes
  for 6.6k frames.
- The cache is 2.65 MB per frame (72 × 72 × 256, bf16): about 17.5 GB on disk.

**Training (estimate, not measured):**
- Each step runs the fusion encoder over 5,184 image tokens, the 6-layer decoder (200 queries, 400
  with DAC) and the 24-layer text encoder on 32 tokens. The ViT is not in the graph.
- A few GB per frame in bf16, so a batch of 4–8 fits in 24 GB.
- 30–80 ms per frame-step, so 10 epochs take about 0.6–1.5 GPU-hours.

**Choosing the prompt costs more than training it.**
- Video inference on HELD-A (about 23.5k frames) takes about 1–2 GPU-hours per candidate (§A5).
- Five candidates would cost more than the training itself.
- Instead, select on an image-level proxy: recall and unmatched detections at SAM 3's 0.5 cut, on
  sampled HELD-A frames. Run video inference on the final one or two only.

**When 48 GB is needed:** not for training; yes for video inference on long or crowded videos under
"no skips" (§A5).

**Measure it instead of estimating:** a 15-minute full-scale smoke step can ride on the D7 rental
(§E12): the real checkpoint, one batch, peak memory and seconds per step.

### A5. The learned prompt in video inference, and presence: **feasible with work**

**Injecting the prompt:**
- `add_prompt(text=SENTINEL)` sets `find_text_batch[0]` and `text_prompt`
  (`sam3_video_inference.py:868-878`). `has_text_prompt` needs it to be non-None (`:368`).
- The video model encodes the captions once per caption tuple and caches them
  (`sam3_video_base.py:555-566`).
- The batch is `[text, "visual"]` (`sam3_video_inference.py:124`). So the wrapper returns two
  entries, both padded to 32 tokens:
  - the learned features for the sentinel;
  - SAM 3's own features for "visual".

  About 40 lines.
- **Bitwise check:** fed the hand-written prompt's tokens through our path, the wrapper must
  reproduce stock SAM 3 on the smoke video. It can run alongside check 0.

**The runner:** G1's runner accepts only the 35 class names (`studies/rpt/sam3_runner.py:52-55`,
`:332-337`). `run_video` itself takes any list of (name, prompt) pairs (`:283`), so a new entry
point is enough.

**Cost: one pass is not 35× cheaper.**
- **G0, per prompt:** 0.089–0.094 s/frame with no masklets, and 0.118–0.178 with masklets
  (`g0_report.md:163`). Tracking cost grows with the number of masklets.
- **G1** averaged 0.077 s per frame-prompt (34.37 GPU-h / 45,863 frames / 35 prompts), mostly from
  passes that found nothing.
- **A generic pass tracks every object,** so expect at least 0.12 s/frame, and more in crowded
  videos.

| | planners | this estimate |
|---|---|---|
| HELD, 160 videos | 1–2 GPU-h | about 1.6–3 GPU-h or more |
| FIT, 640 videos | 5–6 GPU-h | about 7–14 GPU-h or more |

It is still a few dollars. The D7 check should record seconds per frame and masklets per pass, to
re-cost this.

**Memory goes up, not down.**
- G1 ran out of memory in single-class passes on short, crowded videos (21.5–22.4 GiB;
  `studies/rpt/g1/g1_8x_done_report.md:15-18`).
- One pass with every object holds all masklets at once, so expect more out-of-memory failures on
  24 GB than in G1.
- `max_num_objects` exists (`sam3_video_base.py:301`, `:349-355`), but using it is a non-default
  design choice.

**Presence: sensible, but no help.**
- Presence means "the concept is in this frame". At video time it multiplies every score
  (`sam3_image.py:345-353`; `model_builder.py:739`).
- For "object", or for a class-free prompt trained only on frames with GT, it will sit near 1 on
  almost every VidVRD frame. That is harmless, but it filters nothing; the per-query scores do all
  the work.
- It becomes harmful only if training keeps empty frames (§A3).

**SAM 3's thresholds:**
- The keep cut (0.5) and the new-masklet cut (0.7) (`model_builder.py:747-750`) were tuned for
  named prompts.
- IABCE's soft target, prob^0.25 · IoU^0.75 (`loss_fns.py:371-374`), places matched queries in the
  same regime. That is a reason to keep SAM 3's own loss.

**A gap the plan does not cover: track boxes are mask boxes.**
- G1's adapter takes the tight box of SAM 3's returned mask (`studies/rpt/adapter.py:9-11`,
  `:95-112`), and SAM 3's tracker is seeded from the detector's masks.
- P1's loss trains boxes and scores, not masks, yet the segmentation head also reads the prompt
  (`sam3_image.py:403-411`). A drifting prompt could keep good boxes while its masks degrade.
- **Check:** on HELD-A frames, compare the IoU between the detector's box and its mask's tight box,
  for the learned prompt and the hand-written one.
- **If it drops:** add a box-projection mask loss (BoxInst-style, about 40 lines, with the
  segmentation head on), or raise the KgCoOp weight.

## B. Phase 1, the other parts

### B6. Naming SAM 3's class-free tracks (P4): **feasible with work; faithfulness doubtful**

**There is no given-box path.** Naming lives inline in `End2End_Model`'s test loop
(`models/end2end_model.py:234-265`):

| step | code |
|---|---|
| box → 24 × 24 mask → masked mean of TagCLIP `patch_proj` → L2 norm | `end2end_model.py:234-253` |
| modelB → softmax(100 · cos) | `models/object_classifier.py:58-72` |
| reorder by `new_order` | `end2end_model.py:257-259` |
| fuse with the decoder's class score: 0.6/0.4 for 6 novel classes, 0.3/0.7 for the rest | `:261-264`; RB1's L7 notes red_panda is fused twice and turtle not at all |
| per track: the 35-vectors of detections matching it at IoU > 0.5 | `:303-313` |
| averaged; category = argmax, score = max of the mean | `models/gen_labels.py:605-622` |

**What P4 needs:** about 150 lines and tests. For each SAM 3 track, on its detected frames (or every
k-th):
- the box's pooled `patch_proj` (`studies/rpt/relation.py:24-73` already computes it on demand);
- modelB, loaded as EOV-MMP loads it;
- `new_order`, then the same averaging.

**Faithfulness: the mechanics can be copied exactly; the naming system cannot.**
- Without the decoder score, modelB decides alone. In EOV-MMP it has weight 0.3 on 29 classes.
- EOV-MMP never names with modelB alone, and the repo has no measurement of it. F2 measured
  modelC's head, not modelB: 68.1% top-1, about 0% on novel classes
  (`pilot_analysis/f2/f2_report.md:59-64`).
- modelB was trained on the training videos, so on HELD's base objects it is in-sample, and its
  accuracy there is optimistic.
- So the claim "as in EOV-MMP's SGCls setting" has no code behind it. P4 is a new configuration.

**Why this is the main risk:**
- Gates 2 and 3 are class-aware. G1's +9.80 / +9.16 for the 35-name SAM 3 used the prompt's name,
  which is right by construction for a true object.
- PaD's class-aware recall is its class-agnostic recall × naming accuracy.
- **Example:**
  - R0w's recall is 67.93% (`sam3_screen_report.md` §1), so gate 2's +10 needs 77.9% class-aware.
  - If modelB names 85% of matched tracks correctly, that needs about 92% class-agnostic recall.

**Cheap check, before any rental:** rename G1's 35-name SAM 3 tracks on HELD-A with modelB alone.
Report:
- class-aware recall and pair recall with modelB's names against the prompt names, base and novel
  separately;
- modelB's top-1 on HELD-A GT boxes.

CLIP-L/14@336 and modelB fit a 4 GB GPU in fp16. So this runs on the laptop for $0, after about
half a day of code.

**The triplet score needs a decision.**
- The score is `sbj_scr × obj_scr × pre_scr` (`inference/post_process.py:98`). Only the ranking
  within a video matters, because of the top 200 (`:102-110`).
- The proposal's "capped score" must become a rule, fixed in the amendment: modelB's maximum
  probability, P3's keep probability, or their product.

### B7. The learned keep/drop model (P3): **feasible with work**

**Easy to get from G1-style runner records** (`adapter.build_track`, `adapter.py:95-112`: per-frame
mask box, SAM 3 box, mask, probability, detections, tracker status):
- track length;
- mean and minimum score over detected frames;
- detected and filled fractions;
- box stability (IoU between consecutive boxes, variation of size);
- border contact and size;
- containment in a larger track (screen rule 3);
- mask area / box area, and its stability.

**Needs computation:** CLIP-feature consistency and the naming margin. Both need TagCLIP patch
features per frame, as P4 does; about 50 lines on top of it.

**Worth adding:** the mask fill ratio, against slivers and background; the propagated-only fraction,
against masklets that live only on propagation; relative size.

**Caveats:**
1. **P3 must be fitted on P1's tracks,** from the learned-prompt SAM 3 on HELD-A, not on G1's
   35-name tracks; their scores and their mix of tracks differ. G1's outputs can only prototype the
   code. So P3 waits for P1's HELD-A run.
2. **The naming margin is in-sample on HELD** (modelB).
3. **The sample is small:** about 78 HELD-A videos. Keep the model small (logistic regression or
   shallow trees) and cross-validate by video.

### B8. Relations (P5): **feasible**

**What `gen_feats_test` needs** (`gen_labels.py:642-719`; `adapter.check_repo_tracks`,
`adapter.py:170-183`): per track, a dense trajectory, one of the 35 names, and a float score.

**The name does not enter modelC's features.**
- Only `rel`, `mot`, `clip` and `bbox` features are extracted (`gen_labels.py:647`); the word-vector
  `lan_feat` (`:67-70`) is unused.
- The name and score enter post-processing only (`:675-683`; `post_process.py:40-51`, `:98`), and
  the T1 rules.

**Changes:**
- a class-free adapter, because `tracks_from_sam3` requires a 35-name prompt (`adapter.py:56-61`,
  `:191-206`), taking P4's name and the chosen score;
- no deduplication across prompts, since there is one prompt.

The rest is as in G0 and G1 (`relation.py`).

**Watch:**
- **Pairs:** every ordered pair of tracks is a pair (`gen_labels.py:654-659`), so cost and top-200
  competition grow with the square of the burden. P3 must run before P5.
- **HELD-B:** `gen_feats_test` opens the video's annotation file for width and height
  (`gen_labels.py:645-648`, `:25-29`).
  - It does not matter today, since nothing computes relations on HELD-B.
  - It would trip the HELD-B guard, so any future relation run there must take width and height
    from the frames.

## C. Phase 2 (extension): feasibility and size

### C9. RB1's vendored step-4 code: **feasible with work**, about twice the size planned

**The path is about 3,900 lines:**

| file | lines |
|---|---|
| `end2end_model` | 353 |
| `gen_labels` | 746 |
| `relation_classifier` | 573 |
| `ov_prompt` (model, criterion, matcher, post-processing, transformer) | 2,083 |
| `cli/train` | 155 |

**Changes, file by file** (in `studies/metor_code/`):

| item | in the code today | change |
|---|---|---|
| class-free box targets | GT boxes with labels | loader and target builder with a "teacher" flag |
| matching | class cost on target labels (`ov_prompt/matcher.py:62-83`) | class cost 0 for teacher targets |
| class loss | `label_to_idx[label]` raises for a target without a label (`set_criterion.py:276-277`) | mask queries matched to teacher boxes |
| **CLIP-embedding loss** (missed by the reviews) | target = the label's CLIP query (`set_criterion.py:325-344`) | mask queries matched to teacher boxes |
| generic "object" query | T2: queries are the labels present (`ov_prompt/model.py:309-330`) | an "object" text query in training and at test, so R2 is rerun with it |
| ignore regions | none | zero the class and objectness losses of unmatched queries over ignore boxes |
| objectness on `pred_ious` | **already trained** (row below) | extend IoU targets to teacher boxes, weighted by P3; keep rule at `ov_prompt/post_process.py:194-195` |
| relation half | F3 makes pairs on teacher-only objects negatives | drop those pairs before modelC's loss, in the step-4 loop |
| frame gate (T3) | `len(labels) > 0` | count frames that have only teacher boxes |

**`pred_ious` is already trained:**
- The loss is `loss_iouaware`, a BCE against the IoU on matched queries (`set_criterion.py:383-398`).
- It is on by default (`utils/parser_func.py:262-264`, weight 2).
- Checked on CPU (memory-mapped, parameter names only): both the stage-1 checkpoint's `args` and the
  released end-to-end checkpoint's `config` have `iou_aware=True`, and the latter holds
  `modelA.iou_embed.0–5`.

**But it is not an objectness head.**
- It predicts IoU quality on matched queries and is never trained on background. Reading it as "is
  this an object" for unmatched queries is extrapolation.
- OLN-style objectness learns from many sampled proposals per object; Hungarian matching gives one
  high-IoU query per object.
- So plan a real objectness target: matched queries as soft positives, unmatched queries far from
  any box as negatives, and ignore near boxes. That is C4, and it makes gate 4's "C4-only head"
  baseline a like-for-like comparison.

**Size:** about 500–600 lines of code and 300–400 of tests, not "a few hundred".

**The risk: step 4 has never run on a GPU.**
- The released path crashed on this data (`studies/rb1/step1_report.md` §3), and the fix was tested
  on CPU only (§2).
- RB1's 5-video check comes first, for about $2–5.
- RB1's own unmeasured estimate is 80–110 GPU-hours per configuration (§8).
- Phase 2 needs at least two configurations (R2 with the generic query, then the student): about
  160–220 GPU-hours, or 4–5 days of wall time each on one GPU.

## D. The stretch goal

### D10. Level 2 (relation loss → prompt through soft masks): **doubtful**

**What breaks first:**
1. **Tracking is under `inference_mode`** (§A2). No gradient passes through tracking without
   rewriting the video model.
2. **The steps between are discrete:** the 0.5 and 0.7 cuts, NMS, matching detections to tracks,
   and hotstart.
3. **Boxes are not differentiable here:**
   - masks become tight boxes (adapter);
   - EOV-MMP pools features with a hard 24 × 24 box mask (`gen_labels.py:95-111`;
     `end2end_model.py:235-252`).
4. **modelC is in-sample on FIT,** so a relation loss on FIT feeds back a memorised signal.

**A feasible cousin** (2–3 weeks on a 48 GB GPU):
- fix the tracks with a no-grad run;
- on clip frames, run `forward_grounding` with gradients and take the queries matched to the tracks;
- soft-pool patch features with their masks, feed modelC, and backpropagate to the prompt.

modelC was trained on box-pooled features, though, so this is a distribution shift. It is research,
not something for the thesis plan.

## E. The next session (Part 6)

### E11. Implementable as written? **Step 2: blocker. Steps 1 and 4: feasible with fixes**

**Step 1, the split:**
- **The procedure is ambiguous.** It does not say:
  - whether one generator runs over all strata or there is one per stratum;
  - the order of the strata;
  - whether to use `shuffle` or `permutation`.
- **80/80 is not guaranteed.** If the A/B alternation restarts in each stratum, odd strata give up
  to 4 extra A's (82/78), contradicting 80/80.
  - **Fix:** one generator; strata in a fixed order (novel flag, then length flag); the alternation
    continues across strata. That gives exactly 80/80.
- **"Longer than the median"** needs a source (the frame folders) and a rule for ties. With 160
  videos, the median is the mean of two counts.
- **The novel flag** reads all 160 HELD annotations. That is fine before the guard is installed;
  the amendment should say so.

**The guard (ground rule 3).** A path guard like `studies/rpt/no_test_labels.py:7-13` works for
per-video files: `anno/train/<vid>.json`, `studies/rpt/g1/sam3_out/<vid>.json.gz`, `eov_out/`. It
cannot filter aggregate files that contain HELD-B:
- `train_object_trajectories_gt.json` (`paths.TRAIN_TRAJ`), which is also what P1's loader would
  read for FIT;
- `pilot_analysis/stage2/data/held_gt.json`;
- `studies/rpt/g1/results_futility/{futility,box_style_values_157}.json` and
  `results_descriptive/descriptive.json`;
- `studies/postg1/results_screen/sam3_screen.json`;
- `studies/rpt/g1/viz_unmatched/` and `held_gt_missing3.json`.

**Fix:** before sealing, write hashed FIT-only and HELD-A-only extracts, then block the aggregates by
path.

**HELD-B is sealed, not unseen.** All of these used all 157 videos, HELD-B included:
- G1;
- A29d's futility rule;
- the descriptive analysis and the 20 inspected tracks;
- A30's 1,000-configuration screen.

The amendment should say: HELD-B is fresh for decisions made from now on, and P3's family of rules
was chosen after seeing results on all of HELD.

**Step 2 conflicts with A30.**
- **Task 2 already ran** under A30 (commit `b04b714`; `sam3_screen_report.md`): 0 of 500
  configurations with the class gate pass.
- **Its rule was binding:** "Fail → SAM 3 closes as a detector; no second round of rules on HELD"
  (`monday-pack.md:102`).
- **Re-running on HELD-A is a second round,** with the answer already known. The "restated closing
  rule" narrows a binding rule after its result.
- **"Report only HELD-A's inspected tracks" cannot be followed:** all 20 are already in a committed
  file (`studies/rpt/g1/viz_unmatched_observations.md`).
- **Fix:**
  - no re-run;
  - one amendment as in Summary item 1;
  - for P3's starting filter, keep the rule family (tiny, edge sliver, fragment) and fit thresholds
    later on P1's own HELD-A tracks. Thresholds from the 35-name tracks would not transfer to a
    different detector anyway.

**Step 4, the D7 check:**
- **The chain head is A30** (`b10cf7a7…0aae`), not A29d. The next label is A31, since the trimming
  oracle is skipped.
- **Check 0:** smoke video 1 is a HELD video (`…00008005`), so it may land in HELD-B.
  - **Fix:** use the first G0 smoke video that is in HELD-A. G0 has reference outputs for all three
    (`studies/rpt/g0/sam3_out/`).
- **Code:** the runner and adapter accept only the 35 names (`sam3_runner.py:332-337`;
  `adapter.py:56-61`, `:170-183`).
  - **Fix:** a new entry point that reuses `run_video`, G1's driver and machine scripts, and the
    existing fake worker (`studies/rpt/tests/fake_sam3_worker.py`). About one day.
- **The guard's edge cases:**
  - The prompt does not say what happens if the gain is still below +5 on 40 videos, or is ≤ 0. At
    ≤ 0, "half" passes trivially.
  - **Fix:** in either case, stop and report; the check cannot discriminate.
  - Also define "the better arm" as the one with the higher class-agnostic recall gain.
- **Burden is not gated.** "object" can raise class-agnostic recall by over-segmenting.
  - **Fix:** add a reported, ungated measure, **recall at an equal budget**: each system's confident
    tracks cut by score to the 35-name SAM 3's count per video.
  - If the generic arm passes only through burden, the learned prompt's job is much harder than
    planned.
- **The memory rule:**
  - Arm A puts every object into one pass, so expect more out-of-memory failures than in G1. Each
    skip removes a crowded video from all four systems.
  - **Fix:** use one 48 GB GPU (G1 priced one at $0.457/h) and "no skips" here too. It costs under
    $1 more, removes the bias, and measures the memory the 48 GB plan needs.
- **The cost estimate is too low.**
  - 0.094 s/frame is a figure for passes with no masklets (`g0_report.md:163`).
  - With generic prompts: 20 videos × about 294 frames × 5 passes × 0.12–0.25 s ≈ 1–2 GPU-h. 40
    videos ≈ 2–4 GPU-h. Add about an hour of setup.
  - That is $1–3 for 20 videos.
- **Record for every pass:** seconds per frame, masklet count, and peak memory. These re-cost PaD.

### E12. Order of work: **yes, change it**

1. **The P1 gradient path:** done in this review, on CPU, for $0.
2. **Step 1:** the split, the extracts and the guard, with the fixes above.
3. **The A30 scope amendment** in place of Step 2.
4. **New: the naming check** (§B6) on HELD-A, on the laptop, for $0.
   - If modelB alone loses most of the class-aware gain on G1's tracks, gate 2's +10/+10 is out of
     reach whatever the prompt does.
   - Know this before renting anything.
5. **The D7 check,** on one 48 GB machine. After the arms, on the same machine:
   - the forward_text wrapper's bitwise check (§A5);
   - a P1 full-scale smoke step: real checkpoint, a few FIT frames, one batch; peak memory and
     seconds per step.

   This needs a few FIT frames in the upload and the training extras in a separate venv. Both go
   in the amendment. It costs about $0.20 extra.
6. **Then the P1 code.** P3 waits for P1's HELD-A run (§B7).

## F. Overall

### F13. Effort

**Phase 1:**

| work | working days |
|---|---|
| split, extracts, guard, the two amendments | 1.5 |
| P4 naming module and the naming check | 2 |
| D7 check scripts, run, report | 2 |
| P1 training code: dataset, cache, prompt, loop, image proxy, tests | 4–5 |
| P1 video wrapper and its bitwise check | 1 |
| P1 runs and the HELD-A video evaluation | 1–2 |
| P3: features, model, cross-validation | 2–3 |
| gates 2 and 3: amendments, runs, reports | 2–3 |
| P5 integration, T1, test-session preparation | 2 |
| **total** | **about 18–22 days (4–4.5 weeks), against the planners' 3–4** |

The dollar cost (about $15–40) is in line with the planners'.

**Phase 2:**

| work | days |
|---|---|
| RB1's 5-video GPU check, plus fixes if it crashes | 2–5 |
| Phase 2 code and tests | 6–8 |
| cross-fit teacher (2 P1 trainings) and FIT teacher runs (48 GB, no skips) | 2–3 |
| R2 with the generic query (80–110 GPU-h) | 4–5 of wall time |
| the student (the same) | 4–5 of wall time, in parallel if possible |
| C4-only head and gate 4 | 2–3 |
| **total** | **about 5–7 weeks in sequence, about 4 with parallel machines; against the planners' 3–4** |

The dollar cost is about $100–200.

### F14. The top 3 risks, and the cheapest check for each

| risk | check that retires it | cost |
|---|---|---|
| 1. **modelB alone names SAM 3's tracks too poorly** for the class-aware gates | the naming check on G1's HELD-A tracks (§B6) | $0, laptop, about 1 day |
| 2. **Without class names, SAM 3 does not find VidVRD's objects** at a bearable burden or memory | the D7 check on 48 GB, with recall at an equal budget and memory logged (§E11) | $2–4 |
| 3. **P1's targets teach suppression:** empty frames, gaps, non-exhaustive labels; masks drifting from boxes | counts on the FIT annotations before writing the loader (share of empty sampled frames, share of sampled frames inside a gap); then the mask–box check on the first trained prompt (§A3, §A5) | $0, then part of P1 |

For Phase 2, the top risk is the never-run step-4 GPU path. RB1's 5-video check retires it.

### F15. What the planners got wrong or missed

1. **Protocol.**
   - Part 6 re-runs Task 2 and re-scopes A30's binding verdict after the result.
   - HELD-B is treated as unseen, but G1 and A30 used it (§E11).
2. **Cost and memory of one pass.** It is not 35× cheaper; tracking scales with the number of
   objects. Memory per pass goes up, so the out-of-memory skips in the D7 check are likely to be
   more than "slight" (§A5).
3. **P4 is new code and a new naming system,** not "EOV-MMP's SGCls setting". Under class-aware
   gates, naming is the binding risk.
   - Gate 2's comparison with the 35-name SAM 3 is not like for like: that system's names come from
     its prompts, PaD's from modelB.
   - modelB is in-sample on HELD (§B6).
4. **Track boxes come from masks;** P1's loss trains boxes, and masks are unsupervised (§A5).
5. **P3 must be fitted on P1's tracks,** which fixes the order of work (§B7).
6. **Empty frames and gaps** interact with the presence loss and option (b) (§A3).
7. **Choosing the prompt on HELD-A video runs** costs more than training; use an image-level proxy
   (§A4).
8. **Phase 2:**
   - `pred_ious` is already trained, and it is not objectness;
   - `loss_embed` and the matcher's class cost also need masking;
   - the code is about twice "a few hundred lines" (§C9).
9. **The environment:**
   - SAM 3's training loss needs packages that are not in G1's pinned environment;
   - `normalization="global"` needs `torch.distributed`;
   - plan a separate venv and re-check check 0 (§A3).
10. **The chain head** is A30 and the next label is A31; Part 6 names A29d (§E11).

---

## Appendix: the CPU gradient probe

Run with `CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/sam3/bin/python -I grad_probe.py`. It loads no
checkpoint and does not use SAM 3's ViT; it ran on the laptop's CPU. The script's SHA256 is
`bd1feeb9b565376419e441ed1d58fb4df5f1bf212ad28423cd645d95c3a638e0`.

**The 20-step optimisation check:** the same file up to `report = {}`, then 20 `torch.optim.Adam`
steps (lr 1e-2) on `ctx_tok` with `step(False)`. The loss went 11.589 → 8.961; SAM 3's parameters
were unchanged (`torch.equal` on every tensor).

<details><summary><code>grad_probe.py</code></summary>

```python
"""CPU-only probe for the PaD developer review (2026-11-03). Not SAM 3 at scale: SAM 3's own detector
classes (fusion encoder, decoder with presence token, dot-product scorer, geometry encoder) at their
real d_model=256, random weights, nothing loaded from sam3.pt; a TINY text encoder (VETextEncoder,
width 64, 2 layers) instead of the 24-layer one; no ViT (backbone features are random tensors on a
12x12 map). Question: with every SAM 3 parameter frozen, does SAM 3's shipped box-only loss
(Boxes + IABCEMdetr with presence, Sam3LossWrapper, ODinW's weights) put a non-zero gradient on
(a) CoOp-style context token embeddings fed through the text encoder, and (b) a learned text-encoder
output? Run with: CUDA_VISIBLE_DEVICES= python -I grad_probe.py
"""
import os
import sys
import types

import torch
import torch.nn as nn

# torchmetrics is only used by IABCEMdetr for a logged F1 (loss_fns.py:509); it is not installed in
# the laptop's sam3 env (pyproject's [train] extra). Stub it: it does not touch the loss.
tm = types.ModuleType("torchmetrics")
tm.functional = types.SimpleNamespace(f1_score=lambda *a, **k: torch.tensor(0.0))
sys.modules["torchmetrics"] = tm
# geometry_encoders.py:649 pins a tensor, which needs CUDA; on this CPU-only probe it is a no-op.
torch.Tensor.pin_memory = lambda self, *a, **k: self

import pkg_resources  # noqa: E402
from sam3.model.decoder import TransformerDecoder, TransformerDecoderLayer  # noqa: E402
from sam3.model.model_misc import MultiheadAttention  # noqa: E402,F401
from sam3.model.sam3_image import Sam3Image  # noqa: E402
from sam3.model.text_encoder_ve import VETextEncoder  # noqa: E402
from sam3.model.tokenizer_ve import SimpleTokenizer  # noqa: E402
from sam3.model.vl_combiner import SAM3VLBackbone  # noqa: E402
from sam3.model.data_misc import BatchedFindTarget  # noqa: E402
from sam3.model.model_misc import SAM3Output  # noqa: E402
import sam3.model_builder as B  # noqa: E402
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher  # noqa: E402
from sam3.train.loss.sam3_loss import Sam3LossWrapper  # noqa: E402
from sam3.train.loss.loss_fns import Boxes, IABCEMdetr  # noqa: E402

import sam3.train.loss.loss_fns as LF  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _torch_focal(inputs, targets, alpha, gamma):
    """loss_fns.py's own non-Triton branch (sigmoid_focal_loss, triton=False), summed: the Triton
    kernels (loss_fns.py:153-157) need a GPU."""
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss
    return loss.sum()


LF.triton_sigmoid_focal_loss_reduce = _torch_focal

torch.manual_seed(0)
assert not torch.cuda.is_available(), "CPU only"

bpe = pkg_resources.resource_filename("sam3", "assets/bpe_simple_vocab_16e6.txt.gz")
tok = SimpleTokenizer(bpe_path=bpe)
text = VETextEncoder(d_model=256, tokenizer=tok, width=64, heads=2, layers=2)

# The decoder exactly as model_builder._create_transformer_decoder, except resolution/stride=None so its
# coordinate cache is built lazily on CPU instead of on "cuda" at construction (decoder.py:300-306).
import inspect  # noqa: E402
src = inspect.getsource(B._create_transformer_decoder)
assert "resolution=1008" in src and "stride=14" in src
dec_layer = TransformerDecoderLayer(
    activation="relu", d_model=256, dim_feedforward=2048, dropout=0.1,
    cross_attention=B.MultiheadAttention(num_heads=8, dropout=0.1, embed_dim=256), n_heads=8,
    use_text_cross_attention=True)
decoder = TransformerDecoder(
    layer=dec_layer, num_layers=6, num_queries=200, return_intermediate=True, box_refine=True,
    num_o2m_queries=0, dac=True, boxRPB="log", d_model=256, frozen=False, interaction_layer=None,
    dac_use_selfatt_ln=True, resolution=None, stride=None, use_act_checkpoint=True, presence_token=True)
transformer = B.TransformerWrapper(encoder=B._create_transformer_encoder(), decoder=decoder, d_model=256)

backbone = SAM3VLBackbone(visual=nn.Identity(), text=text, scalp=1)
matcher = BinaryHungarianMatcherV2(focal=True, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0,
                                   alpha=0.25, gamma=2, stable=False)
model = Sam3Image(backbone=backbone, transformer=transformer,
                  input_geometry_encoder=B._create_geometry_encoder(), segmentation_head=None,
                  num_feature_levels=1, o2m_mask_predict=True,
                  dot_prod_scoring=B._create_dot_product_scoring(), use_instance_query=False,
                  multimask_output=True, matcher=matcher)  # supervise_joint_box_scores=False, as the image model
with torch.no_grad():   # torch.empty parameters (positional_embedding, text_projection, ...) are filled
    for name, p in model.named_parameters():        # by sam3.pt in real use; here: small random values
        if not torch.isfinite(p).all() or p.abs().max() > 1e3 or name.endswith(("positional_embedding", "text_projection")):
            p.normal_(0.0, 0.02)
for p in model.parameters():
    p.requires_grad_(False)
n_frozen = sum(p.numel() for p in model.parameters())

# ---- the learned prompts
N_CTX = 4
template = " ".join(["X"] * N_CTX)                      # placeholder words; SOT X X X X EOT pad...
ids = tok([template], context_length=text.context_length)
ctx_tok = nn.Parameter(text.encoder.token_embedding(ids)[0, 1:1 + N_CTX].detach().clone())   # (a) CoOp
ctx_out = None                                                                                    # (b)

SENTINEL = "<learned prompt>"


def coop_text_features(device):
    """(a): SAM 3's own text path (text_encoder_ve.py:236-245, 301-319) with the placeholder token
    embeddings replaced by ctx_tok; calls SAM 3's submodules, edits nothing."""
    enc = text.encoder
    t = tok([template], context_length=text.context_length).to(device)
    x = enc.token_embedding(t)
    x = torch.cat([x[:, :1], ctx_tok[None], x[:, 1 + N_CTX:]], dim=1)
    L = t.shape[1]
    x = x + enc.positional_embedding[:L]
    x = enc.transformer(x, attn_mask=enc.attn_mask[:L, :L])
    x = enc.ln_final(x)
    mask = (t != 0).ne(1)                                 # True = padding, as VETextEncoder inverts it
    mem = text.resizer(x.transpose(0, 1))                 # (seq, b, 256)
    return mask, mem


orig_forward_text = backbone.forward_text


def forward_text(captions, input_boxes=None, additional_text=None, device="cpu"):
    """Instance-level wrapper (the pattern G1's runner uses on the detector): the sentinel caption gets
    the learned features; anything else goes to SAM 3's own forward_text."""
    if list(captions) == [SENTINEL]:
        mask, mem = coop_text_features(device) if ctx_out is None else (out_mask, ctx_out[:, None])
        return {"language_features": mem, "language_mask": mask, "language_embeds": mem}
    return orig_forward_text(captions, input_boxes, additional_text, device)


backbone.forward_text = forward_text

# ---- one fake frame: backbone features on a 12x12 map (the real one is 72x72), 3 GT boxes
H = W = 12
feat = torch.randn(1, 256, H, W)
pos = torch.randn(1, 256, H, W)
gt = torch.tensor([[0.30, 0.40, 0.20, 0.30], [0.70, 0.50, 0.25, 0.40], [0.50, 0.80, 0.10, 0.10]])  # cxcywh
n = len(gt)
target = BatchedFindTarget(
    num_boxes=torch.tensor([n]), boxes=gt, boxes_padded=gt[None], repeated_boxes=gt,
    segments=None, semantic_segments=None, is_valid_segment=None,
    is_exhaustive=torch.tensor([True]), object_ids=torch.arange(n), object_ids_padded=torch.arange(n)[None])
find_input = types.SimpleNamespace(img_ids=torch.tensor([0]), text_ids=torch.tensor([0]))

loss_fn = Sam3LossWrapper(
    loss_fns_find=[Boxes(weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0}),
                   IABCEMdetr(pos_weight=5.0, weight_dict={"loss_ce": 20.0, "presence_loss": 20.0},
                              weak_loss=False, alpha=0.25, gamma=2, use_presence=True, pos_focal=False,
                              pad_n_queries=200, pad_scale_pos=1.0)],
    normalization="local",                    # "global" calls torch.distributed.all_reduce (sam3_loss.py:75)
    matcher=matcher, o2m_matcher=BinaryOneToManyMatcher(alpha=0.3, threshold=0.4, topk=4),
    o2m_weight=2.0, use_o2m_matcher_on_o2m_aux=False)


def step(train_mode: bool):
    model.train(train_mode)
    bo = {"backbone_fpn": [feat], "vision_pos_enc": [pos], **backbone.forward_text([SENTINEL], device="cpu")}
    out = model.forward_grounding(backbone_out=bo, find_input=find_input, find_target=target,
                                  geometric_prompt=model._get_dummy_prompt(num_prompts=1))
    if not train_mode:                         # forward_grounding matches only in training (sam3_image.py:495-496)
        model._compute_matching(out, model.back_convert(target))
    stages = SAM3Output(iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE)
    stages.append([out])
    losses = loss_fn(stages, [model.back_convert(target)])
    return out, losses


report = {}
for name in ("coop_tokens", "output_embedding"):
    if name == "output_embedding":
        with torch.no_grad():
            m, mem = coop_text_features("cpu")
        out_mask = m
        ctx_out = nn.Parameter(mem[:, 0].detach().clone())   # (seq, 256): learn the encoder output itself
        param = ctx_out
    else:
        param = ctx_tok
    for train_mode in (True, False):
        param.grad = None
        out, losses = step(train_mode)
        total = losses["core_loss"]
        total.backward()
        g = param.grad
        report[f"{name}/{'train' if train_mode else 'eval'}"] = dict(
            loss=round(float(total), 4),
            grad_norm=None if g is None else float(g.norm()),
            frozen_param_grads=sum(p.grad is not None for p in model.parameters()),
            keys=sorted(k for k in losses if not k.startswith(("ce_f1", "presence_dec_acc")))[:6],
            has_o2m="pred_logits_o2m" in out, n_aux=len(out.get("aux_outputs", [])))

print("frozen SAM 3 params (this tiny build):", n_frozen)
for k, v in report.items():
    print(k, v)
```

</details>
