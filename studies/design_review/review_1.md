# Design review 1 — the plan "relations decide which objects survive"

Independent review for the plan's author. Code was read in EOV `fdeeeb4` (the code that
made the baseline) and METOR `HEAD`. No GPU, no training, and no change to the pipeline
or the study code.

Test-derived data was used once, for Q7, under a descriptive amendment written first
(**A25e**). It read only D1's `tc_missed.json`. D2's recordings and their derived outputs
were not used, because A25d bars them from design. D2's published aggregate shares are
cited, which A25d allows.

Marks: **AGREE** · **DISAGREE** · **UNSURE**.

---

## 0. Verdict in brief

- **AGREE: the front end is where SGDet novel is lost.** D2 found about 70% of U1
  relations lost before tracking.
- **AGREE: there is real headroom.** Q7 says perfect tracks and names would recover
  **106–108 of the 262 U1 relations (about 41%)** with the released relation head.
  SGDet novel recall would rise from 20.0% to at most about 37.8%, roughly PredCls's
  40.3%.
- **DISAGREE: the relation cue the plan relies on.**
  - The detector's relation head is **not computed at test** (Q1).
  - The relation classifier only scores **pairs of tracks of ≥ 10 frames**
    (`gen_labels.py:659`).

  So "relation-aware *box* keeping" can't be built from existing parts. The workable
  version is relation-aware **tracklet** keeping at a lower cut.
- **DISAGREE: the first step.** Building our own detector first is the most expensive
  and riskiest step, and it doesn't answer the question that decides feasibility: how
  much of the published 16.75 / 15.79 comes from protocol leaks. That can be measured
  more cheaply, with RB1 plus an eval-only protocol audit.
- **UNSURE: whether relation plausibility helps decide objectness at all.** The
  released relation head misses most relations even with perfect tracks: in PredCls,
  C1 is 82% of its novel misses (D1). D1's post-hoc check found most of those are runs
  that continue past the GT, not predicates the head never sees. Meanwhile, the
  detector already computes a class-agnostic quality score that nothing uses (Q3).
- **Recommendation (Q8):**
  1. settle the comparison protocol with cheap measurements;
  2. design the front-end step as tracklet keeping on class-agnostic signals plus
     relation plausibility;
  3. calibrate it on a clean object-level held-out set (ImageNet-VID videos outside
     VidVRD) instead of building a detector first.

---

## 1. Answers

### Q1 — What does the detector's relation query output, and is it used after training?

It is **frame-level, training-only, and not computed at inference.**
- **Training.** One extra learned query, `pred_query_embed = nn.Embedding(1, …)`
  (`models/methods/detectors/ov_prompt/model.py:103`), is appended to the 300 object
  queries (`:340-342`). Its last decoder state goes through
  `pred_embed = MLP(256, 512, 71, 2)` (`:260`) and a sigmoid, giving `new_logit`
  (`:346-347`). That is a multi-hot over the **71 base predicates, one per frame.**
- **The loss** is a weighted BCE against the frame's predicate multi-hot `pred_label`
  (`set_criterion.py:400-419`), built by `gen_frame_pre_label` from every relation in
  the frame.
- **Test.** `forward_test` builds its queries from `self.query_embed.weight` only
  (`model.py:456`). The predicate query is never appended, and `new_logit` is never
  computed (`:454-534`).

So the head shapes the decoder during training and then disappears. Weak point (a) is
**confirmed, and stronger than stated.** It's also not per box. The training-time
self-attention included this extra token, and inference doesn't, which is a small
train/test mismatch.

### Q2 — What would training the detector from scratch on COCO need?

**What exists:**
- the model (`ov_prompt/model.py`) and its build function (`:536-669`);
- the criterion (`set_criterion.py`): focal classification over the classes present,
  L1 + GIoU boxes, `loss_embed`, `loss_iouaware`, and `loss_new` (the frame
  predicates);
- the Hungarian matcher and the post-processor;
- the CUDA deformable-attention op (`ops/`);
- the frozen TagCLIP and CLIP encoders (`vlm/backbones`);
- the text-embedding builders, including one for COCO (`models/util/clip_utils.py`).

**What is missing:**
- **A detector training loop.** EOV-MMP never released steps 1–3 (A22 §1b; METOR
  `docs/32_Not-implemented.md` §1). The only training code is step 4, end to end
  (`cli/train.py`).
- **Loaders.** There is no COCO loader, and no frame-level VidVRD detector loader:
  the released config's `dataset_file = open_vidvrd` has no dataset class in the repo.
- **CLIP features.** The detector's input is the frozen TagCLIP 24 × 24 × 768 patch
  grid, computed online per frame at 336 × 336 (`data_loading/dataset.py:98-119`).
  - For COCO train2017 (~118k images), caching costs about 885 KB per image in fp16,
    about 104 GB in all.
  - The alternative is re-encoding every epoch with two ViT-L passes per image.
- **The distillation target is class-level, not region-level.** `loss_embed` regresses
  each matched query's `pred_embed` onto the same CLIP vector used as that class's
  query (`set_criterion.py:325-343`): the class's text embedding 75% of the time, or
  one exemplar crop from the bank 25% (`model.py:318-330`).
  - So COCO needs base-class text embeddings (cheap) and a COCO crop bank.
  - The bank tool exists only for VidVRD, and **the bank in this repo is a
    reconstruction, not the authors' file** (`tools/build_clip_object_bank.py:20-35`).
- **An open question about the recipe.** The released detector's saved args say:
  `dataset_file = open_vidvrd`, 60 epochs, `lr = 1e-5`, batch 16,
  `pre_trained = imagenet`, an empty `resume`, `new_loss = True` (coefficient 3),
  `iou_aware = True`, `use_adapter = True`. `coco_path` is only an argument default.
  - The EOV-MMP paper says the decoder starts from a COCO detector. **The checkpoint
    shows no evidence of a COCO stage.**
  - A fine-tuning rate of 1e-5 suggests it started from *something*.

  **UNSURE.** If there was no COCO stage, a clean detector could be trained on FIT
  base frames only, which would be cheaper. Which frames the authors used (every frame,
  or every 30th as the paper says) is unknown, and it changes the cost by up to 30×.

**Estimate.** These are ranges, not measurements.

| part | engineering | GPU |
|---|---|---|
| COCO OVD stage (loop, loader, bank, evaluation) | 3–6 weeks; DETR training is finicky and has no reference loop | 50–150 GPU-h |
| VidVRD fine-tune | 1–2 weeks | from a few GPU-h (every 30th frame) to 100+ GPU-h (every frame, 60 epochs) |
| at $0.30–0.60/h (assumed; no price recorded) | | about $20–150 in all |

The bigger risk is not cost but **ending up well below the released detector**. M1's
clean frozen-CLIP stack scored 9.92 novel PredCls against EOV-MMP's 21.65. That gap
isn't attributed yet, but it warns how far a clean rebuild can fall.

### Q3 — For box keeping, what candidates and inputs are realistic?

**Candidates.**
- Use the post-processor's own threshold (`OVPostProcess.pruning_threshold`,
  `post_process.py:132`), lowered to about 0.1, after the dedupe. D2's published
  Table 6 shows the O1b boxes sit around 0.1–0.2: pooled median 0.185; 73% of frames
  ≥ 0.1 and only 34% ≥ 0.3.
- deep_sort ignores scores (every detection has confidence 1, `deep_sort_app.py:55`),
  so everything above the lower cut becomes **tracklets**. The keeping decision
  should be made per tracklet, not per box.

**Inputs the detector already computes and nothing uses:**
- **`pred_ious`:** a class-agnostic IoU-quality score per query, computed at test
  (`model.py:519-524`) and trained (`iou_aware = True`). It is not read by the
  post-processor or the tracker (no reference in `post_process.py` or
  `end2end_model.py`).
- **`pred_embed`:** the query's 768-d CLIP-space embedding, computed at test
  (`model.py:503-504`) and unused downstream.
- The 35-class probabilities, modelB's fused scores, TagCLIP RoI means (already
  computed for deep_sort, `end2end_model.py:235-255`), and the tracklet's length,
  persistence and AFLink link.

**Relational inputs.**
- **modelC itself**, run on extra (candidate tracklet, confident track) pairs. It's
  already loaded at test. Its inputs are per-30-frame-clip location, motion, CLIP RoI
  and box features (`gen_labels.py:14-60`, `:642-719`). It needs ≥ 10 frames of
  overlap (`:659`), which is one more reason the unit is the tracklet.
- **METOR's IEM** (`metor/compat.py:92-127`) refines pair embeddings inside the
  relation classifier, after pairs are formed. It doesn't score candidates.

**UNSURE** whether relation plausibility adds anything over `pred_ious` plus
persistence. Measure it, don't assume it.

### Q4 — How do we train keeping without making novel objects negatives?

**There is no target that removes the bias completely.** In clean training videos, a
novel object is unlabeled, so a base-trained "is this a real object" model sees it as
background.

What I'd defend:
1. **Class-blind inputs.** Use `pred_ious`, persistence, geometry, motion and the
   relation-plausibility features. Leave out class identity and the CLIP class
   probabilities. Novel objects are then judged on the same evidence as base ones.
2. **The label.** "This tracklet matches a base GT object at vIoU ≥ 0.5," with
   unmatched tracklets treated as **unlabeled, not negative** (positive-unlabeled
   learning with an assumed class prior), or simply down-weighted.
3. **Measure the suppression; don't fix it by peeking.** Report novel recall before
   and after keeping, descriptively, on HELD's novel labels. That's the M1 precedent:
   HELD novel is reported, and "no choice may change in response to HELD numbers
   without an amendment" (ov-pairtime A24:118, :409-422). Optionally also on
   ImageNet-VID's novel-class videos (Q5).

I'd **avoid** relation self-training (keeping what the relation head likes). The head
misses most relations even on GT tracks (C1 dominates PredCls, D1), and its scores
aren't calibrated for this use.

### Q5 — Is there a cheaper clean held-out set than our own detector?

**Probably yes, for objects:** ImageNet-VID videos that are not in VidVRD.
- VidVRD's 1,000 videos are drawn from ImageNet-VID. The rest of ImageNet-VID has box
  annotations for 30 classes, and VidVRD's released detector was trained on
  `open_vidvrd` according to its args. If the args are complete, it never saw those
  videos.
- That gives **clean scores to calibrate an object-level keeping step without
  building a detector.**
- **Caveats:**
  - there are **no relation labels**, so only object recall and precision can be
    checked;
  - `person` is not an ImageNet-VID class, so person boxes are unlabeled;
  - several **VidVRD novel classes are ImageNet-VID classes** (antelope, bear, cattle,
    fox, giant_panda, horse, lizard, motorcycle, red_panda, watercraft). Those boxes
    may only be used to measure suppression, never as training positives;
  - the "never saw them" rests on the saved args (Q2's open COCO question). **UNSURE
    until checked.**
- **Cost:** downloading part of ImageNet-VID (the full set is about 86 GB) plus one
  front-end pass, about 3.5 h per 200 videos, measured in D2.

VidOR's validation split would add relation labels, but it's a different domain with
much larger data. **UNSURE** it's worth the cost.

### Q6 — Costs and risks: A vs B vs C

Assumes a 4090 at $0.30–0.60/h; nothing below has been measured except D2's inference
speed.

| | A: own detector, keeping steps, clean step 4 | B: faithful METOR | C: RB1 first |
|---|---|---|---|
| **new code** | detector loop, two loaders, COCO bank, keeping steps | A's detector (100 queries), METOR's eval adapter (unwritten, doc 32 §2), the per-step frame cap (about 10% of videos OOM, doc 32 §4) | none; done and pinned (A23a). Needs its PredCls amendment and the 800 training videos' frames. |
| **engineering** | 2–4 months | 3–5 months | days |
| **GPU** | detector $20–150, plus a clean step-4 run $35–95, plus the keeping steps; **$100–400 in all** | similar to A, or more; "24 GB+, days" for METOR alone (doc 40) | step 4 at about 0.5–0.7 s per frame (2–3× D2's measured 0.23 s inference) × 162k frames → 22–32 h per epoch → **R2 alone about $35–95; R1 + R2 + R3 about $100–290** (A23's 5-video step measures it) |
| **can we interpret the result?** | **low**: new detector, new steps and clean retraining all at once | **low**: the paper expects to fall well short without METOR's own detector (doc 32 §1) | **high**: controlled deltas (R1 − R0, R2 − R1, R3 − R2) |
| **moves toward beating 16.75?** | only if the clean baseline isn't far below | the same | no, but it tells whether "beat 16.75 cleanly" is realistic |

The RB1 cost is my extrapolation; training at one frame per step with a backward pass is
slow. **UNSURE** by ±50%.

### Q7 — Headroom (A25e; descriptive, no decision rule)

**Matched in PredCls: 106–108 of the 262 U1 relations (about 41%).**

| | value |
|---|---|
| U1 relations (D1, novel, SGDet) | 262 |
| matched in PredCls (GT tracks and names) | **106–108** (exact on 153 keys: 48 matched / 105 missed; bounded on 53 keys) |
| missed even with GT tracks: D1's PredCls bucket, exact keys | **C1 83** (79%), A1 10, N1 8, U3 4 |
| exact keys, matched / missed: base-only · involves novel | 41 / 58 · **7 / 47** |
| SGDet novel recall now → if every matched U1 relation were recovered | 20.0% → 37.5–37.8% (PredCls 40.3%) |

**Reading.**
- Even a perfect front end, with the released relation head, recovers at most about
  41% of U1. The rest needs a better relation head.
- **U1 relations that involve a novel object are almost never recovered by perfect
  tracks** (7 of 54 on exact keys). So the front end mostly buys back *base-object*
  relations. That matters for weak point (b), and for whether front-end work moves
  the *novel* score at all.

**Disclosed correction.** The pinned script (`q7_headroom.py`, A25e) reported an
inverted interval (106, 104).
- **Cause 1, a bug:** its upper bound lacked a floor at 0.
- **Cause 2, a broken assumption:** A25e assumed no SGDet hit shares a key. That
  fails for exactly one key, `monkey sit_behind monkey`, frames 450–690: D1 lists
  4 PredCls misses against 1 known entry.

`q7_bounds_note.py` recomputes it, post-pin and disclosed here. It uses the same bounds
on the 205 consistent keys and the assumption-free interval on the one inconsistent
key, giving 106–108. Both outputs are kept.

### Q8 — Recommendation

**Direction:** C narrowed, then a cheaper version of A's idea. Not A as ordered.
1. **Settle the protocol question first. It decides what "beat METOR" means.**
   - **The test path, eval only, about $2–5:**
     - the released model with uniform fusion (RB1's R0u, already planned);
     - the released model over the **whole video** instead of the GT-extent range
       (D2's finding). This needs one flag and an amendment, and it counts as a
       disclosed test evaluation of the baseline.

     These measure how much of 15.79 comes from test-time protocol issues.
   - **Training: RB1's R1 and R2,** clean step 4 on the released detector. Measure
     the cost on 5 videos first, as A23 already plans. This measures the step-4
     training leak. The detector-stage leak stays unknown either way.
2. **Meanwhile, on the CPU:** design the front-end step as relation-aware
   **tracklet** keeping at a lower cut. Inputs: `pred_ious`, persistence, geometry,
   motion and modelC pair plausibility, class-blind. The label is base matches, with
   unmatched tracklets unlabeled (Q4).
3. **Calibrate on ImageNet-VID videos outside VidVRD** (Q5). Check suppression
   descriptively on HELD novel. Build our own detector only if step 1 shows the
   released stack can't support a clean claim.

**The first concrete step:** an amendment for the eval-only protocol audit (R0u plus
the whole-video frame range), then about 7 GPU-hours.

**What would change my mind:**
- **R2 ≈ R1 and the test-path audit moves little.** Then the published numbers are
  roughly comparable. The fastest route is front-end work on the released stack:
  the T1 gap is only 0.6 novel points (16.15 vs 16.75).
- **R2 ≪ R1, or the whole-video evaluation drops a lot.** Then beating 16.75 cleanly
  within months isn't realistic. I'd reframe the thesis around a clean benchmark,
  with gains over a clean baseline and the published numbers reported under
  disclosure. Our own detector (A, step 1) then becomes necessary, and its cost is
  justified.
- **If, on HELD, relation plausibility adds nothing over `pred_ious` plus
  persistence,** drop the "relation-aware" framing for the front end and put the
  relation effort into the relation head. That's where Q7 says 59% of U1, and all
  of C1, is lost.

---

## 2. Your weak points

| | verdict | evidence |
|---|---|---|
| (a) where the relation cues come from | **confirmed, and stronger** | the head is not computed at test (Q1); modelC needs ≥ 10-frame pairs (`gen_labels.py:659`); so keep tracklets, not boxes |
| (b) novel objects suppressed | **confirmed** | no clean target avoids it (Q4); class-blind inputs, PU labels and descriptive HELD checks reduce it and measure it. Note Q7: front-end gains go mostly to base-object relations anyway. |
| (c) validation | **confirmed; a cheaper route exists** | ImageNet-VID outside VidVRD (Q5); the released detector's args show VidVRD-only training. UNSURE until checked. |
| (d) starting point | **confirmed** | M1's clean stack 9.92 vs 21.65 PredCls; the cause is unattributed, which is exactly what RB1 attributes |
| (e) headroom | **answered** | about 41% of U1 at best with the released head; novel-involving U1 relations are almost never recovered (7 / 54) |

## 3. Weak points I'd add

- **(f) The comparison target likely shares the protocol leaks.** METOR and EOV-MMP
  share a first author and a lab (`docs/31_Lineage-evidence.md`). The released
  checkpoints' paths (`/data/wyq/…`) match. The leaks:
  - novel boxes as detector targets;
  - novel-favouring fusion weights;
  - a GT-derived test frame range.

  METOR's code is unreleased, so this is **UNSURE**, but it's the default assumption.
  "Beat 16.75" is ill-defined until it's said which protocol both sides use.
- **(g) Recovering O1b needs a much lower cut.** O1b boxes mostly score 0.1–0.2. A
  cut that low multiplies candidates, and deep_sort treats every one as a detection
  (confidence 1, NMS off). The keeping step carries the whole burden of precision.
- **(h) O1a (about 35%) is out of reach for any keeping step,** and the O1a/O1b split
  depends on the floor (D2 Table 4). Plan with "about 70% before tracking, of which
  at most about half can be kept," not with "rescore wins."
- **(i) A25d binds this design.** D2's test recordings can't be used to tune or check
  the keeping step. Every design number must come from training videos, HELD or an
  outside held-out set.
- **(j) "METOR is built on EOV-MMP" holds for the lineage, not the architecture.**
  METOR regresses trajectories (Eq. 5): no tracker and no 0.35 cut. D2's stage
  findings are about EOV's front end and don't transfer to B.

## 4. Records

| file | SHA256 |
|---|---|
| `q7_headroom.py` (pinned by A25e) | `e099e2eb180c7d90b07fa64bf146561390534c9b4b4be294fa0cfca3ed01479d` |
| `q7_headroom.json` (its output, including the inverted interval) | `65befe8170e92350f83e6a8d499ad763435b9dee8aacd64a072182eebd1b7c44` |
| `q7_bounds_note.py` (post-pin correction, disclosed) | `88c0099790f17c69e4d51762aea5bbd3e90bb19e97e3b4139a7eadcf2ee5250c` |
| `q7_bounds_note.json` | `b6fbac86768d250e6c4ef522d10679d8e0b0ac0529aa1468cf050320813409bc` |

- **The prereg after A25e:**
  `7365374bf868dca643913daa46a31e08fe13d339594617ef6170c11f59060114`, committed as
  `a4ab77a` before the count was run.
- **Housekeeping.** The A25e commit now sits on top of the D2 recordings commit
  (`df9ccca`). Dropping the recordings before a push therefore needs a rebase, no
  longer `git reset --hard HEAD~1`. Nothing is pushed.
- **Stop.** Review only. Nothing was built, trained or run on a GPU.
