# Design review 2 — final position on the merged plan

Reply to round 2. CPU only: code reading, checkpoint configs read via memory-mapped loads
(no weights used), and file names. No GPU, no training, and no test-label use. Web
sources for Q2 are listed at the end.

Marks: **AGREE** · **DISAGREE** · **UNSURE** · **AMEND** (agree, with a change).

**Final position: I agree with the merged plan, with five amendments** (§3). The most
important: the "two-tier" tracking must extend tracks **offline**, or the baseline tracks
are not intact (§2 d). And the keeping step must be **refit** on R2's outputs, not
reused (§2 b).

---

## 1. Concessions

- **Your reading (a) is right, and my review 1 implied otherwise.** The evaluator
  filters GT and predictions **by predicate split only**:
  `eval_relation_detection_openvoc(target_split_pred=…)`, with `target_split_traj`
  left at `"all"`, keeps a triplet iff its predicate is in the split and both objects
  are in *any* class (`inference/video_relation_detection_openvoc.py:339-384`). Every
  scoring path in the chain calls it that way: `cli/common.py:168`,
  `time_check/tc_run.py:76`, `stage1c/eval_from_cache.py:148`.
  - So the 201 base-object U1 relations **are** novel-split relations. Recovering them
    raises novel mAP, by up to +17.8 recall points at perfect tracks (Q7).
  - The 7 / 54 result means weak point (b) costs little *in score*: the gain comes
    mostly from base objects. It still matters for the claim's integrity, so keep the
    descriptive suppression report.

---

## 2. Your points (b)–(e)

### (b) RB1 in parallel, not as a gate — **AGREE, with one amendment**

Its result changes the framing, not the method, and R2 is the natural clean base.
**But the keeping step can't simply be "applied" to R2:**
- Step 4 retrains the detector end to end: A22 §1b found 308 of 755 detector tensors
  changed. It also retrains modelC.
- So R2's box scores, `pred_ious`, tracklets and pair plausibilities all differ from
  R0's. A keeping model fitted on R0's outputs would be miscalibrated on R2, which is
  E1's failure again.
- **Amendment:** pre-register the *recipe* (features, model class, hyperparameters,
  thresholds rule), and **refit it per base** on the same outside training videos. That
  needs one more recording pass on the outside set with R2's stack, about 10–15
  GPU-hours.
- **Budget:** RB1 dominates the cost (Q6). Run A23's 5-video measurement before
  committing to the full R1–R3.

### (c) No affordable clean relation-level check on the released stack — **AGREE, with one partial option**

- **On R0:** HELD is in-sample for modelC and for step 4. So relation-level effects can
  only be seen at the test session. Confirmed.
- **The one partly cleaner, affordable option:** an extra R2 variant trained on **FIT
  only** (640 videos).
  - HELD is then clean for step 4 and modelC, the components the relation features
    come from.
  - It is **not** clean for the stage-1 detector: its saved config says
    `dataset_file = open_vidvrd`, presumably all 800 training videos, so HELD scores
    stay in-sample-inflated.
  - The *with versus without relation features* comparison shares that bias in both
    arms, so its **delta** is still informative.
  - Cost: about 80% of one R2 run.
- **VidOR** has relations, but a different vocabulary and much larger data.
  **UNSURE** it's worth it.
- **My view:** if the "relation-aware" framing matters to the author, R2-FIT is the
  only pre-test relation-level evidence on offer.

### (d) Two-tier association — **AMEND: feasible only as offline extension**

- **Online extension breaks the baseline.** A box that updates a track changes:
  - its Kalman state (`deep_sort/track.py:164`; with confidence 1, NSA sets the
    measurement noise to zero, `kalman_filter.py:148-149`);
  - its EMA appearance feature (`track.py:167-170`);
  - its hit and miss counters (`:174-177`).

  Each of these changes which confident boxes the track wins later. So ByteTrack-style
  online extension and "baseline tracks intact by construction" are incompatible.
- **What works:**
  1. **Tier 1, unchanged:** confident boxes (> 0.35) go through today's deep_sort and
     AFLink exactly as now. Verify identical tracks with D2's recorder machinery
     (`studies/d2/recorders.py`) on non-test videos; A25d binds only the test
     recordings.
  2. **Tier 2, candidates:** queries with cut score in [0.1, 0.35].
     - Re-run the same post-processor with `pruning_threshold = 0.1`
       (`post_process.py:132`) on the same detector outputs.
     - Drop boxes overlapping a tier-1 box at IoU > 0.8 (the repo's own dedupe
       threshold, `gen_labels.py:149`).
     - Track them with a **separate** deep_sort instance into candidate tracklets.
  3. **Offline attachment:** a candidate tracklet whose start or end lies within the
     AFLink window of a tier-1 track end (`thrT` (0, 30), `thrS` 75,
     `end2end_model.py:289-291`) is offered as an extension. Otherwise it stands
     alone.
  4. **The keeping step decides on three things:**
     - candidate tracklets;
     - proposed extensions;
     - tier-1 tracks the fixed filter removed (coverage, range). That is D2's O3,
       about 15% of U1.

     The tracks the filter kept stay as they are.
- **Work:** about 2–3 weeks for a wrapper in a new folder, with tests that tier 1 is
  bit-identical. No vendored file changes.

### (e) Pre-registered relation-feature gate — **AGREE, with specifics**

- **Metric:** object-level keeping on the outside held-out split. Average precision of
  the keep decisions, where a tracklet is positive if it matches a base-class GT track
  at vIoU ≥ 0.5. Use **base GT only**; novel-class tracks are reported separately as
  suppression.
- **Arms:**
  - (A) `pred_ious`, persistence, length, geometry, motion, AFLink link probability;
  - (B) = A + modelC plausibility: max base-predicate score over partners,
    interactiveness, and the number of plausible partners.
- **Gate:** the framing stays only if B − A ≥ a margin fixed in advance (my suggestion:
  +2 AP points) **and** the paired video-bootstrap 95% CI's lower bound is > 0.
- **Sample:** pick outside videos with ≥ 2 annotated objects, so relation features get
  a fair chance. Many ImageNet-VID videos have one object, and VidVRD's videos were
  chosen for relations.
- **Caveat on R0:** the interactiveness head was trained with novel-predicate
  positives (A23, T6). On R0, the relation features carry that leak; on R2 they're
  clean. Report both.

---

## 3. The merged plan, amended

Steps 0–5 as you wrote them, plus:
1. **Step 0 (A26):** R0's own numbers already exist (session 3 and D1), so the new runs
   are three:
   - R0u;
   - R0 over the whole video;
   - R0u over the whole video.

   The whole-video setting changes both the detected frames and the 15% rule's base
   (`min_l`, `end2end_model.py:314`). That is the consistent clean protocol. Implement
   it as a wrapper that overrides `begin_fid`/`end_fid` to [0, `video_len`), not a
   pipeline edit.
2. **Step 1 (A27):**
   - tracking is two-tier with **offline** extension (§2 d);
   - the keeping step decides on the three candidate types;
   - the recipe is **refit per base** (§2 b);
   - the features follow the leak rules in Q4.
3. **Step 2:**
   - add an R2 recording pass on the outside set, for the refit;
   - outside videos need ≥ 2 annotated objects;
   - ImageNet-VID access is uncertain (Q2), so pre-register a fallback.
4. **Step 3:** optionally add **R2-FIT** for a partly clean relation-level check on
   HELD (§2 c).
5. **Step 4:** pre-register **every** configuration of the single test session in one
   amendment, and disclose its place in the order of test evaluations. Each new test
   run adds to the chain's count.

---

## 4. Answers

### Q1 — RB1 in parallel?

**Yes,** with the refit amendment and the 5-video cost check first (§2 b).

### Q2 — ImageNet-VID: which videos, how to get them, the other components' training data

**Which videos.** VidVRD's 1,000 videos, counted from file names only:

| | from VID train | from VID val |
|---|---:|---:|
| VidVRD train | 717 | 83 |
| VidVRD test | 176 | 24 |
| total in VidVRD | 893 | 107 |

ImageNet-VID's final release has 3,862 train and 555 val snippets (sources below). That
leaves **2,969 train and 448 val snippets outside VidVRD.** Each file name
(`ILSVRC2015_{train,val}_XXXXXXXX`) maps one-to-one to a snippet id, so the exclusion is
by id.

**Access. Practical problem: UNSURE we can get it legitimately.**
- image-net.org's 2015 downloads page sends VID to an external page,
  `bvisionweb1.cs.unc.edu/ilsvrc2015/download-videos-3j16.php`, "and read carefully the
  term of use for the VID task". **That host no longer resolves** (checked 2026-10-11).
- No official per-video download exists. The dataset is one archive, commonly cited at
  about 86 GB.
- One third-party listing of a split `ILSVRC2015_VID.tar.gz` turned up. Its provenance
  is unverified.
- **Terms:** ImageNet's general terms are non-commercial research and education only,
  with no redistribution. The VID-specific terms were on the dead host, so I couldn't
  read them.
- **Recommendation:** have the author request access through image-net.org or the
  ILSVRC organisers, or confirm a legitimate mirror and its terms, **before** A27 is
  final. Pre-register a fallback outside set. Candidates, which I haven't verified
  here: TAO (multi-object, many categories, sparse annotation) or another
  multi-object video set with exhaustive boxes. **UNSURE** which is best.

**What the other components' saved configs show.** Read via memory-mapped loads:

| component | file | saved config |
|---|---|---|
| object classifier (modelB) | `vidvrd_backboneViT-L_14@336px_lr0.01vision-guided.pth` | `dataset = vidvrd`, `train_split = base`, 20 epochs at lr 0.01, saved at epoch 3 |
| relation classifier (modelC, stage 2) | `…_stage2_new_L14_e2e.pth` | `dataset = vidvrd`, `src_split = base`, 30 epochs at lr 1e-4, saved at epoch 19 |
| end-to-end, step 4 (released) | `…_end2end_base-001.pth` | `dataset = vidvrd` / `open_vidvrd`, `train_split = base`, saved at epoch 6 |
| detector, stage 1 | `checkpoint_vidvrd0059_new_1e-5.pth` | `dataset_file = open_vidvrd`, 60 epochs, `pre_trained = imagenet`, empty `resume` (review 1, Q2) |
| AFLink | `AFLink_epoch20.pth` | a bare state dict, no config. The vendored `aflink/config.py` points to MOT17 (pedestrians), StrongSORT's release. |

**Conclusion.** Every VidVRD-specific component records VidVRD as its data. "base"
names the class split, not a subset of videos. Nothing points to other ImageNet-VID
videos. Still unknown:
- whether the detector had a COCO stage (review 1, Q2);
- CLIP's web pretraining.

### Q3 — The two-tier association: feasible, and how much work?

**Feasible as offline extension**, about **2–3 weeks**, with no vendored file changed
(§2 d). Online extension is not compatible with intact baseline tracks.

### Q4 — Which features leak class identity?

| feature | leak | use in keeping |
|---|---|---|
| the 35-class probabilities, argmax, the track name, modelB scores, modelC's subject/object heads (`relation_classifier.py:506-509`) | **direct** | exclude |
| max class probability (the cut score) | **implicit:** base classes score higher. D2 Table 6: novel O1b boxes sit lower. | exclude, or test both and report suppression |
| `pred_embed` (768-d, CLIP space), TagCLIP RoI means | **implicit:** class identity is linearly decodable | exclude the raw vectors |
| similarity of `pred_embed` to class text | needs a vocabulary: base-only in training and all classes at test is a shift; novel names in training are a leak | avoid. A vocabulary-free alternative: similarity to a generic "an object" prompt against a background prompt. UNSURE how useful. |
| `pred_ious` (`model.py:519-524`), persistence, length, geometry, motion, AFLink probability | **none by construction** (trained class-agnostically on base boxes) | the baseline feature set |
| modelC predicate scores and interactiveness | **no names; appearance enters** through its CLIP RoI inputs | scalars only (max, count); also carries T6's leak on R0 |

### Q5 — Can modelC give a class-blind plausibility score?

**Yes, and it already computes one.**
- **The predicate path is name-free.** Its inputs are 4 CLIP RoI features per slot,
  box features, and location and motion features (`relation_classifier.py:134-154`).
  The predicate text side is a prompt learner over predicate names conditioned on the
  visual embedding (`:500-505`).
- **No class ids enter.** A language feature from the subject and object class ids
  exists (`gen_labels.py:67-70`, `_extract_lan_feat`), but it is **not** among the
  feature types used in training or test: `['rel_feat', 'mot_feat', 'clip_feat',
  'bbox_feat']` (`gen_labels.py:502`, `:647`).
- **The class-blind signals:**
  - `interactiveness`, a per-slot "is there a relation" score (`:510`, multiplied in at
    `:524`);
  - the max over base-predicate scores.

  Both are free of object names. They're not free of appearance, and on R0
  interactiveness carries T6.

### Q6 — Total cost and time for steps 0–4; the biggest risk

$ at $0.30–0.60 per GPU-hour (assumed; no price recorded). D2's 3.5 h per 200 test
videos is the only measured speed.

| step | GPU | money | calendar | note |
|---|---|---|---|---|
| 0, A26 audit (3 runs; the whole video is +13% frames) | about 11–12 h | $4–8 | about 1 week | the wrapper and the amendment |
| 1, A27 design (CPU) | — | — | 2–3 weeks | |
| 2, outside set: two passes (R0, R2) of about 400–600 videos | about 20–30 h | $6–18 | 3–5 weeks | includes the two-tier wrapper and the keeping model; **plus access time for ImageNet-VID** |
| 3, RB1 (R1, R2, R3) | about 330–480 h (extrapolated) | $100–290 | 2–3 weeks of GPU wall time, in parallel | measure on 5 videos first; R2-FIT adds about $30–75 |
| 4, test session (keeping on R0 and R2 × 2 ranges, plus baselines) | about 20 h | $6–12 | 1 week | |
| **total** | | **about $120–330** (+$30–75 with R2-FIT) | **about 2.5–3.5 months** | RB1 is about 80% of the money |

**The biggest risk: the realized gain is small.** A back-of-envelope, not a prediction:
- Keeping can address D2's O1b and O3 (51% of U1, about 134 relations).
- If PredCls recoverability is independent of the stage, about 41% of those (Q7) are
  worth recovering: about 55 relations, about +9 recall points at *perfect* keeping.
  That independence is unverifiable: crossing D2's stages with PredCls is barred by
  A25d.
- A realistic third to a half of that gives about +3–4.5 points. False candidate pairs
  also compete in each video's top-200 ranking.
- Using the slope between today's SGDet and PredCls points (15.8 mAP at 20% recall,
  21.7 at 40%), that's **about +1 novel mAP on R0**.
- That's of the same order as the 0.6 gap between T1 (16.15) and METOR (16.75). So
  "beat 16.75 on the published protocol" is **plausible, not likely**. On R2 it's
  unknown.

**Other risks:**
- ImageNet-VID access;
- RB1's cost range;
- object-level calibration on ImageNet-VID may not transfer to VidVRD's
  relation-dense scenes, which the ≥ 2-object rule only partly fixes.

---

## Sources (Q2)

- [ILSVRC2015 downloads page (image-net.org)](https://image-net.org/challenges/LSVRC/2015/2015-downloads.php): the VID link goes to `bvisionweb1.cs.unc.edu/ilsvrc2015/download-videos-3j16.php`, which no longer resolves.
- [ImageNet download and terms page](https://image-net.org/download): non-commercial research and educational use.
- [Seq-NMS for Video Object Detection (arXiv 1602.08465)](https://arxiv.org/pdf/1602.08465) and [Object Detection in Videos by High Quality Object Linking (arXiv 1801.09823)](https://arxiv.org/pdf/1801.09823): VID final release, 3,862 / 555 / 937 snippets.
- [ImageNet VID listing (awesomepapers.io)](https://awesomepapers.io/computer-vision/datasets/imagenet-vid): a third-party split archive; provenance unverified.

**Stop.** Review only. Nothing was built or trained, and no GPU was used.
