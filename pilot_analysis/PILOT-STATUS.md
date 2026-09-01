# EOV-MMP diagnostic pilot — status handoff

**For:** the Claude chat conversation planning this pilot study.
**Written:** 2026-08-26. **Protocol being executed:** [`../eov-mmp-pilot-prompt.md`](../eov-mmp-pilot-prompt.md).

This file records what has actually been measured, what those measurements say,
and what remains. Everything under **§B Done** is finished and reproducible;
everything under **§D Not done** is not started. Nothing here required a GPU.

**Sections are lettered A–F on purpose.** The protocol's phases are numbered
0–4, so a numbered section here would read as a phase number. It isn't one:
§ letters are this document's structure, and each subsection names the protocol
phase it belongs to where one applies.

---

## A. Environment and provenance

*Protocol ground rule 5, **not** Phase 0. Phase 0 is the sanity gate and has
not been run — see §B.1 for the part of it answered by code reading, and §D.1
for what still needs a GPU.*

| | |
|---|---|
| GPU | **NVIDIA GeForce GTX 1650, 4096 MiB** — driver 610.57.01, CUDA UMD 13.3 |
| CPU / RAM | 12 cores / 7 GB (WSL2) |
| Disk | 880 GB free |
| Python env | conda `repro-next`, Python 3.12.13 |
| Torch | 2.11.0+cu128, torchvision 0.26.0+cu128, `cuda.is_available() == True` |
| Repo | `ov-vidvrd-lab` @ `3e75adc`, clean except untracked `_paper/`, `eov-mmp-pilot-prompt.md` |
| Data | **all present locally**, 47 GB — frames decoded, annotations, authors' CLIP bank, ECC files |
| Checkpoints | **all present locally**, 9.1 GB — incl. two trained end-to-end models |

The rows above are read off this system (`nvidia-smi`, `python -V`,
`torch.__version__`, `git rev-parse`, `du`).

**Measured 2026-08-26, correcting an earlier claim in this file.** An earlier
version said "the GPU almost certainly cannot run this model" on the arithmetic
that two CLIP ViT-L/14@336px encoders (~2.7 GB) plus end-to-end weights
(~2.5 GB) exceed 4 GB before any forward pass. `cli/evaluate.py` was then
actually run (log kept locally at `pilot_analysis/logs/evaluate-oom-probe.txt`,
not committed). **It does not OOM.**

```
19:15:34  process start
19:24:43  checkpoint loaded (epoch 6), text embeddings built, prompt learner init
19:24:43  eval[all]: 0%|  | 0/200   <-- entered the loop, working on video 1
19:30:34  killed by a 900 s timeout, still on video 1
```

**9 minutes to build the model and load the checkpoint; then >6 minutes on the
first video without finishing it.** Nothing errored.

Why: WSL2 lets CUDA allocations spill into system RAM instead of failing, which
`docs/10_Known-issues.md §2` had already observed ("survived only by spilling
into system RAM under WSL2, which is why a probe that should fail in seconds
took six minutes"). So the correct statement is **not** "cannot run" but:

> **This machine runs the model without erroring, at unusable throughput.** The
> 4 GB card plus 7 GB of system RAM thrash rather than OOM.

A second, longer probe pinned the rate down. `dump_predictions.py --limit 1`
with a 40-minute cap (log local, `pilot_analysis/logs/dump-probe-1video.txt`): model loaded
in **9 minutes**, then **31 minutes inside video 1 without finishing it**. So
the per-video cost is **> 31 min**, and for 200 videos × 2 splits Phase 1 here
would take **> 200 hours (> 8 days)** — against a few dollars of rented GPU time.

The practical conclusion is unchanged — rent a ≥16 GB card — but the reason is
throughput, not a hard memory failure. That distinction matters if anyone later
tries to "just run it locally overnight" on the strength of the old claim.

Consequence: this machine is a full-fidelity **CPU analysis box** (all data and
all checkpoints are here), and every model-execution phase must run elsewhere.

Ground rule 5's raw artifacts are captured under `pilot_analysis/logs/`, which
is **gitignored** — they are machine-specific and one transfer log reached
8.85 MB. `commands.md` there lists every command, so all of it regenerates:
`pip-freeze.txt` (118 packages), `nvidia-smi.txt`, `git-state.txt` (commit
`3e75adc`, `git status --porcelain`, and an empty `git diff` — no tracked file
has been modified). The upstream `../EOV-MMP-VidVRD` tree has not been touched
at any point; all pilot work lives in `pilot_analysis/`.

---

## B. Done

### B.1 Phase 0, partial — segment length and pipeline shape

Answered by reading code, no run required:

- **Segment length = 30 frames.** `args.clip_len`, default 30, asserted in
  [`tests/test_config.py:42`](../tests/test_config.py#L42) ("VidVRD-II segment
  length"), consumed as `CLIP_LEN` in
  [`inference/post_process.py:5`](../inference/post_process.py#L5).
- **Segments are a partition, not a sliding window.**
  [`process_pred`](../inference/post_process.py#L33-L39) slices
  `[clip_id*30 : (clip_id+1)*30]` off the tracklet pair, indexed from the pair's
  own `begin_fid`. There is no overlap and no stride. *The pilot prompt's
  description of a "sliding-segment" pipeline is wrong for this code* — worth
  correcting in the writeup, since it changes what the oracle-merge baseline
  should be compared against.
- **Vocabulary splits:** `configs/VidVRD_pred_class_spilt_info_v2.json` →
  71 base + 61 novel predicates (+ `__background__`);
  `configs/VidVRD_class_spilt_info.json` → 25 base + 10 novel object classes.
  The 61 novel predicates match the paper's protocol as recorded in
  `docs/20_Benchmark-analysis.md`.
- **Prediction format** is emitted by
  [`format_()`](../inference/post_process.py#L102-L122): per instance
  `{triplet: [sbj_cls, pre_cls, obj_cls], score, sub_traj, obj_traj, duration:
  [begin, end]}`, truncated to `args.max_per_video`. It carries everything
  Phase 1 asks for; subject/object *scores* are folded into `score`
  (`sbj_scr * obj_scr * pre_scr`) rather than reported separately.
- **Metric** is vIoU ≥ 0.5, `ov = min(subject_vIoU, object_vIoU)`, greedy
  highest-score-first matching, AP averaged per video —
  [`eval_detection_scores`](../inference/video_relation_detection_openvoc.py#L118-L147).
  Pure NumPy, **runs on CPU**.

### B.2 Phase 3.1 — GT duration statistics (complete)

Reproduce: `python pilot_analysis/scripts/gt_duration_stats.py`
Source: `data/vidvrd/anno/test/` — 200 videos, **4,835 GT relation instances**.

| | all | base | novel |
|---|---|---|---|
| instances | 4,835 | 4,230 | 605 |
| **span > 1 segment** (>30f) | **77.4%** | 78.1% | 72.4% |
| span > 2 segments (>60f) | 48.2% | 48.5% | 45.6% |
| span > 4 segments (>120f) | 24.2% | 24.8% | 19.8% |
| median duration | 60 | 60 | 60 |
| mean duration | 107.2 | 108.9 | 95.3 |

Duration range 30–1200 frames. Segments spanned (ceil dur/30):

```
1: 1095 (22.6%)   4: 448 ( 9.3%)   7: 116 (2.4%)
2: 1411 (29.2%)   5: 368 ( 7.6%)   8:  93 (1.9%)
3:  710 (14.7%)   6: 187 ( 3.9%)   9+: 407 (8.4%)
```

**Headline: the greedy merge is load-bearing for 77.4% of the test metric.**
That is the empirical justification for testing H2 at all, and it is now
established without touching a GPU.

### B.3 Annotation grid vs. segment grid (new finding, not in the protocol)

Same script. **Every GT duration is a multiple of 15, not 30**: 2,209 of 4,835
(45.7%) have `duration % 30 == 15`, and 1,520 of 4,835 (31.4%) have
`begin_fid % 30 == 15`.

So GT extents sit on a **15-frame** grid while predicted extents sit on a
**30-frame** grid anchored to the tracklet's own start. A systematic
misalignment exists by construction.

### B.4 Is that misalignment a ceiling? — No. (negative result)

Reproduce: `python pilot_analysis/scripts/temporal_grid_ceiling.py`

Question asked: assuming *perfect boxes*, what is the best vIoU a grid-aligned
prediction can achieve against each GT instance? (Idealisation: constant
per-frame box area, so vIoU reduces to `I / (P + G − I)` on frame counts. The
real evaluator area-weights, so this is not a bound on the real metric —
stated because it is load-bearing.)

| grid phase | mean best vIoU | unmatchable (<0.5) |
|---|---|---|
| 0 | 0.858 | **0 / 4,835 (0.0%)** |
| 15 | 0.804 | **0 / 4,835 (0.0%)** |

Zero instances are made unmatchable by the grid. A 30-frame GT instance at
offset 15 is best covered by the enclosing 60-frame block:
`30 / (60 + 30 − 30) = 0.5` exactly, and the evaluator tests `ov >= threshold`,
so it passes.

**But 414 instances (8.6%) sit at exactly 0.5 with zero margin** — for those,
any box imperfection at all drops them below threshold. Margin distribution
(phase 0): ≤0.50 → 8.6%, ≤0.70 → 12.1%.

Verdict: temporal quantisation is a **fragility affecting ~8.6% of GT**, not a
ceiling. An initial hypothesis that the grid was a hard ceiling was tested and
**refuted** — do not put it in the thesis as a claim.

### B.5 Where the merge loss must actually come from (code reading, unverified)

Since quantisation is exonerated, these are the mechanisms H2 should target.
All are read from [`association()`](../inference/post_process.py#L54-L101) and
are **hypotheses, not measurements**:

1. **Exact string match + break on first gap.** A next-block candidate is
   accepted only if `curr_clip['pre_cls'] == next_clip['pre_cls']`, and
   `if not success: break` abandons the chain at the first block that fails.
   One weak segment fragments an instance permanently and irrecoverably.
2. **Mean-score aggregation penalises correct merges.**
   `pre_scr = sum(curr_scores) / len(curr_scores)`, then
   `score = sbj_scr * obj_scr * pre_scr`. A correctly merged 8-segment instance
   is diluted by its weakest blocks and ranks *below* a short confident
   fragment — a systematic bias against exactly the long instances that make up
   77% of GT. This directly informs the Phase 3.2(b) oracle-merge design: the
   max-vs-mean choice the protocol says to "document" is not a detail, it is
   part of what is being measured. **Run oracle-merge both ways.**
3. **`clip_top_n` caps the ceiling before you compute it — but less tightly
   than first stated.** Only the top-N predicates per block survive
   `process_pred`. An earlier version of this file said N=3; **that was wrong**
   — 3 is a value used in `tests/test_inference_postprocess.py`, while the real
   default is **20** ([`parser_func.py:77`](../utils/parser_func.py#L77)), as
   confirmed by the config dump in
   the config dump in the local `pilot_analysis/logs/evaluate-oom-probe.txt`.
   Top-20 of 132 is a much weaker
   constraint, so the Phase 3.2(c) "segment-level ceiling" is closer to a true
   ceiling than feared. Still worth saying it is top-20, not unrestricted.
4. **`format_` truncates to `args.max_per_video`** — an additional loss point
   after merging, applied per video.

---

### B.6 The official evaluator already supports Phases 2 and 3 (saves work)

[`eval_relation_detection_openvoc()`](../inference/video_relation_detection_openvoc.py#L339)
is the entry point, and it already has the two hooks the protocol assumes must
be built:

- **Subgroup filtering is built in.** `target_split_traj` and
  `target_split_pred` filter *both* the GT and the prediction sets by category
  before evaluating, reading the split from a JSON path
  (`pred_cls_split_info_path`). So Phase 2's per-group mAP needs **no new AP
  code at all** — write a split-info JSON whose `cls2split` maps each predicate
  to `static_spatial` / `trajectory_dynamic` / `appearance_dynamic` and pass it
  in. This satisfies the protocol's ground rule 2 exactly as written.
- **Per-GT hit attribution exists but is unreachable.**
  `rt_hit_infos=True` routes to
  [`evaluate_v2`](../inference/video_relation_detection_openvoc.py#L272), which
  returns a 4th value `det_infos`, and
  [`eval_detection_scores_v2`](../inference/video_relation_detection_openvoc.py#L234)
  additionally builds `gt2det_ids` (for each GT instance, which prediction
  matched it) and stamps `pred_relation['hit_gt']`. That is precisely what
  Phase 2.3's confusion table and Phase 3.3's fragmentation count need.
  **However:** `eval_relation_detection_openvoc` computes `hit_infos` and then
  discards it — its `if rt_hit_infos: return hit_infos` branch is **commented
  out** and the function returns `mean_ap, rec_at_n` unconditionally
  ([lines 391-398](../inference/video_relation_detection_openvoc.py#L391-L398)).
  Call `evaluate_v2` directly, or wrap it, rather than reimplementing.

### B.7 Phase 2.1 — the predicate partition (built; needs your review)

Reproduce: `python pilot_analysis/scripts/build_predicate_partition.py`

All **132 predicates classified, none left over**. Structure made this
rule-based rather than hand-listed: 25 predicates are single tokens, the other
107 are `{verb}_{spatial}` over 13 verbs and 14 spatial terms, so verbs and
spatials are classified once each and compounds are derived (`dynamic` if
*either* component needs time). Full per-predicate record with rationale:
`pilot_analysis/predicate_partition.json`.

Partitioned on **two axes**, not one — see §C.1 for why:

| group | preds | base | novel | test instances | share |
|---|---|---|---|---|---|
| `geometric_static` | 43 | 25 | 18 | 2,331 | 48.2% |
| `geometric_dynamic` | 78 | 42 | 36 | 2,183 | 45.1% |
| `appearance_static` | 4 | 2 | 2 | 179 | 3.7% |
| `appearance_dynamic` | 7 | 2 | 5 | 142 | 2.9% |
| **total** | **132** | **71** | **61** | **4,835** | 100% |

Independently reassuring: these land within ~1.5 points of
`docs/20_Benchmark-analysis.md`'s five-row taxonomy (46.5% trajectory, 3.7%
single-frame appearance, 3.0% appearance-over-time), which was built separately
and by different reasoning. The remaining gap is taxonomy detail, not error.
A second cross-check agrees to one video: **143 of 200 test videos (71.5%)
contain zero `appearance_dynamic` instances**, against that document's
independently derived 142/200. Since mAP is averaged over videos, for ~71% of
the metric a perfect temporal-appearance model and a blind one are
indistinguishable.

**Five predicates are flagged `review: true` — judgement calls that need your
eye before any Phase 2 number is quoted:**

| predicate | assigned | split | inst | the doubt |
|---|---|---|---|---|
| `drive` | `appearance_static` | novel | 3 | arguably geometric, given `person inside vehicle` |
| `feed` | `appearance_dynamic` | novel | 9 | needs appearance change, but slow and contact-like |
| `play` | `appearance_dynamic` | base | 79 | diffuse activity label; boundary with `fight` genuinely unclear |
| `pull` | `appearance_dynamic` | novel | 3 | contact plus induced motion; partly recoverable from boxes |
| `touch` | `appearance_dynamic` | base | 38 | a single frame may suffice, but onset is what is annotated |

The one that can actually move a number is `play` (79 base instances, 55% of all
`appearance_dynamic` base instances). The other four are ≤9 instances each.

Also emitted, in the evaluator's own `cls2split` format:

- `pilot_analysis/splits/pred_split_group.json` → `target_split_pred=<group>`
- `pilot_analysis/splits/pred_split_group_x_ov.json` → `target_split_pred=<group>__novel`

### B.8 The Phase 2 evaluation path is validated end-to-end (CPU)

Per §B.6, the official evaluator can filter by an arbitrary category split, so
Phase 2 needs **no new AP code**. Verified by feeding the GT relations back in
as predictions (score 1.0) through
`eval_relation_detection_openvoc(pred_cls_split_info_path=<the file above>,
target_split_pred=<group>__<base|novel>)`:

```
target_split_pred                    mAP    R@50   R@100
geometric_static__base            1.0000  0.8496  0.9536
geometric_static__novel           1.0000  1.0000  1.0000
geometric_dynamic__base           1.0000  0.7858  0.8801
geometric_dynamic__novel          1.0000  1.0000  1.0000
appearance_static__base           1.0000  1.0000  1.0000
appearance_static__novel          1.0000  1.0000  1.0000
appearance_dynamic__base          1.0000  1.0000  1.0000
appearance_dynamic__novel         1.0000  1.0000  1.0000
```

mAP is exactly 1.0 in all eight cells, so the split files, the GT filtering and
the AP computation are all wired correctly. Recall below 1.0 on the two base
geometric groups is **correct, not a bug**: those videos hold more than 50/100
GT instances, so R@50/R@100 truncate. This is the ground rule 2 requirement
("do not write your own AP computation") satisfied — the group numbers come out
of the repo's own evaluator.

Ready to consume Phase 1 predictions the moment they exist.

### B.9 Phase 4 is feasible — the interface found, and it is one line

Reproduce: `python pilot_analysis/scripts/gt_trajectories.py`
(needs the `repro-next` env; it imports the repo's own interpolation helpers)

The protocol asked to "find that interface". Traced in
[`models/end2end_model.py`](../models/end2end_model.py#L279-L337), the test path is:

```
modelA (detector) -> deep_sort -> AFLink -> format_trajectories_test(temp)
                                                    |
                                            `trajectories`   <-- SUBSTITUTE HERE
                                                    |
                              gen_feats_test(video_name, trajectories, 'test',
                                             patch_proj, global_proj, w, h)
                                                    |
                                            modelC (relation classifier)
```

Replacing `trajectories` at that single point bypasses the detector, the tracker
and AFLink while leaving the relation classifier untouched. The schema, read off
[`format_trajectories_test`](../models/gen_labels.py#L586):

```python
{'tid': int, 'category': str, 'score': float,
 'trajectory': {frame_id: [x1, y1, x2, y2], ...},   # DENSE over the span
 'begin_fid': int, 'end_fid': int}                  # end_fid == max(fid) + 1
```

GT annotations supply all of it (`score = 1.0`). Built and validated for all 200
videos:

| | |
|---|---|
| GT trajectories | **583** (4 rejected by the repo's own 65%-coverage rule) |
| ordered trajectory pairs | **1,392** (420 dropped, overlap < 10 frames) |
| segments for modelC | **8,508** |
| pairs per video | min 2, median 2, max 144 |

**One catch found on the way.** 43 GT trajectories have gaps — the object leaves
and re-enters frame — and `gen_feats_test` indexes `trajectory[fid]` for every
frame in the span, so it would `KeyError`. The detected path never hits this
because `format_trajectories_test` runs `add_initial_frames()` and
`interpolate_and_adjust_frames()` first. The script therefore runs GT through
**the repo's own two functions** rather than inventing a fix, which also keeps
GT and detected trajectories under identical downstream treatment.

**A second catch, needing a decision on the GPU run.** `add_initial_frames()`
prepends two frames *before* the first, so a GT trajectory starting at frame 0
gets `begin_fid = -2`. That affects **447 of 583** trajectories. The dict lookup
survives, but `gen_feats_test`'s feature extractor will be asked for frames that
do not exist. `--mode strict` skips the pre-roll and gives near-identical counts
(1,396 pairs vs 1,392), so it is the safer default for Phase 4 — at the cost of
no longer being byte-identical to the detected path. Worth noting this is
arguably a live bug in the *detected* path too, for any track starting at frame 0.

**Phase 4 still needs a GPU** — `gen_feats_test` consumes `patch_proj` /
`global_proj`, which are CLIP features. But it skips modelA (the 3.7 GB
component), so it may need materially less VRAM than Phase 1. Worth measuring.

### B.10 H1's premise is confirmed, and can be stated far more precisely

Traced through [`models/relation_classifier.py`](../models/relation_classifier.py).
The text side of the predicate classifier is a concatenation of **four**
embeddings ([`split_text_embeddings`](../models/relation_classifier.py#L465)):

| # | component | source |
|---|---|---|
| 1 | `"An image of a person or object {name} something."` | frozen CLIP text, `.detach()` |
| 2 | `"An image of something {name} a person or object."` | frozen CLIP text, `.detach()` |
| 3 | `"An image of the visual relation {name} between two entities."` | frozen CLIP text, `.detach()` |
| 4 | learned prompt, CoCoOp-style, instance-conditioned | **learned** |

Three of four are frozen CLIP text encodings of a hand-written template
([`build_clip_fixed_prompts`](../models/relation_classifier.py#L401)). The fourth
is learned — but what is learned is `self.ctx`, a set of context vectors
**shared across all classes**, plus a `meta_net` conditioned on the image. The
only class-specific input anywhere is the tokenised class **name**.

So H1's premise holds, in a stronger form than H1 states it: **for a novel
predicate the model has no learned per-class parameter at all.** Its entire
representation is CLIP's frozen text encoding of its name. If CLIP maps two
predicate names close together, no visual evidence can separate them.

At inference on the novel split the model scores **only** the 61 novel
predicates ([L511](../models/relation_classifier.py#L511):
`scores_[:, :, novel_pids] = pre_scores`), so what matters is separability
*within* those 61.

### B.11 H1's direction is REVERSED — measured on CPU

Reproduce: `python pilot_analysis/scripts/clip_text_separability.py`
(CLIP ViT-L/14@336px is cached at `~/.cache/clip`, and the text tower runs on CPU)

Given §B.10, H1 is checkable without running the model: encode all 132 predicate
names in the three real templates, and measure how separable they are. Cosine
nearest-neighbour similarity **within the 61 novel predicates** (higher = more
confusable):

| group ∩ novel | n | mean NN similarity |
|---|---|---|
| `geometric_dynamic` | 36 | **0.9599** ← least separable |
| `geometric_static` | 18 | 0.9513 |
| `appearance_static` | 2 | 0.9441 |
| `appearance_dynamic` | 5 | **0.9268** ← most separable |
| *control:* the 35 **object** names | 35 | **0.8560** |

**H1 predicts the appearance rows are worst. They are the best.** The ordering is
monotonically the opposite of H1's prediction.

The reason is visible in the worst offenders — it is the **compositional** names
that collapse, and the collapse is on the *spatial* component, not the verb:

```
fly_front    ~ fly_right     0.9798        move_away  ~ move_right   0.9795 (base)
jump_behind  ~ jump_past     0.9789        move_past  ~ move_behind  0.9783 (base)
move_past    ~ move_away     0.9788        jump_away  ~ jump_right   0.9750 (base)
creep_away   ~ creep_past    0.9752        run_toward ~ run_past     0.9696 (base)
```

**38 of 61 novel predicates have a base predicate above 0.95 similarity.** CLIP
essentially cannot tell `move_away` from `move_right`.

So the frozen-CLIP bottleneck H1 suspects is **real and severe** — predicate
names sit at 0.954 mean NN similarity against 0.856 for object names — but it
lands on the **spatial component of compositional names**, not on verbs.
Single-token action words (`bite`, `kick`, `fight`) are the *most* distinctive
things in the vocabulary. The problem is not that CLIP fails to understand
`bite`; it is that CLIP cannot separate `fly_front` from `fly_right`.

**Caveats, all load-bearing:**

- This measures the 3 frozen components, **not** the learned 4th. The learned
  context could in principle re-separate them, though it is class-agnostic.
- Text separability is a **necessary-condition proxy, not mAP**. Poor
  separability bounds what the model can do; good separability guarantees
  nothing.
- `appearance_dynamic` is **n=5** and `appearance_static` **n=2**. The
  appearance rows are weakly powered — the same problem as §C.2. The
  `geometric_dynamic` vs `geometric_static` contrast (36 vs 18) is the solid one.
- All cosines in CLIP text space are compressed near 1; the *ordering* and the
  object-name control carry the information, not the absolute values.

**Falsifiable prediction for the GPU run:** if this mechanism is right, Phase
2.3's confusion table will show novel-predicate errors concentrated **within the
same verb family** (`fly_front` mistaken for `fly_right`), not across verbs. That
is a sharp, pre-registered prediction — record it before running Phase 1.

### B.12 H2 quantified by simulation — and its criterion is measuring the wrong thing

Reproduce: `python pilot_analysis/scripts/merge_simulation.py`

GT instances are cut into 30-frame segments with perfect boxes, perfect
predicate and score 1.0; each segment is then dropped with probability *p* to
simulate one weak clip. Everything except the merge is exact, so the gap is
attributable to the merge rule alone. All mAP from the repo's evaluator.

| p(drop) | greedy `association()` | oracle merge | **gap** | fragmented ≥2 | lost entirely |
|---|---|---|---|---|---|
| 0.00 | 0.9966 | 0.9966 | 0.0000 | 1.6% | — |
| 0.05 | 0.8427 | 0.9549 | **0.1122** | 1.0% | — |
| 0.10 | 0.7337 | 0.9070 | **0.1733** | 0.8% | **22.9%** |
| 0.20 | 0.5683 | 0.8289 | **0.2606** | 0.3% | — |

**The greedy merge is extremely fragile.** Dropping 5% of segments costs
**11.2 mAP** against an oracle merge on the same damaged input; 10% costs
**17.3**. The protocol's H2 threshold is "oracle-merge gain ≥ 2 mAP" — this
simulation clears it by 5–13×, for any plausible drop rate. The mechanism is
confirmed as §B.5.1, break-at-first-gap.

Same simulation restricted to the **novel** predicate split — the pattern holds,
slightly smaller, consistent with novel instances being shorter (§B.2):

| p(drop) | greedy | oracle | gap |
|---|---|---|---|
| 0.05 | 0.8741 | 0.9651 | 9.1 mAP |
| 0.10 | 0.8033 | 0.9421 | 13.9 mAP |
| 0.20 | 0.6625 | 0.8771 | 21.5 mAP |

**But H2's other criterion will read "not supported" while this is happening.**
The protocol offers "or fragmentation affects ≥ 25% of long GT instances".
Measured fragmentation is **0.3–1.6%**. The greedy merge does not fragment — it
**truncates, and the truncated remnant then fails vIoU ≥ 0.5 against the full GT
extent, so it scores zero rather than counting as a fragment.** At p=0.10,
**22.9% of long GT instances are lost entirely** and only 0.8% are fragmented.

**Recommendation:** replace H2's fragmentation criterion with a **loss rate** —
the share of long GT instances with *zero* matching prediction. Reporting
fragmentation as specified would produce a false negative on a hypothesis the
first criterion supports strongly.

**Two things this simulation cannot tell you:**

- **max vs mean scoring is unresolved.** `oracle-mean` and `oracle-max` are
  identical to four decimals here because every simulated score is 1.0 — there
  is no variance to aggregate. §B.5.2 stands as an untested hypothesis until
  real scores exist.
- ***p* is a free parameter.** The real per-segment miss rate is unknown until
  Phase 1. The table gives the cost as a function of it, not the cost.

**A side benefit — §B.5.4 is now measured, not hypothesised.** At p=0.00 the
result is 0.9966, not 1.0: **218 of 4,835 instances are lost in 3 videos** where
predictions exceed `max_per_video`, whose default really is 200
([`parser_func.py:80`](../utils/parser_func.py#L80)). `format_`'s truncation is a
real loss point, worth ~0.34 mAP even with perfect predictions.

**Two bugs found and fixed while building this**, both worth knowing because the
real Phase 3 code would have hit them:

1. A fragmentation counter based on **triplet-string match plus any temporal
   overlap** reported 60.9% fragmentation with nothing dropped. VidVRD videos
   contain several object instances of the same category, so one triplet string
   names several concurrent GT instances and every prediction matches all of
   them. The count must use the evaluator's own criterion —
   `min(subject vIoU, object vIoU) >= 0.5` against *that* GT instance. This is
   the same reason §B.6's `gt2det_ids` matters.
2. Oracle merge over a **union extent with a hole** violates `format_`'s
   `assert len(sbj_traj) == duration[1] - duration[0]`. The hole is filled by
   linear interpolation between bracketing frames — a documented choice, since
   the union extent is what the protocol asks for.

### B.13 The Phase 1 dump wrapper — written, unit-tested, not yet run for real

`pilot_analysis/scripts/dump_predictions.py` — a wrapper around
`cli/common.py:_predict_split` that changes nothing in the repo and adds the two
dumps the protocol needs, so one GPU pass produces both files instead of two:

- `pilot_analysis/preds/final_merged_{all,novel}.json` — post-merge instances
- `pilot_analysis/preds/segments_raw_{all,novel}.json` — **pre-merge** per-segment
  predictions (`segment_index`, `frame_range`, `triplet`, `confidence`)
- `pilot_analysis/preds/metrics.json` — mAP/R@50/R@100 plus the exact config used

It has `--limit N` for the protocol's sanity gate, prints per-video timing and
**extrapolates the full-run duration** so the "ask before exceeding 2 hours" rule
can actually be applied.

**Attempted here and abandoned.** `--limit 1` was run with a 40-minute cap
(log local, `pilot_analysis/logs/dump-probe-1video.txt`): the model loaded in 9 minutes,
then **31 minutes inside video 1 without completing it**, killed by the cap.
`pilot_analysis/preds/` was never created. So this run measured throughput
(§A) but did **not** validate the dump code — it never reached it.

**Unit-tested instead**, which does not need a GPU:
`python pilot_analysis/scripts/test_dump_logic.py` drives synthetic
`pre_preds`/`pair_data` through the **real** `process_pred`, `extract_segments`,
`association()` and `format_`. 15 checks, all passing:

- one record per (clip, top-n) pair; `segment_index` covers every clip
- `frame_range` tiles the span with no overlap, and is **absolute** (offset by
  `begin_fid`), verified at offset 45 — the §B.3 grid case
- the ragged tail ends at the true end, not at a multiple of 30
- `format_` accepts the output and its own length assert holds
- everything is JSON-serialisable

**One thing the test caught that matters.** `association()` mutates `clip_rels`
in place — it rewrites `duration[1]` to the end of the merged chain and
overwrites `pre_scr` with the chain mean. Re-extracting after it runs yields
segments with spans of 60+ frames where the originals were 30, i.e. a silently
corrupted `segments_raw.json` that would look plausible. The wrapper dumps
before calling `association()`; the test now asserts that ordering is
load-bearing rather than incidental. (The first version of that assertion had
the logic backwards and failed — the code was right, the test was wrong.)

What remains unexercised is only the model-driven part: whether `pre_preds` and
`pair_data` arrive in the shapes assumed. That is what `--limit 5` on real
hardware is for.

### B.15 Phase 3.1's other half — and its stated expectation is wrong too

Reproduce: `python pilot_analysis/scripts/duration_by_predicate.py`
Outputs: `pilot_analysis/duration_by_predicate.csv` (132 rows),
`pilot_analysis/duration_histogram.png`

The protocol asks for "per-predicate mean duration (join with the partition —
**expectation: dynamic predicates last longer**)". Measured:

| group | preds | instances | mean duration | median | mean segments |
|---|---|---|---|---|---|
| `geometric_static` | 43 | 2,331 | 133.3 | 75 | 4.44 |
| `geometric_dynamic` | 78 | 2,183 | **75.0** | 60 | 2.50 |
| `appearance_static` | 4 | 179 | 114.0 | 75 | 3.80 |
| `appearance_dynamic` | 7 | 142 | **164.5** | 98 | **5.48** |
| **all static** | | 2,510 | **131.9** | 75 | 4.40 |
| **all dynamic** | | 2,325 | **80.5** | 60 | 2.68 |

**Dynamic predicates last 0.61× as long as static ones — the expectation is
refuted, and by a wide margin.** It is obvious in hindsight: `lie_next_to`
(304 frames), `taller` (200), `larger` (200) are persistent states that hold for
most of a video, while `walk_left` is a brief action.

**But the 2×2 shows what the binary split hides, again.** `appearance_dynamic`
is the **longest-lasting group of all** (164.5 frames, 5.48 segments), driven by
`play` at 205.6 frames. Collapsed into one "dynamic" bucket it is invisible
underneath 2,183 short `geometric_dynamic` instances.

**And that creates a confound H1 and H2 do not account for.** The
`appearance_dynamic` group is simultaneously:

- the group with the **weakest CLIP text prior** (H1's target — though §B.11
  says its *names* are actually the most distinctive, so the weakness must come
  from the visual side, not the text side), and
- the group with the **longest instances**, at 5.48 segments mean, hence the
  most exposed to the greedy merge's break-at-first-gap failure (H2's target).

So a poor `appearance_dynamic` result **cannot be attributed to H1 without first
controlling for H2**. Phase 4 (GT trajectories) does not fix this — it removes
detection error, not merge error. The clean design is to measure
`appearance_dynamic` under *oracle merge*, which isolates H1's contribution from
H2's. That comparison is available from the Phase 1 dumps at no extra GPU cost,
and it is not in the protocol.

### B.16 Phase 3.2(c) — half of H2's loss is mislocalisation, half is lost content

Reproduce: `python pilot_analysis/scripts/eval_at_threshold.py`
(~15 min on CPU; it re-runs the simulation twice)

**First, a blocker the protocol does not anticipate.**
`eval_relation_detection_openvoc` **hardcodes `viou_threshold=0.5`** at both of
its call sites
([L387, L389](../inference/video_relation_detection_openvoc.py#L387-L389)) and
exposes no parameter for it. The underlying `evaluate()` / `evaluate_v2()` do
accept the threshold, so Phase 3.2(c) ("evaluate with temporal-IoU threshold
relaxed") is impossible through the wrapper and needs its category-filtering
logic replicated with the threshold plumbed through.
`pilot_analysis/scripts/eval_at_threshold.py` does exactly that and nothing
else — the AP computation is still the repo's own `evaluate()`.

Swept over the merge simulation (§B.12):

| vIoU thr | greedy (p=.05) | oracle | gap | greedy (p=.10) | oracle | gap |
|---|---|---|---|---|---|---|
| **0.5** | 0.8427 | 0.9549 | 0.1122 | 0.7337 | 0.9070 | 0.1733 |
| 0.4 | 0.8730 | 0.9616 | 0.0886 | 0.7839 | 0.9230 | 0.1391 |
| 0.3 | 0.9041 | 0.9739 | 0.0697 | 0.8421 | 0.9480 | 0.1059 |
| 0.2 | 0.9201 | 0.9807 | 0.0606 | 0.8709 | 0.9648 | 0.0940 |
| 0.1 | 0.9248 | 0.9817 | 0.0569 | 0.8800 | 0.9680 | 0.0880 |

**Reading it, at p=0.10.** Greedy recovers from 0.7337 to 0.8800 — **14.6 mAP**
— purely by relaxing how precisely the temporal extent must match. Those
predictions were *there*, with the right triplet, just mislocalised in time. But
the gap to oracle never closes: **8.8 mAP remains even at vIoU 0.1**, which is
content the greedy merge genuinely destroyed rather than misplaced.

So the 17.3 mAP gap at the standard threshold splits roughly in half:

- **~8.5 mAP — temporal mislocalisation.** The prediction exists and is correct;
  its boundaries are wrong. Recoverable by better temporal decoding alone.
- **~8.8 mAP — lost content.** `association()` truncated at a gap and the
  material after it was never emitted at any threshold.

**This is the strongest argument yet for the research direction.** Roughly half
of H2's cost needs no better classifier and no better detector — only a temporal
extent that is *decoded* rather than produced by a greedy heuristic. That half is
addressable by a change confined to one function.

Same caveats as §B.12: *p* is a free parameter until Phase 1, and the simulation
has no score variance so it cannot speak to max-vs-mean aggregation.

---

### B.17 Phase 0's protocol check — done against the paper, and it found two things

The protocol's Phase 0 asks to "confirm which vocabulary split configuration the
config selects **and that it matches the paper's novel-split protocol**". Only
the first half had been done. The paper (`_paper/2409.12499v2.pdf`, p8) states it
exactly:

> "Training is performed on the base categories and testing is performed under
> two settings: (1) Novel-split evaluation involves **novel object categories for
> trajectory detection**, and **all object categories** along with novel
> relationship categories for relationship classification. (2) All-split
> evaluation involves all object categories and all relationship categories."

**Finding 1 — a possible code/paper mismatch on objects, worth settling at Phase 0.**
The paper says novel-split *relationship classification* uses **all** object
categories. In the code, the novel branch of `Model.forward`
([L511-L521](../models/relation_classifier.py#L511-L521)) restricts subject and
object scores to `novel_oids`:

```python
scores_ = torch.zeros([sbj_scores.shape[0], 35]).cuda()
scores_[:, self.obj_text_encoder.novel_oids] = sbj_scores
```

while `split_text_embeddings(split='novel')` likewise takes only the novel object
text embeddings. Meanwhile the evaluator is called with
`target_split_traj="all"` (the default), so GT and predictions are *filtered*
permissively on objects while the model can only emit novel ones.

Stated carefully, because this is a fast read of subtle code: **the code appears
to restrict object classification to novel categories where the paper says all.**
It is not proof of a bug — the intended reading of "all object categories" may
be about the evaluator's filter rather than the classifier's output space. But it
is exactly the question Phase 0 exists to answer, and it should be settled by
inspection before Phase 1 numbers are interpreted, since it bears directly on
the novel-split mAP.

**Finding 2 — Phase 4 is the paper's own SGCls/PredCls, and is already published.**
The paper evaluates three tasks (p8): SGDet (detect trajectories, then classify),
**SGCls** (classify objects *within provided GT trajectories*, then predict
relationships), and **PredCls** (GT trajectories *and* GT object categories
provided). Phase 4 — "bypass the trajectory detector, feed GT trajectories into
the relationship-classification stage" — **is SGCls/PredCls under another name.**

The paper's Table (p10) for EOV-MMP on VidVRD:

| split | SGDet mAP | SGCls mAP | PredCls mAP |
|---|---|---|---|
| novel | 15.04 | 17.96 | **21.65** |
| all | 26.34 | 31.95 | **39.83** |

**Which means the two-way attribution Phase 4 asks for is already computable,
with no compute at all** — and it comes out three-way:

| novel split | mAP | attributable to |
|---|---|---|
| SGDet | 15.04 | — |
| → SGCls | 17.96 | **+2.92 = trajectory detection error** |
| → PredCls | 21.65 | **+3.69 = object classification error** |
| ceiling | 21.65 | the remaining error is **relationship classification** |

On the all split: +5.61 detection, +7.88 object classification.

Two consequences:

1. **Phase 4 becomes a reproduction check, not a new measurement.** It has a
   published target (21.65 novel / 39.83 all) to validate the GT-trajectory
   substitution against — a much stronger gate than "does it run".
2. **§B.9's builder implements PredCls, not SGCls.** It assigns the GT
   `category` with `score = 1.0`, i.e. it supplies both trajectory *and* class.
   For SGCls you would supply GT boxes and let modelB classify them. Target the
   right number: **21.65 for PredCls-style GT trajectories**, not 17.96.

Also worth noting for the write-up: the paper's headline novel/all SGDet mAP of
15.04 / 26.34 is what §C.3's tolerance discussion is about, and the same table
confirms R@50/R@100 of 16.03/18.18 (novel) and 16.48/19.54 (all).

---

---

### B.20 MEASURED: stride 30 destroys the model. The stride question is settled.

First real GPU run, Colab T4, 2026-09-01. Sanity gate only -- `--limit 5`, the
same 5 videos in both conditions (the batch planner sorts by video id, so the cut
falls in the same place).

| | all mAP | novel mAP | s/video |
|---|---|---|---|
| `--frame_stride 30` | **~0.8** | ~0.0 | 74.6 |
| `--frame_stride 1` | **36.55** | **41.02** | 87.4 |

**~45x the accuracy for 17% more runtime.** The stride-30 figure is the printed
0.02 corrected for the scoring bug (it was averaged over all 200 GT videos while
only 5 had predictions); the stride-1 figures are scored correctly over the 5.

**The qualitative difference confirms the mechanism predicted in SS B.1/SS B.5.**
Sample instances from the two runs:

```
stride 30:  ['person','stand_above','bicycle']  duration=[60, 90]   len(sub_traj)=30
stride  1:  ['person','ride','bicycle']         duration=[0, 105]   len(sub_traj)=105
```

At stride 30 the output is a **single 30-frame segment, never merged**. That is
what identical per-frame CLIP tensors produce: the detector runs per frame on
`patch_[img_id]` and is deterministic, so boxes are constant across a block,
trajectories become step functions, and `association()` has nothing coherent to
chain. At stride 1 the same pair merges into a 105-frame instance and the
predicate shifts from a geometric guess (`stand_above`) to the semantically
correct `ride`.

This retires the prediction in SS A/SS E as **confirmed by measurement** rather
than by code reading.

**What it means for the paper.** The Implementation Details say frames are
sampled every 30; the released code has that sampling commented out. We now know
the commented-out state is not an oversight -- with these checkpoints stride 30
is unusable. Either the sentence describes training-time clip sampling, or the
released checkpoints were not produced as the paper describes. Worth asking the
author directly.

**What it means for the plan.** Stride 1 is the configuration. The feature-cache
idea (SS on HuggingFace staging) is therefore dead: at stride 1 a cached-feature
dataset is 100.3 GiB against 7.40 GiB of frames.

**Caveats, stated because the numbers are tempting.** Five videos, scored over
those five. The published 26.88 / 15.64 are over 200. 36.55 > 26.88 says nothing
about reproduction -- per-video AP varies enormously and novel (41.02) coming out
above all (36.55) is backwards from the usual pattern, which is what a 5-video
sample looks like. The only safe conclusion is that the pipeline runs correctly
end to end.

**Runtime.** 87.4 s/video at stride 1 -> 291 min per split, ~9.7 h for both. That
is more than a free Colab session and probably more than a day's quota. The
`all` and `novel` passes repeat the entire detector, tracker and CLIP pipeline
when only `modelC`'s text embeddings differ; fusing them would roughly halve it.

---

### B.18 A trap: `--test_traj gt` looks like Phase 4 and does nothing

`utils/parser_func.py` defines `--train_traj`, `--val_traj` and `--test_traj`,
**all defaulting to `"gt"`** ([L84-92](../utils/parser_func.py#L84-L92)), and the
config dump duly prints `test_traj='gt'`. Anyone reading that would reasonably
conclude the evaluation is already running on ground-truth trajectories, and
that Phase 4 is already done.

**It is not. Those three flags are dead.** Grepping the whole repo, no live code
reads any of them — the only other occurrence is a commented-out line in
`tools/upstream_data_scripts/gen_trajs.py`. The test path builds trajectories
from modelA + deep_sort + AFLink regardless of what these are set to.

So Phase 4 still requires the substitution in §B.9, and any Phase 1 number is a
*detected*-trajectory number despite what the logged config says. Worth stating
explicitly in the thesis, because the logged config is misleading on exactly the
point Phase 4 turns on.

### B.19 Partition signed off; H1 retired for H1'; power checked; one new confound

**Reviewed and confirmed 2026-08-27.** All five flagged predicates settled — two
moved, three kept. `review: true` is now empty in
`predicate_partition.json`; full table in
[`predicate_partition_table.md`](predicate_partition_table.md).

| predicate | change | inst |
|---|---|---|
| `touch` | appearance_dynamic → **appearance_static** | 38 base |
| `drive` | appearance_static → **geometric_static** | 3 novel |
| `play`, `feed`, `pull` | kept as assigned | 79 base / 9 / 3 novel |

Final counts (was → is):

| group | preds | base | novel | instances | share |
|---|---|---|---|---|---|
| `geometric_static` | 44 | 25 | 19 | 2,334 | 48.3% |
| `geometric_dynamic` | 78 | 42 | 36 | 2,183 | 45.1% |
| `appearance_static` | 4 | 3 | 1 | 214 | 4.4% |
| `appearance_dynamic` | 6 | 1 | 5 | 104 | 2.2% |

**H1 is retired and replaced by H1'**: test `geometric_dynamic` vs
`geometric_static` on both splits, where the mass (45%/48%) and the measured CLIP
degeneracy (§B.11: 0.9599 vs 0.9513 NN similarity) both are. Appearance groups
become descriptive with small-*n* caveats; no criterion depends on them.
Criterion: dynamic-group mAP ≤ 0.5 × static-group mAP, with §B.11's
within-verb-family confusion prediction as the mechanism check.

**Power check — H1' is properly powered, unlike H1.** Videos are the unit mAP
averages over:

| group | split | inst | videos | SE at p≈0.15 |
|---|---|---|---|---|
| `geometric_static` | base | 2,068 | **175** | 0.027 |
| `geometric_static` | novel | 266 | **76** | 0.041 |
| `geometric_dynamic` | base | 1,877 | **151** | 0.029 |
| `geometric_dynamic` | novel | 306 | **81** | 0.040 |
| *appearance_static* | *novel* | *8* | *8* | *0.126* |
| *appearance_dynamic* | *novel* | *25* | *20* | *0.080* |

**SE ~0.03–0.04, a 2–3× improvement on the old H1's ±0.08.** H1' can be decided.
The appearance rows confirm why they must stay descriptive.

**Two problems with the criterion, both new.**

**(1) The 2× threshold is a large leap from the mechanism evidence.** The measured
NN-similarity gap between the two groups is **0.0086** (0.9599 vs 0.9513). §B.11
supports a *direction*, not a magnitude, and with SE ≈ 0.04 the study can resolve
effects far smaller than 2×. A 2× bar therefore risks reporting "not supported"
on a real, well-measured effect. **Recommendation:** pre-register the
*direction* plus the ratio with a confidence interval, and treat ≤0.5× as one
labelled landmark rather than the pass/fail line.

**(2) A confound: label granularity, not verb semantics.** `geometric_dynamic`
spreads 2,183 instances over **78** predicates; `geometric_static` spreads 2,334
over **44**. That is roughly **half the data per label and nearly twice the
classes to separate**, before CLIP is implicated at all. Restricting to compound
`{verb}_{spatial}` predicates only does not remove it:

| group (compounds only) | preds | inst | inst/pred |
|---|---|---|---|
| `geometric_static` | 35 | 1,884 | **53.8** |
| `geometric_dynamic` | 72 | 2,106 | **29.2** |

If dynamic scores worse, granularity is a complete alternative explanation with a
different fix. **The vocabulary's own structure supplies the control**: every
compound predicate is one of 13 verb families × 8–11 spatial variants.

| dynamic families | variants | inst/variant | | static families | variants | inst/variant |
|---|---|---|---|---|---|---|
| `walk` | 11 | 92.5 | | `stand` | 9 | 118.2 |
| `move` | 11 | 31.1 | | `sit` | 8 | 49.6 |
| `swim` | 7 | 43.4 | | `lie` | 9 | 24.3 |
| `run` | 11 | 14.1 | | `stop` | 8 | 23.2 |
| `fly` | 10 | 11.1 | | | | |
| `jump` | 11 | 9.2 | | | | |
| `creep` | 10 | 7.4 | | | | |

**Recommendation: report AP per verb family, not only per group.** It holds
variant count roughly constant within each family, gives four matched pairs
(`stand`/`walk`, `sit`/`run`, `lie`/`fly`, `stop`/`creep`), and is organised
exactly the way §B.11's within-family confusion prediction needs to be tested.
`fall` (1 variant, 1 instance) should be excluded.

**Also updated by the reassignment:** `appearance_dynamic` mean duration rose to
188.5 frames / 6.28 segments (§B.15 rebuilt), making it still the
longest-lasting group and so still the most H2-exposed — the §B.15 confound
stands, now affecting only descriptive figures.

---

## C. Seven corrections the protocol needs before spending GPU time

### C.1 H1's static/dynamic split is the wrong cut for this model

`docs/20_Benchmark-analysis.md` already partitions the VidVRD test set by
*evidence required*, measured from the annotations:

| Evidence required | Instances | Share |
|---|---|---|
| Box trajectory only (`walk_left`, `run_past`) | 2,250 | 46.5% |
| Single frame + box geometry (`stand_behind`) | 2,013 | 41.6% |
| Box dimensions alone (`taller`) | 250 | 5.2% |
| Single frame appearance (`watch`, `ride`) | 179 | 3.7% |
| **Appearance change over time** (`bite`, `kick`, `fight`) | **143** | **3.0%** |

The protocol's binary `static_spatial` / `dynamic_kinematic` cut merges the
46.5% row with the 3.0% row. EOV-MMP **gets box trajectories for free**, so
those two behave nothing alike: predicates in the first row are decidable from
geometry the model already has, and only the last row needs the temporal visual
modelling H1 is really about. A binary partition will report "dynamic is worse"
and explain nothing.

**Recommendation — superseded, read this version.** An earlier draft of this
file proposed a three-way split (`static_spatial` / `trajectory_dynamic` /
`appearance_dynamic`). Building it (§B.7) showed the cut is really **two
independent binary axes**, and collapsing them to three groups loses the one
that matters:

| | **static** (one frame) | **dynamic** (needs time) |
|---|---|---|
| **geometric** (boxes suffice) | 48.2% | 45.1% |
| **appearance** (needs pixels) | 3.7% | 2.9% |

- The **evidence axis** asks whether the model *has* the information. EOV-MMP is
  handed box trajectories, so the whole geometric column — 93.3% of instances —
  is answerable from its input.
- The **time axis** asks whether temporal modelling is needed.

H1 ("CLIP is weak on verbs") is a claim about the **appearance column**, and
sharpest about `appearance_dynamic`. Report the 2×2. A three-way split still
merges `appearance_static` into a static bucket dominated by geometry, which
hides exactly the CLIP-verb-grounding signal H1 is after.

State the H1 decision criterion against `appearance_dynamic` specifically — but
first read §C.2, which is about whether it can be stated at all.

### C.2 H1's decision criterion cannot be tested on the novel split

This is the most consequential finding in this file and it was not anticipated
by the protocol.

The protocol's H1 criterion is *"H1 supported if dynamic-novel mAP ≤ 0.5 ×
static-novel mAP"*. Measured per group on the novel split, with the count of
test videos that contain at least one such instance — the unit mAP averages
over:

| group ∩ novel | instances | videos (of 200) |
|---|---|---|
| `geometric_static` | 263 | 76 (38.0%) |
| `geometric_dynamic` | 306 | 81 (40.5%) |
| `appearance_static` | 11 | **11 (5.5%)** |
| `appearance_dynamic` | **25** | **20 (10.0%)** |

**H1's cleanest target is 25 GT instances spread over 20 videos, and 15 of those
20 videos contain exactly one instance** (per-video counts:
`2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1`).

Per-video AP over a single GT instance is `1/rank` or 0. So
`appearance_dynamic__novel` mAP is an average of ~20 near-binary draws: at a
true value around 0.15 the standard error is roughly
`sqrt(0.15·0.85/20) ≈ 0.08` — **±8 mAP at one sigma**, against a criterion that
turns on a factor of two. The measurement cannot decide the hypothesis. Pooling
the whole appearance column on novel gives 36 instances over 28 videos, which is
not materially better.

**Recommendations, in order of preference:**

1. **Test H1 on the `base` split, where the power exists** (`appearance_dynamic`
   base: 117 instances, 47 videos; `appearance_static` base: 168 instances, 96
   videos), and treat the novel-split appearance numbers as descriptive.
   Open-vocabulary generalisation is then a *separate* base-vs-novel comparison
   within each group, which is a cleaner design anyway.
2. **Report the novel-split appearance figures with explicit intervals**, or as
   raw hit counts out of 25 rather than as an mAP, and drop the ≤0.5× criterion
   there.
3. Do **not** report `appearance_dynamic__novel` mAP as a bare number against a
   threshold. It will not replicate.

Worth stating plainly in the thesis: this is a **property of the benchmark, not
of the method**. VidVRD's novel split has essentially no appearance-dynamic
mass, so no experiment on it can settle whether CLIP's weakness on verbs is what
limits open-vocabulary VidVRD. That is a finding in its own right, and it argues
for either a different benchmark (VidOR) or a targeted re-annotation.

### C.3 The Phase 1 reproduction gate may trip on a working checkpoint

The protocol's gate is mAP 26.34 all / 15.04 novel, tolerance ±0.5. The
author's own reported numbers for the supplied checkpoint
`baseline_..._end2end_base-001.pth` are **26.88 all / 15.64 novel**
(`docs/10_Known-issues.md §1`) — **+0.54 and +0.60, both just outside
tolerance**. The second checkpoint, `..._train42-003.pth`, is further out
(24.33 / 13.70, epoch 2).

**Recommendation:** target the checkpoint's reported figures, or widen tolerance
to ±1.0, and say which in the writeup. Otherwise the gate fires a "hard stop on
reproduction failure" on a checkpoint that is behaving correctly.

### C.4 One further note

`docs/20_Benchmark-analysis.md` also measured that **no novel predicate on this
benchmark requires a genuinely unseen concept**: of 61 novel predicates, 44
(72.1%) have both `{verb}` and `{spatial}` components present in base, and all
61 have at least one. The split tests recombination, not vocabulary. This does
not invalidate H1, but it bounds what a positive H1 result means.

### C.5 H2's fragmentation criterion will produce a false negative

Measured in §B.12: the greedy merge costs 11–26 mAP against an oracle, while
fragmentation sits at 0.3–1.6% — far below the protocol's "≥ 25% of long GT
instances" bar. The merge truncates rather than fragments, and the remnant
scores zero instead of counting as a piece.

**Recommendation:** replace it with the **loss rate** (share of long GT
instances with no matching prediction at all), which was 22.9% at p=0.10 in
simulation. Keep the ≥2 mAP oracle-gain criterion as the primary test.

### C.6 H1 and H2 should be pre-registered before Phase 1 runs

§B.11 produces a sharp prediction (novel-predicate confusions fall *within* a
verb family, not across verbs) and §B.12 predicts the oracle-merge gain clears
2 mAP comfortably. Both were derived without touching the model. Writing them
down *now*, before Phase 1 produces any number, is what separates a prediction
from a post-hoc story — and §C.2 means the H1 result will be statistically
fragile enough that this matters.

---

## D. Not done

### D.1 Not done, needs a ≥16 GB GPU

| Protocol phase | What it needs |
|---|---|
| **Phase 0** sanity gate | Inference on 3–5 test videos; confirm shapes, no vocab mismatch |
| **Phase 0** `frame_stride` decision | **Settle this first — see §E** |
| **Phase 1** reproduction | Full inference over 200 test videos, both splits; save `preds/final_merged.json` and `preds/segments_raw.json` |
| **Phase 2.2** per-predicate AP | Needs Phase 1 predictions (the *evaluator* itself is CPU) |
| **Phase 2.3** confusion table | Needs Phase 1 predictions |
| **Phase 3.2** merge-loss (a)(b)(c) | Needs Phase 1 predictions, pre- and post-merge |
| **Phase 3.3** fragmentation count | Needs Phase 1 predictions |
| **Phase 4** GT-trajectory oracle | Inference with GT trajectories injected |

Hardware: 16 GB minimum; a T4 suffices. Colab works but its ephemeral disk must
re-decode 42 GB of frames each session (30–40 min per `README.md`); a rented box
with persistent disk avoids that.

### D.2 Not done, but doable on CPU here

*Everything that was on this list is now done — see §B.7 through §B.13. What
remains needs either you or a GPU.*

**Yours:** review the five flagged predicates in §B.7. `play` (79 base
instances) is the only one that can move a number, and §C.2 makes the base
split the one that matters.

**Nothing else on this machine is blocking.** The remaining CPU-side work —
running the Phase 2 group breakdown, the oracle-merge comparison and the
fragmentation/loss count — is written and tested (§B.8, §B.12); it needs Phase 1
predictions as input, nothing more.

### D.3 Not done, and explicitly out of scope

- **Nothing has been trained, and nothing can be.** The training code for
  steps 1–3 is unreleased; the supplied checkpoints are outputs, not a recipe
  (`docs/10_Known-issues.md §1`).
- **No number in this file came from running the model.** Every figure here is
  from annotations or source code.

---

## E. The one decision to make before the first GPU run

**`--frame_stride 1` vs `--frame_stride 30`.**

The paper's Implementation Details say frames are sampled every 30; the shipped
code encodes every frame. Over the 52,104 test frames × 2 encoders that is
**104,208 ViT-L/14@336 forwards at stride 1 versus ~3,474 at stride 30** — a
~30× difference in the cost of every subsequent phase. **Which one the released
checkpoints were trained with is unknown** (`docs/10_Known-issues.md §3`), and
the two do not give identical results.

Run 5 videos both ways in Phase 0, compare, commit to one, and record it — every
later number is a delta against that choice. Also print measured throughput
there, since no inference runtime for this repo has ever been measured.

---

## F. Reproducing everything in §B

```bash
conda activate repro-next            # or any Python 3.10+; these need only stdlib
cd <repo root>
# stdlib only -- any Python 3.10+
python pilot_analysis/scripts/gt_duration_stats.py          # §B.2, §B.3
python pilot_analysis/scripts/temporal_grid_ceiling.py      # §B.4
python pilot_analysis/scripts/build_predicate_partition.py  # §B.7

# need the project env (conda activate repro-next) -- still CPU only
python pilot_analysis/scripts/gt_trajectories.py            # §B.9
python pilot_analysis/scripts/clip_text_separability.py     # §B.11  (~2 min, CPU)
python pilot_analysis/scripts/merge_simulation.py           # §B.12  (~6 min, CPU)
python pilot_analysis/scripts/duration_by_predicate.py     # §B.15  (needs matplotlib)
python pilot_analysis/scripts/eval_at_threshold.py         # §B.16  (~15 min, CPU)
python pilot_analysis/scripts/test_dump_logic.py           # §B.13  (15 checks)

# needs a >=16GB GPU -- NOT run yet
python pilot_analysis/scripts/dump_predictions.py \
    --ckpt_path output/ckpt/baseline_fbce_vidvrd_bs1_lr1e-05_dim512_none_rel_mot_clip_bbox_end2end_base-001.pth \
    --path_AFLink output/ckpt/AFLink_epoch20.pth --limit 5   # §B.13, sanity gate
```

The first three import nothing but the standard library. The next three need
the project environment but still use **no GPU** — `clip_text_separability.py`
runs CLIP's text tower on CPU from the cache at `~/.cache/clip`.

§B.8 is the one exception — it imports the repo's evaluator, so it needs the
project environment (torch is imported transitively via `utils.paths`, but no
GPU is used):

```bash
conda activate repro-next
python - <<'EOF'
import json
from utils import paths
from inference.video_relation_detection_openvoc import eval_relation_detection_openvoc as ev
gt = json.load(open(paths.TEST_RELATION_GT))
pred = {v: [dict(r, score=1.0) for r in rels] for v, rels in gt.items()}
for g in ("geometric_static","geometric_dynamic","appearance_static","appearance_dynamic"):
    for sp in ("base","novel"):
        m, r = ev(target_split_traj="all", target_split_pred=f"{g}__{sp}",
                  pred_cls_split_info_path="pilot_analysis/splits/pred_split_group_x_ov.json",
                  prediction_results=pred)
        print(f"{g}__{sp:5} mAP {m:.4f}  R@50 {r[50]:.4f}  R@100 {r[100]:.4f}")
EOF
```
