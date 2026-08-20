# Research directions — Open-VidVRD

Compiled 2026-08-20. Companion to `OPEN_VIDVRD_LANDSCAPE.md`, which covers what
the five existing frameworks do and how they differ.

Every measurement below was computed from `data/vidvrd/annotations/` in this
workspace and can be re-run. Claims taken from papers are quoted.

---

## 0. What Phase-0 reading killed

The first idea considered was **motion-conditioned prompt tuning on CLIP** — on
the basis that RePro (ICLR'23) conditions prompts on motion and no CLIP-branch
method appeared to. Reading the two adjacent papers in full mostly closed it.

**OpenVidVRD** (arXiv 2503.09416, Mar 2025) already:

- extracts motion: *"for every subject-object trajectory pair we extract **motion
  patterns m_s,o encoding their relative positional dynamics and motion trends
  across timestamps**"*, aggregated as `M_s,o = phi_mot({m_s,o})`
- injects it into the spatial Transformer: `Ṽ_k = STran(f̃_vt + R_k + P_k + M_s,o)`
- uses **conditional prompts**: *"learnable continuous prompts ... and learnable
  conditional prompts (dynamically adapting to visual cues)"*
- explicitly discusses CoCoOp's Meta-Net and its limitations

**ICASSP'25 "Decomposed Prompt Learning and Relation Adjustment"** decomposes
predicates into **actional + spatial** patterns (`walk_left` -> action `walk` +
spatial `left`) and shares prompt components between base and novel. This is
*linguistic* decomposition of the label, not trajectory motion — so it does not
collide with motion directly, but it does claim the base→novel prompt
generalisation angle.

**Residual distinction, judged too thin to headline:** OpenVidVRD adds motion as
a *feature in the visual stream*; RePro's motion *selects which prompt bank* is
used. Real, but likely to read as incremental to a reviewer.

**Conclusion: prompting is crowded.** RePro, MMP, EOV-MMP, OpenVidVRD and the
ICASSP paper are all in it. A new prompting mechanism needs to clear five
existing ones.

---

## Idea 1 — The compositional gap nobody measures  ★ recommended

### The measurement

Computed from the annotations (train 800 / test 200 videos):

```
unique triplets            train = 2961    test = 1011
test triplets NEVER seen in train        = 258   (25.5% of test triplets)
their share of test instances            = 432/4835 = 8.9%
of those, all three components seen individually = 258  (100%)
```

Every unseen test triplet is a **pure composition failure**: the subject, the
predicate and the object each appear in training, just never together.

Split by predicate type:

```
unseen composition, predicate is NOVEL :  74 triplets
unseen composition, predicate is BASE  : 184 triplets   <-- the interesting half
                                          318 test instances
```

Examples where the model saw the predicate *and* both objects during training,
but never that combination:

```
<lizard     lie_behind    lizard   >   12 instances   [base predicate]
<lizard     creep_left    lizard   >   10 instances   [base]
<zebra      chase         zebra    >    5 instances   [base]
<monkey     creep_beneath monkey   >    4 instances   [base]
<red_panda  jump_left     red_panda>    4 instances   [base]
<dog        watch         monkey   >    4 instances   [base]
```

### Why this is a contribution

1. **Orthogonal to open-vocabulary.** These are not novel categories. Failure
   here is compositional, not vocabulary-driven. The entire field reports
   base/novel splits and **has never separated "novel category" from "novel
   composition"**.

2. **The field's founding paper names this problem and then doesn't test it.**
   RePro's abstract: *"conventional prompt tuning is easily biased to certain
   **subject-object combinations** and motion patterns."* Compositional bias is
   the stated motivation; the benchmark split does not measure it.

3. **New evaluation axis + diagnosis + method** is the shape of a top-venue
   paper, rather than a sixth prompting mechanism.

4. **Cheap to falsify.** Re-split the existing test set into
   composition-seen vs composition-unseen, run any existing method, report the
   gap. If the gap is large there is a paper; if small, that is known in days.

### Next steps if pursued

- build the split for VidVRD; repeat on VidOR (10k videos, richer combinations)
- confirm the gap is not explained by frequency alone (see §Idea 3 caveat)
- check the image-SGG literature for "zero-shot triplet" precedent and cite it
  honestly — the setting exists for images; the contribution is video +
  open-vocabulary

---

## Idea 2 — Fixed temporal granularity is unexamined

Every method inherits VidVRD-II's 2017 choice: **30-frame segments, stride 15**.
Verified across the codebases in this workspace (RePro `SEG_LEN, SEG_STRIDE =
30, 15`; MMP feature dirs named `train_gt_30`, `test_gt_30`).

But predicates have very different natural durations — `walk_past` is transient,
`stand_behind` persists. One granularity for all of them is an untested
assumption. Adaptive or multi-scale temporal windows appear in none of the five.

Cheap first experiment: measure the duration distribution of each predicate in
the annotations, and check how many relation instances are shorter than one
segment or span many.

---

## Idea 3 — The published numbers do not reconcile

Same method, same benchmark, different papers:

| Method | novel mAP per METOR | novel mAP per ICASSP'25 |
|---|---|---|
| RePro | 5.87 | 6.10 |
| MMP / OV-MMP | **12.15** | **16.56** |

A 4.4 mAP discrepancy for MMP — larger than most papers' entire claimed gain.
ICASSP's own result (19.90) would top METOR's 16.74 if the tables were
comparable; they are not.

Supporting evidence of protocol instability, all verified:

- **Model selection on the test set, field-wide.** RePro's `SegmentEvaluater`
  and MMP's `scripts/train.py` (two test datasets at lines 69-72; keep-best-mAP
  at 205-206).
- **METOR reports SGDet only**, arguing *"SGCls and PredCls rely on pre-detected
  trajectories"*. RePro reports all three.
- **MMP evaluates against both `meta` (detected) and `gt` trajectories**;
  RePro's in-training evaluator uses the detected path only.
- **A subset ground-truth file sat at the canonical path in this very repo**
  (20 videos where 200 were expected) until it was found and fixed on
  2026-08-20 — an illustration of how silently these numbers can drift.

A single-protocol re-evaluation of all five would be a benchmark/analysis
contribution. Lower ceiling than a method paper, higher citation potential, and
this workspace is unusually well equipped for it (four codebases, a verified
refactor, an equivalence harness).

**Caveat:** requires all five to actually run, which is a large engineering
commitment. Best treated as the analysis section of Idea 1 rather than the whole
paper.

---

## Idea 4 — Data efficiency of open-vocabulary transfer

VidVRD trains on 800 videos. Nobody reports how performance scales with the
amount of base supervision. If open-vocabulary transfer largely holds at 25% of
the base data, that is a useful, cheap result and a natural ablation to attach
to a larger paper.

---

## Idea 5 — Other observations that could seed something

- **base/novel is a SEMANTIC split, not a frequency split.** Measured:
  base 71 categories spanning 5..1504 instances; novel 61 spanning 2..339.
  Novel categories are *not* the rare ones (`sit_behind`=339 is novel;
  `fly_toward`=5 is base). So long-tail methods will not close this gap — worth
  stating explicitly, since it is easy to assume otherwise.
- **The association stage is crude and unexamined.** Cascaded methods stitch
  per-segment predictions with a greedy algorithm and a hand-set
  `linkage_threshold = 0.5`. Errors introduced there are never isolated.
- **METOR's stated future work is "audio and 3D information"** — flagged so it
  is not mistaken for open ground.

---

## Sources

Local: `repro/2302.00268v1.pdf`, `UASAN/3664647.3681061.pdf`,
`MMP_OV_VidVRD/AAAI_2024_...pdf`, `EOV-MMP-VidVRD/2409.12499v2.pdf`,
`METOR/2505.06663v1.pdf`, `_papers/2503.09416.pdf` (OpenVidVRD),
`_papers/2025_ICASSP.pdf` (Decomposed Prompt Learning).

External: [CoCoOp, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Conditional_Prompt_Learning_for_Vision-Language_Models_CVPR_2022_paper.pdf)
